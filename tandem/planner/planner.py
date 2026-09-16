"""The planner entry point: two stages, only one of which involves a model.

Stage A asks a small language model a small question — what does the user want? — and
gets six short fields back. Stage B expands that intent into steps in plain Python.

Splitting it this way is what makes a 1.5B (or a 0.5B) usable here. Asking one to emit a
sixteen-step plan produced confident nonsense; asking it to fill in six fields is well
inside what it can do, and generation drops from tens of seconds to a few hundred
milliseconds. Everything that has to be *correct* is deterministic.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from .backends import (
    Planner,
    RuleBasedPlanner,
    absent_object,
    describe_intent,
    get_planner,
)
from .expander import ExpansionError, expand
from .prompt import build_intent_prompt, repair_intent_prompt
from .schema import normalize_intent, validate_plan

log = logging.getLogger(__name__)

_FENCE_OPEN = "```"


# --------------------------------------------------------------------------------------
# JSON recovery
# --------------------------------------------------------------------------------------


def extract_json(text: str) -> dict | list | None:
    """Pull the first complete JSON value out of raw model output.

    Tolerates ```json fences, leading chatter and trailing commentary by scanning for a
    balanced brace run rather than trusting the model to stop cleanly.
    """
    if not text:
        return None

    candidate = text
    if _FENCE_OPEN in candidate:
        start = candidate.find(_FENCE_OPEN) + len(_FENCE_OPEN)
        if candidate[start:start + 4].lower().startswith("json"):
            start += 4
        end = candidate.find(_FENCE_OPEN, start)
        candidate = candidate[start:end] if end != -1 else candidate[start:]

    stripped = candidate.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    for opener, closer in (("{", "}"), ("[", "]")):
        for value in _iter_balanced(stripped, opener, closer):
            return value
    return None


def _iter_balanced(text: str, opener: str = "{", closer: str = "}"):
    """Yield every top-level balanced JSON value in ``text``, in order."""
    index = 0
    length = len(text)
    while index < length:
        start = text.find(opener, index)
        if start == -1:
            return
        depth = 0
        in_string = False
        escaped = False
        end = -1
        for cursor in range(start, length):
            char = text[cursor]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    end = cursor
                    break
        if end == -1:
            return
        try:
            yield json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
        index = end + 1


#: Last resort when the model emitted an unterminated object: read the fields we can see.
_FIELD = re.compile(
    r'"(place|pour|open_drawer|close_drawer|refuse|paraphrase)"\s*:\s*'
    r'(\[[^\]]*\]|"(?:[^"\\]|\\.)*"|true|false|null)'
)


def salvage_intent_fields(text: str) -> dict | None:
    """Recover intent keys from truncated or malformed JSON."""
    found: dict[str, Any] = {}
    for key, raw in _FIELD.findall(text or ""):
        try:
            found[key] = json.loads(raw)
        except json.JSONDecodeError:
            continue
    return found or None


# --------------------------------------------------------------------------------------
# stage A
# --------------------------------------------------------------------------------------


def scene_guard(instruction: str, intent: dict) -> dict:
    """Overrule stage A when the instruction names something that is not in the scene.

    Small instruction-tuned models are relentlessly helpful: asked for a knife they will
    happily plan to fetch one. The inventory is fixed, so this check is a lookup rather
    than a judgement, and it belongs in code for the same reason step sequencing does.
    """
    missing = absent_object(instruction)
    if missing and not intent.get("refuse"):
        intent = dict(intent)
        intent["refuse"] = f"there is no {missing} in the scene"
        intent["place"] = []
        intent["pour"] = False
    return intent


def parse_intent(
    instruction: str,
    *,
    backend: Planner | str | None = None,
    max_attempts: int = 2,
    guard: bool = True,
) -> tuple[dict, dict]:
    """Ask the model what the user wants. Returns ``(intent, meta)``.

    ``meta`` carries the backend identity, the wall-clock cost and whether the answer had
    to be salvaged, so the caller can tell a clean parse from a rescued one. ``guard``
    applies the scene inventory check on top of the model's answer; turn it off to
    measure the model on its own.
    """
    engine: Planner = backend if hasattr(backend, "generate") else get_planner(
        backend if isinstance(backend, str) else None
    )
    setter = getattr(engine, "set_context", None)
    if callable(setter):
        setter(instruction, None)

    system, user = build_intent_prompt(instruction)
    elapsed_ms = 0.0
    attempts = 0
    problem = ""
    intent: dict | None = None
    salvaged = False

    for attempt in range(max(1, max_attempts)):
        attempts = attempt + 1
        started = time.perf_counter()
        try:
            raw = engine.generate(system, user)
        except Exception as exc:  # noqa: BLE001 - a dead backend must not kill the demo
            elapsed_ms += (time.perf_counter() - started) * 1000.0
            log.warning("stage A backend failed: %s", exc)
            problem = f"backend error: {exc}"
            break
        elapsed_ms += (time.perf_counter() - started) * 1000.0

        parsed = extract_json(raw)
        if not isinstance(parsed, dict):
            rescued = salvage_intent_fields(raw)
            if rescued:
                parsed, salvaged = rescued, True
        if isinstance(parsed, dict):
            intent = normalize_intent(parsed)
            if guard:
                intent = scene_guard(instruction, intent)
            break

        problem = "no JSON object in the output"
        if attempt + 1 < max(1, max_attempts):
            log.info("stage A produced no intent, retrying")
            system, user = repair_intent_prompt(instruction, problem)

    info = engine.info()
    meta = {
        "backend": info.get("backend"),
        "model": info.get("model"),
        "device": info.get("device"),
        "precision": info.get("precision"),
        "stage_a_ms": round(elapsed_ms, 2),
        "tokens_per_s": round(getattr(engine, "last_tokens_per_s", 0.0), 2),
        "attempts": attempts,
        "salvaged": salvaged,
        "problem": problem or None,
    }
    return (intent if intent is not None else {}), meta


# --------------------------------------------------------------------------------------
# stage A + stage B
# --------------------------------------------------------------------------------------


def plan(
    instruction: str,
    world: dict | None = None,
    *,
    backend: Planner | str | None = None,
    max_repairs: int = 2,
    fallback_to_rules: bool = True,
) -> dict:
    """Parse the instruction, then expand it into a validated plan against ``world``.

    A refusal short-circuits: ``{"steps": [], "refusal": ..., "_meta": {...}}``. Otherwise
    the returned plan is always valid, because stage B checks its own output.

    ``fallback_to_rules`` covers stage A coming back empty or unusable — the keyword
    parser takes over. It is never hidden: ``_meta.backend`` then reads ``rules`` and
    ``_meta.fell_back_from`` names the backend that could not answer.
    """
    intent, meta = parse_intent(instruction, backend=backend, max_attempts=max_repairs)

    fell_back_from = None
    if fallback_to_rules and meta.get("backend") != "rules" and not _usable(intent):
        log.info("stage A returned nothing usable; parsing with keywords instead")
        rules = RuleBasedPlanner()
        intent = rules.parse_intent(instruction)
        fell_back_from = meta.get("backend")
        meta.update(rules.info())

    if not intent:
        intent = normalize_intent({})
    if not intent.get("paraphrase"):
        intent["paraphrase"] = describe_intent(intent)

    errors: list[str] = []
    if not _usable(intent):
        # Stage A came back with nothing to act on. Say so rather than shipping a plan
        # that just parks both arms and calls it done.
        problem = meta.get("problem") or "I did not understand that instruction"
        result = {"steps": [], "refusal": problem}
        intent["refuse"] = problem
        result["instruction"] = instruction
        result["goal_summary"] = intent.get("paraphrase", "")
        result["_meta"] = _build_meta(meta, intent, errors, fell_back_from)
        return result

    if intent.get("refuse"):
        result: dict[str, Any] = {"steps": [], "refusal": intent["refuse"]}
    else:
        try:
            result = expand(intent, world)
        except ExpansionError as exc:
            # Stage B failing its own check is a bug in us, never in the model. Surface it
            # loudly rather than shipping a plan the executor cannot run.
            log.error("%s", exc)
            errors = [str(exc)]
            result = {"steps": [], "refusal": "the planner could not build a safe sequence"}
        if not result.get("steps") and not result.get("refusal"):
            result["refusal"] = "there is nothing left to do"

    result["instruction"] = instruction
    result.setdefault("goal_summary", intent.get("paraphrase", ""))
    result["_meta"] = _build_meta(meta, intent, errors, fell_back_from)
    return result


def replan(plan_dict: dict, world: dict | None, executed_step_ids: list[int] | None = None) -> dict:
    """Rebuild the remaining work from the *current* world, reusing the parsed intent.

    The executor calls this after a failed step and after a voice barge-in. Stage A does
    not run again: the user's goal has not changed, only the state of the table has.
    """
    meta = dict((plan_dict or {}).get("_meta") or {})
    intent = normalize_intent(meta.get("intent") or {})
    instruction = (plan_dict or {}).get("instruction", "")
    executed = list(executed_step_ids or [])

    if intent.get("refuse"):
        result: dict[str, Any] = {"steps": [], "refusal": intent["refuse"]}
        errors: list[str] = []
    else:
        errors = []
        try:
            result = expand(intent, world)
        except ExpansionError as exc:
            log.error("%s", exc)
            errors = [str(exc)]
            result = {"steps": [], "refusal": "the planner could not rebuild a safe sequence"}
        if not result.get("steps") and not result.get("refusal"):
            result["refusal"] = "there is nothing left to do"

    result["instruction"] = instruction
    result.setdefault("goal_summary", intent.get("paraphrase", ""))
    # Stage A did not run, so its cost is zero for this plan; keep the identity of the
    # backend that originally parsed the intent.
    carried = {**meta, "stage_a_ms": 0.0, "attempts": 0}
    result["_meta"] = _build_meta(carried, intent, errors, meta.get("fell_back_from"))
    result["_meta"]["replanned"] = True
    result["_meta"]["executed_step_ids"] = executed
    return result


def _usable(intent: dict | None) -> bool:
    """True when stage A said something actionable, rather than shrugging."""
    if not intent:
        return False
    return bool(
        intent.get("place")
        or intent.get("pour")
        or intent.get("refuse")
        or intent.get("open_drawer")
        or intent.get("close_drawer")
    )


def _build_meta(source: dict, intent: dict, errors: list[str], fell_back_from: str | None) -> dict:
    return {
        "backend": source.get("backend"),
        "model": source.get("model"),
        "device": source.get("device"),
        "precision": source.get("precision"),
        "stage_a_ms": round(float(source.get("stage_a_ms") or 0.0), 2),
        "latency_ms": round(float(source.get("stage_a_ms") or 0.0), 2),
        "tokens_per_s": source.get("tokens_per_s", 0.0),
        "attempts": source.get("attempts", 0),
        "salvaged": bool(source.get("salvaged")),
        "intent": intent,
        "refused": bool(intent.get("refuse")),
        "fell_back_from": fell_back_from,
        "valid": not errors,
        "errors": errors,
        "replanned": False,
    }


__all__ = [
    "plan",
    "replan",
    "parse_intent",
    "extract_json",
    "salvage_intent_fields",
    "validate_plan",
]
