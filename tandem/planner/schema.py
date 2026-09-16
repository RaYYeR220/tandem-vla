"""Plan vocabulary, validation and repair.

The planner emits plans; the gate consumes them. This module is the shared contract
between the two: it owns the skill table from ``docs/INTERFACES.md``, a validator that
catches everything a plan can get wrong *symbolically*, and a repair pass that absorbs
the formatting slop a small instruction-tuned model produces.

Reachability is deliberately out of scope here — that is the gate's job, and it needs
live geometry to answer it.
"""

from __future__ import annotations

import re
from typing import Any

ARMS: tuple[str, ...] = ("left", "right")

#: Loose props plus the articulated drawer.
OBJECTS: tuple[str, ...] = ("plate", "mug", "bottle", "spoon", "fork", "drawer")

#: Place-setting targets on the table.
SLOT_NAMES: tuple[str, ...] = ("slot_plate", "slot_fork", "slot_spoon", "slot_mug")

#: A place target meaning "any clear spot on the table". The executor resolves the exact
#: pose; the planner only needs somewhere to put a thing down that is not a place setting,
#: which is how the bottle leaves the gripper after a pour.
STAGING: str = "staging"

#: Objects the user can ask to have laid out at the place setting.
PLACEABLE: tuple[str, ...] = ("plate", "fork", "spoon", "mug")

#: The slot each prop belongs in once the table is set.
OBJECT_SLOT: dict[str, str] = {
    "plate": "slot_plate",
    "fork": "slot_fork",
    "spoon": "slot_spoon",
    "mug": "slot_mug",
}

#: The only verbs a plan may contain. ``args`` is the exact, complete key set.
SKILLS: dict[str, dict[str, Any]] = {
    "open_drawer": {"args": ("arm",), "doc": "pull the drawer fully open"},
    "close_drawer": {"args": ("arm",), "doc": "push the drawer shut"},
    "pick": {"args": ("arm", "object"), "doc": "grasp and lift object"},
    "place": {"args": ("arm", "object", "target"), "doc": "put the held object at a slot"},
    "handoff": {
        "args": ("from_arm", "to_arm", "object"),
        "doc": "transfer the held object in the shared zone",
    },
    "hold": {"args": ("arm", "object"), "doc": "keep the held object still for the other arm"},
    "pour": {
        "args": ("arm", "source", "target"),
        "doc": "tilt source over target; the other arm must hold target",
    },
    "home": {"args": ("arm",), "doc": "retract to the rest pose"},
}

#: Argument keys whose value must name an arm.
_ARM_KEYS = ("arm", "from_arm", "to_arm")

#: Argument keys whose value must name an object.
_OBJECT_KEYS = ("object", "source")


def skill_table() -> str:
    """One line per skill, for prompts."""
    return "\n".join(
        f"{name}({', '.join(spec['args'])}) - {spec['doc']}" for name, spec in SKILLS.items()
    )


# --------------------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------------------


def _known_objects(world: dict | None) -> set[str]:
    if not world:
        return set(OBJECTS)
    known = set(world.get("objects", {}) or {})
    if "drawer" in world:
        known.add("drawer")
    return known or set(OBJECTS)


def _known_slots(world: dict | None) -> set[str]:
    """Valid ``place`` targets: the place setting, plus the open staging area."""
    if not world:
        return set(SLOT_NAMES) | {STAGING}
    return (set(world.get("slots", {}) or {}) or set(SLOT_NAMES)) | {STAGING}


def _known_arms(world: dict | None) -> set[str]:
    if not world:
        return set(ARMS)
    return set(world.get("arms", {}) or {}) or set(ARMS)


def _initial_grip(world: dict | None, arms: set[str]) -> dict[str, str | None]:
    grip: dict[str, str | None] = {a: None for a in arms}
    for arm, state in ((world or {}).get("arms", {}) or {}).items():
        if arm in grip and isinstance(state, dict):
            grip[arm] = state.get("holding")
    return grip


def validate_plan(plan: Any, world: dict | None = None) -> list[str]:
    """Return a list of human-readable problems with ``plan``. Empty means valid.

    Checks structure, the skill vocabulary, exact argument keys, known names, and runs a
    symbolic dry-run of gripper occupancy so a plan that picks with a full hand or pours
    with nobody holding the mug is rejected before it ever reaches the executor.
    """
    errors: list[str] = []

    if not isinstance(plan, dict):
        return [f"plan must be an object, got {type(plan).__name__}"]

    steps = plan.get("steps")
    if steps is None:
        return ["plan has no 'steps' key"]
    if not isinstance(steps, list):
        return [f"'steps' must be a list, got {type(steps).__name__}"]

    if not steps:
        # An empty plan is only meaningful as a refusal.
        if not plan.get("refusal"):
            errors.append("plan has no steps and no refusal")
        return errors

    arms = _known_arms(world)
    objects = _known_objects(world)
    slots = _known_slots(world)
    grip = _initial_grip(world, arms)

    seen_ids: set[Any] = set()
    for index, step in enumerate(steps):
        pos = index + 1
        if not isinstance(step, dict):
            errors.append(f"step {pos}: must be an object, got {type(step).__name__}")
            continue

        step_id = step.get("id")
        if step_id != pos:
            errors.append(f"step {pos}: id must be {pos} (ids are 1-based and contiguous), got {step_id!r}")
        if step_id in seen_ids:
            errors.append(f"step {pos}: duplicate id {step_id!r}")
        seen_ids.add(step_id)

        skill = step.get("skill")
        if skill not in SKILLS:
            errors.append(f"step {pos}: unknown skill {skill!r}")
            continue

        args = step.get("args")
        if not isinstance(args, dict):
            errors.append(f"step {pos}: '{skill}' needs an args object with keys {list(SKILLS[skill]['args'])}")
            continue

        expected = set(SKILLS[skill]["args"])
        got = set(args)
        for missing in sorted(expected - got):
            errors.append(f"step {pos}: '{skill}' is missing arg '{missing}'")
        for extra in sorted(got - expected):
            errors.append(f"step {pos}: '{skill}' got unexpected arg '{extra}'")
        if expected - got:
            continue

        # --- names ---
        bad_name = False
        for key in _ARM_KEYS:
            if key in args and args[key] not in arms:
                errors.append(f"step {pos}: {args[key]!r} is not an arm ({'/'.join(sorted(arms))})")
                bad_name = True
        for key in _OBJECT_KEYS:
            if key in args and args[key] not in objects:
                errors.append(f"step {pos}: unknown object {args[key]!r}")
                bad_name = True
        if skill == "place" and args.get("target") not in slots:
            errors.append(f"step {pos}: unknown slot {args.get('target')!r}")
            bad_name = True
        if skill == "pour" and args.get("target") not in objects:
            errors.append(f"step {pos}: unknown pour target {args.get('target')!r}")
            bad_name = True
        if bad_name:
            continue

        # --- symbolic gripper dry-run ---
        if skill == "pick":
            arm, obj = args["arm"], args["object"]
            if grip[arm] is not None:
                errors.append(f"step {pos}: the {arm} arm is already holding {grip[arm]!r}, it cannot pick {obj!r}")
            else:
                grip[arm] = obj
        elif skill == "place":
            arm, obj = args["arm"], args["object"]
            if grip[arm] != obj:
                errors.append(
                    f"step {pos}: the {arm} arm is holding {grip[arm]!r}, it cannot place {obj!r}"
                )
            else:
                grip[arm] = None
        elif skill == "handoff":
            src, dst, obj = args["from_arm"], args["to_arm"], args["object"]
            if src == dst:
                errors.append(f"step {pos}: handoff needs two different arms")
            elif grip[src] != obj:
                errors.append(
                    f"step {pos}: the {src} arm is holding {grip[src]!r}, it cannot hand off {obj!r}"
                )
            elif grip[dst] is not None:
                errors.append(
                    f"step {pos}: the {dst} arm is already holding {grip[dst]!r}, it cannot receive {obj!r}"
                )
            else:
                grip[src] = None
                grip[dst] = obj
        elif skill == "hold":
            arm, obj = args["arm"], args["object"]
            if grip[arm] != obj:
                errors.append(f"step {pos}: the {arm} arm is holding {grip[arm]!r}, it cannot hold {obj!r}")
        elif skill == "pour":
            arm, source, target = args["arm"], args["source"], args["target"]
            if grip[arm] != source:
                errors.append(
                    f"step {pos}: the {arm} arm is holding {grip[arm]!r}, it cannot pour from {source!r}"
                )
            others = [a for a in arms if a != arm]
            if not any(grip[a] == target for a in others):
                errors.append(f"step {pos}: nothing is holding {target!r}, the other arm must hold it to pour")
        elif skill in ("open_drawer", "close_drawer"):
            arm = args["arm"]
            if grip[arm] is not None:
                errors.append(
                    f"step {pos}: the {arm} arm is holding {grip[arm]!r}, it cannot work the drawer handle"
                )

    return errors


# --------------------------------------------------------------------------------------
# repair
# --------------------------------------------------------------------------------------

_ARM_ALIASES = {
    "a": "left",
    "arm_a": "left",
    "arma": "left",
    "arm a": "left",
    "l": "left",
    "left": "left",
    "left_arm": "left",
    "leftarm": "left",
    "left arm": "left",
    "b": "right",
    "arm_b": "right",
    "armb": "right",
    "arm b": "right",
    "r": "right",
    "right": "right",
    "right_arm": "right",
    "rightarm": "right",
    "right arm": "right",
}

#: Everyday words a model reaches for, mapped onto the scene vocabulary. Anything absent
#: here (a knife, a napkin) stays as written so the validator can reject it by name.
_OBJECT_ALIASES = {
    "cup": "mug",
    "mug": "mug",
    "coffee cup": "mug",
    "teacup": "mug",
    "glass": "mug",
    "dish": "plate",
    "plate": "plate",
    "dinner plate": "plate",
    "saucer": "plate",
    "bottle": "bottle",
    "water bottle": "bottle",
    "waterbottle": "bottle",
    "water": "bottle",
    "jug": "bottle",
    "pitcher": "bottle",
    "carton": "bottle",
    "spoon": "spoon",
    "teaspoon": "spoon",
    "tablespoon": "spoon",
    "fork": "fork",
    "drawer": "drawer",
    "cutlery drawer": "drawer",
}

_SKILL_ALIASES = {
    "pick": "pick",
    "pick_up": "pick",
    "pickup": "pick",
    "grasp": "pick",
    "grab": "pick",
    "take": "pick",
    "lift": "pick",
    "place": "place",
    "put": "place",
    "put_down": "place",
    "putdown": "place",
    "drop": "place",
    "release": "place",
    "handoff": "handoff",
    "hand_off": "handoff",
    "hand over": "handoff",
    "handover": "handoff",
    "transfer": "handoff",
    "pass": "handoff",
    "hold": "hold",
    "steady": "hold",
    "pour": "pour",
    "fill": "pour",
    "home": "home",
    "retract": "home",
    "rest": "home",
    "open_drawer": "open_drawer",
    "opendrawer": "open_drawer",
    "open drawer": "open_drawer",
    "open": "open_drawer",
    "close_drawer": "close_drawer",
    "closedrawer": "close_drawer",
    "close drawer": "close_drawer",
    "close": "close_drawer",
    "shut": "close_drawer",
}

_ARG_KEY_ALIASES = {
    "from": "from_arm",
    "to": "to_arm",
    "src_arm": "from_arm",
    "dst_arm": "to_arm",
    "source_arm": "from_arm",
    "target_arm": "to_arm",
    "obj": "object",
    "item": "object",
    "object_name": "object",
    "src": "source",
    "dst": "target",
    "destination": "target",
    "slot": "target",
    "goal": "target",
    "hand": "arm",
    "gripper": "arm",
}

#: Every key that may legitimately live inside ``args``, used to hoist a flattened step.
_ALL_ARG_KEYS = {k for spec in SKILLS.values() for k in spec["args"]}


def _norm(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_")


def _fix_arm(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    key = _norm(value)
    return _ARM_ALIASES.get(key, _ARM_ALIASES.get(key.replace("_", " "), value))


def _fix_object(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    key = _norm(value).replace("_", " ").strip()
    for article in ("the ", "a ", "an ", "my ", "some "):
        if key.startswith(article):
            key = key[len(article):]
            break
    return _OBJECT_ALIASES.get(key.strip(), value)


#: Ways a model refers to "just put it down somewhere out of the way".
_STAGING_WORDS = {"staging", "aside", "side", "counter", "worktop", "anywhere", "clear spot", "down"}


def _fix_slot(value: Any) -> Any:
    """Accept ``mug``, ``the mug slot`` or ``slot_mug`` and return a canonical slot name."""
    if not isinstance(value, str):
        return value
    key = _norm(value)
    if key in SLOT_NAMES:
        return key
    if key in _STAGING_WORDS:
        return STAGING
    key = key.replace("_", " ")
    if key in _STAGING_WORDS:
        return STAGING
    key = key[4:] if key.startswith("the ") else key
    key = key[:-5] if key.endswith(" slot") else key
    key = key[5:] if key.startswith("slot ") else key
    obj = _OBJECT_ALIASES.get(key, key)
    return OBJECT_SLOT.get(obj, value)


def _fix_id(value: Any, fallback: int) -> int:
    if isinstance(value, bool):
        return fallback
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        digits = "".join(ch for ch in value if ch.isdigit())
        if digits:
            return int(digits)
    return fallback


def repair_plan(plan: Any) -> dict:
    """Coerce common model-output slop into the plan schema. Never invents steps."""
    if isinstance(plan, list):
        plan = {"steps": plan}
    if not isinstance(plan, dict):
        return {"steps": []}

    _STEP_KEYS = ("id", "skill", "args", "rationale", "action")
    out: dict[str, Any] = {k: v for k, v in plan.items() if k not in ("steps",)}

    raw_steps = plan.get("steps")
    if not raw_steps:
        raw_steps = plan.get("plan") or plan.get("actions") or []
        out.pop("plan", None)
        out.pop("actions", None)
    if not raw_steps and ("skill" in plan or "action" in plan):
        # A model that answered with one bare step instead of a list. Re-read it as a
        # one-step plan; that is the step it wrote, not one we made up.
        raw_steps = [{k: v for k, v in plan.items() if k != "steps"}]
        for key in _STEP_KEYS:
            out.pop(key, None)
    if isinstance(raw_steps, dict):
        raw_steps = [raw_steps]
    if not isinstance(raw_steps, list):
        raw_steps = []

    steps: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_steps):
        if not isinstance(raw, dict):
            continue
        step = dict(raw)

        skill = step.get("skill", step.get("action", step.get("name")))
        step.pop("action", None)
        step.pop("name", None)
        if isinstance(skill, str):
            skill = _SKILL_ALIASES.get(_norm(skill), _norm(skill))
        step["skill"] = skill

        args = step.get("args")
        if not isinstance(args, dict):
            args = {}
        else:
            args = dict(args)
        # Hoist arguments a model flattened onto the step itself.
        for key in list(step):
            if key in ("id", "skill", "args", "rationale"):
                continue
            canonical = _ARG_KEY_ALIASES.get(_norm(key), _norm(key))
            if canonical in _ALL_ARG_KEYS:
                args.setdefault(canonical, step[key])
            step.pop(key, None)
        # Canonicalise key spellings inside args too.
        for key in list(args):
            canonical = _ARG_KEY_ALIASES.get(_norm(key), _norm(key))
            if canonical != key:
                args[canonical] = args.pop(key)

        for key in _ARM_KEYS:
            if key in args:
                args[key] = _fix_arm(args[key])
        for key in _OBJECT_KEYS:
            if key in args:
                args[key] = _fix_object(args[key])
        if "target" in args:
            args["target"] = _fix_object(args["target"]) if skill == "pour" else _fix_slot(args["target"])

        step["args"] = args
        step["id"] = _fix_id(step.get("id"), index + 1)
        rationale = step.get("rationale")
        step["rationale"] = rationale if isinstance(rationale, str) else ""
        steps.append({"id": step["id"], "skill": step["skill"], "args": step["args"], "rationale": step["rationale"]})

    # Renumber only when the model's own numbering is not already 1-based and contiguous.
    if [s["id"] for s in steps] != list(range(1, len(steps) + 1)):
        for index, step in enumerate(steps):
            step["id"] = index + 1

    out["steps"] = steps
    if "refusal" in out and not isinstance(out["refusal"], str):
        out.pop("refusal")
    for key in ("instruction", "goal_summary"):
        if key in out and not isinstance(out[key], str):
            out.pop(key)
    return out


# --------------------------------------------------------------------------------------
# intent — stage A's output, the only thing the language model is asked to produce
# --------------------------------------------------------------------------------------

#: Keys of a well-formed intent, with their default when the model omits one.
INTENT_DEFAULTS: dict[str, Any] = {
    "place": [],
    "pour": False,
    "open_drawer": None,
    "close_drawer": None,
    "refuse": None,
    "paraphrase": "",
}

_TRUE_WORDS = {"true", "yes", "y", "1", "on"}
_FALSE_WORDS = {"false", "no", "n", "0", "off"}
_NULL_WORDS = {"null", "none", "nil", "unspecified", "unknown", "n/a", ""}


def _tri_bool(value: Any) -> bool | None:
    """Coerce to True / False / None, because models write booleans as words."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        key = value.strip().lower()
        if key in _TRUE_WORDS:
            return True
        if key in _FALSE_WORDS:
            return False
        if key in _NULL_WORDS:
            return None
    return None


#: Small models treat ``refuse`` as a box that must be filled and write things like
#: "nothing to refuse" into it. Those are not refusals, they are the absence of one.
_EMPTY_REFUSAL = re.compile(r"^\s*(?:nothing\b|no\b|none\b|n/?a\b|-+|refuse\b)", re.IGNORECASE)


def _is_real_refusal(text: str) -> bool:
    stripped = text.strip()
    if not stripped or stripped.lower() in _NULL_WORDS:
        return False
    return not _EMPTY_REFUSAL.match(stripped)


def normalize_intent(raw: Any) -> dict[str, Any]:
    """Coerce whatever stage A produced into the intent schema.

    Like ``repair_plan`` this only ever reinterprets what the model wrote — it will not
    add an object to ``place`` that the model did not name.
    """
    intent = dict(INTENT_DEFAULTS)
    if not isinstance(raw, dict):
        return intent

    places = raw.get("place", raw.get("places", raw.get("objects", raw.get("items"))))
    if isinstance(places, str):
        places = [p for p in re.split(r"[,;/]| and ", places) if p.strip()]
    cleaned: list[str] = []
    if isinstance(places, list):
        for item in places:
            name = _fix_object(item)
            if name in PLACEABLE and name not in cleaned:
                cleaned.append(name)
    # Keep the canonical left-to-right setting order so plans read the same way every time.
    intent["place"] = [obj for obj in PLACEABLE if obj in cleaned]

    intent["pour"] = bool(_tri_bool(raw.get("pour", raw.get("fill"))))
    intent["open_drawer"] = _tri_bool(raw.get("open_drawer", raw.get("open")))
    intent["close_drawer"] = _tri_bool(raw.get("close_drawer", raw.get("close")))

    refuse = raw.get("refuse", raw.get("refusal"))
    if isinstance(refuse, str) and _is_real_refusal(refuse):
        intent["refuse"] = refuse.strip()

    paraphrase = raw.get("paraphrase", raw.get("summary", raw.get("goal_summary")))
    intent["paraphrase"] = paraphrase.strip() if isinstance(paraphrase, str) else ""
    return intent


def intent_is_empty(intent: dict) -> bool:
    """True when stage A understood no actionable request."""
    return (
        not intent.get("place")
        and not intent.get("pour")
        and not intent.get("open_drawer")
        and not intent.get("close_drawer")
        and not intent.get("refuse")
    )
