"""Stage-A backends: local OpenVINO, cloud API, and a zero-dependency keyword parser.

All three satisfy the same tiny protocol — ``generate(system, user) -> str`` plus
``info()`` — so the rest of the stack never learns which one is running. What it *can*
learn is the truth about it: ``info()`` always reports the backend that actually produced
the text, and every call records its own latency and throughput.

Since the restructure these backends only ever produce an *intent* — six short fields —
never a plan. Step sequencing is ``expander.py``'s job and never involves a model.

The keyword parser exists so the demo has no hard dependency on a model download or an
API key: a judge can clone the repo and watch the arms set the table immediately.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .schema import PLACEABLE, normalize_intent

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IR_DIR = REPO_ROOT / "models" / "planner-ov"

#: The smaller export. Intent parsing is easy enough that this is often the better demo
#: default; ``benchmark.py`` is what decides, not taste.
SMALL_IR_DIR = REPO_ROOT / "models" / "planner-ov-0.5b"


@runtime_checkable
class Planner(Protocol):
    """Anything that can turn a (system, user) pair into raw model text."""

    def generate(self, system: str, user: str) -> str: ...

    def info(self) -> dict[str, Any]: ...


class _BasePlanner:
    """Shared latency bookkeeping."""

    def __init__(self) -> None:
        self.last_latency_ms: float = 0.0
        self.last_tokens_per_s: float = 0.0
        self.last_output_tokens: int = 0

    def set_context(self, instruction: str, world: dict | None) -> None:
        """Optional hook: backends that parse from structured state override this."""

    def metrics(self) -> dict[str, float]:
        return {
            "latency_ms": round(self.last_latency_ms, 2),
            "tokens_per_s": round(self.last_tokens_per_s, 2),
            "output_tokens": self.last_output_tokens,
        }


# --------------------------------------------------------------------------------------
# 1. OpenVINO — the default
# --------------------------------------------------------------------------------------


def _detect_precision(ir_dir: Path) -> str:
    """Report the weight format actually baked into the IR, not what we hoped for."""
    config = ir_dir / "openvino_config.json"
    if config.exists():
        try:
            data = json.loads(config.read_text(encoding="utf-8"))
            dtype = data.get("dtype") or data.get("quantization_config", {}).get("dtype")
            bits = data.get("quantization_config", {}).get("bits")
            if isinstance(dtype, str) and dtype:
                return dtype.upper()
            if bits:
                return f"INT{bits}"
        except (json.JSONDecodeError, OSError, AttributeError):
            pass
    xml = ir_dir / "openvino_model.xml"
    if xml.exists():
        try:
            with xml.open("r", encoding="utf-8", errors="ignore") as fh:
                head = fh.read(400_000)
            for marker, name in (('"u4"', "INT4"), ('"i4"', "INT4"), ('"nf4"', "NF4"),
                                 ('"u8"', "INT8"), ('"i8"', "INT8")):
                if f"element_type={marker}" in head:
                    return name
            if 'element_type="f16"' in head:
                return "FP16"
        except OSError:
            pass
    return "unknown"


class OpenVINOPlanner(_BasePlanner):
    """Local inference through ``openvino_genai.LLMPipeline`` on an exported IR."""

    def __init__(self, ir_dir: str | Path | None = None, device: str | None = None) -> None:
        super().__init__()
        import openvino_genai as ov_genai  # imported lazily so the rules path stays free

        self.ir_dir = Path(ir_dir or os.environ.get("TANDEM_PLANNER_MODEL") or DEFAULT_IR_DIR)
        self.device = device or os.environ.get("TANDEM_OV_DEVICE", "AUTO")
        # An intent is six short fields. Capping generation here is most of why stage A
        # costs a few hundred milliseconds instead of the best part of a minute.
        self.max_new_tokens = int(os.environ.get("TANDEM_PLANNER_MAX_TOKENS", "128"))
        self._genai = ov_genai
        self._pipe = ov_genai.LLMPipeline(str(self.ir_dir), self.device)
        self._precision = _detect_precision(self.ir_dir)
        self._model_name = self.ir_dir.name

    def _chat_prompt(self, system: str, user: str) -> str:
        """Apply the model's chat template, falling back to the ChatML layout Qwen uses."""
        try:
            tokenizer = self._pipe.get_tokenizer()
            return tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                add_generation_prompt=True,
            )
        except Exception:  # noqa: BLE001 - template support varies across IR exports
            return (
                f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{user}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

    def generate(self, system: str, user: str) -> str:
        config = self._genai.GenerationConfig()
        config.max_new_tokens = self.max_new_tokens
        config.do_sample = False
        config.repetition_penalty = 1.05

        started = time.perf_counter()
        result = self._pipe.generate(self._chat_prompt(system, user), config)
        self.last_latency_ms = (time.perf_counter() - started) * 1000.0

        text = str(result)
        try:
            tokens = int(result.perf_metrics.get_num_generated_tokens())
        except Exception:  # noqa: BLE001 - perf metrics are optional on some builds
            tokens = max(1, len(text) // 4)
        self.last_output_tokens = tokens
        self.last_tokens_per_s = tokens / max(self.last_latency_ms / 1000.0, 1e-6)
        return text

    def info(self) -> dict[str, Any]:
        return {
            "backend": "openvino",
            "model": self._model_name,
            "device": self.device,
            "precision": self._precision,
        }


# --------------------------------------------------------------------------------------
# 2. Cloud — OpenAI-compatible fallback
# --------------------------------------------------------------------------------------


class CloudPlanner(_BasePlanner):
    """Any OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 45.0,
    ) -> None:
        super().__init__()
        self.base_url = (base_url or os.environ.get("TANDEM_LLM_BASE_URL")
                         or "https://api.venice.ai/api/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("TANDEM_LLM_API_KEY", "")
        self.model = model or os.environ.get("TANDEM_LLM_MODEL", "qwen3-4b")
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError("TANDEM_LLM_API_KEY is not set; the cloud planner cannot run")

    def generate(self, system: str, user: str) -> str:
        import requests

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
            "max_tokens": 200,
        }
        started = time.perf_counter()
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            timeout=self.timeout,
        )
        self.last_latency_ms = (time.perf_counter() - started) * 1000.0
        response.raise_for_status()
        body = response.json()

        text = body["choices"][0]["message"]["content"] or ""
        usage = body.get("usage") or {}
        self.last_output_tokens = int(usage.get("completion_tokens") or max(1, len(text) // 4))
        self.last_tokens_per_s = self.last_output_tokens / max(self.last_latency_ms / 1000.0, 1e-6)
        return text

    def info(self) -> dict[str, Any]:
        return {
            "backend": "cloud",
            "model": self.model,
            "device": self.base_url,
            "precision": "remote",
        }


# --------------------------------------------------------------------------------------
# 3. Rules — no model, no network
# --------------------------------------------------------------------------------------

#: Nouns people reach for that simply are not on this table. Matching one is a refusal.
_ABSENT = (
    "knife", "knives", "napkin", "bowl", "chopstick", "candle", "salt", "pepper",
    "wine", "beer", "coffee", "tea", "sugar", "butter", "bread", "apple", "banana",
    "towel", "phone", "book", "scissors", "kettle", "pan", "pot", "straw", "tray",
    "soup", "cereal", "milk", "juice", "cake", "egg", "eggs", "sandwich",
)

#: Everyday words for the four things that can go on the place setting.
_PLACE_WORDS = {
    "plate": "plate", "plates": "plate", "dish": "plate", "saucer": "plate",
    "mug": "mug", "mugs": "mug", "cup": "mug",
    "spoon": "spoon", "spoons": "spoon", "teaspoon": "spoon",
    "fork": "fork", "forks": "fork",
}

_SET_TABLE = re.compile(r"\b(?:set|lay)\s+(?:the\s+|a\s+)?table\b|\bplace\s+setting\b|\btable\s+for\s+\w+\b")
_POUR = re.compile(r"\b(?:pour|fill|refill|top\s*up)\b|\bsome\s+water\b|\bwater\s+(?:in|into)\b")
_OPEN_DRAWER = re.compile(r"\bopen\w*\b[^.]{0,20}\bdrawer\b|\bdrawer\b[^.]{0,20}\bopen\b")
_CLOSE_DRAWER = re.compile(r"\b(?:close|shut)\w*\b[^.]{0,20}\bdrawer\b|\bdrawer\b[^.]{0,20}\b(?:close|shut)\w*\b")


class RuleBasedPlanner(_BasePlanner):
    """Deterministic stage-A substitute: keywords and regexes, no weights, no network."""

    def __init__(self) -> None:
        super().__init__()
        self._instruction = ""
        self._world: dict | None = None

    def set_context(self, instruction: str, world: dict | None) -> None:
        self._instruction = instruction
        self._world = world

    def generate(self, system: str, user: str) -> str:
        instruction = self._instruction or _instruction_from_prompt(user)
        started = time.perf_counter()
        intent = self.parse_intent(instruction)
        self.last_latency_ms = (time.perf_counter() - started) * 1000.0
        text = json.dumps(intent)
        self.last_output_tokens = max(1, len(text) // 4)
        self.last_tokens_per_s = self.last_output_tokens / max(self.last_latency_ms / 1000.0, 1e-6)
        return text

    def parse_intent(self, instruction: str) -> dict[str, Any]:
        """Same six fields the language model is asked for, from keywords instead."""
        text = (instruction or "").lower().strip()

        absent = _first_absent(text)
        if absent:
            return normalize_intent({
                "refuse": f"there is no {absent} in the scene",
                "paraphrase": instruction.strip(),
            })

        pour = bool(_POUR.search(text))
        named = [
            _PLACE_WORDS[word]
            for word in re.findall(r"[a-z]+", text)
            if word in _PLACE_WORDS
        ]
        if _SET_TABLE.search(text) and not named:
            named = list(PLACEABLE)
        if pour:
            # The pour leg fetches and returns the mug itself.
            named = [obj for obj in named if obj != "mug"]
        place = [obj for obj in PLACEABLE if obj in named]

        open_drawer = True if _OPEN_DRAWER.search(text) else None
        close_drawer = True if _CLOSE_DRAWER.search(text) else None

        intent = {
            "place": place,
            "pour": pour,
            "open_drawer": open_drawer,
            "close_drawer": close_drawer,
            "refuse": None,
            "paraphrase": "",
        }
        if not place and not pour and open_drawer is None and close_drawer is None:
            intent["refuse"] = "I could not turn that into anything these two arms can do"
        intent["paraphrase"] = describe_intent(intent)
        return normalize_intent(intent)

    def info(self) -> dict[str, Any]:
        return {
            "backend": "rules",
            "model": "keyword-parser",
            "device": "CPU",
            "precision": "n/a",
        }


def describe_intent(intent: dict) -> str:
    """One readable line for the dashboard."""
    if intent.get("refuse"):
        return intent["refuse"]
    bits = []
    if intent.get("place"):
        bits.append("lay out " + ", ".join(intent["place"]))
    if intent.get("pour"):
        bits.append("fill the mug with water")
    if intent.get("open_drawer"):
        bits.append("open the drawer")
    if intent.get("close_drawer"):
        bits.append("close the drawer")
    return "; ".join(bits) or "nothing to do"


def absent_object(text: str) -> str | None:
    """Name the first thing an instruction asks for that is not in the scene.

    The inventory is fixed and known, so deciding whether a request is impossible is a
    vocabulary lookup, not a judgement call. Stage A asks the model for a refusal as
    well, but this is what makes the answer reliable — a 1.5B would rather be helpful
    than admit a knife does not exist.
    """
    for noun in _ABSENT:
        if re.search(rf"\b{noun}s?\b", (text or "").lower()):
            return noun
    return None


#: Reads better at the call site inside the keyword parser.
_first_absent = absent_object


def _instruction_from_prompt(user: str) -> str:
    """Recover the instruction from a rendered prompt, for standalone ``generate`` use."""
    matches = re.findall(r"instruction:\s*(.+)", user)
    return matches[-1].strip() if matches else user.strip()


# --------------------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------------------

_CACHE: dict[str, Any] = {}


def reset_planner() -> None:
    """Drop the cached backend, e.g. after changing the environment in a test."""
    _CACHE.clear()


def get_planner(choice: str | None = None, *, cache: bool = True) -> Planner:
    """Pick a backend from ``TANDEM_PLANNER`` (``openvino`` | ``cloud`` | ``rules``).

    Falls back to the keyword parser with a warning when the local IR is missing or the
    cloud credentials are absent. It never pretends: ``info()`` reports what ran.
    """
    choice = (choice or os.environ.get("TANDEM_PLANNER") or "openvino").strip().lower()
    if cache and choice in _CACHE:
        return _CACHE[choice]

    backend: Any
    if choice == "rules":
        backend = RuleBasedPlanner()
    elif choice == "cloud":
        try:
            backend = CloudPlanner()
        except Exception as exc:  # noqa: BLE001 - missing key or requests
            log.warning("cloud planner unavailable (%s); falling back to the keyword parser", exc)
            backend = RuleBasedPlanner()
    else:
        ir_dir = Path(os.environ.get("TANDEM_PLANNER_MODEL") or DEFAULT_IR_DIR)
        if not (ir_dir / "openvino_model.xml").exists():
            log.warning(
                "no OpenVINO IR at %s; falling back to the keyword parser "
                "(run scripts/export_planner.py to enable the local LLM)",
                ir_dir,
            )
            backend = RuleBasedPlanner()
        else:
            try:
                backend = OpenVINOPlanner(ir_dir)
            except Exception as exc:  # noqa: BLE001 - runtime, driver or IR problems
                log.warning("OpenVINO planner failed to start (%s); falling back to rules", exc)
                backend = RuleBasedPlanner()

    if cache:
        _CACHE[choice] = backend
    return backend
