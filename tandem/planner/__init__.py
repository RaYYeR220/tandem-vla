"""Natural language in, validated robot plan out.

    from tandem.planner import plan, replan
    result = plan("set the table for one", world)
    result = replan(result, world_after_a_failure, executed_step_ids=[1, 2, 3])

Two stages. A small language model does semantic parsing only — six short fields saying
what the user wants — and a deterministic expander turns that intent into steps. The
backend for stage A is chosen by ``TANDEM_PLANNER`` (``openvino`` by default, then
``cloud`` or ``rules``) and is always reported back in ``result["_meta"]``.
"""

from __future__ import annotations

from .backends import (
    CloudPlanner,
    OpenVINOPlanner,
    Planner,
    RuleBasedPlanner,
    describe_intent,
    get_planner,
    reset_planner,
)
from .expander import ExpansionError, expand
from .planner import extract_json, parse_intent, plan, replan
from .prompt import build_intent_prompt, build_prompt, summarize_world
from .schema import (
    INTENT_DEFAULTS,
    OBJECT_SLOT,
    PLACEABLE,
    SKILLS,
    SLOT_NAMES,
    STAGING,
    normalize_intent,
    repair_plan,
    validate_plan,
)

__all__ = [
    "plan",
    "replan",
    "parse_intent",
    "expand",
    "get_planner",
    "validate_plan",
    "repair_plan",
    "normalize_intent",
    "describe_intent",
    "build_intent_prompt",
    "build_prompt",
    "summarize_world",
    "extract_json",
    "reset_planner",
    "ExpansionError",
    "Planner",
    "OpenVINOPlanner",
    "CloudPlanner",
    "RuleBasedPlanner",
    "SKILLS",
    "SLOT_NAMES",
    "STAGING",
    "PLACEABLE",
    "OBJECT_SLOT",
    "INTENT_DEFAULTS",
]
