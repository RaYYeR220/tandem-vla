"""Stage-A prompt: semantic parsing only.

The language model is never asked to sequence a plan. Its whole job is to read one
instruction and say what the user wants — which items to lay out, whether to pour, what
to do with the drawer, and whether the request is impossible. That output is six short
fields, so a 1.5B (or a 0.5B) is working well inside its competence and generation stays
in the low hundreds of milliseconds instead of the tens of seconds a full plan costs.

Stage B turns the intent into steps deterministically. See ``expander.py``.
"""

from __future__ import annotations

from .schema import PLACEABLE, SLOT_NAMES, skill_table

#: Everything the model is told about the world. Deliberately a fixed inventory: what the
#: robot can reach changes turn by turn, but what *exists* does not, and reachability is
#: stage B's problem, not the model's.
SCENE = (
    "The robot owns exactly five things: plate, mug, bottle, spoon, fork. "
    "There is also a drawer. NOTHING ELSE EXISTS."
)

SYSTEM = f"""You read one instruction for a table-setting robot and report what the user wants.
{SCENE}
If the instruction needs any other object (knife, napkin, bowl, wine, food), fill in
"refuse". In every other case "refuse" is null.

Answer with ONE JSON object, no prose, these six keys:
"place": ONLY the items the instruction actually names, a subset of {list(PLACEABLE)}; [] if it names none
"pour": true if they want the mug filled with water, else false
"open_drawer": true, false, or null if they did not mention the drawer
"close_drawer": true, false, or null if they did not mention the drawer
"refuse": one short sentence when the request needs something that does not exist; else null
"paraphrase": one short line restating the goal

Never copy items from the examples. Read the instruction."""

EXAMPLES = """instruction: set the table for one
{"place": ["plate", "fork", "spoon", "mug"], "pour": false, "open_drawer": null, "close_drawer": null, "refuse": null, "paraphrase": "lay out the full place setting"}

instruction: just put the plate and the fork out
{"place": ["plate", "fork"], "pour": false, "open_drawer": null, "close_drawer": null, "refuse": null, "paraphrase": "lay out only the plate and the fork"}

instruction: fill my mug please
{"place": [], "pour": true, "open_drawer": null, "close_drawer": null, "refuse": null, "paraphrase": "pour water into the mug"}

instruction: fetch me the salt
{"place": [], "pour": false, "open_drawer": null, "close_drawer": null, "refuse": "there is no salt in the scene", "paraphrase": "fetch the salt"}"""


def build_intent_prompt(instruction: str) -> tuple[str, str]:
    """Return ``(system, user)`` for stage A. No world state — intent is world-independent."""
    return SYSTEM, f"{EXAMPLES}\n\ninstruction: {instruction}\n"


def repair_intent_prompt(instruction: str, problem: str) -> tuple[str, str]:
    """Retry stage A once, naming what went wrong with the first answer."""
    system, user = build_intent_prompt(instruction)
    return system, f"{user}\nYour previous answer was unusable ({problem}). Reply with only the JSON object.\n"


# --------------------------------------------------------------------------------------
# world summary — not part of the stage-A prompt any more, but the dashboard and the
# cloud backend's debugging view both want a readable one-liner per entity.
# --------------------------------------------------------------------------------------


def _xy(pos) -> str:
    try:
        return f"({float(pos[0]):.2f}, {float(pos[1]):.2f})"
    except (TypeError, ValueError, IndexError):
        return "(?, ?)"


def _reach(entity: dict) -> str:
    arms = entity.get("reachable_by") or []
    return ", ".join(arms) if arms else "neither arm"


def summarize_world(world: dict | None) -> str:
    """One compact line per object, slot and arm. No raw float dumps."""
    if not world:
        return "world unknown"

    lines: list[str] = []

    for name, obj in (world.get("objects") or {}).items():
        bits = [f"{name} at {_xy(obj.get('pos', ()))}"]
        if obj.get("held_by"):
            bits.append(f"held by {obj['held_by']}")
        if obj.get("in_drawer"):
            bits.append("in the drawer")
        if obj.get("on_slot"):
            bits.append(f"on {obj['on_slot']}")
        bits.append(f"reachable by: {_reach(obj)}")
        lines.append("; ".join(bits))

    drawer = world.get("drawer")
    if isinstance(drawer, dict):
        state = "open" if drawer.get("is_open") else "closed"
        lines.append(f"drawer {state}; reachable by: {_reach(drawer)}")

    for name, slot in (world.get("slots") or {}).items():
        occupied = slot.get("occupied_by")
        mark = f"holds {occupied}" if occupied else "free"
        lines.append(f"{name} {mark}, reachable by: {_reach(slot)}")

    for name, arm in (world.get("arms") or {}).items():
        holding = arm.get("holding") or "nothing"
        lines.append(f"{name} arm holds {holding}")

    water = world.get("water")
    if isinstance(water, dict):
        lines.append(
            f"water: {water.get('in_bottle', 0)} in bottle, {water.get('in_mug', 0)} in mug"
        )

    return "\n".join(lines)


def build_prompt(instruction: str, world: dict | None = None) -> tuple[str, str]:
    """Back-compat alias for the stage-A prompt. ``world`` is ignored by design."""
    return build_intent_prompt(instruction)


__all__ = [
    "build_intent_prompt",
    "repair_intent_prompt",
    "build_prompt",
    "summarize_world",
    "SCENE",
    "SYSTEM",
    "EXAMPLES",
    "SLOT_NAMES",
    "skill_table",
]
