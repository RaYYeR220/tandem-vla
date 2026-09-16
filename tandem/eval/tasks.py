"""Task definitions and the reference plan used to grade the stack.

`canonical_plan` is the deterministic expansion of a goal into skills, written against the
current world state. It is what the rule-based planner produces and what the language planner
is graded against, and the runner re-derives it whenever a step fails, which is how re-planning
works: the plan is always a function of the scene as it is now, never of the scene it was
written for.
"""

from __future__ import annotations

from ..sim import layout

#: Instruction phrasings used by the evaluation, with the intent each should parse to.
INSTRUCTIONS = [
    ("set the table for one", {"place": ["plate", "fork", "spoon", "mug"], "pour": False}),
    ("set the table and pour me some water",
     {"place": ["plate", "fork", "spoon", "mug"], "pour": True}),
    ("put the plate and the fork out", {"place": ["plate", "fork"], "pour": False}),
    ("just the mug please, and fill it", {"place": ["mug"], "pour": True}),
    ("lay out the cutlery", {"place": ["fork", "spoon"], "pour": False}),
]

#: Instructions that must be refused, with the reason the refusal has to name.
REFUSAL_CASES = [
    ("put the knife next to the plate", "knife"),
    ("pour the wine into the glass", "wine"),
    ("fold the napkin", "napkin"),
    ("put the bowl on the table", "bowl"),
]

DEFAULT_ORDER = ("plate", "fork", "spoon", "mug")


def _fetch(obj: str, world: dict, step_id: int) -> list[dict]:
    """Steps that move one object from wherever it is onto its slot.

    Written to be re-entrant: this is what the runner calls again after a failure, so it has to
    start from wherever the object actually is now -- on its slot already, in a gripper, or on
    the table. Emitting a `pick` for something the arm is already holding is the single most
    common way a re-plan turns one failed step into five.
    """
    info = world["objects"][obj]
    slot = layout.OBJECT_SLOT[obj]
    if info["on_slot"] == slot and info["held_by"] is None:
        return []
    if info["held_by"] is not None:
        return _fetch_from_hand(obj, info["held_by"], world, step_id)
    src_arm = (info["reachable_by"] or ["left"])[0]
    dst_arm = (world["slots"][slot]["reachable_by"] or ["right"])[0]
    steps = [
        {"id": step_id, "skill": "pick", "args": {"arm": src_arm, "object": obj},
         "rationale": f"the {obj} is in the {src_arm} arm's reach"},
    ]
    if dst_arm != src_arm:
        steps.append(
            {"id": step_id + 1, "skill": "handoff",
             "args": {"from_arm": src_arm, "to_arm": dst_arm, "object": obj},
             "rationale": f"{slot} is out of the {src_arm} arm's envelope"}
        )
    steps.append(
        {"id": step_id + len(steps), "skill": "place",
         "args": {"arm": dst_arm, "object": obj, "target": slot},
         "rationale": f"put the {obj} on its place setting"}
    )
    return steps


def canonical_plan(world: dict, intent: dict | None = None) -> dict:
    """Expand an intent into a gated, state-aware skill sequence."""
    intent = intent or {"place": list(DEFAULT_ORDER), "pour": True}
    wanted = [o for o in DEFAULT_ORDER if o in set(intent.get("place", []))]
    pour = bool(intent.get("pour"))
    steps: list[dict] = []
    nid = 1

    # Anything held that this plan has no further use for is put down first, so the drawer and
    # the next pick are not blocked by a gripper that is still full from a failed step.
    for arm_name, arm_info in world["arms"].items():
        held = arm_info["holding"]
        if held and held not in wanted and not (pour and held in ("mug", "bottle")):
            steps.append({"id": nid, "skill": "place",
                          "args": {"arm": arm_name, "object": held, "target": "staging"},
                          "rationale": f"the {arm_name} arm still has the {held} from an "
                                       f"interrupted step"})
            nid += 1

    if any(world["objects"][o]["in_drawer"] for o in wanted) and not world["drawer"]["is_open"]:
        drawer_arm = (world["drawer"]["reachable_by"] or ["left"])[0]
        if world["arms"][drawer_arm]["holding"]:
            steps.append({"id": nid, "skill": "place",
                          "args": {"arm": drawer_arm,
                                   "object": world["arms"][drawer_arm]["holding"],
                                   "target": "staging"},
                          "rationale": "the handle needs a free hand"})
            nid += 1
        steps.append({"id": nid, "skill": "open_drawer", "args": {"arm": drawer_arm},
                      "rationale": "the cutlery is inside it"})
        nid += 1

    # The mug is fetched last when we are going to pour into it, so it is already on its slot
    # and steady before the carton comes over.
    order = [o for o in wanted if not (pour and o == "mug")]
    for obj in order:
        block = _fetch(obj, world, nid)
        steps.extend(block)
        nid += max(1, len(block))

    if pour:
        # Both the carton and the mug start inside the left arm's envelope, so the left arm is
        # necessarily the one doing the pouring. That forces the order: the mug goes across to
        # the right arm first, which then steadies it while the left arm fetches and tips the
        # carton. Getting this wrong is the single most expensive planning mistake in the task —
        # the gate catches it, but only after a wasted pick.
        carton_arm = (world["objects"]["bottle"]["reachable_by"] or ["left"])[0]
        holder_arm = layout.other_arm(carton_arm)
        mug_from = (world["objects"]["mug"]["reachable_by"] or ["left"])[0]
        if world["objects"]["mug"]["held_by"] != holder_arm:
            steps.append(
                {"id": nid, "skill": "pick", "args": {"arm": mug_from, "object": "mug"},
                 "rationale": "the mug has to be in the steadying arm before anything is poured"}
            )
            nid += 1
            if mug_from != holder_arm:
                steps.append(
                    {"id": nid, "skill": "handoff",
                     "args": {"from_arm": mug_from, "to_arm": holder_arm, "object": "mug"},
                     "rationale": f"only the {carton_arm} arm can reach the carton, so the "
                                  f"{holder_arm} arm holds the mug"}
                )
                nid += 1
        steps.extend(
            [
                {"id": nid, "skill": "pick",
                 "args": {"arm": carton_arm, "object": "bottle"},
                 "rationale": "fetch the carton with the arm that can reach it"},
                {"id": nid + 1, "skill": "hold",
                 "args": {"arm": holder_arm, "object": "mug"},
                 "rationale": "hold the mug still while it is filled"},
                {"id": nid + 2, "skill": "pour",
                 "args": {"arm": carton_arm, "source": "bottle", "target": "mug"},
                 "rationale": "fill the mug"},
                {"id": nid + 3, "skill": "place",
                 "args": {"arm": carton_arm, "object": "bottle", "target": "staging"},
                 "rationale": "set the carton down clear of the setting"},
            ]
        )
        nid += 4
        if "mug" in wanted:
            steps.extend(_fetch_from_hand("mug", holder_arm, world, nid))
            nid += 2

    steps.append({"id": nid, "skill": "home", "args": {"arm": "left"}, "rationale": "stand down"})
    steps.append({"id": nid + 1, "skill": "home", "args": {"arm": "right"},
                  "rationale": "stand down"})

    return {
        "instruction": intent.get("instruction", ""),
        "goal_summary": _summary(wanted, pour),
        "steps": steps,
    }


def _fetch_from_hand(obj: str, arm: str, world: dict, step_id: int) -> list[dict]:
    """Put an object the arm is already holding onto its slot, handing over if it must."""
    slot = layout.OBJECT_SLOT[obj]
    dst = (world["slots"][slot]["reachable_by"] or ["right"])[0]
    if dst == arm:
        return [{"id": step_id, "skill": "place",
                 "args": {"arm": arm, "object": obj, "target": slot},
                 "rationale": f"put the {obj} on its place setting"}]
    return [
        {"id": step_id, "skill": "handoff",
         "args": {"from_arm": arm, "to_arm": dst, "object": obj},
         "rationale": f"{slot} is out of the {arm} arm's envelope"},
        {"id": step_id + 1, "skill": "place",
         "args": {"arm": dst, "object": obj, "target": slot},
         "rationale": f"put the {obj} on its place setting"},
    ]


def _summary(wanted: list[str], pour: bool) -> str:
    if not wanted and not pour:
        return "nothing to do"
    bits = []
    if wanted:
        bits.append(", ".join(wanted) + " on the place setting")
    if pour:
        bits.append("water in the mug")
    return "; ".join(bits)


def make_replanner(env, intent: dict | None = None):
    """Runner hook: rebuild the remaining plan from the live world state."""

    def replan(world: dict, done_ids: list[int]) -> dict:
        return canonical_plan(world, intent)

    return replan
