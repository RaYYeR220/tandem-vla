"""Stage B: turn an intent into a step sequence, deterministically.

All the multi-step reasoning lives here, in plain Python, where it is testable and cannot
hallucinate. The expander is *state-aware*: it reads the current world, skips whatever is
already done, and emits only the work that remains. That is what makes it safe to re-run
after every executed step, which is exactly how replanning works — there is no separate
"resume" path, just expand again against the newer world.

It also checks its own homework: ``expand`` runs ``validate_plan`` over what it produced
and raises if that fails. A plan this module cannot validate is a bug in this module.
"""

from __future__ import annotations

from typing import Any

from .schema import (
    OBJECT_SLOT,
    PLACEABLE,
    STAGING,
    normalize_intent,
    validate_plan,
)

#: Used only when the world dict omits an entity. Mirrors the reset layout: the props
#: start in left-arm territory and the place setting is in right-arm territory.
_FALLBACK_OBJECT_REACH = ["left"]
_FALLBACK_SLOT_REACH = ["right"]


class ExpansionError(RuntimeError):
    """The expander produced a plan that does not validate. Always our bug."""


def _other(arm: str) -> str:
    return "right" if arm == "left" else "left"


class _Expansion:
    """Mutable symbolic state while the step list is being built."""

    def __init__(self, world: dict | None) -> None:
        self.world = world or {}
        objects = self.world.get("objects") or {}
        self.objects = objects

        self.reach_obj: dict[str, list[str]] = {}
        self.on_slot: dict[str, str | None] = {}
        self.in_drawer: dict[str, bool] = {}
        for name in ("plate", "mug", "bottle", "spoon", "fork"):
            spec = objects.get(name) or {}
            self.reach_obj[name] = list(spec.get("reachable_by") or []) or list(_FALLBACK_OBJECT_REACH)
            self.on_slot[name] = spec.get("on_slot")
            # A reset scene has the cutlery in the drawer; trust the world when it says so.
            default_in_drawer = name in ("spoon", "fork") and not objects
            self.in_drawer[name] = bool(spec.get("in_drawer", default_in_drawer))

        self.reach_slot: dict[str, list[str]] = {}
        slots = self.world.get("slots") or {}
        for slot in OBJECT_SLOT.values():
            spec = slots.get(slot) or {}
            self.reach_slot[slot] = list(spec.get("reachable_by") or []) or list(_FALLBACK_SLOT_REACH)

        drawer = self.world.get("drawer") or {}
        self.drawer_arms = list(drawer.get("reachable_by") or []) or ["left"]
        self.drawer_open = bool(drawer.get("is_open", False))
        self.opened_drawer = False

        arms = self.world.get("arms") or {}
        self.holding: dict[str, str | None] = {
            "left": (arms.get("left") or {}).get("holding"),
            "right": (arms.get("right") or {}).get("holding"),
        }
        # A mug that already has water in it does not need filling again — that matters
        # when the executor replans after a step that happened to come after the pour.
        self.water_in_mug = float((self.world.get("water") or {}).get("in_mug", 0) or 0)
        self.steps: list[dict[str, Any]] = []

    def already_placed(self, obj: str, slot: str) -> bool:
        """True when the object is resting on its slot, by either of the two signals."""
        if self.on_slot.get(obj) == slot:
            return True
        spec = (self.world.get("slots") or {}).get(slot) or {}
        return spec.get("occupied_by") == obj and self.holder_of(obj) is None

    # -- emit --------------------------------------------------------------------------

    def add(self, skill: str, args: dict[str, Any], rationale: str) -> None:
        self.steps.append(
            {"id": len(self.steps) + 1, "skill": skill, "args": args, "rationale": rationale}
        )

    def holder_of(self, obj: str) -> str | None:
        return next((arm for arm, held in self.holding.items() if held == obj), None)

    def free_arms(self, preferred: list[str] | None = None) -> list[str]:
        order = preferred if preferred else ["left", "right"]
        return [arm for arm in order if self.holding.get(arm) is None]

    # -- primitives --------------------------------------------------------------------

    def ensure_drawer_open(self, why: str) -> bool:
        if self.drawer_open:
            return True
        arm = next(iter(self.free_arms(self.drawer_arms)), None)
        if arm is None:
            # Both hands full: put one thing down in the clear so a hand is free.
            arm = self.stage_something(self.drawer_arms)
            if arm is None:
                return False
        self.add("open_drawer", {"arm": arm}, why)
        self.drawer_open = True
        self.opened_drawer = True
        return True

    def close_drawer(self, why: str) -> None:
        arm = next(iter(self.free_arms(self.drawer_arms)), None)
        if arm is None:
            arm = self.stage_something(self.drawer_arms)
        if arm is None:
            return
        self.add("close_drawer", {"arm": arm}, why)
        self.drawer_open = False

    def stage_something(self, preferred: list[str]) -> str | None:
        """Free a gripper by setting whatever it holds down in the clear."""
        for arm in (preferred or []) + ["left", "right"]:
            held = self.holding.get(arm)
            if held:
                self.place(arm, held, STAGING, f"free the {arm} gripper")
                return arm
        return None

    def place(self, arm: str, obj: str, target: str, why: str) -> None:
        self.add("place", {"arm": arm, "object": obj, "target": target}, why)
        self.holding[arm] = None
        self.in_drawer[obj] = False
        if target == STAGING:
            # Wherever it went down, the arm that put it there can still reach it.
            self.reach_obj[obj] = [arm]
            self.on_slot[obj] = None
        else:
            self.reach_obj[obj] = list(self.reach_slot.get(target) or _FALLBACK_SLOT_REACH)
            self.on_slot[obj] = target

    def acquire(self, obj: str, dest_arms: list[str], why: str) -> str | None:
        """End with ``obj`` in the hand of an arm that can reach ``dest_arms``."""
        if self.in_drawer.get(obj) and not self.drawer_open:
            if not self.ensure_drawer_open(f"the {obj} is inside the drawer"):
                return None

        holder = self.holder_of(obj)
        if holder is None:
            source_arms = self.reach_obj.get(obj) or _FALLBACK_OBJECT_REACH
            # Prefer an arm that reaches the object *and* the destination: no hand-off.
            direct = [a for a in source_arms if a in dest_arms and self.holding.get(a) is None]
            pick_arm = direct[0] if direct else next(iter(self.free_arms(source_arms)), None)
            if pick_arm is None:
                pick_arm = self.stage_something(source_arms)
                if pick_arm is None or pick_arm not in source_arms:
                    return None
            self.add("pick", {"arm": pick_arm, "object": obj}, why)
            self.holding[pick_arm] = obj
            holder = pick_arm

        if holder in dest_arms:
            return holder

        receiver = _other(holder)
        if self.holding.get(receiver) is not None:
            if self.stage_something([receiver]) != receiver:
                return None
        self.add(
            "handoff",
            {"from_arm": holder, "to_arm": receiver, "object": obj},
            f"the {holder} arm cannot reach the target, the {receiver} arm can",
        )
        self.holding[holder] = None
        self.holding[receiver] = obj
        return receiver

    # -- composites --------------------------------------------------------------------

    def lay_out(self, obj: str) -> None:
        slot = OBJECT_SLOT[obj]
        if self.already_placed(obj, slot):
            return  # already done; nothing to replan
        dest = self.reach_slot.get(slot) or _FALLBACK_SLOT_REACH
        arm = self.acquire(obj, dest, f"fetch the {obj}")
        if arm is not None:
            self.place(arm, obj, slot, f"{obj} goes at {slot}")

    def pour_into_mug(self) -> None:
        """Mug steadied in one hand, bottle tilted by the other, then both put down."""
        mug_slot = OBJECT_SLOT["mug"]
        slot_arms = self.reach_slot.get(mug_slot) or _FALLBACK_SLOT_REACH
        bottle_arms = self.reach_obj.get("bottle") or _FALLBACK_OBJECT_REACH

        if self.water_in_mug > 0:
            # The pour already happened. All that can be left is putting things down.
            self.lay_out("mug")
            bottle_holder = self.holder_of("bottle")
            if bottle_holder:
                self.place(bottle_holder, "bottle", STAGING, "put the bottle back down in the clear")
            return

        # The holder must be able to set the full mug down afterwards, so it has to be an
        # arm that reaches slot_mug; the pourer is then whatever is left.
        pourer = next((a for a in bottle_arms if _other(a) in slot_arms), None)
        if pourer is None:
            pourer = bottle_arms[0]
        holder = _other(pourer)

        if self.acquire("mug", [holder], "the mug must be steadied while it fills") != holder:
            return
        if self.holding.get(pourer) is not None and self.stage_something([pourer]) != pourer:
            return
        if self.holder_of("bottle") != pourer:
            if self.holder_of("bottle") is not None:
                return
            self.add("pick", {"arm": pourer, "object": "bottle"}, "take the water bottle")
            self.holding[pourer] = "bottle"

        self.add("hold", {"arm": holder, "object": "mug"}, "keep the mug still")
        self.add(
            "pour",
            {"arm": pourer, "source": "bottle", "target": "mug"},
            "tilt the bottle over the steadied mug",
        )
        self.place(holder, "mug", mug_slot, "set the filled mug at its place")
        self.place(pourer, "bottle", STAGING, "put the bottle back down in the clear")

    def finish(self) -> None:
        for arm in ("left", "right"):
            self.add("home", {"arm": arm}, "retract clear of the table")


def expand(intent: dict, world: dict | None = None) -> dict:
    """Build the step sequence for ``intent`` given the current ``world``.

    Only the outstanding work is emitted: an object already on its slot is skipped, an
    open drawer is not reopened. Re-running this against a newer world is the whole
    replanning mechanism.
    """
    intent = normalize_intent(intent)

    if intent.get("refuse"):
        return {"steps": [], "refusal": intent["refuse"], "goal_summary": intent.get("paraphrase", "")}

    state = _Expansion(world)

    if intent.get("open_drawer") is True:
        state.ensure_drawer_open("the user asked for the drawer open")

    # The pour leg owns the mug end to end, so do not lay it out twice.
    to_lay = [obj for obj in PLACEABLE if obj in intent.get("place", [])]
    if intent.get("pour"):
        to_lay = [obj for obj in to_lay if obj != "mug"]
    for obj in to_lay:
        state.lay_out(obj)

    if intent.get("pour"):
        state.pour_into_mug()

    # Explicit request wins. Otherwise tidy up after ourselves — but only when the drawer
    # was opened to fetch something. If the user asked for it open, it stays open.
    close = intent.get("close_drawer")
    asked_open = intent.get("open_drawer") is True
    if close is True or (close is None and not asked_open and state.opened_drawer and state.drawer_open):
        state.close_drawer("leave the workcell tidy")

    state.finish()

    plan = {
        "goal_summary": intent.get("paraphrase") or _describe(intent),
        "steps": state.steps,
    }

    errors = validate_plan(plan, world)
    if errors:
        raise ExpansionError(
            "the expander emitted a plan that does not validate: " + "; ".join(errors)
        )
    return plan


def _describe(intent: dict) -> str:
    bits = []
    if intent.get("place"):
        bits.append(", ".join(intent["place"]) + " at the place setting")
    if intent.get("pour"):
        bits.append("water in the mug")
    if intent.get("open_drawer"):
        bits.append("drawer open")
    if intent.get("close_drawer"):
        bits.append("drawer closed")
    return "; ".join(bits) or "nothing to do"
