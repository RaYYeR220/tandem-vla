"""The gate: the model proposes, this decides.

Every step a planner emits is checked here against the measured world state before a single
joint moves. Three outcomes:

* ``ALLOW``   — preconditions hold and the arm can physically reach the target.
* ``REWRITE`` — the goal is sound but the assignment is not, e.g. the arm holding the object
  cannot reach the destination. The gate rewrites the step into a hand-off plus a placement
  by the other arm rather than letting the executor discover the problem mid-trajectory.
* ``REFUSE`` — the step cannot be executed safely or at all: an object that is not in the
  scene, a gripper that is already full, a slot already taken, a pour with nobody holding the
  cup. Refusals are terminal for that step and are surfaced to the operator verbatim.

The checks are deliberately boring, deterministic Python. Nothing here consults a model, which
is the point: a language model's confidence is not a safety argument, and the failure modes we
care about — an ungrounded object, an unreachable target, two arms in the same volume — are
exactly the ones it is worst at noticing.
"""

from __future__ import annotations

import numpy as np

from ..sim import layout

SKILL_ARGS = {
    "open_drawer": {"arm"},
    "close_drawer": {"arm"},
    "pick": {"arm", "object"},
    "place": {"arm", "object", "target"},
    "handoff": {"from_arm", "to_arm", "object"},
    "hold": {"arm", "object"},
    "pour": {"arm", "source", "target"},
    "home": {"arm"},
}

#: Objects that exist. Anything else is refused rather than guessed at.
KNOWN_OBJECTS = set(layout.PROPS)
KNOWN_TARGETS = set(layout.SLOTS) | {"staging"}

#: Where an object goes when a plan asks for it to be set down without naming a slot. Each arm
#: has its own clear patch, so "put it down" never turns into a pointless hand-off.
STAGING_XY = layout.STAGING_DROP


def _verdict(step_id, verdict, code, reason, rewrite=None) -> dict:
    out = {"step_id": step_id, "verdict": verdict, "code": code, "reason": reason}
    if rewrite is not None:
        out["rewrite"] = rewrite
    return out


def _reachable(arm: str, xy, executor=None, z: float | None = None) -> bool:
    """Envelope test, tightened by a real IK solve when an executor is available."""
    if not layout.in_reach(arm, xy):
        return False
    if executor is None or z is None:
        return True
    from ..control.poses import solve_pose

    pose = solve_pose(
        executor.ik[arm],
        executor.env.data,
        np.array([xy[0], xy[1], z]),
        0.0,
        arm=arm,
        restarts=4,
    )
    return bool(pose.ok)


def check_step(step: dict, world: dict, *, executor=None) -> dict:
    """Judge one planned step against the current world state."""
    sid = int(step.get("id", 0))
    skill = step.get("skill")
    args = dict(step.get("args", {}))

    if skill not in SKILL_ARGS:
        return _verdict(sid, "REFUSE", "UNKNOWN_SKILL", f"there is no skill called {skill!r}")
    missing = SKILL_ARGS[skill] - set(args)
    if missing:
        return _verdict(
            sid, "REFUSE", "UNKNOWN_SKILL",
            f"{skill} needs {sorted(SKILL_ARGS[skill])}, missing {sorted(missing)}",
        )

    obj = args.get("object") or args.get("source")
    if obj is not None and obj not in KNOWN_OBJECTS:
        return _verdict(
            sid, "REFUSE", "OBJECT_ABSENT",
            f"there is no {obj} in this scene; it holds "
            f"{', '.join(sorted(KNOWN_OBJECTS))}",
        )

    arms = world["arms"]

    if skill == "home":
        return _verdict(sid, "ALLOW", "OK", "")

    if skill in ("open_drawer", "close_drawer"):
        arm = args["arm"]
        if arms[arm]["holding"]:
            return _verdict(
                sid, "REFUSE", "GRIPPER_FULL",
                f"{arm} is holding the {arms[arm]['holding']} and cannot work the handle",
            )
        if arm not in world["drawer"]["reachable_by"]:
            other = layout.other_arm(arm)
            if other in world["drawer"]["reachable_by"]:
                return _verdict(
                    sid, "REWRITE", "OUT_OF_REACH",
                    f"the drawer is on the {other} arm's side",
                    [{"id": sid, "skill": skill, "args": {"arm": other},
                      "rationale": "the drawer is only reachable by that arm"}],
                )
            return _verdict(sid, "REFUSE", "OUT_OF_REACH", "neither arm can reach the drawer")
        if skill == "open_drawer" and world["drawer"]["is_open"]:
            return _verdict(sid, "ALLOW", "OK", "the drawer is already open")
        return _verdict(sid, "ALLOW", "OK", "")

    if skill == "pick":
        arm, o = args["arm"], args["object"]
        info = world["objects"][o]
        if arms[arm]["holding"]:
            return _verdict(
                sid, "REFUSE", "GRIPPER_FULL",
                f"{arm} is already holding the {arms[arm]['holding']}",
            )
        if info["held_by"] is not None:
            return _verdict(
                sid, "REFUSE", "OBJECT_OCCLUDED",
                f"the {o} is already in the {info['held_by']} gripper",
            )
        if info["in_drawer"] and not world["drawer"]["is_open"]:
            return _verdict(
                sid, "REWRITE", "DRAWER_CLOSED",
                f"the {o} is inside a closed drawer",
                [
                    {"id": sid, "skill": "open_drawer",
                     "args": {"arm": world["drawer"]["reachable_by"][0]},
                     "rationale": f"the {o} is inside it"},
                    {"id": sid, "skill": "pick", "args": {"arm": arm, "object": o},
                     "rationale": "now the drawer is open"},
                ],
            )
        if arm not in info["reachable_by"]:
            other = layout.other_arm(arm)
            if other in info["reachable_by"]:
                return _verdict(
                    sid, "REWRITE", "OUT_OF_REACH",
                    f"the {o} is {layout.reach(arm, info['pos']):.2f} m from the {arm} base, "
                    f"past its {layout.R_MAX:.2f} m envelope",
                    [{"id": sid, "skill": "pick", "args": {"arm": other, "object": o},
                      "rationale": f"only the {other} arm can reach the {o}"}],
                )
            return _verdict(
                sid, "REFUSE", "OUT_OF_REACH",
                f"the {o} is outside both arms' envelopes "
                f"(left {layout.reach('left', info['pos']):.2f} m, "
                f"right {layout.reach('right', info['pos']):.2f} m, limit "
                f"{layout.R_MAX:.2f} m)",
            )
        return _verdict(sid, "ALLOW", "OK", "")

    if skill == "place":
        arm, o, target = args["arm"], args["object"], args["target"]
        if target not in KNOWN_TARGETS:
            return _verdict(
                sid, "REFUSE", "UNKNOWN_TARGET",
                f"there is no place called {target!r}; the setting has "
                f"{', '.join(sorted(layout.SLOTS))}",
            )
        if arms[arm]["holding"] != o:
            return _verdict(
                sid, "REFUSE", "GRIPPER_EMPTY",
                f"{arm} is not holding the {o}"
                + (f" (it has the {arms[arm]['holding']})" if arms[arm]["holding"] else ""),
            )
        if target == "staging":
            xy = STAGING_XY[arm]
            reach_ok = layout.in_reach(arm, xy)
        else:
            slot = world["slots"][target]
            occupied = slot["occupied_by"]
            if occupied not in (None, o):
                return _verdict(
                    sid, "REFUSE", "SLOT_OCCUPIED",
                    f"{target} already has the {occupied} on it",
                )
            xy = np.array(slot["pos"][:2])
            reach_ok = arm in slot["reachable_by"]
        if not reach_ok:
            other = layout.other_arm(arm)
            other_ok = (
                layout.in_reach(other, STAGING_XY[other])
                if target == "staging"
                else other in world["slots"][target]["reachable_by"]
            )
            if other_ok:
                return _verdict(
                    sid, "REWRITE", "OUT_OF_REACH",
                    f"{target} is {layout.reach(arm, xy):.2f} m from the {arm} base; "
                    f"handing to {other} first",
                    [
                        {"id": sid, "skill": "handoff",
                         "args": {"from_arm": arm, "to_arm": other, "object": o},
                         "rationale": f"{target} is only reachable by the {other} arm"},
                        {"id": sid, "skill": "place",
                         "args": {"arm": other, "object": o, "target": target},
                         "rationale": "now the right arm holds it"},
                    ],
                )
            return _verdict(
                sid, "REFUSE", "OUT_OF_REACH", f"neither arm can reach {target}",
            )
        return _verdict(sid, "ALLOW", "OK", "")

    if skill == "handoff":
        fa, ta, o = args["from_arm"], args["to_arm"], args["object"]
        if fa == ta:
            return _verdict(sid, "REFUSE", "UNKNOWN_SKILL", "a hand-off needs two arms")
        if arms[fa]["holding"] != o:
            return _verdict(sid, "REFUSE", "GRIPPER_EMPTY", f"{fa} is not holding the {o}")
        if arms[ta]["holding"]:
            return _verdict(
                sid, "REFUSE", "GRIPPER_FULL",
                f"{ta} is holding the {arms[ta]['holding']}; it has no hand free",
            )
        return _verdict(sid, "ALLOW", "OK", "")

    if skill == "hold":
        arm, o = args["arm"], args["object"]
        if arms[arm]["holding"] != o:
            return _verdict(sid, "REFUSE", "GRIPPER_EMPTY", f"{arm} is not holding the {o}")
        return _verdict(sid, "ALLOW", "OK", "")

    if skill == "pour":
        arm, src, tgt = args["arm"], args["source"], args["target"]
        if tgt not in KNOWN_OBJECTS:
            return _verdict(
                sid, "REFUSE", "OBJECT_ABSENT", f"there is nothing called {tgt!r} to pour into",
            )
        if arms[arm]["holding"] != src:
            return _verdict(sid, "REFUSE", "GRIPPER_EMPTY", f"{arm} is not holding the {src}")
        holder = layout.other_arm(arm)
        if arms[holder]["holding"] != tgt:
            return _verdict(
                sid, "REFUSE", "NO_HOLDER_FOR_POUR",
                f"nothing is steadying the {tgt}; the {holder} arm has to hold it before the "
                f"{src} tips over it",
            )
        return _verdict(sid, "ALLOW", "OK", "")

    return _verdict(sid, "REFUSE", "UNKNOWN_SKILL", f"unhandled skill {skill!r}")


def check_plan(plan: dict, world: dict, *, executor=None) -> list[dict]:
    """Static pass over a whole plan. Does not simulate occupancy; the runner does that live."""
    return [check_step(s, world, executor=executor) for s in plan.get("steps", [])]
