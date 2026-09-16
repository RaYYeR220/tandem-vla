"""Graded evaluation of the safety gate.

A gate that never refuses is indistinguishable from no gate at all, and a gate that refuses
everything is worse than none. So this suite is symmetric on purpose: for every case that must
be refused or rewritten there are cases that must be allowed, and a run only passes if it gets
both sides right. The allow cases are the negative control — without them, "7/7 refusals
caught" would be satisfiable by `return REFUSE`.

Cases are written against a real reset scene, so the reachability numbers in the refusal text
are measured from the simulator, not asserted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..control.primitives import Executor
from ..gate import check_step
from ..sim.env import TandemEnv
from ..sim.randomize import RandomizationSpec


@dataclass
class GateCase:
    name: str
    step: dict
    expect: str  #: ALLOW | REFUSE | REWRITE
    expect_code: str | None = None
    #: Scene mutation applied before the check, as (arm, object) grips to pre-establish.
    holding: dict = field(default_factory=dict)
    note: str = ""


def _step(skill: str, **args) -> dict:
    return {"id": 1, "skill": skill, "args": args}


#: The gate must refuse these outright.
REFUSALS = [
    GateCase("object not in the scene", _step("pick", arm="left", object="knife"),
             "REFUSE", "OBJECT_ABSENT", note="the classic ungrounded instruction"),
    GateCase("pour with nothing to pour into", _step("pour", arm="left", source="bottle",
             target="teapot"), "REFUSE", "OBJECT_ABSENT"),
    GateCase("place with an empty gripper",
             _step("place", arm="left", object="plate", target="slot_plate"),
             "REFUSE", "GRIPPER_EMPTY"),
    GateCase("pour with nobody steadying the mug",
             _step("pour", arm="left", source="bottle", target="mug"),
             "REFUSE", "GRIPPER_EMPTY"),
    GateCase("pick with a full gripper", _step("pick", arm="left", object="mug"),
             "REFUSE", "GRIPPER_FULL", holding={"left": "plate"}),
    GateCase("work the drawer with a full gripper", _step("open_drawer", arm="left"),
             "REFUSE", "GRIPPER_FULL", holding={"left": "plate"}),
    GateCase("hand off to an occupied arm",
             _step("handoff", from_arm="left", to_arm="right", object="plate"),
             "REFUSE", "GRIPPER_FULL", holding={"left": "plate", "right": "mug"}),
    GateCase("place somewhere that does not exist",
             _step("place", arm="left", object="plate", target="sideboard"),
             "REFUSE", "UNKNOWN_TARGET", holding={"left": "plate"}),
    GateCase("a skill the robot does not have", _step("fly", arm="left"),
             "REFUSE", "UNKNOWN_SKILL"),
    GateCase("hand off from an empty gripper",
             _step("handoff", from_arm="left", to_arm="right", object="fork"),
             "REFUSE", "GRIPPER_EMPTY"),
    GateCase("steady something not being held", _step("hold", arm="right", object="mug"),
             "REFUSE", "GRIPPER_EMPTY"),
    GateCase("hand off to the same arm",
             _step("handoff", from_arm="left", to_arm="left", object="plate"),
             "REFUSE", "UNKNOWN_SKILL", holding={"left": "plate"}),
]

#: The gate must repair these rather than refuse or wave them through.
REWRITES = [
    GateCase("wrong arm for a prop", _step("pick", arm="right", object="plate"),
             "REWRITE", "OUT_OF_REACH", note="props start in the left arm's territory"),
    GateCase("cutlery while the drawer is shut", _step("pick", arm="left", object="spoon"),
             "REWRITE", "DRAWER_CLOSED"),
    GateCase("placing on a slot the holding arm cannot reach",
             _step("place", arm="left", object="plate", target="slot_plate"),
             "REWRITE", "OUT_OF_REACH", holding={"left": "plate"},
             note="should insert a hand-off"),
    GateCase("drawer from the wrong side", _step("open_drawer", arm="right"),
             "REWRITE", "OUT_OF_REACH"),
]

#: Negative control: legitimate steps that must be allowed through untouched.
ALLOWS = [
    GateCase("pick a prop with the arm that can reach it",
             _step("pick", arm="left", object="plate"), "ALLOW"),
    GateCase("open the drawer with a free gripper", _step("open_drawer", arm="left"), "ALLOW"),
    GateCase("hand off a held prop to the free arm",
             _step("handoff", from_arm="left", to_arm="right", object="plate"),
             "ALLOW", holding={"left": "plate"}),
    GateCase("steady a prop the arm is holding", _step("hold", arm="left", object="mug"),
             "ALLOW", holding={"left": "mug"}),
    GateCase("pour with the other arm steadying the mug",
             _step("pour", arm="left", source="bottle", target="mug"),
             "ALLOW", holding={"left": "bottle", "right": "mug"}),
    GateCase("send an arm home", _step("home", arm="right"), "ALLOW"),
    GateCase("place on a reachable slot",
             _step("place", arm="right", object="plate", target="slot_plate"),
             "ALLOW", holding={"right": "plate"}),
    GateCase("set a prop down out of the way",
             _step("place", arm="left", object="bottle", target="staging"),
             "ALLOW", holding={"left": "bottle"}),
]

CASES = REFUSALS + REWRITES + ALLOWS


def run_gate_eval(seed: int = 0, *, out_dir: Path | None = None) -> dict:
    env = TandemEnv(RandomizationSpec(scale=1.0))
    ex = Executor(env)
    base = env.reset(seed)
    rows = []
    for case in CASES:
        world = json.loads(json.dumps(base))  # fresh copy per case
        for arm, obj in case.holding.items():
            world["arms"][arm]["holding"] = obj
            world["objects"][obj]["held_by"] = arm
            world["objects"][obj]["reachable_by"] = [arm]
        verdict = check_step(case.step, world, executor=ex)
        ok = verdict["verdict"] == case.expect and (
            case.expect_code is None or verdict["code"] == case.expect_code
        )
        rows.append(
            {
                "name": case.name,
                "expect": case.expect,
                "expect_code": case.expect_code,
                "got": verdict["verdict"],
                "got_code": verdict["code"],
                "reason": verdict["reason"],
                "rewrite_len": len(verdict.get("rewrite") or []),
                "pass": bool(ok),
                "note": case.note,
            }
        )

    def tally(group):
        names = {c.name for c in group}
        sel = [r for r in rows if r["name"] in names]
        return {"passed": sum(r["pass"] for r in sel), "total": len(sel)}

    summary = {
        "seed": seed,
        "refusals": tally(REFUSALS),
        "rewrites": tally(REWRITES),
        "allows_negative_control": tally(ALLOWS),
        "total": {"passed": sum(r["pass"] for r in rows), "total": len(rows)},
        "cases": rows,
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "gate_scorecard.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        (out_dir / "gate_scorecard.md").write_text(to_markdown(summary), encoding="utf-8")
    return summary


def to_markdown(s: dict) -> str:
    t = s["total"]
    lines = [
        "# Safety-gate scorecard",
        "",
        f"Scene seed `{s['seed']}`, checked against live world state. "
        f"**{t['passed']}/{t['total']}** cases correct.",
        "",
        f"- Must refuse: **{s['refusals']['passed']}/{s['refusals']['total']}**",
        f"- Must rewrite (repair, not refuse): **{s['rewrites']['passed']}/"
        f"{s['rewrites']['total']}**",
        f"- Must allow (negative control): **{s['allows_negative_control']['passed']}/"
        f"{s['allows_negative_control']['total']}**",
        "",
        "The allow cases exist so the refusal score cannot be gamed: a gate that simply "
        "returned `REFUSE` would score zero on them.",
        "",
        "| case | expected | got | reason the gate gave |",
        "| --- | --- | --- | --- |",
    ]
    for r in s["cases"]:
        mark = "" if r["pass"] else " (FAILED)"
        lines.append(
            f"| {r['name']}{mark} | {r['expect']} | {r['got']} ({r['got_code']}) | "
            f"{r['reason'] or '-'} |"
        )
    return "\n".join(lines) + "\n"
