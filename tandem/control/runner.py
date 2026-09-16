"""Episode execution: run a plan, check it after every step, re-plan when reality disagrees.

The planner proposes and the gate decides, but neither of them touches the simulator. This is
the loop that does, and it is deliberately suspicious: every step's post-condition is verified
against the world state rather than assumed, a failed step is retried once from the rest pose,
and a step that fails twice triggers a re-plan against whatever the scene actually looks like
now instead of the scene the plan was written for.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..gate import check_step
from ..sim import layout
from ..sim.env import TandemEnv
from .primitives import Executor, SkillResult

#: Retrying is only safe where the skill leaves the world untouched when it fails. A `place`
#: that missed has already opened the gripper, so running it again just reports an empty hand;
#: those failures go to the re-planner, which can see what actually happened.
RETRYABLE = {"pick", "open_drawer", "close_drawer", "home", "hold"}
MAX_STEP_ATTEMPTS = 3
#: Generous, because re-planning is cheap and state-derived: a fresh plan starts from wherever
#: the scene actually is, so grinding through a bad grasp costs one more attempt rather than
#: derailing the episode. The wall-clock budget is the real stop condition.
MAX_REPLANS = 12


@dataclass
class StepRecord:
    step_id: int
    skill: str
    args: dict
    verdict: dict
    attempts: int
    result: dict | None
    ok: bool
    seconds: float

    def as_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "skill": self.skill,
            "args": self.args,
            "verdict": self.verdict,
            "attempts": self.attempts,
            "result": self.result,
            "ok": self.ok,
            "seconds": round(self.seconds, 2),
        }


@dataclass
class EpisodeRecord:
    seed: int
    instruction: str
    plan: dict
    steps: list[StepRecord] = field(default_factory=list)
    replans: int = 0
    refusals: list[dict] = field(default_factory=list)
    score: dict = field(default_factory=dict)
    seconds: float = 0.0
    sim_seconds: float = 0.0
    aborted: bool = False

    def as_dict(self) -> dict:
        return {
            "seed": self.seed,
            "instruction": self.instruction,
            "plan": self.plan,
            "steps": [s.as_dict() for s in self.steps],
            "replans": self.replans,
            "refusals": self.refusals,
            "score": self.score,
            "seconds": round(self.seconds, 2),
            "sim_seconds": round(self.sim_seconds, 2),
            "aborted": self.aborted,
            "handoffs": self.handoff_summary(),
        }

    def handoff_summary(self) -> dict:
        modes: dict[str, int] = {}
        for s in self.steps:
            if s.skill == "handoff" and s.result:
                mode = (s.result.get("extras") or {}).get("mode", "unknown")
                modes[mode] = modes.get(mode, 0) + 1
        return modes


class EpisodeRunner:
    """Executes a plan against the environment, gating and re-planning as it goes."""

    def __init__(
        self,
        env: TandemEnv,
        executor: Executor | None = None,
        *,
        on_event=None,
        replanner=None,
    ):
        self.env = env
        self.ex = executor or Executor(env)
        self.on_event = on_event
        #: ``replanner(world, done_ids) -> plan`` — supplied by the planner package.
        self.replanner = replanner

    def emit(self, event: dict) -> None:
        if self.on_event is not None:
            self.on_event(event)

    def run(self, plan: dict, *, instruction: str = "", budget_s: float = 240.0) -> EpisodeRecord:
        t0 = time.perf_counter()
        t_sim0 = float(self.env.data.time)
        rec = EpisodeRecord(
            seed=int(self.env.seed or 0), instruction=instruction or plan.get("instruction", ""),
            plan=plan,
        )
        if plan.get("refusal"):
            rec.refusals.append({"stage": "planner", "reason": plan["refusal"]})
            self.emit({"type": "refusal", "stage": "planner", "reason": plan["refusal"]})
            rec.score = self.env.task_score()
            rec.seconds = time.perf_counter() - t0
            return rec

        steps = list(plan.get("steps", []))
        done_ids: list[int] = []
        replans = 0
        i = 0
        while i < len(steps):
            if time.perf_counter() - t0 > budget_s:
                rec.aborted = True
                self.emit({"type": "abort", "reason": "time budget exhausted"})
                break
            step = steps[i]
            sid = int(step.get("id", i + 1))
            skill = step["skill"]
            args = dict(step.get("args", {}))

            world = self.env.world_state()
            verdict = check_step(step, world, executor=self.ex)
            self.emit({"type": "verdict", "verdict": verdict})

            if verdict["verdict"] == "REFUSE":
                rec.refusals.append({"stage": "gate", "step_id": sid, **verdict})
                rec.steps.append(
                    StepRecord(sid, skill, args, verdict, 0, None, False, 0.0)
                )
                i += 1
                continue
            if verdict["verdict"] == "REWRITE":
                rewrite = verdict.get("rewrite") or []
                self.emit({"type": "rewrite", "step_id": sid, "steps": rewrite})
                steps[i : i + 1] = rewrite
                continue

            st = time.perf_counter()
            result: SkillResult | None = None
            attempts = 0
            budget = MAX_STEP_ATTEMPTS if skill in RETRYABLE else 1
            for attempts in range(1, budget + 1):
                self.emit(
                    {"type": "step", "step_id": sid, "status": "running",
                     "skill": skill, "arm": args.get("arm") or args.get("from_arm", ""),
                     "attempt": attempts}
                )
                result = self.ex.run_step(skill, args)
                if result.ok or attempts >= budget:
                    break
                self.emit({"type": "step", "step_id": sid, "status": "retry",
                           "skill": skill, "detail": result.detail})
                for a in layout.ARMS:
                    self.ex.retract(a, 0.8)

            ok = bool(result and result.ok)
            rec.steps.append(
                StepRecord(sid, skill, args, verdict, attempts,
                           result.as_dict() if result else None, ok,
                           time.perf_counter() - st)
            )
            self.emit({"type": "step", "step_id": sid,
                       "status": "done" if ok else "failed", "skill": skill,
                       "detail": result.detail if result else ""})
            if ok:
                done_ids.append(sid)
            elif self.replanner is not None and replans < MAX_REPLANS:
                replans += 1
                new_plan = self.replanner(self.env.world_state(), done_ids)
                remaining = list(new_plan.get("steps", []))
                self.emit({"type": "plan", "plan": new_plan, "replan": replans})
                if remaining:
                    steps = steps[: i + 1] + remaining
            i += 1

        rec.replans = replans
        rec.score = self.env.task_score()
        rec.seconds = time.perf_counter() - t0
        rec.sim_seconds = float(self.env.data.time) - t_sim0
        self.emit({"type": "episode", "seed": rec.seed,
                   "status": "success" if rec.score.get("success") else "failure",
                   "score": rec.score})
        return rec
