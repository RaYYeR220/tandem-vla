"""Seeded evaluation: the scorecard the whole project is judged on.

Runs the full instruction on N randomized seeds, records every step, and reports subgoal
completion rather than a single pass/fail, because "the plate went down but the pour missed"
is a different result from "the arm never got started" and averaging them into one number
hides which. The per-seed records are written out in full so any line of the summary can be
traced back to the episode that produced it.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from ..control.primitives import Executor
from ..control.runner import EpisodeRunner
from ..sim.env import TandemEnv
from ..sim.randomize import RandomizationSpec
from .tasks import canonical_plan, make_replanner


def _jsonable(o):
    """numpy scalars leak in from the simulator; keep the record writable either way."""
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)

SUBGOALS = ("drawer_opened", "plate_placed", "fork_placed", "spoon_placed", "mug_placed",
            "carton_emptied", "water_in_cup")


@dataclass
class EvalConfig:
    seeds: tuple[int, ...] = tuple(range(20))
    instruction: str = "set the table for one and pour me some water"
    intent: dict | None = None
    dr_scale: float = 1.0
    budget_s: float = 260.0
    speed: float = 1.0
    planner: str | None = None  #: None -> deterministic expansion; else a planner backend name


def run_eval(cfg: EvalConfig, *, out_dir: Path | None = None, verbose: bool = True) -> dict:
    intent = cfg.intent or {"place": ["plate", "fork", "spoon", "mug"], "pour": True}
    env = TandemEnv(RandomizationSpec(scale=cfg.dr_scale))
    ex = Executor(env, speed=cfg.speed)
    episodes = []
    t0 = time.perf_counter()

    for seed in cfg.seeds:
        world = env.reset(seed)
        runner = EpisodeRunner(env, ex, replanner=make_replanner(env, intent))
        rec = runner.run(
            canonical_plan(world, intent), instruction=cfg.instruction, budget_s=cfg.budget_s
        )
        episodes.append(rec)
        if verbose:
            sub = rec.score["subgoals"]
            flags = "".join("+" if sub[k] else "." for k in SUBGOALS)
            print(
                f"  seed {seed:>3}  [{flags}]  {rec.score['completed']}/{rec.score['total']}"
                f"  replans={rec.replans}  {rec.seconds:5.1f}s"
            )

    summary = summarize(episodes, cfg)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "episodes.json").write_text(
            json.dumps([e.as_dict() for e in episodes], indent=2, default=_jsonable),
            encoding="utf-8",
        )
        (out_dir / "scorecard.json").write_text(
            json.dumps(summary, indent=2, default=_jsonable), encoding="utf-8"
        )
        (out_dir / "scorecard.md").write_text(to_markdown(summary), encoding="utf-8")
    summary["wall_seconds"] = round(time.perf_counter() - t0, 1)
    return summary


def summarize(episodes, cfg: EvalConfig) -> dict:
    n = len(episodes)
    per_goal = {
        g: sum(1 for e in episodes if e.score["subgoals"][g]) for g in SUBGOALS
    }
    fracs = [e.score["fraction"] for e in episodes]
    step_stats: dict[str, dict] = {}
    failure_codes: dict[str, int] = {}
    handoff_modes: dict[str, int] = {}
    refusals = 0
    for e in episodes:
        refusals += len(e.refusals)
        for mode, c in e.handoff_summary().items():
            handoff_modes[mode] = handoff_modes.get(mode, 0) + c
        for s in e.steps:
            st = step_stats.setdefault(s.skill, {"attempted": 0, "ok": 0})
            st["attempted"] += 1
            st["ok"] += int(s.ok)
            if not s.ok:
                code = (s.result or {}).get("code") or s.verdict.get("code", "REFUSED")
                key = f"{s.skill}:{code}"
                failure_codes[key] = failure_codes.get(key, 0) + 1
    for st in step_stats.values():
        st["rate"] = round(st["ok"] / max(st["attempted"], 1), 3)

    return {
        "config": {
            "seeds": list(cfg.seeds),
            "n": n,
            "instruction": cfg.instruction,
            "dr_scale": cfg.dr_scale,
            "planner": cfg.planner or "deterministic-expansion",
        },
        "task_success": {
            "all_subgoals": sum(1 for e in episodes if e.score["success"]),
            "place_setting_complete": sum(1 for e in episodes if e.score["setting_success"]),
            "rate_all": round(sum(1 for e in episodes if e.score["success"]) / max(n, 1), 3),
            "rate_setting": round(
                sum(1 for e in episodes if e.score["setting_success"]) / max(n, 1), 3
            ),
        },
        "subgoals": {g: {"count": c, "rate": round(c / max(n, 1), 3)} for g, c in per_goal.items()},
        "subgoal_fraction": {
            "mean": round(statistics.fmean(fracs), 3) if fracs else 0.0,
            "median": round(statistics.median(fracs), 3) if fracs else 0.0,
            "min": round(min(fracs), 3) if fracs else 0.0,
            "max": round(max(fracs), 3) if fracs else 0.0,
        },
        "skills": step_stats,
        "failures": dict(sorted(failure_codes.items(), key=lambda kv: -kv[1])),
        "handoff_modes": handoff_modes,
        "gate_refusals": refusals,
        "replans": sum(e.replans for e in episodes),
        "water": {
            "mean_units_in_mug": round(
                statistics.fmean([e.score["water"]["in_mug"] for e in episodes]), 2
            )
            if episodes
            else 0.0,
            "mean_spilled": round(
                statistics.fmean([e.score["water"]["spilled"] for e in episodes]), 2
            )
            if episodes
            else 0.0,
        },
        "episode_seconds": {
            "mean": round(statistics.fmean([e.seconds for e in episodes]), 1) if episodes else 0,
            "max": round(max([e.seconds for e in episodes]), 1) if episodes else 0,
        },
    }


def to_markdown(s: dict) -> str:
    cfg = s["config"]
    lines = [
        f"# Task scorecard — {cfg['n']} randomized seeds",
        "",
        f"Instruction: **{cfg['instruction']}**  ",
        f"Seeds: `{cfg['seeds']}`  ",
        f"Domain-randomization scale: `{cfg['dr_scale']}`  ",
        f"Planner: `{cfg['planner']}`",
        "",
        "## Task completion",
        "",
        f"- All six subgoals: **{s['task_success']['all_subgoals']}/{cfg['n']}** "
        f"({s['task_success']['rate_all']:.0%})",
        f"- Place setting complete (plate, fork, spoon, mug): "
        f"**{s['task_success']['place_setting_complete']}/{cfg['n']}** "
        f"({s['task_success']['rate_setting']:.0%})",
        f"- Mean subgoal fraction: **{s['subgoal_fraction']['mean']:.2f}** "
        f"(median {s['subgoal_fraction']['median']:.2f}, "
        f"range {s['subgoal_fraction']['min']:.2f}–{s['subgoal_fraction']['max']:.2f})",
        "",
        "## Per-subgoal",
        "",
        "| subgoal | seeds passed | rate |",
        "| --- | --- | --- |",
    ]
    for g, v in s["subgoals"].items():
        lines.append(f"| {g} | {v['count']}/{cfg['n']} | {v['rate']:.0%} |")
    lines += ["", "## Per-skill reliability", "", "| skill | attempted | succeeded | rate |",
              "| --- | --- | --- | --- |"]
    for k, v in sorted(s["skills"].items()):
        lines.append(f"| {k} | {v['attempted']} | {v['ok']} | {v['rate']:.0%} |")
    if s["failures"]:
        lines += ["", "## Failure modes, by count", "", "| skill : code | count |", "| --- | --- |"]
        for k, v in s["failures"].items():
            lines.append(f"| `{k}` | {v} |")
    lines += [
        "",
        "## Other",
        "",
        f"- Hand-off modes: `{s['handoff_modes']}`",
        f"- Gate refusals raised: {s['gate_refusals']}",
        f"- Re-plans triggered: {s['replans']}",
        f"- Water landed in the mug: {s['water']['mean_units_in_mug']} units mean "
        f"(of 10), {s['water']['mean_spilled']} spilled",
        f"- Wall clock per episode: {s['episode_seconds']['mean']} s mean, "
        f"{s['episode_seconds']['max']} s max",
    ]
    return "\n".join(lines) + "\n"
