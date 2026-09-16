"""Graded head-to-head: scripted oracle versus the distilled policy.

    python scripts/eval_policy.py --seeds 20 --seed0 400 --precision fp32,int8

Every seed used here is outside the range the demonstrations were collected from, so no
scene the policy is graded on appeared in its training set.

Two protocols, because they answer different questions.

* **Matched skill trials.** The state is snapshotted immediately before the skill, the
  oracle runs from it, the snapshot is restored bit for bit, and the policy runs from the
  *identical* state. This is the only way to attribute a difference in success rate to the
  controller rather than to where the previous step happened to leave the cell. Picks are
  matched from the post-reset state; places are matched from a scripted pick + hand-off
  that both branches share.
* **End-to-end episodes.** The full canonical plan, gate, replanner and all, run twice per
  seed: once with the scripted executor, once with `PolicyExecutor`, which swaps in the
  learned controller for `pick` and `place` and leaves every other skill scripted. The two
  runs diverge after their first disagreement -- that is the point -- so this measures
  subgoal completion, not a matched comparison.

Running with several precisions repeats the whole thing per IR, which is what turns
"INT8 is 3x faster" into "INT8 is 3x faster and costs this much task success".
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tandem.control.primitives import Executor  # noqa: E402
from tandem.control.runner import EpisodeRunner  # noqa: E402
from tandem.eval.tasks import canonical_plan, make_replanner  # noqa: E402
from tandem.policy.rollout import PolicyExecutor, PolicyRunner  # noqa: E402
from tandem.sim import layout  # noqa: E402
from tandem.sim.env import TandemEnv  # noqa: E402
from tandem.sim.randomize import RandomizationSpec  # noqa: E402

INTENT = {"place": ["plate", "fork", "spoon", "mug"], "pour": True}

#: Props reachable from the left arm at reset, which is where every pick starts.
PICK_OBJECTS = ("plate", "mug", "bottle")


# ------------------------------------------------------------------------- state I/O


@dataclass
class Snapshot:
    """Everything needed to rewind the cell to an exact instant."""

    qpos: np.ndarray
    qvel: np.ndarray
    act: np.ndarray
    ctrl: np.ndarray
    warmstart: np.ndarray
    time: float
    attached: dict
    grasp_transform: dict
    held_tool: dict


def capture(env: TandemEnv, executor: Executor) -> Snapshot:
    d = env.data
    return Snapshot(
        qpos=d.qpos.copy(),
        qvel=d.qvel.copy(),
        act=d.act.copy(),
        ctrl=np.array(env.ctrl),
        warmstart=d.qacc_warmstart.copy(),
        time=float(d.time),
        attached=dict(env.attached),
        grasp_transform={k: (None if v is None else np.array(v))
                         for k, v in executor.grasp_transform.items()},
        held_tool={k: (None if v is None else np.array(v))
                   for k, v in getattr(executor, "held_tool", {}).items()},
    )


def restore(env: TandemEnv, executor: Executor, snap: Snapshot) -> None:
    import mujoco

    d = env.data
    d.qpos[:] = snap.qpos
    d.qvel[:] = snap.qvel
    d.act[:] = snap.act
    d.qacc_warmstart[:] = snap.warmstart
    d.time = snap.time
    env.ctrl[:] = snap.ctrl
    d.ctrl[:] = snap.ctrl
    env.attached = dict(snap.attached)
    executor.grasp_transform = {k: (None if v is None else np.array(v))
                                for k, v in snap.grasp_transform.items()}
    if hasattr(executor, "held_tool"):
        executor.held_tool = {k: (None if v is None else np.array(v))
                              for k, v in snap.held_tool.items()}
    mujoco.mj_forward(env.model, d)


# --------------------------------------------------------------------- skill trials


@dataclass
class Tally:
    attempts: int = 0
    successes: int = 0
    seconds: float = 0.0
    errors_mm: list[float] = field(default_factory=list)
    codes: dict[str, int] = field(default_factory=dict)

    def add(self, result) -> None:
        self.attempts += 1
        self.successes += int(result.ok)
        self.seconds += result.seconds
        self.codes[result.code] = self.codes.get(result.code, 0) + 1
        err = (result.extras or {}).get("err")
        if err is not None:
            self.errors_mm.append(err * 1000.0)

    def as_dict(self) -> dict:
        return {
            "attempts": self.attempts,
            "successes": self.successes,
            "rate": round(self.successes / self.attempts, 3) if self.attempts else None,
            "mean_seconds": round(self.seconds / self.attempts, 2) if self.attempts else None,
            "median_place_err_mm": (
                round(float(np.median(self.errors_mm)), 1) if self.errors_mm else None
            ),
            "codes": self.codes,
        }


def skill_trials(env: TandemEnv, scripted: Executor, learned: PolicyExecutor,
                 seeds: list[int], *, do_places: bool, oracle_cache: dict | None = None) -> dict:
    """Matched pick and place trials on identical initial states.

    ``oracle_cache`` carries the scripted results between precision sweeps. The oracle
    branch is a deterministic function of the seed and the snapshot it starts from, so
    re-running it once per IR would burn minutes to reproduce the same numbers; the cache
    records whether the oracle succeeded, which is all the policy branch needs in order to
    stay on the identical set of trials.
    """
    tallies = {
        ("oracle", "pick"): Tally(), ("policy", "pick"): Tally(),
        ("oracle", "place"): Tally(), ("policy", "place"): Tally(),
    }
    per_object: dict[str, dict[str, Tally]] = {}
    cache = oracle_cache if oracle_cache is not None else {}

    for seed in seeds:
        env.reset(seed)
        base = capture(env, scripted)
        started = time.perf_counter()
        for obj in PICK_OBJECTS:
            table = per_object.setdefault(obj, {"oracle": Tally(), "policy": Tally()})
            key = (seed, obj)
            if key in cache:
                oracle_pick = cache[key]
            else:
                restore(env, scripted, base)
                oracle_pick = scripted.pick("left", obj)
                cache[key] = oracle_pick
            tallies[("oracle", "pick")].add(oracle_pick)
            table["oracle"].add(oracle_pick)

            restore(env, learned, base)
            policy_pick = learned.pick("left", obj)
            tallies[("policy", "pick")].add(policy_pick)
            table["policy"].add(policy_pick)

            if not do_places or not oracle_pick.ok:
                continue
            # Matched place: rebuild the held state with the scripted pick + hand-off, then
            # branch. Both controllers start from the same instant, in the same hand. The
            # staged snapshot is cached, so the ~25 s of scripted setup is paid once no
            # matter how many precisions are being graded.
            staged_key = (seed, obj, "staged")
            if staged_key not in cache:
                restore(env, scripted, base)
                if not scripted.pick("left", obj).ok:
                    cache[staged_key] = None
                elif not scripted.handoff("left", "right", obj).ok:
                    cache[staged_key] = None
                else:
                    cache[staged_key] = capture(env, scripted)
            staged = cache[staged_key]
            if staged is None:
                continue
            slot = layout.OBJECT_SLOT[obj] if obj in layout.OBJECT_SLOT else "staging"

            place_key = (seed, obj, "place")
            if place_key not in cache:
                restore(env, scripted, staged)
                cache[place_key] = scripted.place("right", obj, slot)
            tallies[("oracle", "place")].add(cache[place_key])
            restore(env, learned, staged)
            tallies[("policy", "place")].add(learned.place("right", obj, slot))
        print(
            f"    seed {seed}: oracle pick {tallies[('oracle', 'pick')].successes}/"
            f"{tallies[('oracle', 'pick')].attempts}, policy pick "
            f"{tallies[('policy', 'pick')].successes}/{tallies[('policy', 'pick')].attempts}, "
            f"place {tallies[('oracle', 'place')].successes}/"
            f"{tallies[('oracle', 'place')].attempts} vs "
            f"{tallies[('policy', 'place')].successes}/{tallies[('policy', 'place')].attempts} "
            f"({time.perf_counter() - started:.0f}s)",
            flush=True,
        )

    return {
        "matched": {
            f"{who}_{skill}": tally.as_dict() for (who, skill), tally in tallies.items()
        },
        "pick_by_object": {
            obj: {who: tally.as_dict() for who, tally in table.items()}
            for obj, table in per_object.items()
        },
    }


# ---------------------------------------------------------------------- end to end


def episode_trials(env: TandemEnv, executor: Executor, seeds: list[int], *,
                   budget: float) -> dict:
    subgoals: dict[str, int] = {}
    completed: list[int] = []
    totals: list[int] = []
    per_skill: dict[str, dict[str, int]] = {}
    seconds: list[float] = []

    for seed in seeds:
        world = env.reset(seed)
        if hasattr(executor, "held_tool"):
            executor.held_tool = {a: None for a in layout.ARMS}
        executor.grasp_transform = {a: None for a in layout.ARMS}
        runner = EpisodeRunner(env, executor, replanner=make_replanner(env, INTENT))
        t0 = time.perf_counter()
        record = runner.run(canonical_plan(world, INTENT),
                            instruction="set the table for one", budget_s=budget)
        seconds.append(time.perf_counter() - t0)
        print(f"    seed {seed}: {record.score.get('completed')}/"
              f"{record.score.get('total')} subgoals ({seconds[-1]:.0f}s)", flush=True)
        score = record.score
        completed.append(int(score.get("completed", 0)))
        totals.append(int(score.get("total", 0)))
        for name, value in score.get("subgoals", {}).items():
            subgoals[name] = subgoals.get(name, 0) + int(bool(value))
        for step in record.steps:
            entry = per_skill.setdefault(step.skill, {"attempts": 0, "successes": 0})
            entry["attempts"] += 1
            entry["successes"] += int(step.ok)

    n = max(1, len(seeds))
    total = max(totals) if totals else 0
    return {
        "episodes": len(seeds),
        "subgoal_total": total,
        "mean_subgoals": round(float(np.mean(completed)), 3),
        "per_episode_subgoals": completed,
        "subgoal_rate": {k: round(v / n, 3) for k, v in sorted(subgoals.items())},
        "full_success_rate": round(
            sum(1 for c, t in zip(completed, totals) if t and c == t) / n, 3
        ),
        "per_skill": {
            k: {**v, "rate": round(v["successes"] / max(1, v["attempts"]), 3)}
            for k, v in sorted(per_skill.items())
        },
        "mean_wall_seconds": round(float(np.mean(seconds)), 1),
    }


# -------------------------------------------------------------------------- reports


def markdown(results: dict) -> str:
    lines = ["# Scripted oracle vs distilled policy", ""]
    lines.append(
        f"Seeds {results['seeds'][0]}..{results['seeds'][-1]} "
        f"({len(results['seeds'])} of them), domain randomization scale "
        f"{results['dr_scale']}. None of these seeds appear in the demonstration set "
        f"(seeds {results['train_seed_range'][0]}..{results['train_seed_range'][1]})."
    )
    lines += ["", "## Matched skill trials (identical start state per trial)", ""]
    lines.append("| precision | controller | skill | attempts | successes | rate | "
                 "mean s | median place err (mm) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for precision, block in results["precisions"].items():
        matched = block.get("skills", {}).get("matched", {})
        for key in ("oracle_pick", "policy_pick", "oracle_place", "policy_place"):
            row = matched.get(key)
            if not row or not row["attempts"]:
                continue
            who, skill = key.split("_")
            label = precision if who == "policy" else "n/a"
            lines.append(
                f"| {label} | {who} | {skill} | {row['attempts']} | {row['successes']} | "
                f"{row['rate']:.2f} | {row['mean_seconds']} | "
                f"{row['median_place_err_mm'] if row['median_place_err_mm'] is not None else '-'} |"
            )
    lines += ["", "## End-to-end episodes (full plan, gate and replanner active)", ""]
    lines.append("| precision | controller | episodes | mean subgoals | full success | "
                 "pick rate | place rate | mean wall s |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for precision, block in results["precisions"].items():
        for who in ("oracle", "policy"):
            row = block.get("episodes", {}).get(who)
            if not row:
                continue
            picks = row["per_skill"].get("pick", {})
            places = row["per_skill"].get("place", {})
            label = precision if who == "policy" else "n/a"
            lines.append(
                f"| {label} | {who} | {row['episodes']} | "
                f"{row['mean_subgoals']} / {row.get('subgoal_total', '?')} | "
                f"{row['full_success_rate']:.2f} | {picks.get('rate', '-')} | "
                f"{places.get('rate', '-')} | {row['mean_wall_seconds']} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", type=Path, default=Path("models"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--seed0", type=int, default=400)
    ap.add_argument("--dr", type=float, default=1.0)
    ap.add_argument("--budget", type=float, default=260.0)
    ap.add_argument("--device", default="CPU", help="OpenVINO device for the policy")
    ap.add_argument("--precision", default="fp32",
                    help="comma-separated IR precisions to grade, e.g. fp32,int8")
    ap.add_argument("--execute", type=int, default=8, help="chunk steps executed per inference")
    ap.add_argument("--ensemble", action="store_true", help="temporal ensembling")
    ap.add_argument("--apply", default="acting", choices=("acting", "both"))
    ap.add_argument("--no-places", action="store_true", help="skip the matched place trials")
    ap.add_argument("--no-episodes", action="store_true", help="skip the end-to-end runs")
    ap.add_argument("--train-seeds", default="0,127", help="demonstration seed range, for the note")
    args = ap.parse_args()

    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    env = TandemEnv(RandomizationSpec(scale=args.dr))
    scripted = Executor(env)

    results: dict = {
        "seeds": seeds,
        "dr_scale": args.dr,
        "device": args.device,
        "execute": args.execute,
        "ensemble": args.ensemble,
        "apply": args.apply,
        "train_seed_range": [int(v) for v in args.train_seeds.split(",")],
        "precisions": {},
    }

    oracle_episodes = None
    oracle_cache: dict = {}
    for precision in [p.strip() for p in args.precision.split(",") if p.strip()]:
        ir = args.models / f"policy_{precision}.xml"
        if not ir.exists():
            print(f"skipping {precision}: {ir} not found")
            continue
        print(f"\n== {precision.upper()} on {args.device} ==")
        runner = PolicyRunner(ir, device=args.device, execute=args.execute,
                              ensemble=args.ensemble)
        learned = PolicyExecutor(env, runner, apply=args.apply)
        block: dict = {}

        t0 = time.perf_counter()
        block["skills"] = skill_trials(env, scripted, learned, seeds,
                                       do_places=not args.no_places,
                                       oracle_cache=oracle_cache)
        print(f"  matched trials in {time.perf_counter() - t0:.0f}s")
        for key, row in block["skills"]["matched"].items():
            if row["attempts"]:
                print(f"    {key:<14} {row['successes']:>3}/{row['attempts']:<3} "
                      f"({row['rate']:.0%})")

        if not args.no_episodes:
            if oracle_episodes is None:
                print("  end-to-end: oracle ...")
                oracle_episodes = episode_trials(env, scripted, seeds, budget=args.budget)
            print("  end-to-end: policy ...")
            block["episodes"] = {
                "oracle": oracle_episodes,
                "policy": episode_trials(env, learned, seeds, budget=args.budget),
            }
            for who, row in block["episodes"].items():
                print(f"    {who:<7} mean subgoals {row['mean_subgoals']}/"
                      f"{row['subgoal_total']}, "
                      f"full success {row['full_success_rate']:.0%}")

        block["inference"] = runner.latency_stats()
        results["precisions"][precision] = block

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "policy_scorecard.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    (args.out / "policy_scorecard.md").write_text(markdown(results), encoding="utf-8")
    print(f"\nwrote {args.out}/policy_scorecard.json|.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
