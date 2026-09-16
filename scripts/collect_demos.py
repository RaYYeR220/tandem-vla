"""Generate the two learned-half datasets.

Perception -- randomized scene snapshots with segmentation-derived visibility::

    python scripts/collect_demos.py --perception --seeds 640 --frames-per-seed 16 --workers 4

Policy -- scripted-oracle demonstrations, per-step filtered to successful skills only::

    python scripts/collect_demos.py --episodes 120 --workers 4

Both write compressed npz shards under ``data/`` and a ``manifest.json`` next to them.
Collection is CPU-bound in MuJoCo, so it forks worker processes over disjoint seed ranges;
each worker compiles its own model and writes its own shards.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tandem.perception.collect import SceneSampler  # noqa: E402
from tandem.perception.schema import CAMERAS, IMAGE_SIZE, PROPS  # noqa: E402
from tandem.perception.shards import encode_frames, write_shard  # noqa: E402
from tandem.policy.collect import DemoRecorder, RecordingExecutor  # noqa: E402
from tandem.policy.schema import CHUNK  # noqa: E402
from tandem.sim.env import TandemEnv  # noqa: E402
from tandem.sim.randomize import RandomizationSpec  # noqa: E402

DEFAULT_INTENT = {"place": ["plate", "fork", "spoon", "mug"], "pour": True}


# --------------------------------------------------------------------------- perception


def _perception_worker(args: tuple) -> dict:
    seeds, out_dir, frames_per_seed, shard_size, tag, dr_scale = args
    env = TandemEnv(RandomizationSpec(scale=dr_scale))
    sampler = SceneSampler(env, size=IMAGE_SIZE)
    out_dir = Path(out_dir)

    buffers: dict[str, list] = {c: [] for c in CAMERAS}
    labels: dict[str, list] = {"pos": [], "yaw": [], "visible": [], "pixels": [], "drawer": [],
                               "seed": [], "mode": []}
    written, kept, dropped = [], 0, 0
    shard_index = 0

    def flush() -> None:
        nonlocal shard_index, buffers, labels
        if not labels["seed"]:
            return
        arrays: dict[str, np.ndarray] = {}
        for cam in CAMERAS:
            buffer, offsets = encode_frames(buffers[cam])
            arrays[f"{cam}_jpeg"] = buffer
            arrays[f"{cam}_offsets"] = offsets
        arrays["pos"] = np.stack(labels["pos"]).astype(np.float32)
        arrays["yaw"] = np.stack(labels["yaw"]).astype(np.float32)
        arrays["visible"] = np.stack(labels["visible"]).astype(np.float32)
        arrays["pixels"] = np.stack(labels["pixels"]).astype(np.int32)
        arrays["drawer"] = np.asarray(labels["drawer"], dtype=np.float32)
        arrays["seed"] = np.asarray(labels["seed"], dtype=np.int32)
        arrays["mode"] = np.asarray(labels["mode"])
        path = out_dir / f"perception_{tag}{shard_index:03d}.npz"
        write_shard(path, arrays)
        written.append(str(path))
        shard_index += 1
        buffers = {c: [] for c in CAMERAS}
        labels = {k: [] for k in labels}

    for seed in seeds:
        env.reset(int(seed))
        rng = np.random.default_rng(int(seed) * 7919 + 13)
        for _ in range(frames_per_seed):
            frame = sampler.sample(rng)
            if frame is None:
                dropped += 1
                continue
            for cam in CAMERAS:
                buffers[cam].append(frame.views[cam])
            labels["pos"].append(frame.pos)
            labels["yaw"].append(frame.yaw)
            labels["visible"].append(frame.visible)
            labels["pixels"].append(frame.pixels)
            labels["drawer"].append(frame.drawer_open)
            labels["seed"].append(int(seed))
            labels["mode"].append(frame.mode)
            kept += 1
            if len(labels["seed"]) >= shard_size:
                flush()
    flush()
    sampler.close()
    env.close()
    return {"frames": kept, "dropped": dropped, "shards": written}


def collect_perception(args: argparse.Namespace) -> dict:
    out_dir = Path(args.out or "data/perception")
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.append:
        for stale in out_dir.glob("perception_*.npz"):
            stale.unlink()

    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    chunks = [seeds[i :: args.workers] for i in range(args.workers)]
    jobs = [
        (chunk, str(out_dir), args.frames_per_seed, args.shard_size, f"{args.tag}w{i}_", args.dr)
        for i, chunk in enumerate(chunks)
        if chunk
    ]

    t0 = time.perf_counter()
    if len(jobs) == 1:
        results = [_perception_worker(jobs[0])]
    else:
        with mp.Pool(len(jobs)) as pool:
            results = pool.map(_perception_worker, jobs)
    elapsed = time.perf_counter() - t0

    frames = sum(r["frames"] for r in results)
    if args.append:
        existing = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        frames += int(existing.get("frames", 0))
    manifest = {
        "kind": "perception",
        "frames": frames,
        "dropped": sum(r["dropped"] for r in results),
        "seeds": [args.seed0, args.seed0 + args.seeds - 1],
        "frames_per_seed": args.frames_per_seed,
        "image_size": IMAGE_SIZE,
        "cameras": list(CAMERAS),
        "props": list(PROPS),
        "dr_scale": args.dr,
        "shards": sorted(str(p) for p in out_dir.glob("perception_*.npz")),
        "seconds": round(elapsed, 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"perception: {frames} frames from {args.seeds} seeds in {elapsed:.0f}s "
        f"({len(manifest['shards'])} shards, {manifest['dropped']} dropped)"
    )
    return manifest


# ------------------------------------------------------------------------------ policy


def _policy_worker(args: tuple) -> dict:
    seeds, out_dir, stride, shard_size, tag, dr_scale, budget = args
    from tandem.control.runner import EpisodeRunner
    from tandem.eval.tasks import canonical_plan, make_replanner

    env = TandemEnv(RandomizationSpec(scale=dr_scale))
    out_dir = Path(out_dir)
    samples: list = []
    written: list[str] = []
    step_stats: list[tuple[str, bool, int]] = []
    episodes: list[dict] = []
    shard_index = 0

    def flush() -> None:
        nonlocal shard_index, samples
        if not samples:
            return
        arrays: dict[str, np.ndarray] = {}
        for key in ("overhead", "wrist"):
            buffer, offsets = encode_frames([getattr(s, key) for s in samples])
            arrays[f"{key}_jpeg"] = buffer
            arrays[f"{key}_offsets"] = offsets
        arrays["proprio"] = np.stack([s.proprio for s in samples])
        arrays["cond"] = np.stack([s.cond for s in samples])
        arrays["chunk"] = np.stack([s.chunk for s in samples])
        arrays["skill"] = np.asarray([s.skill for s in samples])
        arrays["arm"] = np.asarray([s.arm for s in samples])
        arrays["seed"] = np.asarray([s.seed for s in samples], dtype=np.int32)
        path = out_dir / f"policy_{tag}{shard_index:03d}.npz"
        write_shard(path, arrays)
        written.append(str(path))
        shard_index += 1
        samples = []

    for seed in seeds:
        world = env.reset(int(seed))
        recorder = DemoRecorder(env, seed=int(seed), stride=stride)
        executor = RecordingExecutor(env, recorder)
        runner = EpisodeRunner(env, executor, replanner=make_replanner(env, DEFAULT_INTENT))
        plan = canonical_plan(world, DEFAULT_INTENT)
        t0 = time.perf_counter()
        try:
            record = runner.run(plan, instruction="set the table for one", budget_s=budget)
            score = record.score
        except Exception as exc:  # a physics blow-up must not take the whole shard with it
            score = {"error": f"{type(exc).__name__}: {exc}"}
        samples.extend(recorder.samples)
        step_stats.extend(recorder.step_stats)
        episodes.append(
            {
                "seed": int(seed),
                "samples": len(recorder.samples),
                "completed": score.get("completed"),
                "total": score.get("total"),
                "seconds": round(time.perf_counter() - t0, 1),
            }
        )
        if len(samples) >= shard_size:
            flush()
    flush()
    env.close()
    return {"shards": written, "steps": step_stats, "episodes": episodes}


def collect_policy(args: argparse.Namespace) -> dict:
    out_dir = Path(args.out or "data/policy")
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("policy_*.npz"):
        stale.unlink()

    seeds = list(range(args.seed0, args.seed0 + args.episodes))
    chunks = [seeds[i :: args.workers] for i in range(args.workers)]
    jobs = [
        (chunk, str(out_dir), args.stride, args.shard_size, f"w{i}_", args.dr, args.budget)
        for i, chunk in enumerate(chunks)
        if chunk
    ]

    t0 = time.perf_counter()
    if len(jobs) == 1:
        results = [_policy_worker(jobs[0])]
    else:
        with mp.Pool(len(jobs)) as pool:
            results = pool.map(_policy_worker, jobs)
    elapsed = time.perf_counter() - t0

    steps = [s for r in results for s in r["steps"]]
    episodes = [e for r in results for e in r["episodes"]]
    per_skill: dict[str, dict[str, int]] = {}
    for skill, ok, ticks in steps:
        entry = per_skill.setdefault(skill, {"attempts": 0, "successes": 0, "ticks": 0})
        entry["attempts"] += 1
        entry["successes"] += int(ok)
        entry["ticks"] += ticks if ok else 0

    manifest = {
        "kind": "policy",
        "samples": sum(e["samples"] for e in episodes),
        "episodes": len(episodes),
        "episodes_with_data": sum(1 for e in episodes if e["samples"] > 0),
        "seeds": [args.seed0, args.seed0 + args.episodes - 1],
        "stride": args.stride,
        "chunk": CHUNK,
        "image_size": IMAGE_SIZE,
        "dr_scale": args.dr,
        "oracle_step_stats": per_skill,
        "per_episode": episodes,
        "shards": sorted(p for r in results for p in r["shards"]),
        "seconds": round(elapsed, 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"policy: {manifest['samples']} samples from {len(episodes)} episodes in {elapsed:.0f}s")
    for skill, entry in sorted(per_skill.items()):
        rate = entry["successes"] / max(1, entry["attempts"])
        print(f"  oracle {skill:<12} {entry['successes']:>4}/{entry['attempts']:<4} ({rate:.0%})")
    return manifest


# -------------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--perception", action="store_true", help="collect the perception dataset")
    ap.add_argument("--seeds", type=int, default=640, help="perception: number of seeds")
    ap.add_argument("--frames-per-seed", type=int, default=16)
    ap.add_argument("--episodes", type=int, default=120, help="policy: oracle episodes to run")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--stride", type=int, default=3, help="policy: control ticks per observation")
    ap.add_argument("--shard-size", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default="", help="shard-name prefix, so a run can extend a set")
    ap.add_argument("--append", action="store_true", help="keep shards already in --out")
    ap.add_argument("--dr", type=float, default=1.0, help="domain-randomization scale")
    ap.add_argument("--budget", type=float, default=260.0, help="policy: seconds per episode")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.perception:
        collect_perception(args)
    else:
        collect_policy(args)
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
