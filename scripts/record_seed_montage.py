"""Render one video showing the same instruction run on N randomized seeds at once.

    python scripts/record_seed_montage.py --seeds 10 --out results/seeds.mp4

Every tile is a different seed: different object positions and yaws, different masses, frictions
and sizes, different lighting and materials. The instruction is identical in all of them. Each
tile carries its seed and, as the run finishes, its subgoal tally — so the scene variation and the
outcome are both legible without trusting a voice-over.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tandem.control.primitives import Executor  # noqa: E402
from tandem.control.runner import EpisodeRunner  # noqa: E402
from tandem.eval.tasks import canonical_plan, make_replanner  # noqa: E402
from tandem.sim.env import TandemEnv  # noqa: E402
from tandem.sim.randomize import RandomizationSpec  # noqa: E402

INK = (24, 27, 28)
PAPER = (246, 251, 253)
MUTED = (110, 131, 138)
TEAL = (142, 156, 31)
ORANGE = (31, 90, 255)
AMBER = (61, 163, 232)
FONT = cv2.FONT_HERSHEY_DUPLEX

TILE_W, TILE_H = 376, 212
PAD = 8
HEADER = 132
FOOTER = 96


def run_seed(env: TandemEnv, seed: int, intent: dict, every: int, budget: float):
    """Run one episode, keeping a downsampled frame every `every` control ticks."""
    world = env.reset(seed)
    frames: list[np.ndarray] = []

    def tick(ex: Executor) -> None:
        if ex.tick_count % every == 0:
            frames.append(env.render("cinematic", TILE_H, TILE_W))

    ex = Executor(env, on_tick=tick)
    rec = EpisodeRunner(env, ex, replanner=make_replanner(env, intent)).run(
        canonical_plan(world, intent), budget_s=budget
    )
    return frames, rec.score


def label_tile(img: np.ndarray, seed: int, score: dict | None, done: bool) -> np.ndarray:
    tile = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()
    cv2.rectangle(tile, (0, 0), (TILE_W - 1, TILE_H - 1), (60, 66, 70), 1)
    cv2.rectangle(tile, (0, 0), (74, 22), INK, -1)
    cv2.putText(tile, f"seed {seed}", (7, 16), FONT, 0.40, PAPER, 1, cv2.LINE_AA)
    if score is not None:
        n, tot = score["completed"], score["total"]
        col = TEAL if n >= tot - 1 else (AMBER if n >= tot // 2 else ORANGE)
        cv2.rectangle(tile, (TILE_W - 62, 0), (TILE_W, 22), INK, -1)
        cv2.putText(tile, f"{n}/{tot}", (TILE_W - 54, 16), FONT, 0.44, col, 1, cv2.LINE_AA)
        if done:
            x = 8
            for ok in score["subgoals"].values():
                cv2.rectangle(tile, (x, TILE_H - 14), (x + 12, TILE_H - 6),
                              TEAL if ok else (70, 76, 80), -1)
                x += 16
    return tile


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--every", type=int, default=9)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--budget", type=float, default=150.0)
    ap.add_argument("--no-pour", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "results/seeds.mp4")
    args = ap.parse_args()

    intent = {"place": ["plate", "fork", "spoon", "mug"], "pour": not args.no_pour}
    instruction = ("set the table for one" if args.no_pour
                   else "set the table for one and pour me some water")
    env = TandemEnv(RandomizationSpec(scale=1.0))

    runs = []
    for i in range(args.seeds):
        seed = args.seed0 + i
        frames, score = run_seed(env, seed, intent, args.every, args.budget)
        runs.append((seed, frames, score))
        print(f"  seed {seed:>3}  {score['completed']}/{score['total']}  "
              f"{len(frames)} frames", flush=True)

    cols = args.cols
    rows = (len(runs) + cols - 1) // cols
    longest = max(len(f) for _, f, _ in runs)
    W = cols * TILE_W + (cols + 1) * PAD
    H = HEADER + rows * TILE_H + (rows + 1) * PAD + FOOTER

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(args.out, fps=args.fps, quality=8, macro_block_size=1)
    total = sum(s["completed"] for _, _, s in runs)
    possible = sum(s["total"] for _, _, s in runs)

    for k in range(longest + args.fps):  # hold the final state for a second
        canvas = np.full((H, W, 3), INK, dtype=np.uint8)
        cv2.putText(canvas, "TANDEM", (PAD + 12, 52), FONT, 1.0, PAPER, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{len(runs)} randomized seeds, one instruction",
                    (PAD + 12, 88), FONT, 0.52, MUTED, 1, cv2.LINE_AA)
        cv2.putText(canvas, f'"{instruction}"', (PAD + 12, 116), FONT, 0.56, AMBER, 1,
                    cv2.LINE_AA)
        for idx, (seed, frames, score) in enumerate(runs):
            r, c = divmod(idx, cols)
            j = min(k, len(frames) - 1)
            done = k >= len(frames) - 1
            tile = label_tile(frames[j], seed, score if done else None, done)
            y = HEADER + PAD + r * (TILE_H + PAD)
            x = PAD + c * (TILE_W + PAD)
            canvas[y : y + TILE_H, x : x + TILE_W] = tile
        cv2.putText(canvas, "position, yaw, mass, friction, size, lighting and materials all "
                            "vary by seed; the instruction does not",
                    (PAD + 12, H - 54), FONT, 0.46, MUTED, 1, cv2.LINE_AA)
        if k >= longest - 1:
            cv2.putText(canvas, f"{total}/{possible} subgoals completed",
                        (PAD + 12, H - 22), FONT, 0.56, PAPER, 1, cv2.LINE_AA)
        writer.append_data(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    writer.close()

    print(f"\n{args.out}  {W}x{H}  {(longest + args.fps) / args.fps:.1f}s  "
          f"{args.out.stat().st_size / 1e6:.1f} MB")
    print(f"total {total}/{possible} subgoals")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
