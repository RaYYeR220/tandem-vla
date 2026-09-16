"""Record an episode to MP4 straight out of the simulator.

    python scripts/record_demo.py --seed 3 --out results/demo.mp4

Renders the cinematic view with the wrist cameras inset and burns in what the system is doing:
the instruction, the step being executed, and the gate's verdict on it. Everything on screen is
read from the same event stream the dashboard consumes, so the overlay cannot drift from what
actually happened.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tandem.control.primitives import Executor  # noqa: E402
from tandem.control.runner import EpisodeRunner  # noqa: E402
from tandem.eval.tasks import canonical_plan, make_replanner  # noqa: E402
from tandem.sim.env import TandemEnv  # noqa: E402
from tandem.sim.randomize import RandomizationSpec  # noqa: E402

INK = (24, 27, 28)
PAPER = (246, 251, 253)
TEAL = (142, 156, 31)
ORANGE = (31, 90, 255)
AMBER = (61, 163, 232)
FONT = cv2.FONT_HERSHEY_DUPLEX


class Overlay:
    """Mutable banner state, updated from runner events."""

    def __init__(self, instruction: str, seed: int):
        self.instruction = instruction
        self.seed = seed
        self.step = "standing by"
        self.arm = ""
        self.verdict = ""
        self.code = ""
        self.reason = ""
        self.status = ""

    def on_event(self, ev: dict) -> None:
        kind = ev.get("type")
        if kind == "verdict":
            v = ev["verdict"]
            self.verdict, self.code, self.reason = v["verdict"], v["code"], v.get("reason", "")
        elif kind == "step":
            self.step = ev.get("skill", "")
            self.arm = ev.get("arm", "") or self.arm
            self.status = ev.get("status", "")
        elif kind == "refusal":
            self.verdict, self.reason = "REFUSE", ev.get("reason", "")


def _panel(img, x, y, w, h, alpha=0.82):
    sub = img[y : y + h, x : x + w]
    box = np.full(sub.shape, INK, dtype=np.uint8)
    img[y : y + h, x : x + w] = cv2.addWeighted(box, alpha, sub, 1 - alpha, 0)


def compose(env: TandemEnv, ov: Overlay, size=(720, 1280)) -> np.ndarray:
    h, w = size
    main = env.render("cinematic", h, w)
    frame = cv2.cvtColor(main, cv2.COLOR_RGB2BGR)

    # Wrist insets, bottom right.
    iw, ih = 224, 168
    for i, cam in enumerate(("left_wrist", "right_wrist")):
        tile = cv2.cvtColor(env.render(cam, ih, iw), cv2.COLOR_RGB2BGR)
        x = w - iw - 24
        y = h - (ih + 12) * (2 - i) - 24
        frame[y : y + ih, x : x + iw] = tile
        cv2.rectangle(frame, (x, y), (x + iw, y + ih), PAPER, 1)
        cv2.putText(frame, cam.replace("_", " "), (x + 8, y + 18), FONT, 0.42, PAPER, 1,
                    cv2.LINE_AA)

    # Header.
    _panel(frame, 0, 0, w, 96)
    cv2.putText(frame, "TANDEM", (28, 44), FONT, 0.85, PAPER, 1, cv2.LINE_AA)
    cv2.putText(frame, f"seed {ov.seed}", (28, 74), FONT, 0.46, AMBER, 1, cv2.LINE_AA)
    cv2.putText(frame, f'"{ov.instruction}"', (168, 44), FONT, 0.62, PAPER, 1, cv2.LINE_AA)
    score = env.task_score()
    cv2.putText(frame, f"subgoals {score['completed']}/{score['total']}", (168, 74), FONT,
                0.46, TEAL if score["completed"] else AMBER, 1, cv2.LINE_AA)

    # Footer: current step and the gate's verdict on it.
    _panel(frame, 0, h - 88, w, 88)
    label = f"{ov.step}  {ov.arm}".strip() or "standing by"
    cv2.putText(frame, label.upper(), (28, h - 50), FONT, 0.62, PAPER, 1, cv2.LINE_AA)
    if ov.verdict:
        col = {"ALLOW": TEAL, "REFUSE": ORANGE, "REWRITE": AMBER}.get(ov.verdict, PAPER)
        cv2.putText(frame, ov.verdict, (28, h - 18), FONT, 0.58, col, 1, cv2.LINE_AA)
        if ov.reason:
            cv2.putText(frame, ov.reason[:92], (150, h - 18), FONT, 0.46, PAPER, 1, cv2.LINE_AA)
    return frame


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("results/demo.mp4"))
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--every", type=int, default=2, help="record one frame per N control ticks")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--instruction", default="set the table for one and pour me some water")
    ap.add_argument("--no-pour", action="store_true")
    args = ap.parse_args()

    intent = {"place": ["plate", "fork", "spoon", "mug"], "pour": not args.no_pour}
    env = TandemEnv(RandomizationSpec(scale=1.0))
    world = env.reset(args.seed)
    ov = Overlay(args.instruction, args.seed)
    frames: list[np.ndarray] = []

    def tick(ex: Executor) -> None:
        if ex.tick_count % args.every == 0:
            frames.append(compose(env, ov, (args.height, args.width)))

    ex = Executor(env, on_tick=tick)
    runner = EpisodeRunner(env, ex, on_event=ov.on_event,
                           replanner=make_replanner(env, intent))
    rec = runner.run(canonical_plan(world, intent), instruction=args.instruction, budget_s=400)

    for _ in range(args.fps):  # let the final state sit on screen
        frames.append(compose(env, ov, (args.height, args.width)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(args.out, fps=args.fps, quality=8, macro_block_size=1)
    for f in frames:
        writer.append_data(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    writer.close()

    sub = rec.score["subgoals"]
    print(f"{args.out}  {len(frames)} frames  {len(frames) / args.fps:.1f}s")
    print("subgoals: " + ", ".join(f"{k}={'ok' if v else 'no'}" for k, v in sub.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
