"""Run the seeded task evaluation and write the scorecard.

    python scripts/evaluate.py --seeds 20 --out results/eval

Every number in the README's results table comes out of this script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tandem.eval.run_eval import EvalConfig, run_eval, to_markdown  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=20, help="number of seeds, starting at --seed0")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--dr", type=float, default=1.0, help="domain-randomization scale, 0..1")
    ap.add_argument("--budget", type=float, default=260.0, help="seconds per episode")
    ap.add_argument("--speed", type=float, default=1.0, help="motion speed multiplier")
    ap.add_argument("--no-pour", action="store_true", help="place setting only")
    ap.add_argument(
        "--perception",
        nargs="?",
        const="models/perception_int8.xml",
        default=None,
        help="read the world from the cameras through this OpenVINO IR instead of from "
             "privileged simulator state",
    )
    ap.add_argument("--out", type=Path, default=Path("results/eval"))
    args = ap.parse_args()

    intent = {
        "place": ["plate", "fork", "spoon", "mug"],
        "pour": not args.no_pour,
    }
    cfg = EvalConfig(
        seeds=tuple(range(args.seed0, args.seed0 + args.seeds)),
        instruction=(
            "set the table for one"
            if args.no_pour
            else "set the table for one and pour me some water"
        ),
        intent=intent,
        dr_scale=args.dr,
        budget_s=args.budget,
        speed=args.speed,
        perception_ir=args.perception,
    )
    print(f"Running {len(cfg.seeds)} seeds at DR scale {cfg.dr_scale} ...")
    summary = run_eval(cfg, out_dir=args.out)
    print()
    print(to_markdown(summary))
    print(f"Written to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
