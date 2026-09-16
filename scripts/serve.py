"""Run the Tandem dashboard's backend.

    python scripts/serve.py --port 8000

Then open http://localhost:8000/ for the landing page and
http://localhost:8000/app.html for the live console.

    python scripts/serve.py --build-scorecard 5

Runs 5 seeds of the canonical task first (real episodes, real time — a few minutes) and
writes ``results/eval/scorecard.json`` / ``.md`` before starting the server, so
``GET /api/scorecard`` and the dashboard's scorecard panel have something real to show.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn  # noqa: E402

from tandem.server.app import app  # noqa: E402
from tandem.server.scorecard import build_scorecard  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--log-level", default="info")
    ap.add_argument(
        "--build-scorecard",
        type=int,
        default=0,
        metavar="N",
        help="run N seeds of the canonical task and write results/eval/scorecard.json before serving",
    )
    ap.add_argument("--scorecard-seed0", type=int, default=0)
    ap.add_argument("--scorecard-dr", type=float, default=1.0)
    ap.add_argument("--scorecard-budget", type=float, default=150.0)
    ap.add_argument("--scorecard-only", action="store_true", help="build the scorecard, then exit")
    args = ap.parse_args()

    if args.build_scorecard > 0:
        print(f"Building scorecard: {args.build_scorecard} seed(s) of the canonical task ...")
        summary = build_scorecard(
            seeds=args.build_scorecard,
            seed0=args.scorecard_seed0,
            dr_scale=args.scorecard_dr,
            budget_s=args.scorecard_budget,
        )
        rate = summary["task_success"]["rate_all"]
        print(f"Scorecard written to results/eval/ — all-subgoals success rate {rate:.0%}")
        if args.scorecard_only:
            return 0

    print(f"Serving on http://{args.host}:{args.port}/  (dashboard: /app.html)")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
