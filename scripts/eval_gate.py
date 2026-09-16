"""Grade the safety gate against its case suite.

    python scripts/eval_gate.py

Runs in about a second and needs no models, no GPU and no credentials.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tandem.eval.gate_eval import run_gate_eval, to_markdown  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/gate"))
    args = ap.parse_args()
    summary = run_gate_eval(args.seed, out_dir=args.out)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(to_markdown(summary))
    t = summary["total"]
    print(f"Written to {args.out}/")
    return 0 if t["passed"] == t["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
