"""Score stage-A intent parsing across backends.

    python tandem/planner/benchmark.py                 # all available backends
    python tandem/planner/benchmark.py --backend 0.5b
    python tandem/planner/benchmark.py --device CPU --repeat 3

Fourteen instructions, ten of which are actionable and four of which must be refused.
A case counts as correct only when every field the case pins down matches — getting
``pour`` right while inventing a plate is a failure, not a partial credit.

The numbers this prints are the ones quoted in the submission, so it measures stage A in
isolation: no fallback to the keyword parser, no stage B.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

if __package__ in (None, ""):  # direct execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tandem.planner.backends import DEFAULT_IR_DIR, SMALL_IR_DIR, RuleBasedPlanner
from tandem.planner.planner import parse_intent

#: ``place`` is scored against a set of acceptable answers, because a couple of these are
#: genuinely ambiguous about whether the mug counts as "laid out" or "filled".
CASES: list[dict] = [
    {
        "instruction": "set the table for one",
        "place": [{"plate", "fork", "spoon", "mug"}],
        "pour": False,
    },
    {
        "instruction": "set the table and pour me some water",
        "place": [{"plate", "fork", "spoon", "mug"}, {"plate", "fork", "spoon"}],
        "pour": True,
    },
    {
        "instruction": "just put the plate and the fork out",
        "place": [{"plate", "fork"}],
        "pour": False,
    },
    {
        "instruction": "put the mug on the table",
        "place": [{"mug"}],
        "pour": False,
    },
    {
        "instruction": "pour me some water",
        "place": [set(), {"mug"}],
        "pour": True,
    },
    {
        "instruction": "fill my mug please",
        "place": [set(), {"mug"}],
        "pour": True,
    },
    {
        "instruction": "I need a spoon and a fork",
        "place": [{"spoon", "fork"}],
        "pour": False,
    },
    {
        "instruction": "open the drawer",
        "place": [set()],
        "pour": False,
        "open_drawer": True,
    },
    {
        "instruction": "close the drawer please",
        "place": [set()],
        "pour": False,
        "close_drawer": True,
    },
    {
        "instruction": "put the plate down and fill the mug",
        "place": [{"plate"}, {"plate", "mug"}],
        "pour": True,
    },
    {"instruction": "get me a knife", "refuse": True},
    {"instruction": "put a napkin next to the plate", "refuse": True},
    {"instruction": "pour me a glass of wine", "refuse": True},
    {"instruction": "bring me a bowl of soup", "refuse": True},
]


def score(case: dict, intent: dict) -> tuple[bool, str]:
    """Return ``(correct, why_not)`` for one case."""
    if case.get("refuse"):
        if intent.get("refuse"):
            return True, ""
        return False, "did not refuse"

    if intent.get("refuse"):
        return False, f"refused: {intent['refuse'][:40]}"

    got = set(intent.get("place") or [])
    if got not in case["place"]:
        want = " | ".join(sorted(str(sorted(p)) for p in case["place"]))
        return False, f"place={sorted(got)} want {want}"
    if bool(intent.get("pour")) != case["pour"]:
        return False, f"pour={intent.get('pour')} want {case['pour']}"
    for key in ("open_drawer", "close_drawer"):
        if key in case and intent.get(key) is not case[key]:
            return False, f"{key}={intent.get(key)} want {case[key]}"
    return True, ""


def run(label: str, backend, repeat: int, verbose: bool, guard: bool = True) -> dict:
    latencies: list[float] = []
    intent_ok = 0
    refuse_ok = 0
    intent_total = sum(1 for c in CASES if not c.get("refuse"))
    refuse_total = sum(1 for c in CASES if c.get("refuse"))
    failures: list[str] = []

    for case in CASES:
        correct = False
        why = ""
        for _ in range(repeat):
            started = time.perf_counter()
            intent, meta = parse_intent(case["instruction"], backend=backend, max_attempts=1, guard=guard)
            latencies.append((time.perf_counter() - started) * 1000.0)
            correct, why = score(case, intent)
            if not correct:
                break
        if correct:
            if case.get("refuse"):
                refuse_ok += 1
            else:
                intent_ok += 1
        else:
            failures.append(f"    {case['instruction']!r}: {why}")
        if verbose:
            mark = "ok  " if correct else "FAIL"
            print(f"  {mark} {case['instruction']}")

    result = {
        "label": label,
        "intent_ok": intent_ok,
        "intent_total": intent_total,
        "refuse_ok": refuse_ok,
        "refuse_total": refuse_total,
        "mean_ms": statistics.mean(latencies) if latencies else 0.0,
        "median_ms": statistics.median(latencies) if latencies else 0.0,
        "max_ms": max(latencies) if latencies else 0.0,
        "failures": failures,
    }
    return result


def make_backend(name: str, device: str | None):
    if name == "rules":
        return RuleBasedPlanner(), "rules (keyword parser)"
    from tandem.planner.backends import OpenVINOPlanner

    ir_dir = SMALL_IR_DIR if name == "0.5b" else DEFAULT_IR_DIR
    if not (ir_dir / "openvino_model.xml").exists():
        raise FileNotFoundError(f"no IR at {ir_dir} - run scripts/export_planner.py")
    return OpenVINOPlanner(ir_dir, device), f"{ir_dir.name} INT4"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score stage-A intent parsing.")
    parser.add_argument("--backend", action="append", choices=["1.5b", "0.5b", "rules"],
                        help="repeatable; default is all three")
    parser.add_argument("--device", default=None, help="OpenVINO device (default: TANDEM_OV_DEVICE or AUTO)")
    parser.add_argument("--repeat", type=int, default=1, help="runs per case, for latency stability")
    parser.add_argument("--raw", action="store_true",
                        help="measure the model alone, without the scene inventory guard")
    parser.add_argument("--both", action="store_true",
                        help="score each model twice: with the guard and without it")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    names = args.backend or ["rules", "0.5b", "1.5b"]
    results = []
    for name in names:
        try:
            backend, label = make_backend(name, args.device)
        except Exception as exc:  # noqa: BLE001 - a missing export must not stop the rest
            print(f"skipping {name}: {exc}")
            continue
        print(f"\n=== {label} ===", flush=True)
        results.append(run(label, backend, max(1, args.repeat), args.verbose, guard=not args.raw))
        if args.both and name != "rules":
            print(f"--- {label}, model alone (no scene guard) ---", flush=True)
            results.append(run(label + " [raw]", backend, max(1, args.repeat), args.verbose, guard=False))

    if not results:
        return 1

    width = max(len(r["label"]) for r in results)
    print(f"\n{'backend'.ljust(width)}  intent   refusal  total    mean ms  median ms")
    print("-" * (width + 48))
    for r in results:
        total = r["intent_ok"] + r["refuse_ok"]
        grand = r["intent_total"] + r["refuse_total"]
        print(
            f"{r['label'].ljust(width)}  "
            f"{r['intent_ok']:>2}/{r['intent_total']:<4} "
            f"{r['refuse_ok']:>2}/{r['refuse_total']:<6} "
            f"{total:>2}/{grand:<6} "
            f"{r['mean_ms']:>8.1f} {r['median_ms']:>10.1f}"
        )

    for r in results:
        if r["failures"]:
            print(f"\n{r['label']} missed:")
            print("\n".join(r["failures"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
