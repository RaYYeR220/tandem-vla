"""Export the planner LLM to an OpenVINO IR directory.

Wraps ``optimum-cli export openvino`` so the demo has one reproducible command:

    python scripts/export_planner.py                  # INT4, models/planner-ov
    python scripts/export_planner.py --int8
    python scripts/export_planner.py --model <hf-id> --out <dir> --force

    # the smaller alternative, scored against the default by tandem/planner/benchmark.py
    python scripts/export_planner.py --model Qwen/Qwen2.5-0.5B-Instruct \
                                     --out models/planner-ov-0.5b

The first run downloads several GB of weights from Hugging Face. Subsequent runs are a
no-op unless ``--force`` is passed. If the export never happens the planner still runs —
it falls back to the rule-based backend — so this script is an upgrade, not a dependency.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_OUT = REPO_ROOT / "models" / "planner-ov"

#: A finished export always contains these; a half-written directory usually does not.
REQUIRED_ARTIFACTS = ("openvino_model.xml", "openvino_model.bin")


def dir_size_mb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def is_complete(path: Path) -> bool:
    return path.is_dir() and all((path / name).exists() for name in REQUIRED_ARTIFACTS)


def build_command(model: str, out: Path, int8: bool, sym: bool, ratio: float, group_size: int) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "optimum.commands.optimum_cli",
        "export",
        "openvino",
        "--model",
        model,
        "--task",
        "text-generation-with-past",
        "--trust-remote-code",
    ]
    if int8:
        cmd += ["--weight-format", "int8"]
    else:
        cmd += [
            "--weight-format",
            "int4",
            "--ratio",
            str(ratio),
            "--group-size",
            str(group_size),
        ]
        if sym:
            cmd += ["--sym"]
    cmd.append(str(out))
    return cmd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export the planner LLM to OpenVINO IR.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id or local path")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output IR directory")
    parser.add_argument("--int8", action="store_true", help="export INT8 weights instead of INT4")
    parser.add_argument("--sym", action="store_true", help="symmetric INT4 quantisation (smaller, slightly lossier)")
    parser.add_argument("--ratio", type=float, default=1.0, help="INT4 weight ratio, 1.0 = quantise everything")
    parser.add_argument("--group-size", type=int, default=128, help="INT4 quantisation group size")
    parser.add_argument("--force", action="store_true", help="re-export even if the output already exists")
    args = parser.parse_args(argv)

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out

    if is_complete(out) and not args.force:
        print(f"already exported: {out}  ({dir_size_mb(out):.1f} MB)")
        print("pass --force to re-export")
        return 0

    if out.exists() and args.force:
        print(f"removing {out}")
        shutil.rmtree(out)

    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = build_command(args.model, out, args.int8, args.sym, args.ratio, args.group_size)
    print("$ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        print(f"export failed with exit code {result.returncode}", file=sys.stderr)
        return result.returncode

    if not is_complete(out):
        print(f"export finished but {out} is missing {REQUIRED_ARTIFACTS}", file=sys.stderr)
        return 1

    precision = "INT8" if args.int8 else "INT4"
    print(f"\nexported {args.model} -> {out}")
    print(f"precision: {precision}")
    print(f"size: {dir_size_mb(out):.1f} MB")
    print("\nset TANDEM_PLANNER=openvino (the default) to use it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
