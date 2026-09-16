"""Judge-facing Intel OpenVINO inference benchmark.

    python scripts/benchmark_intel.py [--models DIR] [--devices CPU,GPU,NPU] [--iters N] [--out results/]

Prints the OpenVINO device/host report first (device-agnostic: it reports
whatever this host's OpenVINO Core actually enumerates, with an explicit
caveat when the CPU isn't Intel-validated hardware), then discovers every
``*.xml`` IR under ``--models``, groups variants by stem (``policy_fp32.xml``,
``policy_fp16.xml``, ``policy_int8.xml``, ...), synthesizes random inputs
straight from each IR's declared shapes/dtypes, and benchmarks the full
{variant} x {device} x {LATENCY, THROUGHPUT} grid.

If no models exist yet, it builds, exports, quantizes and benchmarks a small
stand-in convnet on the fly so the deliverable always demonstrates a real,
measured OpenVINO run - clearly labeled SELF-TEST MODEL throughout.

A missing device (no NPU on this host, say) is a normal, reported outcome and
never turns into a non-zero exit code; only genuine errors do.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import openvino as ov
import torch
import torch.nn as nn

from tandem.bench import ovutil, runner

DEFAULT_MODELS_DIR = REPO_ROOT / "models"
DEFAULT_OUT_DIR = REPO_ROOT / "results"
DEFAULT_ITERS = 200
DEFAULT_WARMUP = 20


class SelfTestPolicy(nn.Module):
    """Stand-in for the real vision+proprio policy - same I/O shape, throwaway weights.

    4-layer CNN over a 3x128x128 image plus a 12-float proprio vector -> 96
    outputs, matching the real policy's interface so this benchmark always
    demonstrates a genuine OpenVINO run even before a trained model exists.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.proprio = nn.Linear(12, 64)
        self.head = nn.Linear(64 * 8 * 8 + 64, 96)

    def forward(self, image: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        feat = torch.flatten(self.conv(image), 1)
        p = torch.relu(self.proprio(proprio))
        return self.head(torch.cat([feat, p], dim=-1))


def build_self_test_models(out_dir: Path) -> dict[str, Path]:
    """Export + quantize the stand-in policy; returns {"fp32", "fp16", "int8"} -> IR path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model = SelfTestPolicy().eval()
    example = {"image": torch.randn(1, 3, 128, 128), "proprio": torch.randn(1, 12)}

    fp32_path = ovutil.export_ir(model, example, out_dir / "policy_fp32.xml", fp16=False)
    fp16_path = ovutil.export_ir(model, example, out_dir / "policy_fp16.xml", fp16=True)

    rng = np.random.default_rng(0)
    calib_samples = [
        {
            "image": rng.standard_normal((1, 3, 128, 128)).astype(np.float32),
            "proprio": rng.standard_normal((1, 12)).astype(np.float32),
        }
        for _ in range(64)
    ]
    int8_path = ovutil.quantize_int8(fp32_path, calib_samples, out_dir / "policy_int8.xml", subset_size=64)

    return {"fp32": fp32_path, "fp16": fp16_path, "int8": int8_path}


def discover_models(models_dir: Path) -> dict[str, dict[str, Path]]:
    """Group every top-level ``*.xml`` in models_dir by base stem -> {variant_label: path}.

    Only the top level is scanned on purpose: ``models/`` is shared with other
    pipeline stages that keep their own exports in subdirectories (e.g. an
    exported planner LLM plus its tokenizer/detokenizer IRs) and this bench's
    own variants belong directly under ``models/`` per the naming convention
    (``policy_fp32.xml``, ...), so a subdirectory is never mistaken for one of
    them. Anything a plain ``ov.Core`` still can't read is skipped with a
    printed note rather than crashing discovery.
    """
    groups: dict[str, dict[str, Path]] = {}
    core = ov.Core()
    for xml_path in sorted(models_dir.glob("*.xml")):
        try:
            core.read_model(xml_path)
        except Exception as exc:
            reason = str(exc).splitlines()[0][:160]
            print(f"  (skipping {xml_path}: not a plain IR this bench can read - {type(exc).__name__}: {reason})")
            continue
        base, suffix = ovutil.split_precision_suffix(xml_path.stem)
        groups.setdefault(base, {})[suffix or xml_path.stem] = xml_path
    return groups


def _safe_name(text: str) -> str:
    return re.sub(r"[^\w.-]+", "_", text)


def _prepare_static(variant_path: Path, static_dir: Path, label: str) -> tuple[Path, dict[str, np.ndarray]]:
    """Synthesize inputs from the IR's own shapes/dtypes, freeze those shapes, return both."""
    inputs = ovutil.synthesize_inputs(variant_path)
    shapes = {name: list(arr.shape) for name, arr in inputs.items()}
    static_dir.mkdir(parents=True, exist_ok=True)
    static_path = ovutil.reshape_static(variant_path, shapes, static_dir / f"{_safe_name(label)}.xml")
    return static_path, inputs


def _compute_speedups(rows: list[dict[str, Any]]) -> None:
    """Add "speedup_vs_fp32_cpu" in place: baseline is the FP32-precision CPU row, same hint."""
    baseline_by_hint: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not row["skipped"] and row["device"] == "CPU" and row["precision"] == "FP32":
            baseline_by_hint.setdefault(row["hint"], row)

    for row in rows:
        baseline = baseline_by_hint.get(row["hint"])
        if row["skipped"] or baseline is None:
            row["speedup_vs_fp32_cpu"] = None
            continue
        if row["hint"] == "LATENCY":
            base_mean, this_mean = baseline["latency_ms"]["mean"], row["latency_ms"]["mean"]
            row["speedup_vs_fp32_cpu"] = round(base_mean / this_mean, 3) if this_mean else None
        else:
            base_fps = baseline["throughput_fps"]
            row["speedup_vs_fp32_cpu"] = round(row["throughput_fps"] / base_fps, 3) if base_fps else None


def run_sweep_for_groups(
    groups: dict[str, dict[str, Path]],
    static_dir: Path,
    devices: list[str] | None,
    warmup: int,
    iters: int,
) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    for group_name, variants in groups.items():
        try:
            representative = variants.get("fp32") or next(iter(variants.values()))
            static_variants: dict[str, Path] = {}
            group_inputs: dict[str, np.ndarray] | None = None
            for variant_label, path in variants.items():
                static_path, inputs = _prepare_static(path, static_dir, f"{group_name}__{variant_label}")
                static_variants[variant_label] = static_path
                if path == representative:
                    group_inputs = inputs
            if group_inputs is None:  # pragma: no cover - representative always iterated above
                group_inputs = ovutil.synthesize_inputs(representative)
        except Exception as exc:
            # A model this bench can't confidently prepare (foreign shape signature,
            # unsupported dtype, ...) is a reported skip, not a run-ending crash.
            reason = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
            print(f"  (skipping group '{group_name}': could not prepare static inputs - {reason})")
            all_rows.append({"model": group_name, "device": "-", "hint": "-", "skipped": True, "reason": reason})
            continue

        rows = runner.sweep(static_variants, group_inputs, devices=devices, warmup=warmup, iters=iters)
        tag = "SELF-TEST:" if "_selftest" in group_name.lower() else ""
        for row in rows:
            row["model"] = f"{tag}{group_name}_{row['model']}"
        _compute_speedups(rows)
        all_rows.extend(rows)
    return all_rows


def _print_device_report(report: dict[str, Any]) -> None:
    print("=" * 78)
    print("HOST / DEVICE REPORT")
    print("=" * 78)
    print(f"processor      : {report['processor']}")
    print(f"physical cores : {report['physical_cores']}")
    print(f"logical cores  : {report['logical_cores']}")
    print(f"RAM            : {report['ram_gb']} GB")
    print(f"OS             : {report['os']}")
    print(f"intel_validated: {report['intel_validated']}")
    if not report["intel_validated"]:
        print()
        print("CAVEAT: " + report["caveat"])
    print()
    for entry in report["devices"]:
        print(f"- {entry['device']}: {entry.get('full_device_name', '?')}")
        for key in (
            "device_type",
            "device_architecture",
            "optimization_capabilities",
            "range_for_async_infer_requests",
        ):
            if key in entry:
                print(f"    {key}: {entry[key]}")
    print()


def _slugify_host() -> str:
    import platform

    name = platform.node() or "unknown-host"
    return _safe_name(name).strip("_") or "unknown-host"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", default=str(DEFAULT_MODELS_DIR), help="directory to scan for *.xml IR variants")
    parser.add_argument("--devices", default=None, help="comma-separated OpenVINO devices, e.g. CPU,GPU,NPU (default: autodetect)")
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS, help="timed iterations per benchmark")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help="warmup iterations per benchmark")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR), help="output directory for results")
    args = parser.parse_args(argv)

    try:
        models_dir = Path(args.models)
        if not models_dir.is_absolute():
            models_dir = REPO_ROOT / models_dir
        out_dir = Path(args.out)
        if not out_dir.is_absolute():
            out_dir = REPO_ROOT / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        devices = [d.strip() for d in args.devices.split(",") if d.strip()] if args.devices else None

        report = ovutil.device_report()
        _print_device_report(report)

        groups = discover_models(models_dir)
        if not groups:
            print(f"no *.xml models found under {models_dir} - building the SELF-TEST MODEL instead")
            print("(4-layer CNN, 3x128x128 image + 12-float proprio -> 96 outputs; matches the real policy's I/O shape)")
            print()
            self_test_variants = build_self_test_models(models_dir / "_selftest")
            groups = {"_selftest/policy": self_test_variants}

        static_dir = out_dir / "_static_models"
        rows = run_sweep_for_groups(groups, static_dir, devices, args.warmup, args.iters)

        host_slug = _slugify_host()
        json_path = out_dir / f"benchmark_{host_slug}.json"
        csv_path = out_dir / f"benchmark_{host_slug}.csv"
        md_path = out_dir / f"benchmark_{host_slug}.md"

        json_path.write_text(json.dumps({"host": report, "results": rows}, indent=2), encoding="utf-8")
        runner.to_csv(rows, csv_path)

        table_md = runner.to_markdown(rows)
        md_lines = ["# Intel OpenVINO benchmark", ""]
        if not report["intel_validated"]:
            md_lines += [f"> **CAVEAT:** {report['caveat']}", ""]
        if any("_selftest" in str(g).lower() for g in groups):
            md_lines += [
                "**SELF-TEST MODEL** - no trained policy weights existed yet at run time; "
                "rows tagged `SELF-TEST:` are a stand-in convnet with the real policy's I/O shape, "
                "not the production model.",
                "",
            ]
        md_lines += [table_md, ""]
        md_path.write_text("\n".join(md_lines), encoding="utf-8")

        print(table_md)
        print()
        print(f"wrote {json_path}")
        print(f"wrote {csv_path}")
        print(f"wrote {md_path}")

        skipped = sum(1 for r in rows if r["skipped"])
        ran = len(rows) - skipped
        print(f"\n{ran} benchmarks ran, {skipped} combination(s) skipped (see the reason column/field).")
        return 0
    except Exception as exc:  # a missing device is a skip, not this; this is a genuine failure
        print(f"benchmark_intel.py failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
