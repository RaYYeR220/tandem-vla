"""Latency/throughput benchmarking and reporting over compiled OpenVINO models."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import openvino as ov

from tandem.bench.ovutil import detect_precision, model_size_mb

_EMPTY_LATENCY = {"mean": float("nan"), "p50": float("nan"), "p90": float("nan"), "p99": float("nan"), "std": float("nan")}


def _latency_stats(samples_ns: Sequence[int]) -> dict[str, float]:
    arr = np.asarray(samples_ns, dtype=np.float64) / 1e6  # ns -> ms
    return {
        "mean": round(float(arr.mean()), 4),
        "p50": round(float(np.percentile(arr, 50)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
        "p99": round(float(np.percentile(arr, 99)), 4),
        "std": round(float(arr.std()), 4),
    }


def _full_device_name(core: ov.Core, device: str) -> str:
    for candidate in (device, device.split(".")[0]):
        try:
            return str(core.get_property(candidate, "FULL_DEVICE_NAME"))
        except Exception:
            continue
    return device


def benchmark(
    ir_path: str | Path,
    inputs: Mapping[str, np.ndarray],
    *,
    device: str,
    hint: str = "LATENCY",
    warmup: int = 20,
    iters: int = 200,
) -> dict[str, Any]:
    """Synchronous, single-stream latency benchmark of one IR on one device."""
    ir_path = Path(ir_path)
    core = ov.Core()
    model = core.read_model(ir_path)
    precision = detect_precision(model)

    t0 = time.perf_counter()
    compiled = core.compile_model(model, device, {"PERFORMANCE_HINT": hint})
    compile_ms = (time.perf_counter() - t0) * 1000.0

    request = compiled.create_infer_request()
    sample = dict(inputs)
    for _ in range(warmup):
        request.infer(sample)

    samples_ns: list[int] = []
    for _ in range(iters):
        t_start = time.perf_counter_ns()
        request.infer(sample)
        samples_ns.append(time.perf_counter_ns() - t_start)

    latency_ms = _latency_stats(samples_ns)
    mean_s = latency_ms["mean"] / 1000.0
    throughput_fps = round(1.0 / mean_s, 3) if mean_s > 0 else float("inf")

    return {
        "device": device,
        "device_full_name": _full_device_name(core, device),
        "precision": precision,
        "latency_ms": latency_ms,
        "throughput_fps": throughput_fps,
        "iters": iters,
        "compile_ms": round(compile_ms, 3),
        "model_mb": model_size_mb(ir_path),
    }


def benchmark_throughput(
    ir_path: str | Path,
    inputs: Mapping[str, np.ndarray],
    *,
    device: str,
    hint: str = "THROUGHPUT",
    warmup: int = 20,
    iters: int = 200,
) -> dict[str, Any]:
    """Async, multi-request throughput benchmark via ``ov.AsyncInferQueue``."""
    ir_path = Path(ir_path)
    core = ov.Core()
    model = core.read_model(ir_path)
    precision = detect_precision(model)

    t0 = time.perf_counter()
    compiled = core.compile_model(model, device, {"PERFORMANCE_HINT": hint})
    compile_ms = (time.perf_counter() - t0) * 1000.0

    try:
        nireq = int(compiled.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")) or 4
    except Exception:
        nireq = 4
    nireq = max(1, nireq)

    sample = dict(inputs)
    queue = ov.AsyncInferQueue(compiled, nireq)

    for _ in range(warmup):
        queue.start_async(sample)
    queue.wait_all()

    starts: dict[int, int] = {}
    latencies_ns: list[int] = []

    def _on_done(_request: ov.InferRequest, userdata: int) -> None:
        latencies_ns.append(time.perf_counter_ns() - starts[userdata])

    queue.set_callback(_on_done)

    wall_start = time.perf_counter_ns()
    for i in range(iters):
        starts[i] = time.perf_counter_ns()
        queue.start_async(sample, i)
    queue.wait_all()
    wall_s = (time.perf_counter_ns() - wall_start) / 1e9

    throughput_fps = round(iters / wall_s, 3) if wall_s > 0 else float("inf")
    latency_ms = _latency_stats(latencies_ns) if latencies_ns else dict(_EMPTY_LATENCY)

    return {
        "device": device,
        "device_full_name": _full_device_name(core, device),
        "precision": precision,
        "latency_ms": latency_ms,
        "throughput_fps": throughput_fps,
        "iters": iters,
        "compile_ms": round(compile_ms, 3),
        "model_mb": model_size_mb(ir_path),
    }


def sweep(
    models: Mapping[str, Path],
    inputs: Mapping[str, np.ndarray],
    devices: Sequence[str] | None = None,
    *,
    warmup: int = 20,
    iters: int = 200,
) -> list[dict[str, Any]]:
    """Cross {model variant} x {device} x {LATENCY, THROUGHPUT}.

    Combinations a plugin refuses (device absent, precision unsupported, ...)
    are recorded with ``"skipped": True`` and a human-readable ``"reason"``
    instead of raising or vanishing from the report.
    """
    core = ov.Core()
    candidate_devices = list(devices) if devices else list(core.available_devices)

    rows: list[dict[str, Any]] = []
    for name, ir_path in models.items():
        for device in candidate_devices:
            for hint, fn in (("LATENCY", benchmark), ("THROUGHPUT", benchmark_throughput)):
                try:
                    result = fn(ir_path, inputs, device=device, hint=hint, warmup=warmup, iters=iters)
                except Exception as exc:
                    rows.append(
                        {
                            "model": name,
                            "device": device,
                            "hint": hint,
                            "skipped": True,
                            "reason": f"{type(exc).__name__}: {exc}".splitlines()[0][:300],
                        }
                    )
                    continue
                result["model"] = name
                result["hint"] = hint
                result["skipped"] = False
                rows.append(result)
    return rows


_COLUMNS = [
    "model",
    "device",
    "device_full_name",
    "precision",
    "hint",
    "latency_mean_ms",
    "latency_p50_ms",
    "latency_p90_ms",
    "latency_p99_ms",
    "latency_std_ms",
    "throughput_fps",
    "compile_ms",
    "model_mb",
    "iters",
    "speedup_vs_fp32_cpu",
    "skipped",
    "reason",
]


def _flatten(row: Mapping[str, Any]) -> dict[str, Any]:
    flat = dict(row)
    latency = flat.pop("latency_ms", None)
    if isinstance(latency, Mapping):
        for key, value in latency.items():
            flat[f"latency_{key}_ms"] = value
    return flat


def _active_columns(flat_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [c for c in _COLUMNS if any(c in row for row in flat_rows)]


def to_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render sweep rows as a GitHub-flavored Markdown table."""
    if not rows:
        return "_no results_"
    flat_rows = [_flatten(r) for r in rows]
    columns = _active_columns(flat_rows)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in flat_rows:
        cells = []
        for col in columns:
            value = row.get(col, "")
            if value is None:
                value = ""
            elif isinstance(value, float):
                value = f"{value:.3f}" if value == value else "NaN"  # NaN check
            cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def to_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Write sweep rows to a flat CSV file, returning the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat_rows = [_flatten(r) for r in rows]
    columns = _active_columns(flat_rows) or _COLUMNS
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in flat_rows:
            writer.writerow(row)
    return path
