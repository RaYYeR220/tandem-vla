"""Tests for tandem.bench: device introspection, export/quantize/benchmark round-trip, sweep safety."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from tandem.bench import ovutil, runner


class _TinyTwoLayer(nn.Module):
    """Big enough that INT8 packing visibly shrinks the .bin (weights dominate)."""

    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(256, 512)
        self.l2 = nn.Linear(512, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2(torch.relu(self.l1(x)))


def test_device_report_has_cpu_and_correct_intel_flag() -> None:
    report = ovutil.device_report()

    devices = {entry["device"] for entry in report["devices"]}
    assert "CPU" in devices
    assert isinstance(report["intel_validated"], bool)

    # This dev machine is an AMD Ryzen 5 5600X - not Intel-validated hardware.
    assert report["intel_validated"] is False
    assert "amd" in report["processor"].lower()
    assert "caveat" in report and report["caveat"]


def test_export_quantize_benchmark_roundtrip(tmp_path: Path) -> None:
    model = _TinyTwoLayer().eval()
    example_x = torch.randn(1, 256)

    fp32_path = ovutil.export_ir(model, (example_x,), tmp_path / "tiny_fp32.xml", fp16=False)
    static_fp32_path = ovutil.reshape_static(fp32_path, {"x": [1, 256]}, tmp_path / "tiny_fp32_static.xml")

    rng = np.random.default_rng(0)
    calib_samples = [{"x": rng.standard_normal((1, 256)).astype(np.float32)} for _ in range(30)]
    int8_path = ovutil.quantize_int8(static_fp32_path, calib_samples, tmp_path / "tiny_int8.xml", subset_size=30)

    sample_input = {"x": rng.standard_normal((1, 256)).astype(np.float32)}

    fp32_result = runner.benchmark(static_fp32_path, sample_input, device="CPU", warmup=5, iters=20)
    int8_result = runner.benchmark(int8_path, sample_input, device="CPU", warmup=5, iters=20)

    for result in (fp32_result, int8_result):
        assert math.isfinite(result["latency_ms"]["mean"])
        assert result["latency_ms"]["mean"] > 0
        assert math.isfinite(result["throughput_fps"])
        assert result["throughput_fps"] > 0

    assert fp32_result["precision"] == "FP32"
    assert int8_result["precision"] == "INT8"

    fp32_bin_size = static_fp32_path.with_suffix(".bin").stat().st_size
    int8_bin_size = int8_path.with_suffix(".bin").stat().st_size
    assert int8_bin_size < fp32_bin_size * 0.6, (
        f"expected INT8 .bin meaningfully smaller than FP32: {int8_bin_size} vs {fp32_bin_size}"
    )


def test_sweep_records_skip_reason_for_bogus_device(tmp_path: Path) -> None:
    model = _TinyTwoLayer().eval()
    example_x = torch.randn(1, 256)
    fp32_path = ovutil.export_ir(model, (example_x,), tmp_path / "tiny_fp32.xml", fp16=False)
    static_path = ovutil.reshape_static(fp32_path, {"x": [1, 256]}, tmp_path / "tiny_fp32_static.xml")

    sample_input = {"x": np.random.default_rng(1).standard_normal((1, 256)).astype(np.float32)}

    rows = runner.sweep(
        {"tiny": static_path},
        sample_input,
        devices=["NOT_A_REAL_DEVICE_XYZ"],
        warmup=2,
        iters=5,
    )

    assert len(rows) == 2  # LATENCY + THROUGHPUT hints, both skipped
    for row in rows:
        assert row["skipped"] is True
        assert row["reason"]
        assert row["device"] == "NOT_A_REAL_DEVICE_XYZ"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
