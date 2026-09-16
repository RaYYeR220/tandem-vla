"""OpenVINO helpers shared by the benchmark runner and CLI.

Device introspection is written to be brutally honest about what hardware it
is actually talking to: it never claims Intel-validated results on hardware
that is not Intel, and it never hides a plugin property just because one
device doesn't expose it.
"""

from __future__ import annotations

import os
import platform
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import nncf
import numpy as np
import openvino as ov

try:
    import psutil
except ImportError:  # host may not have it; every reader below tolerates None
    psutil = None  # type: ignore[assignment]


# Properties requested from every device. Plugins differ in what they expose,
# so each read is wrapped individually and silently skipped on failure.
_DEVICE_PROPERTIES = (
    "FULL_DEVICE_NAME",
    "OPTIMIZATION_CAPABILITIES",
    "DEVICE_ARCHITECTURE",
    "RANGE_FOR_ASYNC_INFER_REQUESTS",
    "DEVICE_TYPE",
)

_PRECISION_SUFFIXES = ("fp32", "fp16", "int8")


def _cpu_vendor() -> str:
    """Best-effort CPU vendor/model string, Windows and Linux."""
    ident = os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor()
    if ident:
        return ident
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        try:
            for line in cpuinfo.read_text(errors="ignore").splitlines():
                if line.lower().startswith("vendor_id"):
                    return line.split(":", 1)[-1].strip()
        except OSError:
            pass
    return "unknown"


def _host_info() -> dict[str, Any]:
    vendor = _cpu_vendor()
    intel_validated = "genuineintel" in vendor.lower().replace(" ", "")

    physical_cores = logical_cores = None
    ram_gb: float | None = None
    if psutil is not None:
        try:
            physical_cores = psutil.cpu_count(logical=False)
            logical_cores = psutil.cpu_count(logical=True)
            ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
        except Exception:
            pass
    if logical_cores is None:
        logical_cores = os.cpu_count()

    info: dict[str, Any] = {
        "processor": vendor,
        "physical_cores": physical_cores,
        "logical_cores": logical_cores,
        "ram_gb": ram_gb,
        "os": platform.platform(),
        "intel_validated": intel_validated,
    }
    if not intel_validated:
        info["caveat"] = (
            f"Host CPU is '{vendor}', not an Intel part. This run is NOT on "
            "Intel-validated hardware (no Core Ultra Series 2/3 here). OpenVINO's "
            "CPU/GPU plugins still execute and the numbers below are real "
            "measurements, but they characterize whatever silicon actually ran "
            "them, not Intel's target hardware. Any 'GPU' or 'NPU' device listed "
            "is reported by FULL_DEVICE_NAME below so it can't be mistaken for "
            "an Intel iGPU/NPU."
        )
    return info


def device_report() -> dict[str, Any]:
    """Enumerate every OpenVINO device plus host identification.

    Returns ``{"devices": [...], "processor": ..., "intel_validated": bool,
    "caveat": str | omitted, ...}``. Each device entry always has a "device"
    key and whichever of FULL_DEVICE_NAME / OPTIMIZATION_CAPABILITIES /
    DEVICE_ARCHITECTURE / RANGE_FOR_ASYNC_INFER_REQUESTS / DEVICE_TYPE that
    plugin actually answers.
    """
    core = ov.Core()
    devices: list[dict[str, Any]] = []
    for device in core.available_devices:
        entry: dict[str, Any] = {"device": device}
        for prop in _DEVICE_PROPERTIES:
            try:
                value = core.get_property(device, prop)
            except Exception:
                continue
            entry[prop.lower()] = str(value) if hasattr(value, "name") else value
        devices.append(entry)

    report = _host_info()
    report["devices"] = devices
    return report


def _dynamic_batch_shape(tensor: Any) -> ov.PartialShape:
    """Every dim from the example tensor, except dim 0 (batch) left dynamic.

    Trace-based ``ov.convert_model`` otherwise leaves *every* dimension of
    every input fully dynamic - including feature/spatial dims that a
    downstream MatMul or Conv needs concretely - which breaks shape inference
    the moment ``reshape_static`` tries to pin a batch size. Freezing every
    non-batch dim to what the example actually used avoids that trap.
    """
    dims = list(tensor.shape)
    if dims:
        dims[0] = -1
    return ov.PartialShape(dims)


def export_ir(
    torch_module: Any,
    example_inputs: Mapping[str, Any] | Sequence[Any],
    out_path: str | Path,
    *,
    fp16: bool = True,
) -> Path:
    """Convert a torch module to OpenVINO IR and save it next to a .bin.

    ``example_inputs`` matches the module's ``forward`` signature: a dict of
    ``{arg_name: tensor}`` for keyword-style tracing, or a tuple/list of
    positional tensors. A bare tensor is also accepted for single-input models.
    Only the batch dimension (dim 0 of each input) is left dynamic in the
    resulting IR; every other dimension is pinned to what the example used.
    """
    import torch

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch_module = torch_module.eval()

    if isinstance(example_inputs, Mapping):
        example_input: Any = dict(example_inputs)
        input_shapes: Any = {name: _dynamic_batch_shape(t) for name, t in example_input.items()}
    elif isinstance(example_inputs, (tuple, list)):
        example_input = tuple(example_inputs)
        input_shapes = [_dynamic_batch_shape(t) for t in example_input]
    else:
        example_input = (example_inputs,)
        input_shapes = [_dynamic_batch_shape(example_inputs)]

    with torch.no_grad():
        ov_model = ov.convert_model(torch_module, example_input=example_input, input=input_shapes)
    ov.save_model(ov_model, out_path, compress_to_fp16=fp16)
    return out_path


def quantize_int8(
    ir_path: str | Path,
    calib_samples: Sequence[Mapping[str, np.ndarray]],
    out_path: str | Path,
    *,
    subset_size: int = 300,
) -> Path:
    """NNCF post-training INT8 quantization calibrated on real sample inputs.

    ``calib_samples`` is a list of dicts mapping IR input names to numpy
    arrays of the shape/dtype that model input expects.
    """
    ir_path = Path(ir_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    core = ov.Core()
    model = core.read_model(ir_path)
    samples = list(calib_samples)
    dataset = nncf.Dataset(samples, transform_func=lambda sample: sample)
    quantized = nncf.quantize(model, dataset, subset_size=min(subset_size, len(samples)))
    ov.save_model(quantized, out_path, compress_to_fp16=False)
    return out_path


def reshape_static(
    ir_path: str | Path,
    shapes: Mapping[str, Sequence[int]],
    out_path: str | Path,
) -> Path:
    """Freeze an IR's input shapes to fixed static dimensions.

    NPU compilation requires static shapes; CPU/GPU accept them too, so this
    is used unconditionally before benchmarking to keep every device on equal
    footing. Existing weight precision (FP32/FP16/INT8) is left untouched.
    """
    ir_path = Path(ir_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    core = ov.Core()
    model = core.read_model(ir_path)
    model.reshape({name: ov.PartialShape(list(shape)) for name, shape in shapes.items()})
    ov.save_model(model, out_path, compress_to_fp16=False)
    return out_path


def model_size_mb(ir_path: str | Path) -> float:
    """Combined .xml + .bin size in MiB (0 if the .bin is missing/embedded)."""
    ir_path = Path(ir_path)
    total = ir_path.stat().st_size
    bin_path = ir_path.with_suffix(".bin")
    if bin_path.exists():
        total += bin_path.stat().st_size
    return round(total / (1024**2), 4)


def detect_precision(model: "ov.Model | str | Path") -> str:
    """Dominant weight precision read straight from the IR's constants.

    Filenames aren't trusted - a model can be handed to us with any name - so
    this inspects the actual constant element types baked into the graph:
    any INT8 constant means the model was quantized, any FP16-only constants
    mean a half-precision export, otherwise it's FP32.
    """
    if isinstance(model, (str, Path)):
        model = ov.Core().read_model(Path(model))
    seen: Counter[str] = Counter()
    for op in model.get_ordered_ops():
        if op.get_type_name() == "Constant":
            seen[str(op.get_element_type()).lower()] += 1
    keys = " ".join(seen)
    if "int8" in keys:
        return "INT8"
    if "float16" in keys:
        return "FP16"
    return "FP32"


def synthesize_inputs(ir_path: str | Path, *, seed: int = 0) -> dict[str, np.ndarray]:
    """Random inputs matching an IR's declared input names, shapes and dtypes.

    Dynamic dimensions (batch size, most commonly) are pinned to 1 so any
    model - however it was exported - gets a concrete, runnable sample.
    """
    core = ov.Core()
    model = core.read_model(Path(ir_path))
    rng = np.random.default_rng(seed)
    samples: dict[str, np.ndarray] = {}
    for inp in model.inputs:
        shape = [dim.get_length() if dim.is_static else 1 for dim in inp.get_partial_shape()]
        dtype = inp.get_element_type().to_dtype()
        name = inp.get_any_name()
        if np.issubdtype(dtype, np.floating):
            samples[name] = rng.standard_normal(size=shape).astype(dtype)
        elif dtype == np.bool_:
            samples[name] = rng.integers(0, 2, size=shape).astype(dtype)
        elif np.issubdtype(dtype, np.integer):
            samples[name] = rng.integers(0, 8, size=shape).astype(dtype)
        else:
            samples[name] = np.zeros(shape, dtype=dtype)
    return samples


def split_precision_suffix(stem: str) -> tuple[str, str | None]:
    """Split ``policy_fp32`` -> ``("policy", "fp32")``; unmatched -> (stem, None)."""
    for suffix in _PRECISION_SUFFIXES:
        marker = f"_{suffix}"
        if stem.endswith(marker):
            return stem[: -len(marker)], suffix
    return stem, None
