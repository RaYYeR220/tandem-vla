"""Check a device can compile a model without taking the process down with it.

Some OpenVINO plugin failures are not Python exceptions. On this development machine the GPU
plugin aborts inside `clBuildProgram` while compiling the policy transformer, and the whole
interpreter dies with a native stack trace — a `try`/`except` around `compile_model` never gets
a chance to run. A benchmark script that a reviewer is told to run must not do that.

So each device is probed in a short-lived subprocess first. If the probe dies, the device is
reported as unusable for that model, with the reason, and the main sweep skips it. The report
still names the device and still says what happened; nothing is silently dropped.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

#: Probe timeout. Compiling a small IR is seconds; anything longer is a hang, not a slow build.
TIMEOUT_S = 120


def _probe_source(ir_path: str, device: str) -> str:
    return (
        "import json, sys\n"
        "import numpy as np\n"
        "import openvino as ov\n"
        "core = ov.Core()\n"
        f"m = core.read_model({ir_path!r})\n"
        "shapes = {}\n"
        "for i in m.inputs:\n"
        "    ps = i.get_partial_shape()\n"
        "    dims = [1 if d.is_dynamic else d.get_length() for d in ps]\n"
        "    shapes[i.get_any_name()] = dims\n"
        "m.reshape({k: ov.PartialShape(v) for k, v in shapes.items()})\n"
        f"c = core.compile_model(m, {device!r})\n"
        "r = c.create_infer_request()\n"
        "feed = {}\n"
        "for i in c.inputs:\n"
        "    t = i.get_element_type().to_dtype()\n"
        "    feed[i.get_any_name()] = np.zeros([d.get_length() for d in i.get_partial_shape()], dtype=t)\n"
        "r.infer(feed)\n"
        "print(json.dumps({'ok': True}))\n"
    )


def probe_device(ir_path: Path, device: str, *, timeout: float = TIMEOUT_S) -> tuple[bool, str]:
    """Return (usable, reason). A crash, a hang and a refusal are all just 'not usable'."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _probe_source(str(ir_path), device)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"probe timed out after {timeout:.0f}s compiling for {device}"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        detail = next(
            (ln.strip() for ln in reversed(tail) if ln.strip() and "0x" not in ln[:4]),
            f"exit code {proc.returncode}",
        )
        return False, f"{device} plugin failed to compile this model: {detail[:200]}"
    try:
        return bool(json.loads(proc.stdout.strip().splitlines()[-1])["ok"]), ""
    except Exception:  # noqa: BLE001 - a malformed probe result is still a failure
        return False, f"{device} probe returned no result"


def usable_devices(
    ir_path: Path, devices: list[str], *, timeout: float = TIMEOUT_S
) -> tuple[list[str], list[dict]]:
    """Split `devices` into the ones that can run `ir_path` and the ones that cannot."""
    ok: list[str] = []
    skipped: list[dict] = []
    for device in devices:
        if device == "CPU":  # the reference path; if it cannot run, the report should say so loudly
            ok.append(device)
            continue
        usable, reason = probe_device(ir_path, device, timeout=timeout)
        if usable:
            ok.append(device)
        else:
            skipped.append({"device": device, "model": ir_path.stem, "reason": reason})
    return ok, skipped
