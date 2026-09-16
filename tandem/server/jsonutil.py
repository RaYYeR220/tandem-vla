"""JSON encoding that survives whatever numpy scalars slip into a world/verdict/result dict.

``tandem.sim`` and ``tandem.control`` are privileged-state code, not wire-format code, and a
couple of paths hand back ``numpy.bool_`` / ``numpy.floating`` instead of a plain Python
scalar (e.g. a ``bool(x) and y > z`` expression where ``y > z`` is a numpy comparison — the
``and`` returns the second operand as-is). Stock ``json.dumps`` raises on those. Rather than
edit modules this server does not own, every telemetry write in this package goes through
``dumps()`` here, which coerces numpy scalars/arrays to native types and leaves everything
else to the standard encoder.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def dumps(obj: Any) -> str:
    return json.dumps(obj, default=json_default, ensure_ascii=False)


def sse_line(obj: Any) -> str:
    """One SSE ``data:`` frame carrying one newline-delimited JSON event."""
    return f"data: {dumps(obj)}\n\n"
