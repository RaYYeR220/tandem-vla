"""The no-credentials transcriber: replays a script instead of listening.

Every hackathon judge without a Speechmatics key, and every CI run, needs the
rest of the product (barge-in, the gate, the executor) to see the exact same
event traffic a live session would produce. This module is how: it plays back
a fixed list of ``(t_seconds, text, is_final)`` lines at their scheduled
wall-clock offsets, so the timing looks and feels like real speech.

It never claims to be Speechmatics. ``info()`` always reports
``{"backend": "scripted", "live": False}`` and every event it emits is
honestly missing a ``latency_ms`` — there is no audio pipeline to time.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from typing import ClassVar

from .events import FinalEvent
from .events import PartialEvent

#: (t_seconds, text, is_final). Mirrors the three demo commands in assets/audio/,
#: including a stop-vocabulary line so the barge-in demo works without a microphone.
DEFAULT_SCRIPT: list[tuple[float, str, bool]] = [
    (0.3, "set", False),
    (0.7, "set the table", False),
    (1.1, "set the table for one", False),
    (1.6, "set the table for one, please.", True),
    (3.0, "pour", False),
    (3.4, "pour me some", False),
    (3.9, "pour me some water.", True),
    (5.5, "stop", False),
    (5.9, "stop", True),
    (6.3, "stop - put the plate down first.", True),
]


class ScriptedTranscriber:
    """Replays a fixed transcript in real time. Not live ASR — say so everywhere.

    Args:
        script: ``(t_seconds, text, is_final)`` tuples, offsets relative to the
            start of ``stream()``. Defaults to :data:`DEFAULT_SCRIPT`.
    """

    DEFAULT_SCRIPT: ClassVar[list[tuple[float, str, bool]]] = DEFAULT_SCRIPT

    def __init__(self, script: list[tuple[float, str, bool]] | None = None) -> None:
        self.script = list(script) if script is not None else list(self.DEFAULT_SCRIPT)

    def info(self) -> dict[str, Any]:
        return {
            "backend": "scripted",
            "live": False,
            "label": "REPLAY - not live ASR",
            "lines": len(self.script),
        }

    async def stream(self, queue: asyncio.Queue) -> None:
        """Emit the script onto ``queue`` at its scheduled wall-clock offsets."""
        start = time.monotonic()
        for t_seconds, text, is_final in self.script:
            elapsed = time.monotonic() - start
            delay = t_seconds - elapsed
            if delay > 0:
                await asyncio.sleep(delay)
            t = round(time.monotonic() - start, 3)
            event = FinalEvent(text=text, t=t) if is_final else PartialEvent(text=text, t=t)
            await queue.put(event.to_dict())
