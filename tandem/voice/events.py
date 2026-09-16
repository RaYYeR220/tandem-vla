"""The four event shapes a transcriber may put on the voice queue.

Frozen against ``docs/INTERFACES.md`` § "Voice events":

    {"type": "partial", "text": "put the plate on the", "t": 1.9}
    {"type": "final",   "text": "put the plate on the table", "t": 2.4, "latency_ms": 310}
    {"type": "barge_in","text": "stop",                        "t": 5.1}
    {"type": "error",   "detail": "..."}

A transcriber builds one of these dataclasses and calls ``.to_dict()`` before
putting the result on the queue — the queue itself only ever carries plain
dicts, so any consumer downstream (executor, SSE bridge) can `json.dumps` an
event without knowing this module exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Union


@dataclass
class PartialEvent:
    """An in-flight transcript that may still change."""

    text: str
    t: float

    def to_dict(self) -> dict[str, Any]:
        return {"type": "partial", "text": self.text, "t": self.t}


@dataclass
class FinalEvent:
    """A transcript segment the backend will not revise further.

    ``latency_ms`` is omitted when the backend has no honest number to report
    (e.g. the scripted replay, which has no audio pipeline to time).
    ``stop`` is set by the barge-in policy in ``service.py`` when the text
    matches the stop vocabulary; a bare transcriber never sets it.
    """

    text: str
    t: float
    latency_ms: float | None = None
    stop: bool = False

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": "final", "text": self.text, "t": self.t}
        if self.latency_ms is not None:
            out["latency_ms"] = self.latency_ms
        if self.stop:
            out["stop"] = True
        return out


@dataclass
class BargeInEvent:
    """Raised alongside a `final` that arrives while the executor is busy.

    The executor treats this as a request to halt at the next safe point and
    replan; ``stop`` mirrors whether the text also matched the stop vocabulary.
    """

    text: str
    t: float
    stop: bool = False

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": "barge_in", "text": self.text, "t": self.t}
        if self.stop:
            out["stop"] = True
        return out


@dataclass
class ErrorEvent:
    """Connection, auth, or audio failure. Never a fabricated transcript."""

    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"type": "error", "detail": self.detail}


VoiceEvent = Union[PartialEvent, FinalEvent, BargeInEvent, ErrorEvent]


def to_dict(event: VoiceEvent) -> dict[str, Any]:
    """Free-function form of ``event.to_dict()``, for call sites that only hold the union type."""
    return event.to_dict()
