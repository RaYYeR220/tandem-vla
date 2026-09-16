"""Streams ``VoiceService`` events onto the hub as ``transcript`` telemetry.

Every event is stamped with what is actually listening (``VoiceService.info()``) so the
dashboard can label a scripted replay as a replay instead of implying live ASR — that
labelling has to survive per-event, not just at ``/api/status`` check time, because a stream
started ten minutes ago is still exactly as live or as scripted as when it started.

A ``final`` transcript becomes the pending instruction (``run_manager.last_instruction``) for
the next ``/api/run`` call — it does not auto-start one; the operator still presses Run. A
``barge_in`` (a final that arrived while an episode is running) aborts the in-flight episode
the same way ``/api/stop`` does, per ``docs/INTERFACES.md``: "the executor treats it as a
request to halt at the next safe point and replan."
"""

from __future__ import annotations

import asyncio
import logging

from ..voice.service import VoiceService
from .hub import hub
from .session import run_manager
from .status import get_voice_service

log = logging.getLogger(__name__)


async def stream_source(source: str) -> None:
    voice: VoiceService = get_voice_service()
    info = voice.info()
    hub.publish(
        {
            "type": "transcript",
            "stage": "status",
            "backend": info.get("backend"),
            "live": bool(info.get("live", False)),
            "label": info.get("label", info.get("backend", "")),
        }
    )

    queue: asyncio.Queue = asyncio.Queue()
    voice.set_busy(run_manager.running)

    async def forward() -> None:
        while True:
            event = await queue.get()
            etype = event.get("type")
            out = {
                "type": "transcript",
                "stage": etype,
                "text": event.get("text", event.get("detail", "")),
                "t": event.get("t"),
                "live": bool(info.get("live", False)),
            }
            if "latency_ms" in event:
                out["latency_ms"] = event["latency_ms"]
            if event.get("stop"):
                out["stop"] = True
            hub.publish(out)

            if etype == "final":
                run_manager.last_instruction = event.get("text", "")
            if etype == "barge_in":
                run_manager.request_abort(reason=f"voice barge-in: {event.get('text', '')}")

    forwarder = asyncio.create_task(forward())
    try:
        await voice.run(queue, source)
    except Exception as exc:  # noqa: BLE001 - surface it, never fabricate a transcript instead
        log.exception("voice stream failed")
        hub.publish({"type": "transcript", "stage": "error", "text": str(exc)})
    finally:
        forwarder.cancel()
        voice.set_busy(False)
