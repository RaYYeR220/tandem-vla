"""The app's single entry point into the voice layer.

Picks a transcriber — Speechmatics when credentials exist, the scripted
replay otherwise — runs it against a source, and layers the barge-in policy
and the run's evidence log on top of whichever backend is doing the
listening. Nothing outside this module talks to a transcriber directly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any
from typing import Union

from .events import BargeInEvent
from .offline import ScriptedTranscriber
from .speechmatics_rt import SpeechmaticsTranscriber

logger = logging.getLogger(__name__)

#: Case-insensitive words/phrases that mean "halt right now". Checked as a
#: substring of the final transcript, so "please stop" and "wait a second"
#: both match.
STOP_WORDS: tuple[str, ...] = ("stop", "wait", "hold on", "abort", "freeze", "cancel")

Transcriber = Union[SpeechmaticsTranscriber, ScriptedTranscriber]


def _matches_stop_vocabulary(text: str) -> bool:
    lowered = text.strip().lower()
    return any(word in lowered for word in STOP_WORDS)


class VoiceService:
    """Runs one transcriber against one source and layers barge-in on top.

    Args:
        api_key: Speechmatics API key. Defaults to the `SPEECHMATICS_API_KEY`
            env var; when neither is set, the service runs on the scripted
            replay and says so.

    `source` passed to `run()` is one of:
      - `"mic"`: live microphone (Speechmatics only)
      - a path to a `.wav` file: streamed through Speechmatics at wall-clock speed
      - `"scripted"`: always the no-credentials replay, even if a key is set —
        the deterministic demo/CI path
    """

    def __init__(self, *, api_key: str | None = None) -> None:
        self._api_key = api_key if api_key is not None else os.environ.get("SPEECHMATICS_API_KEY")
        self._busy = False
        self.transcript_log: list[dict[str, Any]] = []
        self._transcriber: Transcriber = self.get_transcriber()

    def get_transcriber(self) -> Transcriber:
        """Speechmatics when a key is present, else the scripted fallback. Logs which."""
        if self._api_key:
            logger.info("voice: SPEECHMATICS_API_KEY set - using live Speechmatics realtime ASR")
            return SpeechmaticsTranscriber(api_key=self._api_key)
        logger.info("voice: no SPEECHMATICS_API_KEY set - using scripted replay (not live ASR)")
        return ScriptedTranscriber()

    def info(self) -> dict[str, Any]:
        """The truth about what is actually listening right now."""
        return self._transcriber.info()

    def set_busy(self, busy: bool) -> None:
        """The executor calls this while a plan is running, so a barge-in can be detected."""
        self._busy = busy

    async def run(self, queue: asyncio.Queue, source: str | Path) -> None:
        """Stream `source` through the active transcriber onto `queue`.

        Every event is logged to `transcript_log`, checked against the stop
        vocabulary, and — if a `final` arrives while `set_busy(True)` — followed
        by an extra `barge_in` event, before being forwarded to `queue`.
        """
        internal: asyncio.Queue = asyncio.Queue()
        producer = asyncio.create_task(self._produce(internal, source))
        try:
            while True:
                event = await internal.get()
                if event is None:  # sentinel: producer finished
                    break
                await self._dispatch(event, queue)
        finally:
            await producer

    async def _produce(self, internal: asyncio.Queue, source: str | Path) -> None:
        try:
            if isinstance(self._transcriber, ScriptedTranscriber):
                if source != "scripted":
                    logger.warning(
                        "voice: source=%r requested but no credentials are set - replaying scripted demo instead",
                        source,
                    )
                await self._transcriber.stream(internal)
            elif source == "scripted":
                # Explicit override: run the replay even though a live backend is available.
                await ScriptedTranscriber().stream(internal)
            elif source == "mic":
                await self._transcriber.stream_microphone(internal)
            else:
                await self._transcriber.stream_file(Path(source), internal)
        finally:
            await internal.put(None)

    async def _dispatch(self, event: dict[str, Any], queue: asyncio.Queue) -> None:
        event = dict(event)
        if event.get("type") == "final" and _matches_stop_vocabulary(event.get("text", "")):
            event["stop"] = True

        self.transcript_log.append(event)
        await queue.put(event)

        if event.get("type") == "final" and self._busy:
            barge = BargeInEvent(
                text=event.get("text", ""),
                t=event.get("t", 0.0),
                stop=event.get("stop", False),
            ).to_dict()
            self.transcript_log.append(barge)
            await queue.put(barge)

    def dump_log(self, path: str | Path) -> None:
        """Write `transcript_log` as newline-delimited JSON — the submission's evidence artifact."""
        with open(path, "w", encoding="utf-8") as fh:
            for event in self.transcript_log:
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")
