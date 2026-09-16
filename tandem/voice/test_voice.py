"""Tests for the voice layer. No credentials, no microphone, no network.

Run with:  python -m pytest tandem/voice/test_voice.py -v
"""

from __future__ import annotations

import asyncio
import json
import wave
from pathlib import Path

from tandem.voice.offline import ScriptedTranscriber
from tandem.voice.service import VoiceService
from tandem.voice.speechmatics_rt import SpeechmaticsTranscriber


def _drain(queue: asyncio.Queue) -> list[dict]:
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return events


def _make_silent_wav(path: Path, *, seconds: float = 1.0, sample_rate: int = 16000) -> None:
    """A locally generated 16kHz mono PCM16 WAV of digital silence."""
    n_samples = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * n_samples)


# --------------------------------------------------------------------------------------
# ScriptedTranscriber: shapes and ordering
# --------------------------------------------------------------------------------------


def test_scripted_transcriber_emits_partials_then_final_in_order():
    script = [
        (0.0, "put the plate", False),
        (0.02, "put the plate on the", False),
        (0.04, "put the plate on the table", True),
    ]

    async def main() -> list[dict]:
        transcriber = ScriptedTranscriber(script)
        queue: asyncio.Queue = asyncio.Queue()
        await transcriber.stream(queue)
        return _drain(queue)

    events = asyncio.run(main())

    assert [e["type"] for e in events] == ["partial", "partial", "final"]
    # Every event carries the shape from docs/INTERFACES.md.
    for event in events:
        assert set(event) <= {"type", "text", "t", "latency_ms", "stop"}
        assert isinstance(event["text"], str)
        assert isinstance(event["t"], (int, float))
    assert events[-1]["text"] == "put the plate on the table"
    # Timestamps are non-decreasing (real-time playback, scheduled offsets).
    assert [e["t"] for e in events] == sorted(e["t"] for e in events)
    # Scripted replay has no audio pipeline to time honestly, so it never fabricates a latency.
    assert "latency_ms" not in events[-1]


def test_scripted_transcriber_info_reports_replay_not_live():
    info = ScriptedTranscriber().info()
    assert info == {
        "backend": "scripted",
        "live": False,
        "label": "REPLAY - not live ASR",
        "lines": len(ScriptedTranscriber.DEFAULT_SCRIPT),
    }


# --------------------------------------------------------------------------------------
# Barge-in policy (VoiceService)
# --------------------------------------------------------------------------------------


def test_final_while_busy_emits_extra_barge_in_event():
    async def main() -> list[dict]:
        service = VoiceService(api_key=None)
        service._transcriber = ScriptedTranscriber([(0.0, "put the plate on the table", True)])
        service.set_busy(True)
        queue: asyncio.Queue = asyncio.Queue()
        await service.run(queue, "scripted")
        return _drain(queue)

    events = asyncio.run(main())

    assert [e["type"] for e in events] == ["final", "barge_in"]
    assert events[1]["text"] == events[0]["text"]
    assert "stop" not in events[0] and "stop" not in events[1]


def test_final_while_idle_emits_no_barge_in():
    async def main() -> list[dict]:
        service = VoiceService(api_key=None)
        service._transcriber = ScriptedTranscriber([(0.0, "put the plate on the table", True)])
        # busy is False by default: no set_busy(True) call.
        queue: asyncio.Queue = asyncio.Queue()
        await service.run(queue, "scripted")
        return _drain(queue)

    events = asyncio.run(main())
    assert [e["type"] for e in events] == ["final"]


def test_stop_vocabulary_sets_stop_flag_case_insensitively():
    async def main() -> list[dict]:
        service = VoiceService(api_key=None)
        service._transcriber = ScriptedTranscriber(
            [
                (0.0, "STOP right now", True),
                (0.02, "please hold on a second", True),
                (0.04, "pour me some water", True),
            ]
        )
        service.set_busy(True)
        queue: asyncio.Queue = asyncio.Queue()
        await service.run(queue, "scripted")
        return _drain(queue)

    events = asyncio.run(main())
    finals = [e for e in events if e["type"] == "final"]
    barge_ins = [e for e in events if e["type"] == "barge_in"]

    assert [f.get("stop", False) for f in finals] == [True, True, False]
    assert [b.get("stop", False) for b in barge_ins] == [True, True, False]


# --------------------------------------------------------------------------------------
# SpeechmaticsTranscriber: honest no-key path
# --------------------------------------------------------------------------------------


def test_stream_file_without_api_key_emits_clean_error_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.delenv("SPEECHMATICS_API_KEY", raising=False)
    wav_path = tmp_path / "silence.wav"
    _make_silent_wav(wav_path)

    async def main() -> list[dict]:
        transcriber = SpeechmaticsTranscriber(api_key=None)
        queue: asyncio.Queue = asyncio.Queue()
        await transcriber.stream_file(wav_path, queue)
        return _drain(queue)

    events = asyncio.run(main())

    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert "SPEECHMATICS_API_KEY" in events[0]["detail"]


def test_speechmatics_info_never_claims_a_key_it_does_not_have(monkeypatch):
    monkeypatch.delenv("SPEECHMATICS_API_KEY", raising=False)
    info = SpeechmaticsTranscriber(api_key=None).info()
    assert info["backend"] == "speechmatics"
    assert info["has_api_key"] is False


# --------------------------------------------------------------------------------------
# dump_log
# --------------------------------------------------------------------------------------


def test_dump_log_round_trips(tmp_path):
    async def main() -> VoiceService:
        service = VoiceService(api_key=None)
        service._transcriber = ScriptedTranscriber(
            [
                (0.0, "pour me some", False),
                (0.02, "pour me some water", True),
            ]
        )
        queue: asyncio.Queue = asyncio.Queue()
        await service.run(queue, "scripted")
        return service

    service = asyncio.run(main())
    log_path = tmp_path / "transcript_log.ndjson"
    service.dump_log(log_path)

    lines = log_path.read_text(encoding="utf-8").splitlines()
    round_tripped = [json.loads(line) for line in lines]

    assert round_tripped == service.transcript_log
    assert [e["type"] for e in round_tripped] == ["partial", "final"]
