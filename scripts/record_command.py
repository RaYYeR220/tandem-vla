"""Command-line helper for the voice layer's live paths.

    python scripts/record_command.py --record out.wav --seconds 6
    python scripts/record_command.py --transcribe out.wav
    python scripts/record_command.py --record out.wav --transcribe out.wav

`--record` only needs `sounddevice` and writes a 16kHz mono PCM16 WAV — no
Speechmatics key required. `--transcribe` streams a WAV through the real,
live Speechmatics realtime transcriber (`tandem.voice.speechmatics_rt`) and
prints partials/finals with their timing as they arrive. Without
`SPEECHMATICS_API_KEY` set, it prints exactly that and stops — it never
pretends to transcribe.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # allow `python scripts/record_command.py` from anywhere

from tandem.voice.speechmatics_rt import SpeechmaticsTranscriber  # noqa: E402

SAMPLE_RATE = 16000


def record(out_path: Path, seconds: float) -> None:
    """Record `seconds` of mono 16kHz PCM16 audio from the default input device."""
    try:
        import sounddevice as sd
    except Exception as exc:
        print(f"sounddevice is not available: {exc}", file=sys.stderr)
        raise SystemExit(1)

    print(f"recording {seconds:.1f}s @ {SAMPLE_RATE}Hz mono -> {out_path}  (speak now)")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(audio.tobytes())
    print(f"wrote {out_path}")


async def _transcribe(path: Path) -> None:
    api_key = os.environ.get("SPEECHMATICS_API_KEY")
    if not api_key:
        print("SPEECHMATICS_API_KEY is not set - nothing will be sent to Speechmatics.")
        print("Set it (a .env file works) to transcribe for real. Until then, the rest of")
        print("the product runs on tandem.voice.offline.ScriptedTranscriber (REPLAY, not live ASR).")
        return

    print(f"streaming {path} to Speechmatics (model=enhanced, language=en)...")
    transcriber = SpeechmaticsTranscriber(api_key=api_key)
    queue: asyncio.Queue = asyncio.Queue()
    start = time.monotonic()

    async def _print_events() -> None:
        while True:
            event = await queue.get()
            if event is None:  # sentinel: the transcriber is done
                return
            elapsed = time.monotonic() - start
            kind = event["type"]
            if kind == "partial":
                print(f"[{elapsed:6.2f}s] partial: {event['text']}")
            elif kind == "final":
                latency = event.get("latency_ms")
                latency_str = f"{latency:.0f}ms" if latency is not None else "n/a"
                print(f"[{elapsed:6.2f}s] FINAL:   {event['text']}   (latency {latency_str})")
            elif kind == "error":
                print(f"[{elapsed:6.2f}s] ERROR:   {event['detail']}", file=sys.stderr)
            else:
                print(f"[{elapsed:6.2f}s] {kind}: {event}")

    printer = asyncio.create_task(_print_events())
    await transcriber.stream_file(path, queue)
    await queue.put(None)
    await printer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record or transcribe a spoken command.")
    parser.add_argument("--record", metavar="OUT.WAV", help="record from the microphone to this WAV file")
    parser.add_argument("--seconds", type=float, default=6.0, help="recording length in seconds (default 6)")
    parser.add_argument("--transcribe", metavar="FILE.WAV", help="stream this WAV through live Speechmatics ASR")
    args = parser.parse_args(argv)

    if not args.record and not args.transcribe:
        parser.print_help()
        return 1

    if args.record:
        record(Path(args.record), args.seconds)

    if args.transcribe:
        asyncio.run(_transcribe(Path(args.transcribe)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
