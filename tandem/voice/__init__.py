"""Speech in, transcript events out.

This package is the only place in the product that touches a microphone, a WAV
file, or the Speechmatics realtime API. Everything downstream — the executor,
the dashboard — consumes the four event shapes in ``docs/INTERFACES.md`` §
"Voice events" and never talks to a transcriber directly.
"""

from __future__ import annotations

from .events import BargeInEvent
from .events import ErrorEvent
from .events import FinalEvent
from .events import PartialEvent
from .events import VoiceEvent
from .events import to_dict
from .offline import ScriptedTranscriber
from .service import VoiceService
from .speechmatics_rt import SpeechmaticsTranscriber

__all__ = [
    "PartialEvent",
    "FinalEvent",
    "BargeInEvent",
    "ErrorEvent",
    "VoiceEvent",
    "to_dict",
    "ScriptedTranscriber",
    "SpeechmaticsTranscriber",
    "VoiceService",
]
