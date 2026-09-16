"""The truth panel: what is actually running, read straight from the modules that know.

Every field here comes straight from the module that owns the answer — nothing is guessed,
cached-and-forgotten, or hand-typed. If the planner fell back to keywords or the voice
service has no credentials, this is where that becomes visible instead of silently
disappearing.
"""

from __future__ import annotations

from typing import Any

from ..bench import ovutil
from ..planner import get_planner
from ..voice.service import VoiceService

_voice_singleton: VoiceService | None = None


def get_voice_service() -> VoiceService:
    """One VoiceService for the process — constructing it decides live-vs-scripted once."""
    global _voice_singleton
    if _voice_singleton is None:
        _voice_singleton = VoiceService()
    return _voice_singleton


def status_payload() -> dict[str, Any]:
    return {
        "device": ovutil.device_report(),
        "planner": get_planner().info(),
        "voice": get_voice_service().info(),
    }
