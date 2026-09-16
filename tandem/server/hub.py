"""Broadcast hub: bridges events from the (blocking, worker-thread) sim loop onto every
subscribed SSE connection running on the asyncio event loop.

One publisher (the run manager's worker thread, or the voice bridge coroutine), many
subscribers (one per open ``/api/stream`` connection, including reconnects). A reconnecting
client is handed a snapshot of the last-known ``state``/``plan``/``verdict``/``infer`` events
before it starts receiving live ones, so the dashboard is never blank after a refresh.
"""

from __future__ import annotations

import asyncio
from typing import Any

#: Event types worth replaying to a freshly (re)connected subscriber.
_SNAPSHOT_TYPES = ("state", "plan", "verdict", "infer", "episode", "refusal")


class EventHub:
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._subscribers: set[asyncio.Queue] = set()
        self._last: dict[str, dict[str, Any]] = {}

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    # ------------------------------------------------------------------ publish

    def publish(self, event: dict[str, Any]) -> None:
        """Thread-safe from anywhere: a worker thread, a voice coroutine, the loop itself."""
        loop = self._loop
        if loop is None:
            return
        loop.call_soon_threadsafe(self._publish_sync, event)

    def _publish_sync(self, event: dict[str, Any]) -> None:
        etype = event.get("type")
        if etype in _SNAPSHOT_TYPES:
            self._last[etype] = event
        dead = []
        for q in self._subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # A stalled client should not stall the sim; drop the oldest instead.
                try:
                    q.get_nowait()
                    q.put_nowait(event)
                except Exception:
                    dead.append(q)
        for q in dead:
            self._subscribers.discard(q)

    # ------------------------------------------------------------------ subscribe

    async def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._subscribers.add(q)
        for etype in _SNAPSHOT_TYPES:
            snap = self._last.get(etype)
            if snap is not None:
                q.put_nowait(snap)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    def reset_snapshot(self) -> None:
        """Clear cached snapshots before a new episode starts. Thread-safe, like ``publish``.

        Without this, a reconnecting client between episodes can be handed a Frankenstate:
        e.g. a fresh ``plan`` that refused before any step ran, replayed alongside a stale
        ``episode``/``verdict`` cached from the *previous* run (which may not emit a new
        ``episode``/``verdict`` at all if it also refuses immediately — see
        ``tandem.control.runner.EpisodeRunner.run``, which returns before its final
        ``episode`` emit when ``plan.get("refusal")`` is set). Wiping the cache the moment a
        new run starts, right before ``TandemEnv.reset`` publishes the first fresh ``state``,
        guarantees every cached type a reconnect can see belongs to the current episode only.
        """
        loop = self._loop
        if loop is None:
            self._last.clear()
            return
        loop.call_soon_threadsafe(self._last.clear)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)


#: One hub for the whole process. A hackathon dashboard is single-tenant by design; every
#: request handler and the worker thread share this instance.
hub = EventHub()
