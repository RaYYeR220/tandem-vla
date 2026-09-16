"""The FastAPI app: routes only. Everything with actual behaviour lives in
``session.py`` (the run manager), ``voice_bridge.py``, ``status.py`` and ``scorecard.py``.

    python scripts/serve.py --port 8000

Serves the dashboard's REST + SSE surface under ``/api`` and the static frontend
(``web/index.html``, ``web/app.html``) at ``/``.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from ..planner import get_planner
from . import scorecard as scorecard_mod
from .hub import hub
from .jsonutil import sse_line
from .session import ALLOWED_CAMS, DEFAULT_BUDGET_S, run_manager
from .status import status_payload
from .voice_bridge import stream_source

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = REPO_ROOT / "web"


# ---------------------------------------------------------------------------- request bodies


class RunRequest(BaseModel):
    seed: int = 0
    instruction: str
    dr_scale: float = 1.0
    planner: str | None = None  # "openvino" | "cloud" | "rules"; None -> TANDEM_PLANNER / default
    budget_s: float = DEFAULT_BUDGET_S


class VoiceRequest(BaseModel):
    source: str = "scripted"  # "mic" | "scripted" | a .wav path


class CameraRequest(BaseModel):
    cam: str


class ScorecardRequest(BaseModel):
    seeds: int = 5
    seed0: int = 0
    dr_scale: float = 1.0
    budget_s: float = 150.0
    no_pour: bool = False


# ---------------------------------------------------------------------------- app factory


def create_app() -> FastAPI:
    app = FastAPI(title="Tandem console")

    @app.on_event("startup")
    async def _startup() -> None:
        hub.bind_loop(asyncio.get_running_loop())
        # Warm the planner backend off the event loop so the first /api/run or /api/status
        # is not the call that pays for loading the IR (~4s). Failures here are non-fatal —
        # get_planner() itself already falls back to the keyword parser and info() will
        # report that honestly.
        try:
            await run_in_threadpool(get_planner)
        except Exception:  # noqa: BLE001
            log.exception("planner warm-up failed; it will retry on first use")

    # ------------------------------------------------------------------ run control

    @app.post("/api/run")
    async def api_run(req: RunRequest) -> dict[str, Any]:
        started = run_manager.try_start(
            seed=req.seed,
            instruction=req.instruction,
            dr_scale=req.dr_scale,
            planner=req.planner,
            budget_s=req.budget_s,
        )
        if not started:
            raise HTTPException(status_code=409, detail="an episode is already running")
        return {"status": "started", "seed": req.seed, "instruction": req.instruction}

    @app.post("/api/stop")
    async def api_stop() -> dict[str, Any]:
        stopped = run_manager.request_abort(reason="operator stop")
        return {"stopped": stopped}

    @app.post("/api/camera")
    async def api_camera(req: CameraRequest) -> dict[str, Any]:
        try:
            run_manager.set_camera(req.cam)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"cam": run_manager.current_cam}

    @app.get("/api/cameras")
    async def api_cameras() -> dict[str, Any]:
        return {"cameras": list(ALLOWED_CAMS), "current": run_manager.current_cam}

    # ------------------------------------------------------------------ voice

    @app.post("/api/voice")
    async def api_voice(req: VoiceRequest) -> dict[str, Any]:
        asyncio.create_task(stream_source(req.source))
        return {"status": "started", "source": req.source}

    # ------------------------------------------------------------------ telemetry stream

    @app.get("/api/stream")
    async def api_stream(request: Request) -> StreamingResponse:
        queue = await hub.subscribe()

        async def event_gen():
            try:
                yield "retry: 2000\n\n"
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=15.0)
                        yield sse_line(event)
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"
            finally:
                hub.unsubscribe(queue)

        return StreamingResponse(
            event_gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ------------------------------------------------------------------ status / scorecard

    @app.get("/api/status")
    async def api_status() -> dict[str, Any]:
        return await run_in_threadpool(status_payload)

    @app.get("/api/scorecard")
    async def api_scorecard() -> JSONResponse:
        data = await run_in_threadpool(scorecard_mod.read_scorecard)
        if data is None:
            return JSONResponse({"available": False}, status_code=200)
        return JSONResponse({"available": True, "scorecard": data}, status_code=200)

    @app.post("/api/scorecard/run")
    async def api_scorecard_run(req: ScorecardRequest) -> dict[str, Any]:
        """Build the scorecard now. Blocking (real seeds take real seconds) — call it once
        from the operator console, not from a page load."""
        summary = await run_in_threadpool(
            scorecard_mod.build_scorecard,
            seeds=req.seeds,
            seed0=req.seed0,
            dr_scale=req.dr_scale,
            budget_s=req.budget_s,
            no_pour=req.no_pour,
        )
        return {"status": "done", "scorecard": summary}

    # ------------------------------------------------------------------ static frontend

    if WEB_DIR.exists():
        app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

    return app


app = create_app()
