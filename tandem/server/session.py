"""Runs one episode at a time in a worker thread and feeds the hub.

The sim is synchronous, CPU-bound MuJoCo — it has no business on the asyncio event loop, so
``POST /api/run`` hands it to a plain ``threading.Thread`` and returns immediately. Camera
frames and world-state snapshots are captured from inside the executor's own ``on_tick``
hook, throttled by wall clock so the dashboard gets a steady ~12-15 fps video feed and a
~10 Hz state/subgoal readout without slowing the control loop down to match — the tick still
fires at 50 Hz, this just skips most of them for rendering purposes.

Only one run lives at a time (this is a single-operator demo console, not a fleet manager).
A voice barge-in or the ``/api/stop`` control both work the same way: set the executor's own
``aborted`` flag, which every primitive already polls every tick (see
``tandem.control.primitives``) — this module does not invent a second abort mechanism.
"""

from __future__ import annotations

import base64
import io
import logging
import threading
import time
from typing import Any

from ..control.primitives import Executor
from ..control.runner import EpisodeRunner
from ..planner import plan as planner_plan
from ..planner import replan as planner_replan
from ..sim.env import TandemEnv
from ..sim.randomize import RandomizationSpec
from .hub import hub

log = logging.getLogger(__name__)

#: Every camera the scene actually defines (tandem/sim/scene.py + assets/scene_table.xml,
#: assets/so101/so101.xml). The dashboard's camera switcher may only ask for one of these.
ALLOWED_CAMS = ("overhead", "front", "cinematic", "left_wrist", "right_wrist")

DEFAULT_BUDGET_S = 180.0
FRAME_SIZE = (360, 480)  # height, width — matches the brief's 480x360 frame spec
FRAME_FPS = 13.0
STATE_HZ = 10.0
JPEG_QUALITY = 70


class Busy(RuntimeError):
    """Raised when a run is requested while one is already in flight."""


class RunManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.running: bool = False
        self.env: TandemEnv | None = None
        self.ex: Executor | None = None
        self.thread: threading.Thread | None = None
        self.current_cam: str = "overhead"
        self.last_instruction: str = ""
        self._last_state_t: float = 0.0
        self._last_frame_t: float = 0.0

    # ------------------------------------------------------------------ control

    def try_start(
        self,
        *,
        seed: int,
        instruction: str,
        dr_scale: float,
        planner: str | None,
        budget_s: float = DEFAULT_BUDGET_S,
    ) -> bool:
        with self._lock:
            if self.running:
                return False
            self.running = True
        self.last_instruction = instruction
        self.thread = threading.Thread(
            target=self._run,
            args=(seed, instruction, dr_scale, planner, budget_s),
            daemon=True,
        )
        self.thread.start()
        return True

    def set_camera(self, cam: str) -> None:
        if cam not in ALLOWED_CAMS:
            raise ValueError(f"unknown camera {cam!r}; choose one of {ALLOWED_CAMS}")
        self.current_cam = cam

    def request_abort(self, reason: str) -> bool:
        if not self.running or self.ex is None:
            return False
        self.ex.aborted = True
        hub.publish({"type": "abort", "reason": reason})
        return True

    # ------------------------------------------------------------------ worker thread

    def _run(
        self, seed: int, instruction: str, dr_scale: float, planner: str | None, budget_s: float
    ) -> None:
        self._last_state_t = 0.0
        self._last_frame_t = 0.0
        hub.reset_snapshot()
        try:
            env = TandemEnv(RandomizationSpec(scale=dr_scale))
            self.env = env
            world = env.reset(seed)
            hub.publish({"type": "state", "world": world, "score": env.task_score()})
            hub.publish(
                {"type": "transcript", "stage": "final", "text": instruction, "source": "run"}
            )

            plan_dict = planner_plan(instruction, world, backend=planner)
            hub.publish({"type": "plan", "plan": plan_dict})
            self._emit_infer(plan_dict)

            ex = Executor(env, on_tick=self._on_tick)
            self.ex = ex

            def replanner(w: dict, done_ids: list[int]) -> dict:
                new_plan = planner_replan(plan_dict, w, done_ids)
                self._emit_infer(new_plan)
                return new_plan

            runner = EpisodeRunner(env, ex, on_event=hub.publish, replanner=replanner)
            runner.run(plan_dict, instruction=instruction, budget_s=budget_s)
            hub.publish({"type": "state", "world": env.world_state(), "score": env.task_score()})
        except Exception as exc:  # noqa: BLE001 - a dead episode must reach the dashboard, not the log alone
            log.exception("episode crashed")
            hub.publish({"type": "error", "detail": str(exc)})
        finally:
            self.running = False
            self.ex = None

    def _emit_infer(self, plan_dict: dict) -> None:
        meta = plan_dict.get("_meta") or {}
        hub.publish(
            {
                "type": "infer",
                "model": "planner",
                "backend": meta.get("backend"),
                "device": meta.get("device"),
                "precision": meta.get("precision"),
                "ms": meta.get("latency_ms"),
            }
        )

    def _on_tick(self, ex: Executor) -> None:
        now = time.monotonic()
        if now - self._last_state_t >= 1.0 / STATE_HZ:
            self._last_state_t = now
            env = self.env
            if env is not None:
                try:
                    hub.publish({"type": "state", "world": env.world_state(), "score": env.task_score()})
                except Exception:  # noqa: BLE001 - a bad tick must not kill the episode
                    log.exception("state snapshot failed")
        if now - self._last_frame_t >= 1.0 / FRAME_FPS:
            self._last_frame_t = now
            frame = self._capture_frame()
            if frame is not None:
                hub.publish(frame)

    def _capture_frame(self) -> dict[str, Any] | None:
        env = self.env
        if env is None:
            return None
        height, width = FRAME_SIZE
        try:
            image = env.render(self.current_cam, height, width)
        except Exception:  # noqa: BLE001 - a renderer hiccup should not stop the episode
            log.exception("render failed for cam=%s", self.current_cam)
            return None
        try:
            from PIL import Image

            buf = io.BytesIO()
            Image.fromarray(image).save(buf, format="JPEG", quality=JPEG_QUALITY)
            jpeg_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:  # noqa: BLE001
            log.exception("jpeg encode failed")
            return None
        return {"type": "frame", "cam": self.current_cam, "jpeg_b64": jpeg_b64}


#: One run manager for the process — see the module docstring for why this is single-tenant.
run_manager = RunManager()
