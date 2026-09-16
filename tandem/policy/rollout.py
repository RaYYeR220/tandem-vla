"""Closed-loop execution of the distilled policy through OpenVINO.

`PolicyRunner` is the inference loop: render, infer a 16-step chunk, execute the first 8
control ticks, re-infer. `PolicyExecutor` wraps that as a drop-in replacement for the
scripted `pick` and `place` -- same signatures, same post-conditions, same `SkillResult`,
so the episode runner, the gate and the evaluator cannot tell which one they are driving.
Every other skill falls through to the scripted implementation.

Three things are worth being explicit about.

* **The post-condition is not a stopping signal.** The policy runs a fixed horizon and the
  post-condition is checked once at the end, exactly as the evaluator checks it. Nothing
  peeks at simulator state to decide when the motion is done.
* **Only the acting arm is driven, by default.** The network predicts all twelve joint
  targets because that is what ``env.ctrl`` holds, but a one-armed skill has no business
  re-commanding the arm that is steadying a full mug. ``apply="both"`` releases that.
* **Latency is measured, not assumed.** Every inference is timed and the statistics come
  back on the `SkillResult`, so the OpenVINO numbers in the report are the ones the robot
  actually paid.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

import numpy as np
import openvino as ov

from ..control.primitives import Executor, SkillResult
from ..sim import layout
from ..sim.env import TandemEnv
from ..control.poses import grasp_transform
from .schema import (
    ACTION_DIM,
    CHUNK,
    CHUNK_STRIDE,
    EXECUTE,
    IMAGE_SIZE,
    condition_vector,
    denormalize_joints,
    joint_limits,
    normalize_joints,
    wrist_camera,
)

#: Control ticks allowed per skill, sized from the scripted oracle's own step lengths
#: (measured at collection time: ~340 ticks for a pick, ~325 for a place) with headroom.
HORIZON = {"pick": 430, "place": 430}
DEFAULT_HORIZON = 430

#: Exponential decay for temporal ensembling, in ACT's convention: weight exp(-k * age).
ENSEMBLE_DECAY = 0.25


class PolicyRunner:
    """Compiled-IR action-chunking controller."""

    def __init__(
        self,
        ir_path: str | Path,
        *,
        device: str = "CPU",
        execute: int = EXECUTE,
        ensemble: bool = False,
        chunk_stride: int = CHUNK_STRIDE,
        core: ov.Core | None = None,
        performance_hint: str = "LATENCY",
    ):
        self.ir_path = Path(ir_path)
        self.device = device
        self.execute = max(1, min(execute, CHUNK))
        self.chunk_stride = max(1, chunk_stride)
        self.ensemble = ensemble
        self._core = core or ov.Core()
        self._compiled = self._core.compile_model(
            self._core.read_model(self.ir_path), device, {"PERFORMANCE_HINT": performance_hint}
        )
        self._request = self._compiled.create_infer_request()
        self._inputs = [port.get_any_name() for port in self._compiled.inputs]
        self.latencies_ms: list[float] = []

    # ------------------------------------------------------------------ inference

    def infer(
        self, overhead: np.ndarray, wrist: np.ndarray, proprio: np.ndarray, cond: np.ndarray
    ) -> np.ndarray:
        """One forward pass -> ``(CHUNK, 12)`` normalized joint targets."""
        feed = {
            "overhead": _as_chw(overhead),
            "wrist": _as_chw(wrist),
            "proprio": proprio.reshape(1, -1).astype(np.float32),
            "cond": cond.reshape(1, -1).astype(np.float32),
        }
        started = time.perf_counter_ns()
        result = self._request.infer({name: feed[name] for name in self._inputs})
        self.latencies_ms.append((time.perf_counter_ns() - started) / 1e6)
        return np.asarray(next(iter(result.values()))).reshape(CHUNK, ACTION_DIM)

    def latency_stats(self) -> dict[str, float]:
        if not self.latencies_ms:
            return {"inferences": 0}
        arr = np.asarray(self.latencies_ms)
        return {
            "inferences": int(arr.size),
            "mean_ms": round(float(arr.mean()), 3),
            "p50_ms": round(float(np.percentile(arr, 50)), 3),
            "p90_ms": round(float(np.percentile(arr, 90)), 3),
        }

    # ------------------------------------------------------------------ rollout

    def run_skill(
        self,
        env: TandemEnv,
        skill: str,
        arm: str,
        obj: str | None,
        target: str | None = None,
        *,
        ticks: int | None = None,
        apply: str = "acting",
        on_tick: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        """Drive ``env`` for a fixed horizon under the policy. Returns rollout diagnostics."""
        limits = joint_limits(env.model)
        cond = condition_vector(skill, obj, arm, target)
        wrist_cam = wrist_camera(arm)
        act_slice = _actuator_slice(arm)
        horizon = ticks if ticks is not None else HORIZON.get(skill, DEFAULT_HORIZON)

        pending: deque[tuple[int, np.ndarray]] = deque(maxlen=CHUNK)
        step = self.execute * self.chunk_stride if not self.ensemble else self.chunk_stride
        inferences = 0
        for tick in range(horizon):
            if tick % step == 0:
                chunk = self.infer(
                    env.render("overhead", IMAGE_SIZE, IMAGE_SIZE),
                    env.render(wrist_cam, IMAGE_SIZE, IMAGE_SIZE),
                    _proprio(env, limits),
                    cond,
                )
                pending.append((tick, chunk))
                inferences += 1
            action = (
                _blend(pending, tick, self.chunk_stride)
                if self.ensemble
                else _sample(pending[-1][1], tick - pending[-1][0], self.chunk_stride)
            )
            command = denormalize_joints(action, limits)
            if apply == "both":
                env.ctrl[:] = command
            else:
                env.ctrl[act_slice] = command[act_slice]
            env.step(1)
            if on_tick is not None:
                on_tick()
        return {"ticks": horizon, "inferences": inferences, **self.latency_stats()}


def _as_chw(image: np.ndarray) -> np.ndarray:
    return np.transpose(image, (2, 0, 1)).astype(np.float32).reshape(
        1, 3, IMAGE_SIZE, IMAGE_SIZE
    ) / 255.0


def _actuator_slice(arm: str) -> slice:
    return slice(0, 6) if arm == "left" else slice(6, 12)


def _proprio(env: TandemEnv, limits: np.ndarray) -> np.ndarray:
    raw = np.empty(12, dtype=np.float32)
    for i, arm in enumerate(layout.ARMS):
        raw[i * 6 : i * 6 + 5] = env.arm_qpos(arm)
        raw[i * 6 + 5] = float(env.data.qpos[env.index.grip_qpos[arm]])
    return normalize_joints(raw, limits)


def _sample(chunk: np.ndarray, offset: int, stride: int) -> np.ndarray:
    """Read a chunk at a control tick, interpolating between its strided entries."""
    position = offset / stride
    index = int(position)
    frac = position - index
    lo = chunk[min(index, CHUNK - 1)]
    hi = chunk[min(index + 1, CHUNK - 1)]
    return lo * (1.0 - frac) + hi * frac


def _blend(pending: deque[tuple[int, np.ndarray]], tick: int, stride: int) -> np.ndarray:
    """Temporal ensembling: exponentially weighted mean of every chunk covering ``tick``."""
    total = np.zeros(ACTION_DIM, dtype=np.float32)
    weight_sum = 0.0
    for start, chunk in pending:
        offset = tick - start
        if 0 <= offset < CHUNK * stride:
            weight = float(np.exp(-ENSEMBLE_DECAY * offset / stride))
            total += weight * _sample(chunk, offset, stride)
            weight_sum += weight
    return total / max(weight_sum, 1e-6)


class PolicyExecutor(Executor):
    """Scripted executor with ``pick`` and ``place`` delegated to the learned policy."""

    def __init__(self, env: TandemEnv, runner: PolicyRunner, *, apply: str = "acting", **kwargs):
        super().__init__(env, **kwargs)
        self.runner = runner
        self.apply = apply

    def pick(self, arm: str, obj: str, **_: Any) -> SkillResult:
        t0 = time.perf_counter()
        env = self.env
        if env.attached[arm] is not None:
            return SkillResult("pick", arm, False, "GRIPPER_FULL",
                               f"{arm} is already holding {env.attached[arm]}",
                               time.perf_counter() - t0)
        stats = self.runner.run_skill(env, "pick", arm, obj, None, apply=self.apply,
                                      on_tick=self._notify)
        held = env.is_grasped(arm, obj) and env.object_pos(obj)[2] > layout.REST_Z[obj] + 0.010
        if held:
            env.attached[arm] = obj
            self.grasp_transform[arm] = grasp_transform(env, arm, obj)
            # Hand the scripted layer the same tool offset its own grasp would have left,
            # measured from where the jaws actually ended up, so a following handoff or
            # pour solves against the real geometry of the grip the policy achieved.
            gap = self.grip.gap_for_cmd(float(env.data.qpos[env.index.grip_qpos[arm]]))
            self.held_tool[arm] = self.grip.grasp_center_local(
                gap, self.grip.seat_depth_for(float(env.object_pos(obj)[2]), 0.0)
            )
        return SkillResult(
            "pick", arm, bool(held), "OK" if held else "GRASP_FAILED",
            f"{obj} at z={env.object_pos(obj)[2]:.3f}", time.perf_counter() - t0,
            {"object": obj, "policy": stats},
        )

    def place(self, arm: str, obj: str, target: str) -> SkillResult:
        t0 = time.perf_counter()
        env = self.env
        if env.attached[arm] != obj:
            return SkillResult("place", arm, False, "GRIPPER_EMPTY",
                               f"{arm} is not holding {obj}", time.perf_counter() - t0)
        if target not in layout.SLOTS and target != "staging":
            return SkillResult("place", arm, False, "UNKNOWN_TARGET", target,
                               time.perf_counter() - t0)
        stats = self.runner.run_skill(env, "place", arm, obj, target, apply=self.apply,
                                      on_tick=self._notify)
        env.attached[arm] = None
        self.grasp_transform[arm] = None
        self.held_tool[arm] = None
        if target == "staging":
            down = float(env.object_pos(obj)[2]) < layout.REST_Z[obj] + 0.02
            return SkillResult("place", arm, down, "OK" if down else "PLACE_MISSED",
                               f"{obj} set down clear of the setting", time.perf_counter() - t0,
                               {"object": obj, "target": target, "policy": stats})
        placed = env.object_on_slot(obj) == target
        error = float(np.linalg.norm(env.object_pos(obj)[:2] - layout.SLOTS[target][:2]))
        return SkillResult(
            "place", arm, placed, "OK" if placed else "PLACE_MISSED",
            f"{obj} {'on' if placed else 'off'} {target} ({error * 1000:.0f} mm from centre)",
            time.perf_counter() - t0,
            {"object": obj, "target": target, "err": round(error, 4), "policy": stats},
        )

    def _notify(self) -> None:
        self.tick_count += 1
        if self.on_tick is not None:
            self.on_tick(self)
