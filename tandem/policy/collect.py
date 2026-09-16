"""Recording the scripted oracle so the policy can be distilled from it.

The executor already fires ``on_tick`` on every 50 Hz control tick; this module hangs a
recorder off it and off ``run_step``, so every transition is tagged with the skill that
produced it. Two things matter for data quality:

* **Failed steps are dropped.** A pick that ended with the prop still on the table is a
  demonstration of how to fail. The recorder buffers a step's transitions and only commits
  them once the skill reports its post-condition satisfied.
* **Actions are recorded at full rate, observations are not.** The chunk the policy
  predicts has to be consecutive 50 Hz commands or it cannot be replayed, but successive
  camera frames during a 1.2 s ramp are near-identical. Commands are therefore kept for
  every tick and images only every ``stride`` ticks, which cuts the dataset by 3x for no
  measurable loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..control.primitives import Executor, SkillResult
from ..sim.env import TandemEnv
from .schema import (
    ACTION_DIM,
    CHUNK,
    IMAGE_SIZE,
    condition_vector,
    joint_limits,
    normalize_joints,
    wrist_camera,
)

#: Only these skills are distilled. They are the low-level motion the policy is meant to
#: replace; the rest of the plan (hand-offs, pouring) stays with the scripted executor.
RECORDED_SKILLS = ("pick", "place")


@dataclass
class Sample:
    """One training transition."""

    overhead: np.ndarray  #: (128, 128, 3) uint8
    wrist: np.ndarray  #: (128, 128, 3) uint8
    proprio: np.ndarray  #: (12,) normalized joint positions
    cond: np.ndarray  #: (22,) goal conditioning
    chunk: np.ndarray  #: (CHUNK, 12) normalized future commands
    skill: str
    arm: str
    seed: int


@dataclass
class _StepBuffer:
    skill: str
    arm: str
    obj: str | None
    target: str | None
    actions: list[np.ndarray] = field(default_factory=list)
    observations: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=list)


class DemoRecorder:
    """Collects samples from a running episode. One instance per episode."""

    def __init__(self, env: TandemEnv, *, seed: int, stride: int = 3, size: int = IMAGE_SIZE):
        self.env = env
        self.seed = seed
        self.stride = stride
        self.size = size
        self.limits = joint_limits(env.model)
        self.samples: list[Sample] = []
        self.step_stats: list[tuple[str, bool, int]] = []
        self._buffer: _StepBuffer | None = None
        self._tick = 0

    # ------------------------------------------------------------------ hooks

    def begin(self, skill: str, args: dict) -> None:
        if skill not in RECORDED_SKILLS:
            self._buffer = None
            return
        arm = args.get("arm") or args.get("to_arm") or args.get("from_arm") or "left"
        self._buffer = _StepBuffer(
            skill=skill, arm=arm, obj=args.get("object"), target=args.get("target")
        )
        self._tick = 0

    def on_tick(self, executor: Executor) -> None:
        buf = self._buffer
        if buf is None:
            return
        env = self.env
        buf.actions.append(np.array(env.ctrl, dtype=np.float32))
        if self._tick % self.stride == 0:
            overhead = env.render("overhead", self.size, self.size).copy()
            wrist = env.render(wrist_camera(buf.arm), self.size, self.size).copy()
            proprio = self._proprio()
            buf.observations.append((len(buf.actions) - 1, overhead, wrist, proprio))
        self._tick += 1

    def end(self, result: SkillResult) -> None:
        buf, self._buffer = self._buffer, None
        if buf is None:
            return
        self.step_stats.append((buf.skill, bool(result.ok), len(buf.actions)))
        if not result.ok or not buf.actions:
            return
        actions = np.stack(buf.actions)
        normalized = normalize_joints(actions, self.limits)
        cond = condition_vector(buf.skill, buf.obj, buf.arm, buf.target)
        n = len(normalized)
        for index, overhead, wrist, proprio in buf.observations:
            chunk = np.empty((CHUNK, ACTION_DIM), dtype=np.float32)
            for k in range(CHUNK):
                chunk[k] = normalized[min(index + 1 + k, n - 1)]
            self.samples.append(
                Sample(overhead, wrist, proprio, cond, chunk, buf.skill, buf.arm, self.seed)
            )

    # ------------------------------------------------------------------ helpers

    def _proprio(self) -> np.ndarray:
        env = self.env
        raw = np.empty(12, dtype=np.float32)
        for i, arm in enumerate(("left", "right")):
            raw[i * 6 : i * 6 + 5] = env.arm_qpos(arm)
            raw[i * 6 + 5] = float(env.data.qpos[env.index.grip_qpos[arm]])
        return normalize_joints(raw, self.limits)


class RecordingExecutor(Executor):
    """Executor that tells a `DemoRecorder` which skill each tick belongs to.

    The hook sits on the skill methods rather than on ``run_step`` so that skills invoked
    from *inside* another skill are captured too. That matters: a relay hand-off ends with
    the receiving arm picking the object out of the shared zone, and those are the only
    right-arm picks the oracle ever performs. Hooking the dispatcher would have left the
    policy trained on left-arm picks alone.
    """

    def __init__(self, env: TandemEnv, recorder: DemoRecorder, **kwargs):
        super().__init__(env, on_tick=recorder.on_tick, **kwargs)
        self.recorder = recorder

    def pick(self, arm: str, obj: str, **kwargs) -> SkillResult:
        self.recorder.begin("pick", {"arm": arm, "object": obj})
        result = super().pick(arm, obj, **kwargs)
        self.recorder.end(result)
        return result

    def place(self, arm: str, obj: str, target: str) -> SkillResult:
        self.recorder.begin("place", {"arm": arm, "object": obj, "target": target})
        result = super().place(arm, obj, target)
        self.recorder.end(result)
        return result
