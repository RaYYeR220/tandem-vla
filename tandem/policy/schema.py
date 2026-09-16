"""Frozen encoding contract for the action-chunking policy.

The collector, the trainer and the closed-loop runner all index observations and actions
through this module, so there is exactly one definition of what "the 12-d proprioception
vector" or "the goal conditioning vector" means.

Layout of the 12-d proprioception and action vectors matches the simulator's actuator
order exactly, which is what ``TandemEnv.ctrl`` holds::

    [ left j0..j4, left gripper, right j0..j4, right gripper ]

Both are normalized to roughly [-1, 1] by the actuators' own joint limits, so the network
never sees a raw radian and the loss weights every joint comparably.
"""

from __future__ import annotations

import numpy as np

from ..sim import layout

#: Square resolution of each camera the policy sees.
IMAGE_SIZE: int = 128

#: Cameras: the fixed overhead view plus the acting arm's wrist view. Which wrist camera
#: that is depends on the arm the skill is running on, so the second slot is filled in at
#: record and rollout time; the network only ever sees "overhead, wrist".
CAMERA_SLOTS: tuple[str, str] = ("overhead", "wrist")

#: Chunk length the network predicts, and how much of it the runner executes before
#: re-inferring. K = 16 is 0.32 s of control at 50 Hz; M = 8 halves that.
CHUNK: int = 16
EXECUTE: int = 8

#: Skill vocabulary for the conditioning one-hot. The full set is reserved even though the
#: distilled policy is only trained on pick and place, so the encoding never has to change.
SKILLS: tuple[str, ...] = (
    "pick",
    "place",
    "open_drawer",
    "close_drawer",
    "handoff",
    "hold",
    "pour",
    "home",
)

#: Objects a skill can be aimed at.
OBJECTS: tuple[str, ...] = ("plate", "mug", "bottle", "spoon", "fork", "drawer")

#: Place targets. ``none`` covers every skill that has no destination (a pick, a home).
TARGETS: tuple[str, ...] = (
    "none",
    "slot_plate",
    "slot_fork",
    "slot_spoon",
    "slot_mug",
    "staging",
)

ARMS: tuple[str, str] = ("left", "right")

PROPRIO_DIM = 12
ACTION_DIM = 12
COND_DIM = len(SKILLS) + len(OBJECTS) + len(ARMS) + len(TARGETS)  # 8 + 6 + 2 + 6 = 22

#: Per-actuator scaling, filled once from the compiled model's joint limits.
_JOINT_LIMITS: np.ndarray | None = None


def joint_limits(model) -> np.ndarray:
    """``(12, 2)`` lower/upper actuator limits in the canonical control order."""
    global _JOINT_LIMITS
    if _JOINT_LIMITS is None:
        rows = []
        for arm in ARMS:
            names = layout.NAMES[arm]
            for joint in names.joints:
                jid = model.joint(joint).id
                rows.append(model.jnt_range[jid])
            gid = model.joint(names.gripper_joint).id
            rows.append(model.jnt_range[gid])
        _JOINT_LIMITS = np.asarray(rows, dtype=np.float32)
    return _JOINT_LIMITS


def normalize_joints(values: np.ndarray, limits: np.ndarray) -> np.ndarray:
    """Radians -> [-1, 1] against the actuator limits."""
    lo, hi = limits[:, 0], limits[:, 1]
    return (2.0 * (np.asarray(values, dtype=np.float32) - lo) / (hi - lo) - 1.0).astype(np.float32)


def denormalize_joints(values: np.ndarray, limits: np.ndarray) -> np.ndarray:
    """[-1, 1] -> radians."""
    lo, hi = limits[:, 0], limits[:, 1]
    return ((np.asarray(values, dtype=np.float32) + 1.0) * 0.5 * (hi - lo) + lo).astype(np.float32)


def condition_vector(skill: str, obj: str | None, arm: str, target: str | None) -> np.ndarray:
    """Build the 22-d goal conditioning vector.

    Deliberately symbolic: the policy is told *what* it is doing and *where it is meant to
    end up* among the fixed table slots, but never where the object is. Finding the object
    is the visual half of the job and has to come out of the cameras.
    """
    vec = np.zeros(COND_DIM, dtype=np.float32)
    offset = 0
    if skill in SKILLS:
        vec[offset + SKILLS.index(skill)] = 1.0
    offset += len(SKILLS)
    if obj in OBJECTS:
        vec[offset + OBJECTS.index(obj)] = 1.0
    offset += len(OBJECTS)
    vec[offset + ARMS.index(arm)] = 1.0
    offset += len(ARMS)
    vec[offset + TARGETS.index(target if target in TARGETS else "none")] = 1.0
    return vec


def wrist_camera(arm: str) -> str:
    return f"{arm}_wrist"
