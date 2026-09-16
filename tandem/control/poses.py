"""Grasp geometry: where to put the gripper for each object, and how to get there.

The SO-101 is a small five-joint arm, so a strictly top-down grasp is unreachable across a
good part of the table. Every pose request therefore sweeps a short ladder of approach tilts
and returns the first one the IK can actually achieve, which is also what makes the gate's
`OUT_OF_REACH` verdict meaningful: if none of the tilts solves, the arm genuinely cannot do it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..sim import layout
from .ik import ArmIK, IKResult, approach_frame

#: Approach tilts away from vertical, tried in order. Measured against the compiled model,
#: this arm can only hold a near-vertical finger axis at table height, so the ladder is short.
TILT_LADDER = (0.0, 0.20, 0.40)

#: Clearance above the grasp point for the pre-grasp and retreat poses.
APPROACH_CLEARANCE = 0.028
LIFT_HEIGHT = 0.030


@dataclass
class Pose:
    pos: np.ndarray
    rot: np.ndarray
    qpos: np.ndarray
    tilt: float
    ik: IKResult
    #: Residual joint error after the move, when the arm could not reach the solved pose.
    blocked: float = 0.0

    @property
    def ok(self) -> bool:
        return self.ik.ok and self.blocked == 0.0

    @property
    def solvable(self) -> bool:
        return self.ik.ok


def radial_dir(arm: str, xy) -> np.ndarray:
    """Unit vector in the table plane pointing from the arm's base to ``xy``."""
    d = np.asarray(xy, dtype=float)[:2] - layout.ARM_BASE[arm][:2]
    n = np.linalg.norm(d)
    if n < 1e-6:
        return np.array([0.0, 1.0, 0.0])
    return np.array([d[0] / n, d[1] / n, 0.0])


def tilted_frame(arm: str, xy, tilt: float, jaw_yaw: float) -> np.ndarray:
    """Gripper frame that points down, tilted ``tilt`` radians back along the radial axis."""
    approach = -np.array([0.0, 0.0, 1.0]) * np.cos(tilt) - radial_dir(arm, xy) * np.sin(tilt)
    jaw = np.array([np.cos(jaw_yaw), np.sin(jaw_yaw), 0.0])
    return approach_frame(approach, jaw)


def solve_pose(
    ik: ArmIK,
    data,
    pos,
    jaw_yaw: float,
    *,
    arm: str,
    tilts=TILT_LADDER,
    q_init: np.ndarray | None = None,
    restarts: int = 5,
    tool_local: np.ndarray | None = None,
) -> Pose:
    """Best reachable pose that puts the tool point at ``pos`` with jaws along ``jaw_yaw``.

    ``tool_local`` is the offset from the gripper site to the point we actually care about,
    expressed in the site frame — normally the midpoint between the closed jaws.
    """
    pos = np.asarray(pos, dtype=float)
    best: Pose | None = None
    # The pinch point sits ~15 mm off the wrist site along the jaw axis, so flipping the jaw
    # yaw by 180 degrees moves the whole wrist to the other side of the object. For a
    # symmetric grasp both are equivalent to the object and very different to the arm, so try
    # the one that keeps the wrist nearer the base first.
    base = layout.ARM_BASE[arm][:2]
    yaws = [jaw_yaw, jaw_yaw + np.pi]
    if tool_local is not None:
        def wrist_dist(y: float) -> float:
            R = tilted_frame(arm, pos[:2], 0.0, y)
            return float(np.linalg.norm((pos - R @ np.asarray(tool_local))[:2] - base))

        yaws.sort(key=wrist_dist)
    for jy in yaws:
        for tilt in tilts:
            R = tilted_frame(arm, pos[:2], tilt, jy)
            site_target = pos if tool_local is None else pos - R @ np.asarray(tool_local)
            res = ik.solve(data, site_target, R, q_init=q_init, restarts=restarts)
            cand = Pose(pos=site_target, rot=R, qpos=res.qpos, tilt=tilt, ik=res)
            if best is None or (res.pos_err + 0.05 * res.rot_err) < (
                best.ik.pos_err + 0.05 * best.ik.rot_err
            ):
                best = cand
            if res.ok:
                return cand
    assert best is not None
    return best


def grasp_transform(env, arm: str, obj: str) -> np.ndarray:
    """4x4 pose of the object expressed in the gripper frame, captured at grasp time."""
    T_t = np.eye(4)
    T_t[:3, :3] = env.tcp_rot(arm)
    T_t[:3, 3] = env.tcp(arm)
    T_o = np.eye(4)
    T_o[:3, :3] = env.object_rot(obj)
    T_o[:3, 3] = env.object_pos(obj)
    return np.linalg.inv(T_t) @ T_o


def solve_for_object_pose(
    ik: ArmIK,
    data,
    T_obj_des: np.ndarray,
    T_rel: np.ndarray,
    *,
    q_init: np.ndarray | None = None,
    restarts: int = 6,
) -> Pose:
    """IK on the *object's* pose rather than the gripper's.

    Used whenever the thing that has to end up somewhere specific is the payload, not the
    hand — pouring being the obvious case.
    """
    T_tcp = T_obj_des @ np.linalg.inv(T_rel)
    res = ik.solve(
        data,
        T_tcp[:3, 3],
        T_tcp[:3, :3],
        q_init=q_init,
        restarts=restarts,
        rot_weight=0.55,
    )
    return Pose(pos=T_tcp[:3, 3], rot=T_tcp[:3, :3], qpos=res.qpos, tilt=0.0, ik=res)


def pour_pose_candidates(
    mouth: np.ndarray, *, spout_local_z: float, azimuths: int = 12, rolls: int = 6
) -> list[np.ndarray]:
    """Desired carton poses that put its rim over ``mouth``, ordered easiest-first.

    Three things are free here and exploiting all of them is what makes the pour reachable on a
    five-joint arm: which way the carton leans (azimuth), how far above the mug it sits (dz),
    and how it is rolled about its own axis, which changes nothing about where the liquid goes
    but a great deal about whether the wrist can hold the pose.
    """
    out: list[np.ndarray] = []
    for tilt in (1.75, 1.60, 1.90, 1.45, 1.30):
        for dz in (0.045, 0.060, 0.032, 0.080):
            for k in range(azimuths):
                az = 2.0 * np.pi * k / azimuths
                lean = np.array([np.cos(az), np.sin(az), 0.0])
                perp = np.array([-lean[1], lean[0], 0.0])
                K = np.array(
                    [
                        [0.0, -perp[2], perp[1]],
                        [perp[2], 0.0, -perp[0]],
                        [-perp[1], perp[0], 0.0],
                    ]
                )
                R_tilt = np.eye(3) + np.sin(tilt) * K + (1.0 - np.cos(tilt)) * (K @ K)
                rim = R_tilt @ np.array([0.0, 0.0, spout_local_z])
                pos = mouth + np.array([0.0, 0.0, dz]) - rim
                for r in range(rolls):
                    ang = 2.0 * np.pi * r / rolls
                    c, s = np.cos(ang), np.sin(ang)
                    R_roll = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
                    T = np.eye(4)
                    T[:3, :3] = R_tilt @ R_roll
                    T[:3, 3] = pos
                    out.append(T)
    return out


def grasp_geometry(env, obj: str, *, handoff_offset: float = 0.0) -> tuple[np.ndarray, float]:
    """World grasp point and jaw yaw for ``obj`` in its current pose.

    ``handoff_offset`` shifts the grasp along the object's long axis so two grippers can
    hold a thin object at the same time.
    """
    pos = env.object_pos(obj).copy()
    yaw = env.object_yaw(obj)
    axis = np.array([np.cos(yaw), np.sin(yaw), 0.0])

    if obj in ("spoon", "fork"):
        # Grip the handle a little behind its midpoint so the bowl or tines stay clear.
        pos = pos - axis * 0.016 + axis * handoff_offset
        jaw_yaw = yaw + np.pi / 2
    elif obj == "bottle":
        # Square carton: close across a face, not a diagonal.
        jaw_yaw = yaw
    else:
        # Discs and cylinders: close across any diameter. Aim the jaws across the radial
        # direction so the wrist stays near the middle of its roll range.
        jaw_yaw = yaw
    pos[2] += layout.GRASP_OFFSET_Z[obj]
    return pos, float(jaw_yaw)
