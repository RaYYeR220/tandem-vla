"""Gripper calibration.

The jaw command is a hinge angle, not an opening width, and the relationship between the
two is not linear. Rather than hand-tune magic numbers per object we sweep the joint once
against the compiled model, measure the real fingertip gap, and invert the curve.
"""

from __future__ import annotations

import functools

import mujoco
import numpy as np

from ..sim import layout

_FIXED_TIP = "fixed_jaw_sph_tip1"
_MOVING_TIP = "moving_jaw_sph_tip1"


@functools.lru_cache(maxsize=4)
def _curve(model_id: int, arm: str) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover
    raise RuntimeError("use GripperCalibration")


class GripperCalibration:
    """Maps fingertip gap (metres) to jaw command (radians) and back."""

    def __init__(self, model: mujoco.MjModel, index, arm: str = "left", samples: int = 96):
        data = mujoco.MjData(model)
        jid = index.joint[layout.NAMES[arm].gripper_joint]
        qadr = model.jnt_qposadr[jid]
        lo, hi = model.jnt_range[jid]
        gf = index.geom[f"{arm}/{_FIXED_TIP}"]
        gm = index.geom[f"{arm}/{_MOVING_TIP}"]

        cmds = np.linspace(lo, hi, samples)
        gaps = np.empty_like(cmds)
        for i, c in enumerate(cmds):
            data.qpos[qadr] = c
            mujoco.mj_kinematics(model, data)
            gaps[i] = np.linalg.norm(data.geom_xpos[gf] - data.geom_xpos[gm])

        # Where the stationary jaw tip sits in the gripper-site frame. The site is a wrist
        # frame, not the pinch point, so every IK target has to be corrected by this.
        data.qpos[qadr] = 0.0
        mujoco.mj_kinematics(model, data)
        site = index.tcp_site[arm]
        o = np.array(data.site_xpos[site])
        R = np.array(data.site_xmat[site]).reshape(3, 3)
        self.fixed_tip_local = R.T @ (np.array(data.geom_xpos[gf]) - o)

        order = np.argsort(gaps)
        self.cmds = cmds
        self.gaps = gaps
        self._g_sorted = gaps[order]
        self._c_sorted = cmds[order]
        self.min_gap = float(gaps.min())
        self.max_gap = float(gaps.max())

    #: How far inside the jaws, along the finger axis, we aim to seat an object. Seating at
    #: the very tips is a pinch that slips; a centimetre in is a stable hold.
    SEAT_DEPTH = 0.012

    #: Extra jaw gap held open while approaching. Small on purpose: the object has to sit
    #: centred between the jaws on the way down (touch one jaw and you shove the object across
    #: the table), but every millimetre of extra gap is a millimetre the object gets nudged
    #: when the jaws finally close.
    APPROACH_GAP = 0.012

    def seat_depth_for(self, grasp_z: float, support_z: float = 0.0) -> float:
        """How deep the jaws can take an object without pushing the tips into the support.

        Fingertip height is ``grasp_z - seat``, so a short object simply cannot be held any
        deeper than its own standing height allows. Reporting that honestly is what keeps the
        grasp model matched to the hardware instead of to wishful thinking.
        """
        return float(np.clip(grasp_z - support_z - 0.004, 0.0, self.SEAT_DEPTH))

    def grasp_center_local(self, width: float, seat: float | None = None) -> np.ndarray:
        """Where an object of ``width`` ends up, in the gripper-site frame."""
        c = np.array(self.fixed_tip_local, dtype=float)
        c[0] -= self.SEAT_DEPTH if seat is None else float(seat)
        c[2] += 0.5 * float(width)
        return c

    def cmd_for_gap(self, gap: float) -> float:
        g = float(np.clip(gap, self.min_gap, self.max_gap))
        return float(np.interp(g, self._g_sorted, self._c_sorted))

    def gap_for_cmd(self, cmd: float) -> float:
        return float(np.interp(cmd, self.cmds, self.gaps))

    def open_cmd(self, obj: str | None = None, clearance: float | None = None) -> float:
        """Jaw command that clears the object with room to descend around it."""
        if obj is None:
            return self.cmd_for_gap(self.max_gap)
        c = self.APPROACH_GAP if clearance is None else clearance
        return self.cmd_for_gap(layout.GRASP_WIDTH[obj] + c)

    def grasp_cmd(self, obj: str, squeeze: float | None = None) -> float:
        """Jaw command that pinches the object with a firm interference fit."""
        sq = layout.GRIPPER_SQUEEZE if squeeze is None else squeeze
        return self.cmd_for_gap(layout.GRASP_WIDTH[obj] - sq)
