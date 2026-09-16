"""Named geometry of the dinner-table workcell.

Everything downstream (IK targets, the reachability gate, the planner's world model,
the evaluator) reads its coordinates from here so there is exactly one source of truth.
Table surface is the z = 0 plane. +y points away from the two arm bases.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ARMS = ("left", "right")

#: Base of each arm in table coordinates.
ARM_BASE = {
    "left": np.array([-0.20, -0.05, 0.0]),
    "right": np.array([0.20, -0.05, 0.0]),
}

#: Planar annulus each arm can service. Deliberately conservative: the SO-101 can stretch
#: a little further, but grasps become unreliable past R_MAX, so the gate treats anything
#: outside as out of reach rather than letting the executor discover it the hard way.
R_MIN = 0.085
R_MAX = 0.280

#: Nothing behind the shoulder line.
Y_MIN = -0.055

#: Place setting — right-arm territory. Spaced so a 110 mm piece of cutlery lying on its slot
#: does not overlap the plate: they did, and laying the fork used to knock the plate off.
SLOTS = {
    "slot_plate": np.array([0.27, 0.14, 0.006]),
    "slot_fork": np.array([0.160, 0.14, 0.006]),
    "slot_spoon": np.array([0.375, 0.14, 0.006]),
    "slot_mug": np.array([0.285, 0.210, 0.032]),
}

#: Which object each slot is meant to receive (used by the task success check).
SLOT_OBJECT = {
    "slot_plate": "plate",
    "slot_fork": "fork",
    "slot_spoon": "spoon",
    "slot_mug": "mug",
}
OBJECT_SLOT = {v: k for k, v in SLOT_OBJECT.items()}

#: Where the mug is presented to be filled: low, central, and inside both envelopes so one arm
#: can steady it while the other tips the carton over it.
POUR_STATION = np.array([0.0, 0.085])
POUR_STATION_Z = 0.035

#: Shared zone where a hand-off happens. Reachable by both arms, by construction.
HANDOFF_XY = np.array([0.0, 0.075])
HANDOFF_Z = 0.050

#: Where each arm sets something down when asked to clear it out of the way.
STAGING_DROP = {
    "left": np.array([-0.135, 0.055]),
    "right": np.array([0.125, 0.055]),
}

#: Rectangle the loose props are scattered in on reset, further filtered by reachability so
#: that every prop starts inside left-arm-only territory while every slot sits in
#: right-arm-only territory. That makes a hand-off structurally necessary, not decorative.
STAGING_X = (-0.250, -0.050)
STAGING_Y = (0.040, 0.200)

#: Reach band a prop must start in, and the margin by which the right arm must miss it.
STAGING_REACH = (0.135, 0.252)
STAGING_RIGHT_CLEAR = 0.305

#: Minimum free space between two props, on top of their radii. Generous, because the SO-101
#: fingers are long and sweep a corridor on the way down.
PROP_CLEARANCE = 0.030

#: Cabinet footprint to keep props out of.
CABINET_KEEPOUT_X = -0.243
CABINET_KEEPOUT_Y = 0.065

#: Fallback layout used when randomization is switched off.
NOMINAL_PROP_XY = {
    "plate": (-0.100, 0.150),
    "mug": (-0.185, 0.095),
    "bottle": (-0.215, 0.168),
}

#: Drawer, in left-arm territory.
CABINET_XY = np.array([-0.34, 0.18])
DRAWER_OPEN_QPOS = 0.105
DRAWER_OPEN_THRESHOLD = 0.070

PROPS = ("plate", "mug", "bottle", "spoon", "fork")
DRAWER_CONTENTS = ("spoon", "fork")
TABLE_PROPS = ("plate", "mug", "bottle")

#: Approximate half-extent used for placement checks and collision padding.
PROP_RADIUS = {
    "plate": 0.027,
    "mug": 0.023,
    "bottle": 0.022,
    "spoon": 0.058,
    "fork": 0.058,
}

#: Width the jaws must span to grasp the prop, in metres. Everything is sized against the
#: SO-101's *measured* usable opening, which is about 50 mm -- considerably less than the
#: 133 mm of fingertip travel suggests, because the jaw faces close in well ahead of the tips.
GRASP_WIDTH = {
    "plate": 0.048,
    "mug": 0.044,
    "bottle": 0.042,
    "spoon": 0.011,
    "fork": 0.011,
    "drawer": 0.011,
}

#: Height of the grasp point above the object's body frame origin.
GRASP_OFFSET_Z = {
    "plate": 0.000,
    "mug": 0.004,
    "bottle": 0.006,
    "spoon": 0.000,
    "fork": 0.000,
}

#: Surface the object is standing on, used to work out how deep the jaws can seat without
#: driving the fingertips through it.
SUPPORT_Z = {
    "plate": 0.0,
    "mug": 0.0,
    "bottle": 0.0,
    "spoon": 0.016,
    "fork": 0.016,
}

#: Resting height of the body frame when the prop sits on the table.
REST_Z = {
    "plate": 0.0181,
    "mug": 0.0292,
    "bottle": 0.0302,
    "spoon": 0.0033,
    "fork": 0.0033,
}


def reach(arm: str, xy) -> float:
    """Planar distance from ``arm``'s base to ``xy``."""
    p = np.asarray(xy, dtype=float)[:2]
    return float(np.linalg.norm(p - ARM_BASE[arm][:2]))


def in_reach(arm: str, xy) -> bool:
    p = np.asarray(xy, dtype=float)
    if p[1] < Y_MIN:
        return False
    return R_MIN <= reach(arm, p) <= R_MAX


def reaching_arms(xy) -> list[str]:
    return [a for a in ARMS if in_reach(a, xy)]


def other_arm(arm: str) -> str:
    return "right" if arm == "left" else "left"


@dataclass(frozen=True)
class ArmNames:
    """Model element names for one attached SO-101, after the `left/` `right/` prefixing."""

    arm: str

    @property
    def prefix(self) -> str:
        return f"{self.arm}/"

    @property
    def joints(self) -> tuple[str, ...]:
        return tuple(
            self.prefix + j
            for j in (
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
            )
        )

    @property
    def gripper_joint(self) -> str:
        return self.prefix + "gripper"

    @property
    def actuators(self) -> tuple[str, ...]:
        return self.joints + (self.gripper_joint,)

    @property
    def tcp_site(self) -> str:
        return self.prefix + "gripperframe"

    @property
    def wrist_cam(self) -> str:
        return self.prefix + "wrist_cam"


NAMES = {a: ArmNames(a) for a in ARMS}

#: Gripper actuator command range (radians on the jaw hinge).
GRIPPER_OPEN = 1.35
GRIPPER_CLOSED = -0.17
#: Extra closure past the object width, in metres, that produces a firm pinch.
GRIPPER_SQUEEZE = 0.005

#: Jaw servo tuning. Measured, not guessed. The stock gain (kp 998) saturates its torque limit
#: at the slightest interference, and a rigid prop squeezed at the stock 2.94 N*m limit is
#: ejected from the jaws the moment the arm accelerates -- every lift failed until this was
#: changed. Simply lowering the torque limit instead left the jaws too weak to close against
#: their own damping. So the gain comes down and the limit stays: the jaws still travel, and
#: the pinch force is proportional to how far past contact the command is parked.
GRIPPER_KP = 5.0
GRIPPER_KV = 0.8
GRIPPER_FORCE_LIMIT = 2.94

#: How far past the stall point the jaw command is parked, in radians -- with the soft gain,
#: this is the pinch force. Swept against the grasp suite: 0.10 holds almost nothing, 0.16
#: holds the plate but loses the mug, 0.30 is where every prop is retained, and going further
#: buys nothing. The jaw stalls on contact well before the command gets there, so the
#: effective squeeze stays modest regardless.
GRIPPER_FORCE_ERR = 0.30
