"""Runtime scene estimation: cameras in, a `world_state()`-shaped dict out.

This is the drop-in replacement for privileged simulator state. It runs the exported
OpenVINO IR on two camera frames and then *derives* every symbolic field the rest of the
stack consumes -- `reachable_by`, `on_slot`, `in_drawer` -- from the estimated positions
using :mod:`tandem.sim.layout`, with the same tolerances the simulator's own bookkeeping
uses. Nothing here reads a body position out of MuJoCo.

Two inputs are deliberately *not* treated as privileged:

* **Joint encoders.** ``arms`` carries each arm's measured joint angles, gripper command
  and forward-kinematic tool point. A real SO-101 knows all three from its own servos; no
  camera is required and no scene knowledge is involved.
* **Nothing else.** In particular ``held_by`` is inferred from the gripper being closed on
  an estimated object position near the tool point, not from the executor's book-keeping.

One field genuinely cannot be estimated from these two cameras: the water count. The
particles live inside an opaque carton and, once poured, inside a mug seen from above.
The returned dict therefore carries zeros with ``observed: False`` unless a caller passes
a measurement in, and never pretends otherwise.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import openvino as ov

from ..sim import layout
from ..sim.env import GRASP_TOL, PLACE_TOL_XY, PLACE_TOL_Z
from .schema import (
    CAMERAS,
    DRAWER_INDEX,
    IMAGE_SIZE,
    POS_SLICE,
    PROPS,
    VIS_SLICE,
    YAW_SLICE,
    denormalize_pos,
    stack_views,
)

#: Half-extents of the drawer tray, matching ``TandemEnv.object_in_drawer``.
DRAWER_HALF_X = 0.062
DRAWER_HALF_Y = 0.078
DRAWER_Z_RANGE = (0.012, 0.046)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def on_slot_for(name: str, pos: np.ndarray) -> str | None:
    """Which slot an object at ``pos`` counts as resting on, or ``None``.

    Mirrors ``TandemEnv.object_on_slot`` exactly, including its tolerances, but reads an
    estimated position instead of a body pose.
    """
    for slot, slot_pos in layout.SLOTS.items():
        if layout.SLOT_OBJECT[slot] != name:
            continue
        if (
            float(np.linalg.norm(np.asarray(pos)[:2] - slot_pos[:2])) < PLACE_TOL_XY
            and abs(float(pos[2]) - layout.REST_Z[name]) < PLACE_TOL_Z
        ):
            return slot
    return None


def in_drawer_for(pos: np.ndarray, open_frac: float) -> bool:
    """Mirrors ``TandemEnv.object_in_drawer`` against the estimated drawer pose."""
    drawer = drawer_position(open_frac)
    return bool(
        abs(float(pos[0]) - drawer[0]) < DRAWER_HALF_X
        and abs(float(pos[1]) - drawer[1]) < DRAWER_HALF_Y
        and DRAWER_Z_RANGE[0] < float(pos[2]) < DRAWER_Z_RANGE[1]
    )


def drawer_position(open_frac: float) -> np.ndarray:
    """Centre of the drawer tray for a given open fraction, in table coordinates.

    The cabinet is bolted to the table and the drawer runs on a prismatic joint along -y,
    so its pose is a known function of one scalar -- which is exactly the scalar the
    network estimates.
    """
    return np.array(
        [
            layout.CABINET_XY[0],
            layout.CABINET_XY[1] - float(open_frac) * layout.DRAWER_OPEN_QPOS,
        ]
    )


class PerceptionEstimator:
    """Compiled-IR scene estimator producing the frozen world-state schema."""

    def __init__(
        self,
        ir_path: str | Path,
        *,
        device: str = "CPU",
        core: ov.Core | None = None,
        performance_hint: str = "LATENCY",
    ):
        self.ir_path = Path(ir_path)
        self.device = device
        self._core = core or ov.Core()
        model = self._core.read_model(self.ir_path)
        self._compiled = self._core.compile_model(
            model, device, {"PERFORMANCE_HINT": performance_hint}
        )
        self._request = self._compiled.create_infer_request()
        self._input_name = self._compiled.inputs[0].get_any_name()
        self.last_latency_ms: float = float("nan")

    # ------------------------------------------------------------------ raw inference

    def infer(self, views: Mapping[str, np.ndarray]) -> np.ndarray:
        """Two ``HxWx3`` uint8 frames -> the raw 31-vector."""
        batch = stack_views({c: views[c] for c in CAMERAS}).reshape(1, 6, IMAGE_SIZE, IMAGE_SIZE)
        started = time.perf_counter_ns()
        result = self._request.infer({self._input_name: batch})
        self.last_latency_ms = (time.perf_counter_ns() - started) / 1e6
        return np.asarray(next(iter(result.values()))).reshape(-1)

    def decode(self, raw: np.ndarray) -> dict[str, np.ndarray]:
        """Raw network output -> metres, radians and probabilities."""
        yaw_pair = raw[YAW_SLICE].reshape(-1, 2)
        return {
            "pos": denormalize_pos(raw[POS_SLICE].reshape(-1, 3)),
            "yaw": np.arctan2(yaw_pair[:, 1], yaw_pair[:, 0]),
            "visible": _sigmoid(raw[VIS_SLICE]),
            "drawer_open": float(np.clip(_sigmoid(np.array([raw[DRAWER_INDEX]]))[0], 0.0, 1.0)),
        }

    # ------------------------------------------------------------------ world state

    def estimate(
        self,
        views: Mapping[str, np.ndarray],
        arms: Mapping[str, Mapping[str, Any]],
        *,
        t: float = 0.0,
        seed: int | None = None,
        water: Mapping[str, int] | None = None,
    ) -> dict:
        """Produce a dict matching ``TandemEnv.world_state()`` exactly.

        ``arms`` maps each arm name to ``{"qpos": [5 floats], "gripper": float,
        "tcp": [x, y, z]}`` -- proprioception, not scene knowledge.
        """
        decoded = self.decode(self.infer(views))
        return self.world_state(decoded, arms, t=t, seed=seed, water=water)

    def world_state(
        self,
        decoded: Mapping[str, np.ndarray],
        arms: Mapping[str, Mapping[str, Any]],
        *,
        t: float = 0.0,
        seed: int | None = None,
        water: Mapping[str, int] | None = None,
    ) -> dict:
        positions = {name: np.asarray(decoded["pos"][i], dtype=float)
                     for i, name in enumerate(PROPS)}
        yaws = {name: float(decoded["yaw"][i]) for i, name in enumerate(PROPS)}
        visibility = {name: float(decoded["visible"][i]) for i, name in enumerate(PROPS)}
        open_frac = float(decoded["drawer_open"])
        held = self._infer_holders(positions, arms)

        objects: dict[str, dict] = {}
        for name in PROPS:
            pos = positions[name]
            objects[name] = {
                "pos": [round(float(v), 4) for v in pos],
                "yaw": round(yaws[name], 4),
                "held_by": held.get(name),
                "on_slot": on_slot_for(name, pos),
                "in_drawer": in_drawer_for(pos, open_frac),
                "reachable_by": layout.reaching_arms(pos),
                "visible": round(visibility[name], 3),
            }

        slots = {}
        for slot, slot_pos in layout.SLOTS.items():
            want = layout.SLOT_OBJECT[slot]
            slots[slot] = {
                "pos": [round(float(v), 4) for v in slot_pos],
                "occupied_by": want if objects[want]["on_slot"] == slot else None,
                "reachable_by": layout.reaching_arms(slot_pos),
            }

        arm_state = {}
        for arm in layout.ARMS:
            source = arms.get(arm, {})
            holding = next((o for o, a in held.items() if a == arm), None)
            arm_state[arm] = {
                "holding": holding,
                "tcp": [round(float(v), 4) for v in np.asarray(source.get("tcp", (0.0, 0.0, 0.0)))],
                "qpos": [round(float(v), 4) for v in np.asarray(source.get("qpos", [0.0] * 5))],
                "gripper": round(float(source.get("gripper", layout.GRIPPER_OPEN)), 4),
                "busy": bool(source.get("busy", False)),
            }

        water_out = {"in_mug": 0, "in_bottle": 0, "spilled": 0, "observed": False}
        if water is not None:
            water_out = {**{k: int(v) for k, v in water.items()}, "observed": True}

        return {
            "t": round(float(t), 3),
            "seed": seed,
            "objects": objects,
            "drawer": {
                "open_frac": round(open_frac, 3),
                "is_open": bool(open_frac * layout.DRAWER_OPEN_QPOS
                                >= layout.DRAWER_OPEN_THRESHOLD),
                "reachable_by": layout.reaching_arms(drawer_position(open_frac)),
            },
            "slots": slots,
            "arms": arm_state,
            "water": water_out,
            "source": "perception",
            "infer_ms": round(self.last_latency_ms, 3),
            "device": self.device,
        }

    # ------------------------------------------------------------------ derivations

    def _infer_holders(
        self, positions: Mapping[str, np.ndarray], arms: Mapping[str, Mapping[str, Any]]
    ) -> dict[str, str]:
        """An arm holds the nearest estimated object to its tool point, if its jaws are
        closed at all and that object is within a grasp's reach of the tool point."""
        held: dict[str, str] = {}
        claimed: set[str] = set()
        for arm in layout.ARMS:
            source = arms.get(arm)
            if source is None:
                continue
            gripper = float(source.get("gripper", layout.GRIPPER_OPEN))
            if gripper > 0.95 * layout.GRIPPER_OPEN:
                continue
            tcp = np.asarray(source.get("tcp", (0.0, 0.0, 0.0)), dtype=float)
            best, best_distance = None, GRASP_TOL
            for name, pos in positions.items():
                if name in claimed:
                    continue
                distance = float(np.linalg.norm(pos - tcp))
                if distance < best_distance:
                    best, best_distance = name, distance
            if best is not None:
                held[best] = arm
                claimed.add(best)
        return held


def arm_proprioception(env) -> dict[str, dict[str, Any]]:
    """Read the joint encoders and forward kinematics an SO-101 knows about itself.

    Lives here so evaluation code has one obvious call for "everything the robot knows
    without looking", and it is visibly not reading object or slot state.
    """
    return {
        arm: {
            "qpos": [float(v) for v in env.arm_qpos(arm)],
            "gripper": float(env.gripper_cmd(arm)),
            "tcp": [float(v) for v in env.tcp(arm)],
        }
        for arm in layout.ARMS
    }


def render_views(env, *, size: int = IMAGE_SIZE) -> dict[str, np.ndarray]:
    """Grab the two camera frames the estimator expects."""
    return {cam: env.render(cam, size, size) for cam in CAMERAS}


def calibration_samples(
    views: Sequence[Mapping[str, np.ndarray]], input_name: str
) -> list[dict[str, np.ndarray]]:
    """Turn recorded camera pairs into NNCF calibration samples."""
    return [
        {input_name: stack_views({c: v[c] for c in CAMERAS}).reshape(1, 6, IMAGE_SIZE, IMAGE_SIZE)}
        for v in views
    ]
