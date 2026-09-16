"""The dinner-table environment.

Holds the compiled model, the randomizer and the book-keeping that turns raw MuJoCo state
into the world-state dict every other module consumes. Control runs at 50 Hz on top of a
2 ms physics step.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from . import layout
from .randomize import RandomizationSpec, Randomizer
from .scene import Index, build_model

CONTROL_HZ = 50.0
PHYSICS_SUBSTEPS = 10  # 10 * 2 ms = 20 ms
CONTROL_DT = 0.02

#: Retracted pose. Chosen by search over the joint box for a configuration that keeps the
#: gripper ~0.28 m above the table with comfortable margin on every joint limit, so the
#: servos actually hold it instead of resting the fingers on the tabletop.
HOME_QPOS = np.array([0.0, -1.4688, 0.1904, 1.4088, 0.0])

#: A grasp counts when the object's centre is within this of the fingertip midpoint.
GRASP_TOL = 0.045
#: An object counts as placed when it is this close to its slot, and at rest.
PLACE_TOL_XY = 0.035
PLACE_TOL_Z = 0.030
REST_SPEED = 0.05


@dataclass
class StepResult:
    t: float
    contacts: int
    settled: bool


class TandemEnv:
    """Dual SO-101 table-setting cell."""

    def __init__(
        self,
        spec: RandomizationSpec | None = None,
        *,
        render_size: tuple[int, int] = (480, 640),
        model: mujoco.MjModel | None = None,
    ):
        self.model = model if model is not None else build_model()
        self.data = mujoco.MjData(self.model)
        self.index = Index(self.model)
        self.randomizer = Randomizer(self.model, self.index, spec)
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self._render_size = render_size
        self.seed: int | None = None
        self.variation = None
        for arm in layout.ARMS:
            gid = self.index.grip_act[arm]
            self.model.actuator_forcerange[gid] = [
                -layout.GRIPPER_FORCE_LIMIT,
                layout.GRIPPER_FORCE_LIMIT,
            ]
            # MuJoCo position actuator: gainprm[0] = kp, biasprm[1] = -kp, biasprm[2] = -kv.
            self.model.actuator_gainprm[gid, 0] = layout.GRIPPER_KP
            self.model.actuator_biasprm[gid, 1] = -layout.GRIPPER_KP
            self.model.actuator_biasprm[gid, 2] = -layout.GRIPPER_KV
        self.ctrl = np.zeros(self.model.nu)
        #: Which arm is currently believed to hold which object, maintained by the executor.
        self.attached: dict[str, str | None] = {"left": None, "right": None}

    # ------------------------------------------------------------------ lifecycle

    def reset(self, seed: int = 0, *, settle_steps: int = 220) -> dict:
        self.seed = int(seed)
        self.variation = self.randomizer.apply(self.seed)
        d = self.data
        mujoco.mj_resetData(self.model, d)
        idx = self.index

        for arm in layout.ARMS:
            d.qpos[idx.arm_qpos[arm]] = HOME_QPOS
            d.qpos[idx.grip_qpos[arm]] = layout.GRIPPER_OPEN
        d.qpos[idx.drawer_qpos] = 0.0

        for p in layout.PROPS:
            adr = idx.prop_qpos[p]
            pos = self.variation.prop_pos[p]
            yaw = float(self.variation.prop_yaw[p])
            d.qpos[adr : adr + 3] = pos
            d.qpos[adr + 3 : adr + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]

        self._seed_water()

        d.qvel[:] = 0.0
        self.ctrl[:] = self._home_ctrl()
        d.ctrl[:] = self.ctrl
        self.attached = {"left": None, "right": None}
        for _ in range(settle_steps):
            mujoco.mj_step(self.model, d)
        d.qvel[:] = 0.0
        mujoco.mj_forward(self.model, d)
        return self.world_state()

    def _seed_water(self) -> None:
        """Stack the particles inside the carton, filled close to the brim.

        Fill level matters: the tilt needed to pour scales with the gap between the liquid
        surface and the rim, and the arm has only so much wrist travel to give.
        """
        idx = self.index
        d = self.data
        carton = self.variation.prop_pos["bottle"]
        yaw = float(self.variation.prop_yaw["bottle"])
        c, s = np.cos(yaw), np.sin(yaw)
        rng = np.random.default_rng(self.seed + 977)
        cells = [(-0.0072, -0.0072), (0.0072, -0.0072), (-0.0072, 0.0072), (0.0072, 0.0072)]
        for k, adr in enumerate(idx.water_qpos):
            lx, ly = cells[k % 4]
            level = k // 4
            lx += rng.uniform(-0.0008, 0.0008)
            ly += rng.uniform(-0.0008, 0.0008)
            d.qpos[adr : adr + 3] = [
                carton[0] + c * lx - s * ly,
                carton[1] + s * lx + c * ly,
                carton[2] - 0.019 + 0.0126 * level,
            ]
            d.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]

    def _home_ctrl(self) -> np.ndarray:
        c = np.zeros(self.model.nu)
        for arm in layout.ARMS:
            c[self.index.arm_act[arm]] = HOME_QPOS
            c[self.index.grip_act[arm]] = layout.GRIPPER_OPEN
        return c

    # ------------------------------------------------------------------ stepping

    def set_arm_target(self, arm: str, qpos: np.ndarray) -> None:
        self.ctrl[self.index.arm_act[arm]] = qpos

    def set_gripper(self, arm: str, cmd: float) -> None:
        self.ctrl[self.index.grip_act[arm]] = float(cmd)

    def arm_qpos(self, arm: str) -> np.ndarray:
        return np.array(self.data.qpos[self.index.arm_qpos[arm]])

    def arm_target(self, arm: str) -> np.ndarray:
        return np.array(self.ctrl[self.index.arm_act[arm]])

    def gripper_cmd(self, arm: str) -> float:
        return float(self.ctrl[self.index.grip_act[arm]])

    def step(self, n: int = 1) -> StepResult:
        d = self.data
        d.ctrl[:] = self.ctrl
        for _ in range(n * PHYSICS_SUBSTEPS):
            mujoco.mj_step(self.model, d)
        return StepResult(
            t=float(d.time),
            contacts=int(d.ncon),
            settled=bool(np.max(np.abs(d.qvel)) < REST_SPEED),
        )

    # ------------------------------------------------------------------ queries

    def tcp(self, arm: str) -> np.ndarray:
        return np.array(self.data.site_xpos[self.index.tcp_site[arm]])

    def tcp_rot(self, arm: str) -> np.ndarray:
        return np.array(self.data.site_xmat[self.index.tcp_site[arm]]).reshape(3, 3)

    def object_pos(self, name: str) -> np.ndarray:
        return np.array(self.data.xpos[self.index.prop_body[name]])

    def object_yaw(self, name: str) -> float:
        R = np.array(self.data.xmat[self.index.prop_body[name]]).reshape(3, 3)
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def object_rot(self, name: str) -> np.ndarray:
        return np.array(self.data.xmat[self.index.prop_body[name]]).reshape(3, 3)

    def drawer_open_frac(self) -> float:
        return float(
            self.data.qpos[self.index.drawer_qpos] / layout.DRAWER_OPEN_QPOS
        )

    def drawer_is_open(self) -> bool:
        return bool(
            self.data.qpos[self.index.drawer_qpos] >= layout.DRAWER_OPEN_THRESHOLD
        )

    def jaw_contacts(self, arm: str, obj: str) -> int:
        """Number of live contacts between ``arm``'s jaws and ``obj``."""
        idx = self.index
        jaws = set(idx.jaw_geoms[arm])
        body = idx.prop_body[obj]
        n = 0
        for c in self.data.contact[: self.data.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            in_jaw = (g1 in jaws) + (g2 in jaws)
            on_obj = (idx.geom_body[g1] == body) + (idx.geom_body[g2] == body)
            if in_jaw and on_obj:
                n += 1
        return n

    def is_grasped(self, arm: str, obj: str) -> bool:
        """A grasp is real when the jaws actually touch the object and are not wide open."""
        if self.gripper_cmd(arm) > 0.95 * layout.GRIPPER_OPEN:
            return False
        if self.jaw_contacts(arm, obj) >= 1:
            return True
        return float(np.linalg.norm(self.tcp(arm) - self.object_pos(obj))) < GRASP_TOL

    def holder_of(self, obj: str) -> str | None:
        for arm in layout.ARMS:
            if self.attached[arm] == obj and self.is_grasped(arm, obj):
                return arm
        return None

    def water_counts(self) -> dict:
        idx = self.index
        mug = self.object_pos("mug")
        mug_R = self.object_rot("mug")
        carton = self.object_pos("bottle")
        carton_R = self.object_rot("bottle")
        in_mug = in_bottle = spilled = 0
        for bid in idx.water_bodies:
            p = np.array(self.data.xpos[bid])
            local_m = mug_R.T @ (p - mug)
            local_b = carton_R.T @ (p - carton)
            if np.hypot(*local_m[:2]) < 0.020 and -0.026 < local_m[2] < 0.032:
                in_mug += 1
            elif abs(local_b[0]) < 0.017 and abs(local_b[1]) < 0.017 and abs(local_b[2]) < 0.032:
                in_bottle += 1
            else:
                spilled += 1
        return {"in_mug": in_mug, "in_bottle": in_bottle, "spilled": spilled}

    def object_on_slot(self, obj: str) -> str | None:
        p = self.object_pos(obj)
        for slot, sp in layout.SLOTS.items():
            if layout.SLOT_OBJECT[slot] != obj:
                continue
            if (
                np.linalg.norm(p[:2] - sp[:2]) < PLACE_TOL_XY
                and abs(p[2] - layout.REST_Z[obj]) < PLACE_TOL_Z
            ):
                return slot
        return None

    def object_in_drawer(self, obj: str) -> bool:
        p = self.object_pos(obj)
        drawer = np.array(self.data.xpos[self.index.drawer_body])
        return bool(
            abs(p[0] - drawer[0]) < 0.062
            and abs(p[1] - drawer[1]) < 0.078
            and 0.012 < p[2] < 0.046
        )

    def world_state(self) -> dict:
        objects = {}
        for p in layout.PROPS:
            pos = self.object_pos(p)
            objects[p] = {
                "pos": [round(float(v), 4) for v in pos],
                "yaw": round(self.object_yaw(p), 4),
                "held_by": self.holder_of(p),
                "on_slot": self.object_on_slot(p),
                "in_drawer": bool(self.object_in_drawer(p)),
                "reachable_by": layout.reaching_arms(pos),
            }
        drawer_xy = np.array(self.data.xpos[self.index.drawer_body])[:2]
        slots = {}
        for slot, sp in layout.SLOTS.items():
            want = layout.SLOT_OBJECT[slot]
            slots[slot] = {
                "pos": [round(float(v), 4) for v in sp],
                "occupied_by": want if objects[want]["on_slot"] == slot else None,
                "reachable_by": layout.reaching_arms(sp),
            }
        arms = {}
        for arm in layout.ARMS:
            arms[arm] = {
                "holding": self.attached[arm] if self.attached[arm] else None,
                "tcp": [round(float(v), 4) for v in self.tcp(arm)],
                "qpos": [round(float(v), 4) for v in self.arm_qpos(arm)],
                "gripper": round(self.gripper_cmd(arm), 4),
                "busy": False,
            }
        return {
            "t": round(float(self.data.time), 3),
            "seed": self.seed,
            "objects": objects,
            "drawer": {
                "open_frac": round(self.drawer_open_frac(), 3),
                "is_open": bool(self.drawer_is_open()),
                "reachable_by": layout.reaching_arms(drawer_xy),
            },
            "slots": slots,
            "arms": arms,
            "water": {k: int(v) for k, v in self.water_counts().items()},
        }

    # ------------------------------------------------------------------ scoring

    def task_score(self) -> dict:
        """Per-subgoal breakdown of the dinner-table task."""
        placed = {
            obj: bool(self.object_on_slot(obj) is not None)
            for obj in ("plate", "fork", "spoon", "mug")
        }
        water = self.water_counts()
        # The pour is scored as two separate things on purpose. Getting both arms to present a
        # cup and tip a carton over it is one capability; landing the contents inside a 37 mm
        # opening is another, and averaging them into a single flag would hide which one works.
        subgoals = {
            "drawer_opened": bool(self.drawer_is_open()),
            **{f"{k}_placed": bool(v) for k, v in placed.items()},
            "carton_emptied": bool(water["in_bottle"] <= 4),
            "water_in_cup": bool(water["in_mug"] >= 3),
        }
        done = sum(1 for v in subgoals.values() if v)
        return {
            "subgoals": subgoals,
            "completed": done,
            "total": len(subgoals),
            "fraction": round(done / len(subgoals), 3),
            "success": bool(all(subgoals.values())),
            "setting_success": bool(all(placed.values())),
            "water": water,
        }

    # ------------------------------------------------------------------ rendering

    def renderer(self, height: int, width: int) -> mujoco.Renderer:
        key = (height, width)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, height=height, width=width)
        return self._renderers[key]

    def render(
        self, cam: str = "overhead", height: int | None = None, width: int | None = None
    ) -> np.ndarray:
        h = height or self._render_size[0]
        w = width or self._render_size[1]
        r = self.renderer(h, w)
        r.update_scene(self.data, camera=self.index.cam[cam])
        return r.render()

    def close(self) -> None:
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()
