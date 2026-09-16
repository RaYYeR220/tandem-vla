"""Scene sampling for the perception dataset.

Resetting the environment and rendering would only ever show props in the staging
rectangle with both arms parked at home -- the exact 5% of the workspace the estimator is
least useful in. This module re-poses the whole cell between renders: props are teleported
across the table, into the place setting and into mid-air transport poses, the drawer is
opened to a random fraction with its cutlery carried along, and both arms are driven to
random configurations so they genuinely occlude things. A few physics ticks then settle
whatever contact that produced, and the label is read *after* the settle, so it describes
the scene that was actually rendered.

Visibility is not guessed from geometry: a MuJoCo segmentation pass counts how many pixels
each prop actually paints across the two views.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from ..sim import layout
from ..sim.env import HOME_QPOS, TandemEnv
from .schema import CAMERAS, IMAGE_SIZE, PROPS, VISIBILITY_MIN_PIXELS

#: Table region props may be teleported into, in metres.
SCATTER_X = (-0.40, 0.40)
SCATTER_Y = (0.02, 0.28)

#: Spread of the random arm configurations, per joint, around the rest pose. Wide enough to
#: sweep the arm across the table and over the props, narrow enough that the elbow does not
#: routinely fold through the tabletop.
ARM_SIGMA = np.array([0.95, 0.55, 0.70, 0.75, 1.30])

#: Props outside this box are treated as ejected from the cell and the frame is dropped.
VALID_X = (-0.50, 0.50)
VALID_Y = (-0.14, 0.40)
VALID_Z = (-0.05, 0.34)

MODES = ("reset", "scatter", "setting", "transport", "drawer")


@dataclass
class Frame:
    """One rendered observation and its ground truth."""

    views: dict[str, np.ndarray]
    pos: np.ndarray  #: (5, 3) metres, table frame, in PROPS order
    yaw: np.ndarray  #: (5,) radians
    visible: np.ndarray  #: (5,) float 0/1
    pixels: np.ndarray  #: (5,) segmentation pixel counts, both views summed
    drawer_open: float  #: 0..1 fraction of DRAWER_OPEN_QPOS
    mode: str


def _quat_from_yaw(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


class SceneSampler:
    """Produces randomized, physically settled observations from one environment."""

    def __init__(self, env: TandemEnv, *, size: int = IMAGE_SIZE, settle_ticks: int = 6):
        self.env = env
        self.size = size
        self.settle_ticks = settle_ticks
        self.seg = mujoco.Renderer(env.model, height=size, width=size)
        self.seg.enable_segmentation_rendering()
        self._geom_body = np.asarray(env.model.geom_bodyid)

    def close(self) -> None:
        self.seg.close()

    # ------------------------------------------------------------------ scene edits

    def _set_prop(self, name: str, pos: np.ndarray, yaw: float) -> None:
        env = self.env
        adr = env.index.prop_qpos[name]
        dof = env.index.prop_dof[name]
        env.data.qpos[adr : adr + 3] = pos
        env.data.qpos[adr + 3 : adr + 7] = _quat_from_yaw(yaw)
        env.data.qvel[dof : dof + 6] = 0.0

    def _shift_water(self, delta: np.ndarray) -> None:
        """Carry the carton's contents with it so a teleported bottle does not rain."""
        for adr, dof in zip(self.env.index.water_qpos, self.env.index.water_dof):
            self.env.data.qpos[adr : adr + 3] += delta
            self.env.data.qvel[dof : dof + 6] = 0.0

    def _set_drawer(self, frac: float) -> None:
        env = self.env
        before = float(env.data.qpos[env.index.drawer_qpos])
        after = float(np.clip(frac, 0.0, 1.0)) * layout.DRAWER_OPEN_QPOS
        delta = np.array([0.0, -(after - before), 0.0])
        inside = [p for p in layout.DRAWER_CONTENTS if env.object_in_drawer(p)]
        env.data.qpos[env.index.drawer_qpos] = after
        env.data.qvel[env.index.drawer_dof] = 0.0
        for name in inside:
            adr = env.index.prop_qpos[name]
            env.data.qpos[adr : adr + 3] += delta

    def _set_arms(self, rng: np.random.Generator) -> None:
        env = self.env
        for arm in layout.ARMS:
            qpos = HOME_QPOS + rng.normal(0.0, 1.0, 5) * ARM_SIGMA
            for i, joint in enumerate(layout.NAMES[arm].joints):
                lo, hi = env.model.jnt_range[env.index.joint[joint]]
                qpos[i] = float(np.clip(qpos[i], lo + 0.03, hi - 0.03))
            grip = float(rng.uniform(layout.GRIPPER_CLOSED, layout.GRIPPER_OPEN))
            env.data.qpos[env.index.arm_qpos[arm]] = qpos
            env.data.qpos[env.index.grip_qpos[arm]] = grip
            env.data.qvel[env.index.arm_dof[arm]] = 0.0
            env.set_arm_target(arm, qpos)
            env.set_gripper(arm, grip)

    # ------------------------------------------------------------------ pose sampling

    def _scatter_xy(self, rng: np.random.Generator) -> np.ndarray:
        return np.array([rng.uniform(*SCATTER_X), rng.uniform(*SCATTER_Y)])

    def _pose_props(self, mode: str, rng: np.random.Generator) -> None:
        env = self.env
        if mode == "reset":
            return

        bottle_before = env.object_pos("bottle").copy()
        if mode == "setting":
            for name in ("plate", "mug"):
                slot = layout.SLOTS[layout.OBJECT_SLOT[name]]
                xy = slot[:2] + rng.normal(0.0, 0.020, 2)
                self._set_prop(name, np.array([xy[0], xy[1], layout.REST_Z[name]]),
                               rng.uniform(-np.pi, np.pi))
            xy = self._scatter_xy(rng)
            self._set_prop("bottle", np.array([xy[0], xy[1], layout.REST_Z["bottle"]]),
                           rng.uniform(-np.pi, np.pi))
            for name in layout.DRAWER_CONTENTS:
                if rng.random() < 0.6:
                    slot = layout.SLOTS[layout.OBJECT_SLOT[name]]
                    xy = slot[:2] + rng.normal(0.0, 0.020, 2)
                    self._set_prop(name, np.array([xy[0], xy[1], layout.REST_Z[name]]),
                                   rng.uniform(-0.3, 0.3))
        elif mode in ("scatter", "drawer"):
            for name in layout.TABLE_PROPS:
                xy = self._scatter_xy(rng)
                self._set_prop(name, np.array([xy[0], xy[1], layout.REST_Z[name]]),
                               rng.uniform(-np.pi, np.pi))
            if mode == "drawer":
                for name in layout.DRAWER_CONTENTS:
                    if rng.random() < 0.45:
                        xy = self._scatter_xy(rng)
                        self._set_prop(name, np.array([xy[0], xy[1], layout.REST_Z[name]]),
                                       rng.uniform(-np.pi, np.pi))
        elif mode == "transport":
            for name in layout.TABLE_PROPS:
                xy = self._scatter_xy(rng)
                self._set_prop(name, np.array([xy[0], xy[1], layout.REST_Z[name]]),
                               rng.uniform(-np.pi, np.pi))
            lifted = PROPS[rng.integers(len(PROPS))]
            arm = layout.ARMS[rng.integers(2)]
            tcp = self.env.tcp(arm)
            self._set_prop(lifted, tcp + rng.normal(0.0, 0.015, 3), rng.uniform(-np.pi, np.pi))

        delta = env.object_pos("bottle") - bottle_before
        if float(np.linalg.norm(delta)) > 1e-6:
            self._shift_water(delta)

    # ------------------------------------------------------------------ labels

    def _segmentation_pixels(self) -> np.ndarray:
        counts = np.zeros(len(PROPS), dtype=np.int64)
        for cam in CAMERAS:
            self.seg.update_scene(self.env.data, camera=self.env.index.cam[cam])
            seg = self.seg.render()
            ids, kinds = seg[:, :, 0], seg[:, :, 1]
            geoms = np.where(kinds == int(mujoco.mjtObj.mjOBJ_GEOM), ids, -1)
            valid = geoms >= 0
            bodies = np.where(valid, self._geom_body[np.clip(geoms, 0, None)], -1)
            for i, name in enumerate(PROPS):
                counts[i] += int((bodies == self.env.index.prop_body[name]).sum())
        return counts

    # ------------------------------------------------------------------ sampling

    def sample(self, rng: np.random.Generator, mode: str | None = None) -> Frame | None:
        """Re-pose the cell, settle it, render it and read the truth. ``None`` if a prop
        was ejected from the workspace and the frame is not worth labelling."""
        env = self.env
        mode = mode or MODES[rng.integers(len(MODES))]
        self._set_arms(rng)
        self._pose_props(mode, rng)
        self._set_drawer(rng.random() if mode == "drawer" else rng.choice([0.0, rng.random()]))
        mujoco.mj_forward(env.model, env.data)
        env.step(self.settle_ticks)

        pos = np.stack([env.object_pos(p) for p in PROPS]).astype(np.float32)
        if not (
            np.all((pos[:, 0] > VALID_X[0]) & (pos[:, 0] < VALID_X[1]))
            and np.all((pos[:, 1] > VALID_Y[0]) & (pos[:, 1] < VALID_Y[1]))
            and np.all((pos[:, 2] > VALID_Z[0]) & (pos[:, 2] < VALID_Z[1]))
        ):
            return None

        pixels = self._segmentation_pixels()
        views = {cam: env.render(cam, self.size, self.size).copy() for cam in CAMERAS}
        return Frame(
            views=views,
            pos=pos,
            yaw=np.array([env.object_yaw(p) for p in PROPS], dtype=np.float32),
            visible=(pixels >= VISIBILITY_MIN_PIXELS).astype(np.float32),
            pixels=pixels,
            drawer_open=float(env.drawer_open_frac()),
            mode=mode,
        )
