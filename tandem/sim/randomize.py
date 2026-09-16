"""Seeded domain randomization.

One integer seed fixes every perturbation in an episode, so a reported success rate is
reproducible by anyone who clones the repo. Each knob maps to a `MjModel` array, which is
mutated in place — the model is never recompiled mid-run.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import mujoco
import numpy as np

from . import layout


@dataclass
class RandomizationSpec:
    """Per-episode perturbation ranges. ``scale`` = 0 reproduces the nominal scene."""

    scale: float = 1.0

    pos_jitter: float = 0.055  #: metres, prop start position
    yaw_jitter: float = np.pi  #: radians, prop start yaw
    mass_range: tuple[float, float] = (0.55, 1.75)
    friction_range: tuple[float, float] = (0.55, 1.60)
    size_range: tuple[float, float] = (0.93, 1.05)
    light_pos_jitter: float = 0.35
    light_intensity_range: tuple[float, float] = (0.55, 1.35)
    hue_jitter: float = 0.30
    table_shade_range: tuple[float, float] = (0.55, 1.15)
    backdrop_hue_jitter: float = 0.45
    drawer_friction_range: tuple[float, float] = (0.6, 1.9)

    randomize_props: bool = True
    randomize_dynamics: bool = True
    randomize_visual: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


NOMINAL = RandomizationSpec(scale=0.0)


@dataclass
class EpisodeVariation:
    """Everything the randomizer decided, recorded so an episode can be audited."""

    seed: int
    prop_pos: dict = field(default_factory=dict)
    prop_yaw: dict = field(default_factory=dict)
    mass_mult: dict = field(default_factory=dict)
    friction_mult: dict = field(default_factory=dict)
    size_mult: dict = field(default_factory=dict)
    light_scale: float = 1.0
    table_shade: float = 1.0

    def as_dict(self) -> dict:
        d = asdict(self)
        d["prop_pos"] = {k: [round(float(x), 4) for x in v] for k, v in self.prop_pos.items()}
        d["prop_yaw"] = {k: round(float(v), 4) for k, v in self.prop_yaw.items()}
        for key in ("mass_mult", "friction_mult", "size_mult"):
            d[key] = {k: round(float(v), 3) for k, v in d[key].items()}
        return d


def _hue_shift(rgba: np.ndarray, rng: np.random.Generator, amount: float) -> np.ndarray:
    out = np.array(rgba, dtype=float)
    out[:3] = np.clip(out[:3] + rng.uniform(-amount, amount, 3), 0.05, 1.0)
    return out


def _valid_staging(p: np.ndarray) -> bool:
    """A start position must be the left arm's alone, clear of the cabinet, and comfortable."""
    lo, hi = layout.STAGING_REACH
    if not (lo <= layout.reach("left", p) <= hi):
        return False
    if layout.reach("right", p) < layout.STAGING_RIGHT_CLEAR:
        return False
    if p[0] < layout.CABINET_KEEPOUT_X and p[1] > layout.CABINET_KEEPOUT_Y:
        return False
    return True


def _poisson_positions(
    rng: np.random.Generator, names: list[str], radii: list[float], tries: int = 600
) -> list[np.ndarray]:
    """Rejection-sample well-separated xy positions inside the staging region."""
    pts: list[np.ndarray] = []
    for i, name in enumerate(names):
        for _ in range(tries):
            p = np.array(
                [rng.uniform(*layout.STAGING_X), rng.uniform(*layout.STAGING_Y)]
            )
            if not _valid_staging(p):
                continue
            if all(
                np.linalg.norm(p - q)
                > (radii[i] + radii[j] + layout.PROP_CLEARANCE)
                for j, q in enumerate(pts)
            ):
                pts.append(p)
                break
        else:  # fall back to the nominal spot rather than looping forever
            pts.append(np.array(layout.NOMINAL_PROP_XY[name]))
    return pts


class Randomizer:
    """Applies a `RandomizationSpec` to a compiled model + data for one episode."""

    def __init__(self, model: mujoco.MjModel, index, spec: RandomizationSpec | None = None):
        self.model = model
        self.index = index
        self.spec = spec or RandomizationSpec()
        self.base_mass = np.array(model.body_mass)
        self.base_inertia = np.array(model.body_inertia)
        self.base_friction = np.array(model.geom_friction)
        self.base_size = np.array(model.geom_size)
        self.base_rgba = np.array(model.geom_rgba)
        self.base_light_pos = np.array(model.light_pos)
        self.base_light_diffuse = np.array(model.light_diffuse)
        self.base_dof_friction = np.array(model.dof_frictionloss)
        self.base_mat_rgba = np.array(model.mat_rgba)

    def restore(self) -> None:
        m = self.model
        m.body_mass[:] = self.base_mass
        m.body_inertia[:] = self.base_inertia
        m.geom_friction[:] = self.base_friction
        m.geom_size[:] = self.base_size
        m.geom_rgba[:] = self.base_rgba
        m.light_pos[:] = self.base_light_pos
        m.light_diffuse[:] = self.base_light_diffuse
        m.dof_frictionloss[:] = self.base_dof_friction
        m.mat_rgba[:] = self.base_mat_rgba

    def apply(self, seed: int) -> EpisodeVariation:
        self.restore()
        spec = self.spec
        s = float(np.clip(spec.scale, 0.0, 1.0))
        rng = np.random.default_rng(seed)
        m, idx = self.model, self.index
        var = EpisodeVariation(seed=seed)

        table_props = list(layout.TABLE_PROPS)
        radii = [layout.PROP_RADIUS[p] for p in table_props]
        if s > 0 and spec.randomize_props:
            pts = _poisson_positions(rng, table_props, radii)
        else:
            pts = [np.array(layout.NOMINAL_PROP_XY[p]) for p in table_props]
        for p, xy in zip(table_props, pts):
            var.prop_pos[p] = np.array([xy[0], xy[1], layout.REST_Z[p]])
            var.prop_yaw[p] = rng.uniform(-spec.yaw_jitter, spec.yaw_jitter) * s

        # Cutlery keeps its slot in the drawer but jitters a little inside the tray.
        for i, p in enumerate(layout.DRAWER_CONTENTS):
            base = np.array([-0.355, 0.160 + 0.035 * i, 0.0192])
            base[0] += rng.uniform(-0.012, 0.012) * s
            base[1] += rng.uniform(-0.008, 0.008) * s
            var.prop_pos[p] = base
            var.prop_yaw[p] = rng.uniform(-0.18, 0.18) * s

        if spec.randomize_dynamics and s > 0:
            for p in layout.PROPS:
                mm = rng.uniform(*spec.mass_range) ** s
                fm = rng.uniform(*spec.friction_range) ** s
                sm = rng.uniform(*spec.size_range) ** s
                var.mass_mult[p] = mm
                var.friction_mult[p] = fm
                var.size_mult[p] = sm
                bid = idx.prop_body[p]
                m.body_mass[bid] = self.base_mass[bid] * mm
                m.body_inertia[bid] = self.base_inertia[bid] * mm
                for gid in idx.prop_geoms[p]:
                    m.geom_friction[gid, 0] = self.base_friction[gid, 0] * fm
                    if p in ("mug", "bottle"):
                        # Only scale the wall thickness axis; the grasp width must stay
                        # inside the jaw envelope for the task to remain solvable.
                        continue
                    m.geom_size[gid] = self.base_size[gid] * sm
            dj = m.jnt_dofadr[idx.joint["drawer_slide"]]
            m.dof_frictionloss[dj] = self.base_dof_friction[dj] * (
                rng.uniform(*spec.drawer_friction_range) ** s
            )

        if spec.randomize_visual and s > 0:
            var.light_scale = float(rng.uniform(*spec.light_intensity_range) ** s)
            for li in range(m.nlight):
                m.light_pos[li] = self.base_light_pos[li] + rng.uniform(
                    -spec.light_pos_jitter, spec.light_pos_jitter, 3
                ) * s * np.array([1.0, 1.0, 0.35])
                m.light_diffuse[li] = np.clip(
                    self.base_light_diffuse[li] * var.light_scale, 0.04, 1.0
                )
            var.table_shade = float(rng.uniform(*spec.table_shade_range) ** s)
            for name in ("tablemat", "woodmat"):
                mid = idx.material.get(name)
                if mid is not None:
                    c = np.array(self.base_mat_rgba[mid], dtype=float)
                    c[:3] = np.clip(c[:3] * var.table_shade, 0.05, 1.0)
                    m.mat_rgba[mid] = c
            m.geom_rgba[idx.backdrop_geom] = _hue_shift(
                self.base_rgba[idx.backdrop_geom], rng, spec.backdrop_hue_jitter * s
            )
            for name in set(idx.prop_material.values()):
                mid = idx.material.get(name)
                if mid is None:
                    continue
                c = np.array(self.base_mat_rgba[mid], dtype=float)
                c[:3] = np.clip(
                    c[:3] + rng.uniform(-spec.hue_jitter, spec.hue_jitter, 3) * s, 0.06, 1.0
                )
                m.mat_rgba[mid] = c

        mujoco.mj_setConst(m, mujoco.MjData(m))
        return var
