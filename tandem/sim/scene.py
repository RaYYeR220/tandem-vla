"""Assembles the dual SO-101 dinner-table cell.

MuJoCo has no namespacing, so the single-arm SO-101 model is attached twice through
`MjSpec` with a `left/` and `right/` prefix. Compiling is slow enough (19 meshes) that we
do it once and cache the spec; per-episode variation is applied by mutating `MjModel`
arrays instead of rebuilding.
"""

from __future__ import annotations

import functools
from pathlib import Path

import mujoco
import numpy as np

from . import layout

ASSETS = Path(__file__).resolve().parents[2] / "assets"
TABLE_XML = ASSETS / "scene_table.xml"
ARM_XML = ASSETS / "so101" / "so101.xml"

#: Yaw applied to each arm base so both face +y, across the table.
MOUNT_YAW = {"left": 0.0, "right": 0.0}


def build_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(TABLE_XML))
    for arm in layout.ARMS:
        child = mujoco.MjSpec.from_file(str(ARM_XML))
        site = spec.site(f"mount_{arm}")
        spec.attach(child, prefix=f"{arm}/", site=site)
    return spec


@functools.lru_cache(maxsize=1)
def _cached_xml() -> str:
    return build_spec().to_xml()


def build_model() -> mujoco.MjModel:
    """Compile the full cell. Assets are resolved relative to ``assets/so101``."""
    spec = build_spec()
    return spec.compile()


class Index:
    """Cached name -> id lookups for everything the stack touches."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.body = {}
        self.geom = {}
        self.site = {}
        self.joint = {}
        self.actuator = {}
        self.camera = {}
        for kind, obj, store in (
            ("body", mujoco.mjtObj.mjOBJ_BODY, self.body),
            ("geom", mujoco.mjtObj.mjOBJ_GEOM, self.geom),
            ("site", mujoco.mjtObj.mjOBJ_SITE, self.site),
            ("joint", mujoco.mjtObj.mjOBJ_JOINT, self.joint),
            ("actuator", mujoco.mjtObj.mjOBJ_ACTUATOR, self.actuator),
            ("camera", mujoco.mjtObj.mjOBJ_CAMERA, self.camera),
        ):
            n = {
                "body": model.nbody,
                "geom": model.ngeom,
                "site": model.nsite,
                "joint": model.njnt,
                "actuator": model.nu,
                "camera": model.ncam,
            }[kind]
            for i in range(n):
                name = mujoco.mj_id2name(model, obj, i)
                if name:
                    store[name] = i

        # Per-arm actuator / joint / dof slices.
        self.arm_act = {}
        self.arm_qpos = {}
        self.arm_dof = {}
        self.grip_act = {}
        self.grip_qpos = {}
        self.grip_dof = {}
        self.tcp_site = {}
        for arm in layout.ARMS:
            names = layout.NAMES[arm]
            self.arm_act[arm] = np.array([self.actuator[a] for a in names.joints])
            self.grip_act[arm] = self.actuator[names.gripper_joint]
            self.arm_qpos[arm] = np.array(
                [model.jnt_qposadr[self.joint[j]] for j in names.joints]
            )
            self.arm_dof[arm] = np.array(
                [model.jnt_dofadr[self.joint[j]] for j in names.joints]
            )
            self.grip_qpos[arm] = model.jnt_qposadr[self.joint[names.gripper_joint]]
            self.grip_dof[arm] = model.jnt_dofadr[self.joint[names.gripper_joint]]
            self.tcp_site[arm] = self.site[names.tcp_site]

        self.prop_body = {p: self.body[p] for p in layout.PROPS}
        self.prop_qpos = {
            p: model.jnt_qposadr[self.joint[f"{p}_free"]] for p in layout.PROPS
        }
        self.prop_dof = {
            p: model.jnt_dofadr[self.joint[f"{p}_free"]] for p in layout.PROPS
        }
        self.drawer_qpos = model.jnt_qposadr[self.joint["drawer_slide"]]
        self.drawer_dof = model.jnt_dofadr[self.joint["drawer_slide"]]
        self.drawer_body = self.body["drawer"]

        #: Water particles (replicated bodies) that live inside the carton.
        self.water_bodies = [
            self.body[n] for n in sorted(self.body) if n.startswith("water_")
        ]
        self.water_qpos = [
            model.jnt_qposadr[self.joint[n]]
            for n in sorted(self.joint)
            if n.startswith("water_free_")
        ]
        self.water_dof = [
            model.jnt_dofadr[self.joint[n]]
            for n in sorted(self.joint)
            if n.startswith("water_free_")
        ]

        #: Geoms whose colour / size / friction are perturbed by domain randomization.
        self.prop_geoms = {
            p: [
                i
                for n, i in self.geom.items()
                if n.startswith(p) and not n.startswith("water")
            ]
            for p in layout.PROPS
        }
        self.table_geom = self.geom["table_top"]
        self.backdrop_geom = self.geom["backdrop_geom"]
        self.light_ids = list(range(model.nlight))

        #: Geom ids belonging to each arm's jaws, for contact-based grasp detection.
        self.jaw_geoms = {
            a: [i for n, i in self.geom.items() if n.startswith(f"{a}/") and "jaw" in n]
            for a in layout.ARMS
        }
        self.geom_body = np.array(model.geom_bodyid)

        self.material = {}
        for i in range(model.nmat):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, i)
            if name:
                self.material[name] = i
        #: Material driving each prop's appearance, for visual randomization.
        self.prop_material = {
            "plate": "platemat",
            "mug": "mugmat",
            "bottle": "bottlemat",
            "spoon": "metalmat",
            "fork": "metalmat",
        }

        self.cam = {
            "overhead": self.camera["overhead"],
            "front": self.camera["front"],
            "cinematic": self.camera["cinematic"],
            "left_wrist": self.camera[layout.NAMES["left"].wrist_cam],
            "right_wrist": self.camera[layout.NAMES["right"].wrist_cam],
        }


def site_xpos(model, data, site_id) -> np.ndarray:
    return np.array(data.site_xpos[site_id])


if __name__ == "__main__":  # smoke test
    m = build_model()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    idx = Index(m)
    print(f"nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} ncam={m.ncam}")
    for arm in layout.ARMS:
        print(arm, "tcp @", np.round(d.site_xpos[idx.tcp_site[arm]], 4))
    print("cameras", sorted(idx.cam))
