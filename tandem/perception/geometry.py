"""Camera projection for the fixed overhead and front views.

The two cameras are bolted to the scene and the randomizer never touches ``cam_pos``,
``cam_quat`` or ``cam_fovy`` -- it varies lights, materials, masses and sizes only. That
makes the world-to-pixel map a constant of the cell, which is worth exploiting twice:

* at training time, the true image-space location of every prop can be computed in closed
  form and used to supervise the network's attention maps directly, instead of hoping the
  position loss alone teaches it where to look;
* at review time, it gives a cheap way to overlay ground truth on a rendered frame and
  confirm that a label really does sit on the object it names.

Nothing here runs at inference. The exported IR takes pixels and returns metres; this
module only ever builds training targets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import CAMERAS


@dataclass(frozen=True)
class Camera:
    """A pinhole camera in MuJoCo's convention: -z forward, +y up, +x right."""

    position: np.ndarray  #: (3,) world position
    rotation: np.ndarray  #: (3, 3) columns are the camera axes in world coordinates
    fovy: float  #: vertical field of view, degrees

    def project(self, points: np.ndarray) -> np.ndarray:
        """World points ``(..., 3)`` -> normalized image coordinates ``(..., 2)``.

        The output is ``(u, v)`` in [-1, 1] with ``u`` running left to right and ``v``
        running *top to bottom*, which is the orientation of the network's attention grid.
        Points behind the camera come back clamped rather than mirrored.
        """
        local = (np.asarray(points, dtype=np.float64) - self.position) @ self.rotation
        depth = np.maximum(-local[..., 2], 1e-4)
        scale = 1.0 / np.tan(np.radians(self.fovy) / 2.0)
        u = scale * local[..., 0] / depth
        v = -scale * local[..., 1] / depth
        return np.stack([u, v], axis=-1).astype(np.float32)


def cameras(model, index) -> dict[str, Camera]:
    """Read the perception cameras out of a compiled model."""
    import mujoco

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    out: dict[str, Camera] = {}
    for name in CAMERAS:
        cam_id = index.cam[name]
        out[name] = Camera(
            position=np.array(data.cam_xpos[cam_id], dtype=np.float64),
            rotation=np.array(data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3),
            fovy=float(model.cam_fovy[cam_id]),
        )
    return out


def default_cameras() -> dict[str, Camera]:
    """Compile the cell once and read its camera poses."""
    from ..sim.scene import Index, build_model

    model = build_model()
    return cameras(model, Index(model))


def keypoint_targets(positions: np.ndarray, cams: dict[str, Camera]) -> np.ndarray:
    """``(N, P, 3)`` metres -> ``(N, len(CAMERAS), P, 2)`` normalized image coordinates."""
    return np.stack([cams[name].project(positions) for name in CAMERAS], axis=1)
