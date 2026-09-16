"""Frozen encoding contract for the scene-state network.

Everything that has to agree between the collector, the trainer, the exporter and the
runtime estimator lives here: which cameras are read, at what resolution, the order of the
props in the output vector, and the affine map between metres in the table frame and the
normalized targets the network actually regresses.

Normalization is a fixed, hand-set box around the reachable workspace -- not dataset
statistics -- so a model trained on one collection run stays compatible with a model
trained on another, and so the exported IR has no hidden dependency on a scaler file.

    normalized = (metres - POS_CENTER) / POS_SCALE          # nominally in [-1, 1]
    metres     = normalized * POS_SCALE + POS_CENTER

The box spans x in [-0.45, 0.45], y in [-0.10, 0.36], z in [-0.02, 0.30] metres, which
covers the cabinet on the far left, the place setting on the right, the shoulder line at
the near edge and the full lift height of both arms.
"""

from __future__ import annotations

import numpy as np

#: Cameras the network sees, in the order they are stacked.
CAMERAS: tuple[str, str] = ("overhead", "front")

#: Square input resolution per camera.
IMAGE_SIZE: int = 128

#: Props the network localizes, in output order.
PROPS: tuple[str, ...] = ("plate", "mug", "bottle", "spoon", "fork")
N_PROPS = len(PROPS)

#: Centre and half-extent of the normalization box, in metres, table frame.
POS_CENTER = np.array([0.00, 0.13, 0.14], dtype=np.float32)
POS_SCALE = np.array([0.45, 0.23, 0.16], dtype=np.float32)

#: A prop counts as visible when it paints at least this many pixels across both views
#: in a MuJoCo segmentation pass at IMAGE_SIZE. Roughly 0.1% of one frame.
VISIBILITY_MIN_PIXELS: int = 16

#: Output vector layout, so the trainer and the estimator index it the same way.
POS_SLICE = slice(0, N_PROPS * 3)  # 15 normalized xyz
YAW_SLICE = slice(N_PROPS * 3, N_PROPS * 5)  # 10 (cos, sin) pairs
VIS_SLICE = slice(N_PROPS * 5, N_PROPS * 6)  # 5 visibility logits
DRAWER_INDEX = N_PROPS * 6  # 1 drawer-open logit
OUTPUT_DIM = N_PROPS * 6 + 1


def normalize_pos(pos_m: np.ndarray) -> np.ndarray:
    """Metres in the table frame -> normalized regression target."""
    return (np.asarray(pos_m, dtype=np.float32) - POS_CENTER) / POS_SCALE


def denormalize_pos(pos_n: np.ndarray) -> np.ndarray:
    """Normalized network output -> metres in the table frame."""
    return np.asarray(pos_n, dtype=np.float32) * POS_SCALE + POS_CENTER


def stack_views(views: dict[str, np.ndarray]) -> np.ndarray:
    """``{cam: HxWx3 uint8}`` -> ``(2, 3, H, W)`` float32 in [0, 1].

    The two cameras are kept as separate images rather than concatenated into six
    channels: they look at the cell from completely unrelated viewpoints, so a filter that
    straddles both at the same pixel coordinate would be mixing unrelated geometry. The
    trunk is shared across the views instead, and the pooled features are concatenated.
    """
    out = np.empty((len(CAMERAS), 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    for i, cam in enumerate(CAMERAS):
        img = views[cam]
        out[i] = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0
    return out
