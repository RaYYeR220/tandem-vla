"""Learned scene-state estimation: two cameras in, a `world_state()`-shaped dict out.

    from tandem.perception import PerceptionEstimator, render_views, arm_proprioception

    est = PerceptionEstimator("models/perception_int8.xml", device="CPU")
    world = est.estimate(render_views(env), arm_proprioception(env), t=env.data.time)

`world` is schema-identical to `TandemEnv.world_state()`, so the planner, the gate and the
executor run against it unchanged -- with the one documented exception of the water count,
which these two cameras cannot see and which the estimator flags rather than invents.
"""

from __future__ import annotations

from .estimator import (
    PerceptionEstimator,
    arm_proprioception,
    drawer_position,
    in_drawer_for,
    on_slot_for,
    render_views,
)
from .model import PerceptionNet, split_outputs
from .schema import CAMERAS, IMAGE_SIZE, OUTPUT_DIM, PROPS, denormalize_pos, normalize_pos

__all__ = [
    "CAMERAS",
    "IMAGE_SIZE",
    "OUTPUT_DIM",
    "PROPS",
    "PerceptionEstimator",
    "PerceptionNet",
    "arm_proprioception",
    "denormalize_pos",
    "drawer_position",
    "in_drawer_for",
    "normalize_pos",
    "on_slot_for",
    "render_views",
    "split_outputs",
]
