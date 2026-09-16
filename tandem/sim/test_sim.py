"""Tests for the properties the rest of the project relies on.

These are not smoke tests. Each one pins a claim made in the README: that the cell compiles to
the shape everything else indexes into, that a seed reproduces an episode exactly, and above all
that the hand-off is forced by the geometry rather than by the plan.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ..control.gripper import GripperCalibration
from ..control.ik import ArmIK, approach_frame
from . import layout
from .env import TandemEnv
from .randomize import RandomizationSpec


@pytest.fixture(scope="module")
def env() -> TandemEnv:
    return TandemEnv(RandomizationSpec(scale=1.0))


def test_cell_compiles_to_the_expected_shape(env: TandemEnv) -> None:
    assert env.model.nu == 12  # five arm joints plus a jaw, twice
    for arm in layout.ARMS:
        assert len(env.index.arm_act[arm]) == 5
    for cam in ("overhead", "front", "cinematic", "left_wrist", "right_wrist"):
        assert cam in env.index.cam
    assert len(env.index.water_bodies) == 10


def test_a_seed_reproduces_an_episode(env: TandemEnv) -> None:
    a = env.reset(7)
    b = env.reset(7)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    c = env.reset(8)
    assert json.dumps(c, sort_keys=True) != json.dumps(a, sort_keys=True)


def test_world_state_is_serialisable(env: TandemEnv) -> None:
    json.dumps(env.reset(3))  # numpy scalars here break the dashboard and the eval records


@pytest.mark.parametrize("seed", list(range(12)))
def test_the_handoff_is_forced_by_the_geometry(env: TandemEnv, seed: int) -> None:
    """No prop is ever reachable by the right arm; no slot is ever reachable by the left.

    This is the claim the whole bimanual story rests on. If the two envelopes ever overlapped
    the goal region, each arm could finish items on its own and the hand-off would be
    decoration. Cutlery at the back of a shut drawer is outside *both* envelopes, which is the
    point of the drawer — it has to be pulled forward first.
    """
    world = env.reset(seed)
    for name, info in world["objects"].items():
        assert "right" not in info["reachable_by"], f"{name}: {info['reachable_by']}"
    for name, slot in world["slots"].items():
        assert "left" not in slot["reachable_by"], f"{name}: {slot['reachable_by']}"
    for name in layout.TABLE_PROPS:
        assert world["objects"][name]["reachable_by"] == ["left"], name


def test_opening_the_drawer_brings_the_cutlery_into_reach(env: TandemEnv) -> None:
    from ..control.primitives import Executor

    env.reset(0)
    ex = Executor(env)
    assert ex.open_drawer("left").ok
    world = env.world_state()
    for name in layout.DRAWER_CONTENTS:
        assert world["objects"][name]["reachable_by"] == ["left"], name


def test_the_shared_zone_is_reachable_by_both() -> None:
    assert layout.reaching_arms(layout.HANDOFF_XY) == ["left", "right"]
    assert layout.reaching_arms(layout.POUR_STATION) == ["left", "right"]


def test_gripper_calibration_is_monotonic_and_measured(env: TandemEnv) -> None:
    cal = GripperCalibration(env.model, env.index)
    assert cal.min_gap < 0.01 < cal.max_gap
    gaps = [cal.gap_for_cmd(c) for c in np.linspace(cal.cmds[0], cal.cmds[-1], 40)]
    assert all(b >= a - 1e-9 for a, b in zip(gaps, gaps[1:]))
    # Every prop must fit inside the jaws with room to approach.
    for prop, width in layout.GRASP_WIDTH.items():
        assert width + cal.APPROACH_GAP < cal.max_gap, prop


def test_seat_depth_never_drives_the_fingertips_through_the_table(env: TandemEnv) -> None:
    cal = GripperCalibration(env.model, env.index)
    for grasp_z in (0.003, 0.010, 0.020, 0.050):
        seat = cal.seat_depth_for(grasp_z, 0.0)
        assert grasp_z - seat >= 0.0


def test_ik_reaches_a_table_point(env: TandemEnv) -> None:
    env.reset(0)
    ik = ArmIK(env.model, env.index, "left")
    target = np.array([-0.16, 0.12, 0.035])
    res = ik.solve(env.data, target, approach_frame(np.array([0.0, 0.0, -1.0])), restarts=6)
    assert res.ok and res.pos_err < 5e-3


def test_task_score_reports_every_subgoal(env: TandemEnv) -> None:
    env.reset(0)
    score = env.task_score()
    assert set(score["subgoals"]) == {
        "drawer_opened", "plate_placed", "fork_placed", "spoon_placed", "mug_placed",
        "carton_emptied", "water_in_cup",
    }
    assert score["total"] == len(score["subgoals"])
    json.dumps(score)


def test_cameras_render(env: TandemEnv) -> None:
    env.reset(0)
    img = env.render("overhead", 120, 160)
    assert img.shape == (120, 160, 3) and img.dtype == np.uint8
    assert img.std() > 5  # not a blank frame
