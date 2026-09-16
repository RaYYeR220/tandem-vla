"""Learned visuomotor control: an action-chunking transformer distilled from the oracle.

    from tandem.policy import PolicyExecutor, PolicyRunner

    runner = PolicyRunner("models/policy_int8.xml", device="CPU")
    executor = PolicyExecutor(env, runner)     # drop-in for control.primitives.Executor

`PolicyExecutor` subclasses the scripted executor and overrides `pick` and `place` only;
every other skill, and every post-condition check, is the scripted one, so the episode
runner and the evaluator cannot tell the two apart except by the success rate.
"""

from __future__ import annotations

from .model import ActionChunkPolicy
from .rollout import PolicyExecutor, PolicyRunner
from .schema import (
    ACTION_DIM,
    CHUNK,
    COND_DIM,
    EXECUTE,
    IMAGE_SIZE,
    PROPRIO_DIM,
    SKILLS,
    condition_vector,
    denormalize_joints,
    joint_limits,
    normalize_joints,
)

__all__ = [
    "ACTION_DIM",
    "CHUNK",
    "COND_DIM",
    "EXECUTE",
    "IMAGE_SIZE",
    "PROPRIO_DIM",
    "SKILLS",
    "ActionChunkPolicy",
    "PolicyExecutor",
    "PolicyRunner",
    "condition_vector",
    "denormalize_joints",
    "joint_limits",
    "normalize_joints",
]
