"""The skill layer: the only motions the planner is allowed to ask for.

Each skill is a closed-loop routine over the simulator, reports whether it actually achieved
its post-condition, and leaves the cell in a state the next skill can start from. Failures
are returned, never raised — the executor needs to decide whether to retry, re-plan or stop.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

from ..sim import layout
from ..sim.env import CONTROL_DT, HOME_QPOS, TandemEnv
from .gripper import GripperCalibration
from .ik import ArmIK
from .poses import (
    APPROACH_CLEARANCE,
    LIFT_HEIGHT,
    Pose,
    grasp_geometry,
    grasp_transform,
    pour_pose_candidates,
    radial_dir,
    solve_for_object_pose,
    solve_pose,
)


@dataclass
class SkillResult:
    skill: str
    arm: str
    ok: bool
    code: str = "OK"
    detail: str = ""
    seconds: float = 0.0
    extras: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "skill": self.skill,
            "arm": self.arm,
            "ok": self.ok,
            "code": self.code,
            "detail": self.detail,
            "seconds": round(self.seconds, 2),
            **({"extras": self.extras} if self.extras else {}),
        }


#: Opt-in: attempt a true in-air exchange for elongated props instead of the shared-zone relay.
DIRECT_HANDOFF = os.environ.get("TANDEM_HANDOFF", "relay").lower() == "direct"

#: Joint error, in radians, above which we call a commanded pose unreached.
BLOCKED_JOINT_ERR = 0.10


def _smoothstep(u: float) -> float:
    return u * u * (3.0 - 2.0 * u)


class Executor:
    """Runs skills against a `TandemEnv`."""

    def __init__(self, env: TandemEnv, *, on_tick=None, speed: float = 1.0):
        self.env = env
        self.ik = {a: ArmIK(env.model, env.index, a) for a in layout.ARMS}
        self.grip = GripperCalibration(env.model, env.index)
        self.on_tick = on_tick
        self.speed = speed
        self.aborted = False
        self.tick_count = 0
        #: Pose of the held object in the gripper frame, captured when a grasp closes.
        self.grasp_transform: dict[str, np.ndarray | None] = {a: None for a in layout.ARMS}
        #: Tool offset measured at grasp time, used for every pose while the object is held.
        self.held_tool: dict[str, np.ndarray | None] = {a: None for a in layout.ARMS}

    def retract(self, arm: str, seconds: float = 0.9) -> None:
        """Park the arm at the rest pose.

        Every skill travels laterally from here. The fingers are long and the reachable
        envelope is only ~80 mm tall, so a direct low move between two points on the table
        sweeps a corridor straight through whatever else is standing there.
        """
        self.move_joint(arm, HOME_QPOS, seconds)

    def tool(self, obj: str, *, approach: bool = False, arm: str | None = None) -> np.ndarray:
        """Offset from the gripper site to the point where ``obj`` will be pinched.

        With ``approach`` the offset is measured against the *open* jaw gap, so the descent
        brings the object down the middle of the jaws rather than scraping one of them.
        """
        if obj == "drawer":
            grasp_z = float(
                self.env.data.site_xpos[self.env.index.site["drawer_handle_site"]][2]
            )
            support = 0.0
        else:
            grasp_z = float(self.env.object_pos(obj)[2]) + layout.GRASP_OFFSET_Z[obj]
            support = layout.SUPPORT_Z.get(obj, 0.0) if self.env.object_in_drawer(obj) else 0.0
        seat = self.grip.seat_depth_for(grasp_z, support)
        width = layout.GRASP_WIDTH[obj]
        if approach:
            if arm is not None:
                # Centre on the gap the jaws are actually holding right now.
                width = self.grip.gap_for_cmd(
                    float(self.env.data.qpos[self.env.index.grip_qpos[arm]])
                )
            else:
                width += self.grip.APPROACH_GAP
        return self.grip.grasp_center_local(width, seat)

    # ------------------------------------------------------------------ motion

    def _tick(self, n: int = 1) -> None:
        self.env.step(n)
        self.tick_count += n
        if self.on_tick is not None:
            self.on_tick(self)

    def hold(self, seconds: float) -> None:
        for _ in range(max(1, int(seconds / CONTROL_DT / self.speed))):
            if self.aborted:
                return
            self._tick()

    def move_joint(
        self, arm: str, goal: np.ndarray, seconds: float = 1.1, *, settle: float = 0.5
    ) -> float:
        """Ramp the joint targets and then wait for the servos to actually get there.

        Returns the residual joint error. A non-zero residual means something is in the way --
        the caller can treat that as a failed motion rather than assuming the pose was reached.
        """
        start = self.env.arm_target(arm)
        # Stretch the move for large reconfigurations so a held object is not flung.
        travel = float(np.linalg.norm(np.asarray(goal) - start))
        seconds = max(seconds, 0.55 * travel)
        steps = max(2, int(seconds / CONTROL_DT / self.speed))
        for i in range(1, steps + 1):
            if self.aborted:
                return float("nan")
            self.env.set_arm_target(arm, start + (goal - start) * _smoothstep(i / steps))
            self._tick()
        err = float(np.max(np.abs(self.env.arm_qpos(arm) - goal)))
        waited = 0.0
        while err > 0.02 and waited < settle and not self.aborted:
            self._tick()
            waited += CONTROL_DT
            err = float(np.max(np.abs(self.env.arm_qpos(arm) - goal)))
        return err

    def close_on_object(self, arm: str, obj: str, *, force_err: float | None = None) -> float:
        """Close the jaws until the servo is actually pressing on the object.

        Commanding a fixed opening assumes the object is exactly the size we think it is.
        Under domain randomization it is not: an 8% smaller plate slips straight through a
        preset gap. Contact alone is not enough either -- the jaw geometry registers a touch
        several millimetres before the pinch loads up. So we keep closing past first contact
        until the servo's own tracking error says it is pushing, which is a direct read of
        grip force (kp x error / lever arm) and needs no knowledge of the object at all.
        """
        env = self.env
        target_err = layout.GRIPPER_FORCE_ERR if force_err is None else force_err
        qadr = env.index.grip_qpos[arm]
        vadr = env.index.grip_dof[arm]
        floor = self.grip.cmd_for_gap(self.grip.min_gap)

        def settle(limit: int = 30) -> float:
            n = 0
            while n < limit and abs(float(env.data.qvel[vadr])) > 0.05 and not self.aborted:
                self._tick()
                n += 1
            return float(env.data.qpos[qadr])

        # Walk the command down in fixed increments, letting the jaw catch up each time, until
        # the jaw stops moving between rounds. Where it stops is the object; the command sits
        # `target_err` beyond it, and that error is the pinch force.
        prev = float(env.data.qpos[qadr])
        cmd = env.gripper_cmd(arm)
        for _ in range(26):
            if self.aborted:
                break
            cmd = max(floor, float(env.data.qpos[qadr]) - target_err)
            env.set_gripper(arm, cmd)
            self._tick(2)
            actual = settle()
            stalled = abs(actual - prev) < 0.012
            prev = actual
            if stalled and (env.jaw_contacts(arm, obj) >= 1 or cmd <= floor):
                break

        self.hold(0.2)
        held_gap = self.grip.gap_for_cmd(float(env.data.qpos[qadr]))
        self.held_tool[arm] = self.grip.grasp_center_local(
            held_gap, self.grip.seat_depth_for(float(env.object_pos(obj)[2]), 0.0)
        )
        return held_gap

    def move_gripper(
        self, arm: str, cmd: float, seconds: float = 0.45, *, settle: float = 0.9
    ) -> float:
        """Ramp the jaw command and wait for the jaw to reach it.

        The jaw servo is deliberately soft, which makes it slow. Returning before it has
        arrived means the next pose is computed against an opening the gripper does not have,
        and the grasp lands off-centre by half the difference.
        """
        env = self.env
        qadr = env.index.grip_qpos[arm]
        start = env.gripper_cmd(arm)
        steps = max(2, int(seconds / CONTROL_DT / self.speed))
        for i in range(1, steps + 1):
            if self.aborted:
                return float("nan")
            env.set_gripper(arm, start + (cmd - start) * _smoothstep(i / steps))
            self._tick()
        waited = 0.0
        while (
            abs(float(env.data.qpos[qadr]) - cmd) > 0.04
            and waited < settle
            and not self.aborted
        ):
            self._tick()
            waited += CONTROL_DT
        return float(env.data.qpos[qadr])

    def go(
        self,
        arm: str,
        pos,
        jaw_yaw: float,
        *,
        seconds: float = 1.1,
        tilts=None,
        q_init: np.ndarray | None = None,
        tool: str | None = None,
        approach: bool = False,
    ) -> Pose:
        kw = {} if tilts is None else {"tilts": tilts}
        if tool is not None:
            if (
                not approach
                and self.env.attached.get(arm) == tool
                and self.held_tool[arm] is not None
            ):
                kw["tool_local"] = self.held_tool[arm]
            else:
                kw["tool_local"] = self.tool(tool, approach=approach, arm=arm)
        pose = solve_pose(
            self.ik[arm],
            self.env.data,
            pos,
            jaw_yaw,
            arm=arm,
            q_init=q_init if q_init is not None else self.env.arm_target(arm),
            **kw,
        )
        if pose.ok:
            residual = self.move_joint(arm, pose.qpos, seconds)
            if residual == residual and residual > BLOCKED_JOINT_ERR:
                # The IK solution was fine; the arm could not get there. Something is in the
                # way, and pretending otherwise just moves the failure one step later.
                pose.blocked = float(residual)
        return pose

    def follow(self, arm: str, waypoints, jaw_yaw: float, *, seconds_each: float = 0.8,
               tool: str | None = None):
        """Move through a Cartesian path, seeding each solve with the previous solution."""
        q = self.env.arm_target(arm)
        last: Pose | None = None
        tl = None if tool is None else self.tool(tool, arm=arm)
        for wp in waypoints:
            pose = solve_pose(
                self.ik[arm], self.env.data, wp, jaw_yaw, arm=arm, q_init=q, restarts=4,
                tool_local=tl,
            )
            if not pose.ok:
                return pose
            self.move_joint(arm, pose.qpos, seconds_each)
            q = pose.qpos
            last = pose
        return last

    # ------------------------------------------------------------------ skills

    def home(self, arm: str) -> SkillResult:
        t0 = time.perf_counter()
        self.move_joint(arm, HOME_QPOS, 1.1)
        self.env.attached[arm] = None if not self.env.attached[arm] else self.env.attached[arm]
        return SkillResult("home", arm, True, seconds=time.perf_counter() - t0)

    def open_drawer(self, arm: str, *, pull: float = layout.DRAWER_OPEN_QPOS) -> SkillResult:
        t0 = time.perf_counter()
        env = self.env
        handle = np.array(env.data.site_xpos[env.index.site["drawer_handle_site"]])
        jaw_yaw = np.pi / 2  # jaws close across the bar, which runs along x
        self.retract(arm, 0.7)

        self.move_gripper(arm, self.grip.open_cmd("drawer"), 0.9, settle=1.6)
        pre = handle + np.array([0.0, 0.0, APPROACH_CLEARANCE])
        pose = self.go(arm, pre, jaw_yaw, seconds=1.2, tool="drawer", approach=True)
        if not pose.ok:
            return SkillResult(
                "open_drawer", arm, False, "OUT_OF_REACH",
                f"no approach to the handle (err {pose.ik.pos_err:.3f} m)",
                time.perf_counter() - t0,
            )
        grab = handle + np.array([0.0, 0.0, 0.002])
        pose = self.go(arm, grab, jaw_yaw, seconds=0.9, q_init=pose.qpos, tool="drawer",
                       approach=True)
        if not pose.ok:
            return SkillResult(
                "open_drawer", arm, False, "OUT_OF_REACH", "cannot descend onto the handle",
                time.perf_counter() - t0,
            )
        self.move_gripper(arm, self.grip.grasp_cmd("drawer"), 0.4)
        self.hold(0.14)

        start = np.array(env.data.site_xpos[env.index.site["drawer_handle_site"]])
        waypoints = [
            start + np.array([0.0, -pull * f, 0.0]) for f in (0.3, 0.6, 0.85, 1.0)
        ]
        self.follow(arm, waypoints, jaw_yaw, seconds_each=0.55, tool="drawer")
        self.move_gripper(arm, self.grip.open_cmd("drawer", clearance=0.045), 0.35)
        self.hold(0.1)
        tcp = env.tcp(arm)
        self.go(arm, tcp + np.array([0.0, -0.01, 0.075]), jaw_yaw, seconds=0.8)
        self.move_joint(arm, HOME_QPOS, 0.9)

        opened = env.drawer_is_open()
        return SkillResult(
            "open_drawer",
            arm,
            opened,
            "OK" if opened else "DRAWER_STUCK",
            f"drawer at {env.drawer_open_frac():.0%} open",
            time.perf_counter() - t0,
            {"open_frac": round(env.drawer_open_frac(), 3)},
        )

    def close_drawer(self, arm: str) -> SkillResult:
        t0 = time.perf_counter()
        env = self.env
        handle = np.array(env.data.site_xpos[env.index.site["drawer_handle_site"]])
        jaw_yaw = np.pi / 2
        push_from = handle + np.array([0.0, -0.05, 0.004])
        pose = self.go(arm, push_from + np.array([0, 0, APPROACH_CLEARANCE]), jaw_yaw,
                       seconds=1.0, tool="drawer")
        if not pose.ok:
            return SkillResult("close_drawer", arm, False, "OUT_OF_REACH", "", time.perf_counter() - t0)
        self.move_gripper(arm, self.grip.grasp_cmd("drawer"), 0.3)
        self.go(arm, push_from, jaw_yaw, seconds=0.7, q_init=pose.qpos, tool="drawer")
        target = handle + np.array([0.0, 0.02, 0.004])
        self.follow(arm, [push_from + np.array([0, 0.03 * i, 0]) for i in range(1, 4)] + [target],
                    jaw_yaw, seconds_each=0.45, tool="drawer")
        self.move_joint(arm, HOME_QPOS, 0.9)
        closed = not env.drawer_is_open()
        return SkillResult(
            "close_drawer", arm, closed, "OK" if closed else "DRAWER_STUCK",
            f"drawer at {env.drawer_open_frac():.0%} open", time.perf_counter() - t0
        )

    def pick(self, arm: str, obj: str, *, lift: float = LIFT_HEIGHT) -> SkillResult:
        t0 = time.perf_counter()
        env = self.env
        if env.attached[arm] is not None:
            return SkillResult("pick", arm, False, "GRIPPER_FULL",
                               f"{arm} is already holding {env.attached[arm]}", time.perf_counter() - t0)

        grasp, jaw_yaw = grasp_geometry(env, obj)
        self.retract(arm, 0.7)
        self.move_gripper(arm, self.grip.open_cmd(obj), 0.9, settle=1.6)
        pre = grasp + np.array([0.0, 0.0, APPROACH_CLEARANCE])
        pose = self.go(arm, pre, jaw_yaw, seconds=1.2, tool=obj, approach=True)
        if not pose.ok:
            return SkillResult("pick", arm, False, "OUT_OF_REACH",
                               f"{obj} is {layout.reach(arm, grasp):.3f} m from the {arm} base",
                               time.perf_counter() - t0)
        # Re-read the pose: approaching can nudge the prop, and a stale target misses it.
        grasp, jaw_yaw = grasp_geometry(env, obj)
        pose = self.go(arm, grasp, jaw_yaw, seconds=0.85, q_init=pose.qpos, tool=obj,
                       approach=True)
        if not pose.ok:
            return SkillResult("pick", arm, False, "OUT_OF_REACH",
                               f"cannot descend onto {obj}", time.perf_counter() - t0)
        self.close_on_object(arm, obj)

        env.attached[arm] = obj
        self.grasp_transform[arm] = grasp_transform(env, arm, obj)
        up = grasp + np.array([0.0, 0.0, lift])
        lifted = self.go(arm, up, jaw_yaw, seconds=0.9, q_init=pose.qpos, tool=obj)
        if not lifted.ok:  # straight up can leave the envelope; settle for a tilt-back
            back = grasp - radial_dir(arm, grasp[:2]) * 0.03 + np.array([0, 0, lift * 0.6])
            lifted = self.go(arm, back, jaw_yaw, seconds=0.9, q_init=pose.qpos, tool=obj)
        self.hold(0.12)

        if env.attached[arm] == obj:
            self.retract(arm, 1.0)
            self.hold(0.15)
        held = env.is_grasped(arm, obj) and env.object_pos(obj)[2] > layout.REST_Z[obj] + 0.010
        if not held:
            env.attached[arm] = None
            self.grasp_transform[arm] = None
            self.held_tool[arm] = None
        return SkillResult(
            "pick", arm, held, "OK" if held else "GRASP_FAILED",
            f"{obj} at z={env.object_pos(obj)[2]:.3f}", time.perf_counter() - t0,
            {"object": obj, "z": round(float(env.object_pos(obj)[2]), 4)},
        )

    def place(self, arm: str, obj: str, target: str) -> SkillResult:
        t0 = time.perf_counter()
        env = self.env
        if env.attached[arm] != obj:
            return SkillResult("place", arm, False, "GRIPPER_EMPTY",
                               f"{arm} is not holding {obj}", time.perf_counter() - t0)
        if target in layout.SLOTS:
            slot = layout.SLOTS[target].copy()
            drop = np.array([slot[0], slot[1], layout.REST_Z[obj] + 0.010])
        elif target == "staging":
            # "Put it down somewhere sensible": a clear patch both arms can service.
            xy = layout.STAGING_DROP[arm]
            drop = np.array([xy[0], xy[1], layout.REST_Z[obj] + 0.010])
        else:
            return SkillResult("place", arm, False, "UNKNOWN_TARGET", target, time.perf_counter() - t0)

        # Carry the object with the gripper's current jaw alignment.
        _, jaw_yaw = grasp_geometry(env, obj)
        self.retract(arm, 0.8)
        over = drop + np.array([0.0, 0.0, APPROACH_CLEARANCE])
        pose = self.go(arm, over, jaw_yaw, seconds=1.4, tool=obj)
        if not pose.ok:
            return SkillResult("place", arm, False, "OUT_OF_REACH",
                               f"{target} is {layout.reach(arm, drop):.3f} m from the {arm} base",
                               time.perf_counter() - t0)
        # Close the loop on where the object actually ended up in the jaws: the grasp is never
        # perfectly centred, and a 1 cm bias is the difference between on-slot and off-slot.
        bias = env.object_pos(obj)[:2] - env.tcp(arm)[:2]
        drop = drop - np.array([bias[0], bias[1], 0.0])
        pose = self.go(arm, drop, jaw_yaw, seconds=0.85, q_init=pose.qpos, tool=obj)
        self.move_gripper(arm, self.grip.open_cmd(obj), 0.4)
        env.attached[arm] = None
        self.grasp_transform[arm] = None
        self.held_tool[arm] = None
        self.hold(0.2)
        self.go(arm, drop + np.array([0, 0, APPROACH_CLEARANCE]), jaw_yaw, seconds=0.8, tool=obj)
        self.move_joint(arm, HOME_QPOS, 0.9)
        self.hold(0.3)

        if target == "staging":
            down = float(env.object_pos(obj)[2]) < layout.REST_Z[obj] + 0.02
            return SkillResult(
                "place", arm, down, "OK" if down else "PLACE_MISSED",
                f"{obj} set down clear of the setting", time.perf_counter() - t0,
                {"object": obj, "target": target},
            )
        placed = env.object_on_slot(obj) == target
        err = float(np.linalg.norm(env.object_pos(obj)[:2] - layout.SLOTS[target][:2]))
        return SkillResult(
            "place", arm, placed, "OK" if placed else "PLACE_MISSED",
            f"{obj} {'on' if placed else 'off'} {target} ({err * 1000:.0f} mm from centre)",
            time.perf_counter() - t0,
            {"object": obj, "target": target, "err": round(err, 4)},
        )

    def handoff(self, from_arm: str, to_arm: str, obj: str) -> SkillResult:
        """Transfer in the shared zone. Falls back to a table relay if the catch is unreachable.

        The two grippers cross at ninety degrees on a round object, or sit a few centimetres
        apart along a thin one, so the jaws never contend for the same surface.
        """
        t0 = time.perf_counter()
        env = self.env
        if env.attached[from_arm] != obj:
            return SkillResult("handoff", from_arm, False, "GRIPPER_EMPTY",
                               f"{from_arm} is not holding {obj}", time.perf_counter() - t0)
        if env.attached[to_arm] is not None:
            return SkillResult("handoff", to_arm, False, "GRIPPER_FULL",
                               f"{to_arm} is holding {env.attached[to_arm]}", time.perf_counter() - t0)

        thin = obj in ("spoon", "fork")
        if not (thin and DIRECT_HANDOFF):
            # Measured, not assumed: with the two bases 0.40 m apart and both wrists forced
            # near-vertical, the arms' own links collide above the exchange point long before
            # the jaws meet. Transfers therefore go through the shared zone -- the object
            # still changes arms, which is what the task requires, but it touches down in the
            # middle instead of passing hand to hand. Set TANDEM_HANDOFF=direct to attempt the
            # in-air version for elongated props.
            return self._relay_handoff(from_arm, to_arm, obj, t0, "shared zone")
        self.retract(to_arm, 0.7)
        present = np.array([layout.HANDOFF_XY[0], layout.HANDOFF_XY[1], layout.HANDOFF_Z])
        _, jaw_yaw = grasp_geometry(env, obj)
        pose = self.go(from_arm, present, jaw_yaw, seconds=1.5, tool=obj)
        if not pose.ok:
            lower = present - np.array([0.0, 0.0, 0.018])
            pose = self.go(from_arm, lower, jaw_yaw, seconds=1.2, tool=obj)
        if not pose.ok:
            return self._relay_handoff(from_arm, to_arm, obj, t0, "presenting")
        self.hold(0.25)

        catch, catch_yaw = grasp_geometry(env, obj, handoff_offset=0.042 if thin else 0.0)
        if not thin:
            catch_yaw += np.pi / 2  # cross the jaws rather than share a diameter
        self.move_gripper(to_arm, self.grip.open_cmd(obj), 0.9, settle=1.6)
        pre = catch + np.array([0.0, 0.0, APPROACH_CLEARANCE])
        rpose = self.go(to_arm, pre, catch_yaw, seconds=1.3, tool=obj, approach=True)
        if not rpose.ok:
            return self._relay_handoff(from_arm, to_arm, obj, t0, "catching")
        catch, catch_yaw2 = grasp_geometry(env, obj, handoff_offset=0.042 if thin else 0.0)
        if not thin:
            catch_yaw2 += np.pi / 2
        rpose = self.go(to_arm, catch, catch_yaw2, seconds=0.85, q_init=rpose.qpos, tool=obj,
                        approach=True)
        if not rpose.ok:
            return self._relay_handoff(from_arm, to_arm, obj, t0, "catching")
        self.close_on_object(to_arm, obj)
        env.attached[to_arm] = obj
        self.grasp_transform[to_arm] = grasp_transform(env, to_arm, obj)
        self.move_gripper(from_arm, self.grip.open_cmd(obj, clearance=0.030), 0.4)
        env.attached[from_arm] = None
        self.grasp_transform[from_arm] = None
        self.held_tool[from_arm] = None
        self.hold(0.25)
        self.retract(from_arm, 1.0)
        self.hold(0.15)

        got = env.is_grasped(to_arm, obj) and env.object_pos(obj)[2] > layout.REST_Z[obj] + 0.012
        if not got:
            env.attached[to_arm] = None
            self.grasp_transform[to_arm] = None
            self.held_tool[to_arm] = None
        return SkillResult(
            "handoff", to_arm, got, "OK" if got else "HANDOFF_DROPPED",
            f"{obj} transferred {from_arm} -> {to_arm}", time.perf_counter() - t0,
            {"object": obj, "mode": "direct", "from": from_arm, "to": to_arm},
        )

    def _free_handoff_xy(self, obj: str) -> np.ndarray:
        """A clear spot in the shared zone.

        Several objects cross the table through the same place during one episode, and a
        previous transfer that was left behind — or a placement that missed — turns the next
        hand-off into a collision. Nudge along the zone until the ground is clear.
        """
        env = self.env
        base = np.array(layout.HANDOFF_XY, dtype=float)
        others = [p for p in layout.PROPS if p != obj]
        for dx in (0.0, 0.055, -0.055, 0.095, -0.095):
            cand = base + np.array([dx, 0.0])
            if not layout.in_reach("left", cand) or not layout.in_reach("right", cand):
                continue
            clear = all(
                float(np.linalg.norm(env.object_pos(o)[:2] - cand))
                > layout.PROP_RADIUS[o] + layout.PROP_RADIUS[obj] + 0.018
                or env.object_pos(o)[2] > layout.REST_Z[o] + 0.05
                for o in others
            )
            if clear:
                return cand
        return base

    def _relay_handoff(self, from_arm, to_arm, obj, t0, stage) -> SkillResult:
        """Fallback: set the object down in the shared zone and let the other arm collect it."""
        env = self.env
        drop = np.array([*self._free_handoff_xy(obj), layout.REST_Z[obj] + 0.008])
        _, jaw_yaw = grasp_geometry(env, obj)
        self.retract(from_arm, 0.7)
        pose = self.go(from_arm, drop + np.array([0, 0, APPROACH_CLEARANCE]), jaw_yaw,
                       seconds=1.2, tool=obj, approach=True)
        if pose.ok:
            pose = self.go(from_arm, drop, jaw_yaw, seconds=0.8, q_init=pose.qpos, tool=obj)
        if not pose.ok:
            return SkillResult("handoff", from_arm, False, "OUT_OF_REACH",
                               f"neither arm can meet in the shared zone (failed while {stage})",
                               time.perf_counter() - t0, {"object": obj, "mode": "relay"})
        self.move_gripper(from_arm, self.grip.open_cmd(obj), 0.4)
        env.attached[from_arm] = None
        self.grasp_transform[from_arm] = None
        self.held_tool[from_arm] = None
        self.hold(0.3)
        self.retract(from_arm, 0.9)
        res = self.pick(to_arm, obj)
        if not res.ok:
            self.retract(to_arm, 0.8)
            res = self.pick(to_arm, obj)
        return SkillResult(
            "handoff", to_arm, res.ok, "OK" if res.ok else res.code,
            f"{obj} changed arms via the shared zone ({stage})",
            time.perf_counter() - t0,
            {"object": obj, "mode": "relay", "from": from_arm, "to": to_arm},
        )

    def pour(self, arm: str, source: str, target: str) -> SkillResult:
        """Tilt ``source`` over ``target``. The other arm must be holding ``target``."""
        t0 = time.perf_counter()
        env = self.env
        if env.attached[arm] != source:
            return SkillResult("pour", arm, False, "GRIPPER_EMPTY",
                               f"{arm} is not holding {source}", time.perf_counter() - t0)
        holder = layout.other_arm(arm)
        if env.attached[holder] != target:
            return SkillResult("pour", arm, False, "NO_HOLDER_FOR_POUR",
                               f"{holder} must hold the {target} before pouring",
                               time.perf_counter() - t0)

        if self.grasp_transform[arm] is None:
            return SkillResult("pour", arm, False, "GRIPPER_EMPTY",
                               "no grasp transform recorded", time.perf_counter() - t0)
        # Re-measure where the carton is sitting in the jaws right now. It shifts during
        # transport, and pouring from where it was at pick time misses by centimetres.
        T_rel = grasp_transform(env, arm, source)

        # Bring the mug down to the pour station first. Held at the rest pose it sits ~0.28 m
        # up, far above anything the other arm can tip a carton over.
        station = np.array(
            [layout.POUR_STATION[0], layout.POUR_STATION[1], layout.POUR_STATION_Z]
        )
        _, hold_yaw = grasp_geometry(env, target)
        held = self.go(holder, station, hold_yaw, seconds=1.5, tool=target)
        if not held.ok:
            return SkillResult("pour", arm, False, "OUT_OF_REACH",
                               f"the {holder} arm cannot present the {target} to be filled",
                               time.perf_counter() - t0)
        self.hold(0.2)
        mouth = np.array(env.data.site_xpos[env.index.site[f"{target}_mouth"]])

        # Approach from above first, still upright, then tip.
        _, jaw_yaw = grasp_geometry(env, source)
        offset = env.tcp(arm) - env.object_pos(source)
        stage = mouth + np.array([0.0, 0.0, 0.055]) + offset
        staged = self.go(arm, stage, jaw_yaw, seconds=1.4)

        q_seed = staged.qpos if staged.ok else self.env.arm_target(arm)
        # Accept a candidate only if the pose the arm can actually hold still puts the carton's
        # rim over the mug and tipped far enough to empty. A five-joint arm routinely "solves"
        # a 6-DoF target by giving up most of the orientation, which looks fine in the residual
        # and pours onto the tablecloth.
        chosen = None
        chosen_rim = None
        best_tilt = 0.0
        for T_des in pour_pose_candidates(mouth, spout_local_z=0.030):
            cand = solve_for_object_pose(
                self.ik[arm], env.data, T_des, T_rel, q_init=q_seed, restarts=2
            )
            if not cand.solvable:
                continue
            p_tcp, R_tcp = self.ik[arm].fk(cand.qpos, env.data)
            T_tcp = np.eye(4)
            T_tcp[:3, :3] = R_tcp
            T_tcp[:3, 3] = p_tcp
            T_obj = T_tcp @ T_rel
            axis = T_obj[:3, 2]
            tilt_deg = float(np.degrees(np.arccos(np.clip(axis[2], -1.0, 1.0))))
            rim = T_obj[:3, 3] + axis * 0.030
            best_tilt = max(best_tilt, tilt_deg)
            if tilt_deg >= 80.0 and rim[2] > layout.POUR_STATION_Z + 0.045:
                chosen = cand
                chosen_rim = rim
                break
        if chosen is None:
            return SkillResult("pour", arm, False, "OUT_OF_REACH",
                               f"no reachable pose tips the carton over the mug "
                               f"(best tilt {best_tilt:.0f} deg, needs 80)",
                               time.perf_counter() - t0)

        # Move the cup under the spout *before* tipping. Correcting afterwards is too late --
        # the carton empties during the tilt itself -- and the arm holding the cup has the more
        # accurate, lower-amplitude move of the two anyway.
        aim = np.array([chosen_rim[0], chosen_rim[1], layout.POUR_STATION_Z])
        aimed = self.go(holder, aim, hold_yaw, seconds=1.1, tool=target)
        if not aimed.ok:
            return SkillResult("pour", arm, False, "OUT_OF_REACH",
                               f"the {holder} arm cannot bring the {target} under the spout",
                               time.perf_counter() - t0)
        self.hold(0.3)

        before = env.water_counts()
        # Tip in stages, and let the cup chase the spout between them. The carton rotates a
        # little inside the jaws as it goes over -- 12 N of pinch is enough to hold it, not
        # enough to stop it turning -- so where the stream ends up is only known once the arm is
        # actually there. Tracking it with the other arm is what turns a spill into a pour.
        start_q = np.array(env.arm_target(arm))
        # The first few fractions stay under the angle at which the carton starts to empty, so
        # the cup is already tracking the real rim -- not the solved one -- before any water
        # moves. Going straight to the final pose spills the lot during the transit.
        for frac in (0.22, 0.36, 0.46, 0.54, 0.62, 0.71, 0.80, 0.90, 1.0):
            if self.aborted:
                break
            self.move_joint(arm, start_q + (chosen.qpos - start_q) * frac, 0.45)
            self._track_spout(holder, source, target, hold_yaw)
            self.hold(0.12)
        for _ in range(10):
            if self.aborted:
                break
            self._track_spout(holder, source, target, hold_yaw)
            self.hold(0.16)
        if staged.ok:
            self.move_joint(arm, staged.qpos, 1.0)
        self.hold(0.4)

        after = env.water_counts()
        poured = after["in_mug"] - before["in_mug"]
        ok = poured >= 3
        return SkillResult(
            "pour", arm, ok, "OK" if ok else "POUR_MISSED",
            f"{poured} of {len(env.index.water_bodies)} units landed in the {target}",
            time.perf_counter() - t0,
            {"poured": poured, "spilled": after["spilled"], "source": source, "target": target},
        )

    def _track_spout(self, holder: str, source: str, target: str, hold_yaw: float) -> None:
        """Nudge the cup under wherever the carton's rim currently is."""
        env = self.env
        axis = env.object_rot(source)[:, 2]
        rim = env.object_pos(source) + axis * 0.030
        mouth = np.array(env.data.site_xpos[env.index.site[f"{target}_mouth"]])
        err = rim[:2] - mouth[:2]
        if float(np.linalg.norm(err)) < 0.005:
            return
        step = np.clip(err, -0.06, 0.06)
        want = env.object_pos(target)[:2] + step
        aim = np.array([want[0], want[1], layout.POUR_STATION_Z])
        if not layout.in_reach(holder, aim[:2]):
            return
        self.go(holder, aim, hold_yaw, seconds=0.35, tool=target)

    def hold_still(self, arm: str, obj: str, seconds: float = 0.4) -> SkillResult:
        t0 = time.perf_counter()
        ok = self.env.attached[arm] == obj
        if ok:
            self.hold(seconds)
        return SkillResult(
            "hold", arm, ok, "OK" if ok else "GRIPPER_EMPTY",
            f"{arm} steadying {obj}", time.perf_counter() - t0
        )

    # ------------------------------------------------------------------ dispatch

    def run_step(self, skill: str, args: dict) -> SkillResult:
        if skill == "open_drawer":
            return self.open_drawer(args["arm"])
        if skill == "close_drawer":
            return self.close_drawer(args["arm"])
        if skill == "pick":
            return self.pick(args["arm"], args["object"])
        if skill == "place":
            return self.place(args["arm"], args["object"], args["target"])
        if skill == "handoff":
            return self.handoff(args["from_arm"], args["to_arm"], args["object"])
        if skill == "hold":
            return self.hold_still(args["arm"], args["object"])
        if skill == "pour":
            return self.pour(args["arm"], args["source"], args["target"])
        if skill == "home":
            return self.home(args["arm"])
        return SkillResult(skill, args.get("arm", "left"), False, "UNKNOWN_SKILL", skill)
