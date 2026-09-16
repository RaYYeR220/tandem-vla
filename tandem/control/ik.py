"""Per-arm differential inverse kinematics.

The SO-101 has five actuated arm joints, so it cannot realise an arbitrary 6-DoF pose.
Rather than pick a pose and hope, we solve a weighted least-squares problem in which the
position residual dominates and the orientation residual is down-weighted, then report the
residuals back to the caller. The gate uses that report to refuse a motion the arm cannot
actually achieve instead of letting the executor discover it mid-trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from ..sim import layout


@dataclass
class IKResult:
    qpos: np.ndarray  #: the five arm joint angles
    pos_err: float  #: metres
    rot_err: float  #: radians-ish (norm of the orientation residual)
    ok: bool
    iters: int


def approach_frame(approach: np.ndarray, jaw_axis: np.ndarray | None = None) -> np.ndarray:
    """Build a target rotation for the gripper site.

    The site's local +x is the finger direction and its local +z is the jaw-opening
    direction, both measured off the compiled model. ``approach`` is where the fingers
    should point in world coordinates; ``jaw_axis`` is the direction the jaws separate
    along, and is orthogonalised against ``approach``.
    """
    x = np.asarray(approach, dtype=float)
    x = x / (np.linalg.norm(x) + 1e-12)
    if jaw_axis is None:
        ref = np.array([0.0, 0.0, 1.0]) if abs(x[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        z = ref - np.dot(ref, x) * x
    else:
        z = np.asarray(jaw_axis, dtype=float)
        z = z - np.dot(z, x) * x
    n = np.linalg.norm(z)
    if n < 1e-9:  # degenerate request, fall back to any orthogonal axis
        z = np.array([0.0, 0.0, 1.0]) - x[2] * x
        n = np.linalg.norm(z)
    z = z / n
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def _rot_residual(R_cur: np.ndarray, R_des: np.ndarray, axis_only: bool = False) -> np.ndarray:
    """Orientation error in world coordinates.

    With ``axis_only`` we constrain the finger direction alone and leave the roll about it
    free. That matters a lot on a five-joint arm: a rotationally symmetric object does not
    care how the wrist is rolled, and spending a scarce degree of freedom on it is what puts
    otherwise easy poses outside the workspace.
    """
    if axis_only:
        return np.cross(R_cur[:, 0], R_des[:, 0])
    return 0.5 * sum(np.cross(R_cur[:, i], R_des[:, i]) for i in range(3))


class ArmIK:
    """Damped least-squares IK restricted to one arm's five joints."""

    def __init__(self, model: mujoco.MjModel, index, arm: str):
        self.model = model
        self.arm = arm
        self.qadr = np.asarray(index.arm_qpos[arm])
        self.dofadr = np.asarray(index.arm_dof[arm])
        self.site = index.tcp_site[arm]
        self.scratch = mujoco.MjData(model)
        jnt_ids = [index.joint[j] for j in layout.NAMES[arm].joints]
        self.jnt_range = np.array([model.jnt_range[j] for j in jnt_ids])
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def solve(
        self,
        data: mujoco.MjData,
        target_pos,
        target_rot: np.ndarray | None = None,
        *,
        q_init: np.ndarray | None = None,
        rot_weight: float = 0.45,
        iters: int = 160,
        damping: float = 0.08,
        tol_pos: float = 2.5e-3,
        tol_rot: float = 0.30,
        max_step: float = 0.22,
        restarts: int = 6,
        seed: int = 0,
        axis_only: bool = False,
    ) -> IKResult:
        """Solve from the current pose first, falling back to random restarts.

        Two things matter here and they pull against each other. The SO-101's workspace has
        thin regions near the folded configuration where a single gradient descent stalls
        against a joint limit, so restarts are necessary. But a restart can land in a
        completely different elbow configuration, and interpolating to it in joint space
        swings the gripper through a wide arc — which throws whatever it was holding across
        the table. So the seed nearest the current configuration always wins ties, and a
        distant solution has to be clearly better to be chosen at all.
        """
        rng = np.random.default_rng(seed)
        here = np.array(data.qpos[self.qadr])
        anchor = here if q_init is None else np.asarray(q_init, dtype=float)

        def run(q0):
            return self._solve_once(
                data,
                target_pos,
                target_rot,
                q_init=q0,
                rot_weight=rot_weight,
                iters=iters,
                damping=damping,
                tol_pos=tol_pos,
                tol_rot=tol_rot,
                max_step=max_step,
                axis_only=axis_only,
            )

        def score(r: IKResult) -> float:
            travel = float(np.linalg.norm(r.qpos - anchor))
            return r.pos_err + 0.05 * r.rot_err + 0.004 * travel

        first = run(anchor)
        if first.ok and first.pos_err < tol_pos and first.rot_err < tol_rot:
            return first

        best = first
        seeds: list[np.ndarray] = [here]
        for _ in range(max(0, restarts - 2)):
            seeds.append(rng.uniform(self.jnt_range[:, 0], self.jnt_range[:, 1]))
        for q0 in seeds:
            r = run(q0)
            if score(r) < score(best):
                best = r
            if best.ok and best.pos_err < tol_pos and best.rot_err < tol_rot:
                break
        return best

    def _solve_once(
        self,
        data: mujoco.MjData,
        target_pos,
        target_rot: np.ndarray | None = None,
        *,
        q_init: np.ndarray | None = None,
        rot_weight: float = 0.45,
        iters: int = 160,
        damping: float = 0.08,
        tol_pos: float = 2.5e-3,
        tol_rot: float = 0.30,
        max_step: float = 0.22,
        axis_only: bool = False,
    ) -> IKResult:
        s = self.scratch
        s.qpos[:] = data.qpos
        s.qvel[:] = 0.0
        if q_init is not None:
            s.qpos[self.qadr] = q_init
        target_pos = np.asarray(target_pos, dtype=float)

        pos_err = rot_err = np.inf
        used = 0
        for used in range(1, iters + 1):
            mujoco.mj_kinematics(self.model, s)
            mujoco.mj_comPos(self.model, s)
            cur = s.site_xpos[self.site]
            dp = target_pos - cur
            pos_err = float(np.linalg.norm(dp))
            if target_rot is None:
                dr = np.zeros(3)
                rot_err = 0.0
            else:
                R = s.site_xmat[self.site].reshape(3, 3)
                dr = _rot_residual(R, target_rot, axis_only)
                rot_err = float(np.linalg.norm(dr))
            if pos_err < tol_pos and rot_err < tol_rot:
                break

            mujoco.mj_jacSite(self.model, s, self._jacp, self._jacr, self.site)
            J = np.vstack(
                [self._jacp[:, self.dofadr], rot_weight * self._jacr[:, self.dofadr]]
            )
            err = np.concatenate([dp, rot_weight * dr])
            JJt = J @ J.T
            dq = J.T @ np.linalg.solve(JJt + (damping**2) * np.eye(6), err)
            n = np.linalg.norm(dq)
            if n > max_step:
                dq *= max_step / n
            q = s.qpos[self.qadr] + dq
            q = np.clip(q, self.jnt_range[:, 0] + 1e-4, self.jnt_range[:, 1] - 1e-4)
            s.qpos[self.qadr] = q

        return IKResult(
            qpos=np.array(s.qpos[self.qadr]),
            pos_err=pos_err,
            rot_err=rot_err,
            ok=bool(pos_err < 6e-3 and rot_err < 0.55),
            iters=used,
        )

    def fk(self, qpos: np.ndarray, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        s = self.scratch
        s.qpos[:] = data.qpos
        s.qpos[self.qadr] = qpos
        mujoco.mj_kinematics(self.model, s)
        return (
            np.array(s.site_xpos[self.site]),
            np.array(s.site_xmat[self.site]).reshape(3, 3),
        )
