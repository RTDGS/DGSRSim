# -*- coding: utf-8 -*-
"""
leisaac/agents/rule_grasp_agent.py

PAN_ONLY (shoulder_pan only) using "two-vector angle" incremental steering:

Idea:
- Compute signed angle between:
    (1) base -> target vector (in XY)
    (2) base -> arm-facing vector (in XY), approximated by base -> gripper position
- delta_yaw = wrap_pi(yaw_target - yaw_arm)
- Convert to joint increment using one-time pan_sign calibration (handles joint "反向"):
    err_pan = wrap_pi(pan_sign * delta_yaw)
    pan_inc = clip(err_pan * gain, max_step)
- Apply soft-limit guard to stop pushing further into the joint bound.
- Settle when |delta_yaw| < yaw_tol for align_settle_steps.

Env action terms (as per your setup):
  - arm_action: 6-dim DifferentialInverseKinematicsAction (zeros)
  - gripper_action: 2-dim RelativeJointPositionAction ["shoulder_pan","gripper"]
      => action[6] shoulder_pan increment (rad)
      => action[7] gripper increment (kept 0)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch

from leisaac.utils.physics_prims import get_world_xf


def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _mat_to_pos(T: np.ndarray) -> np.ndarray:
    return np.array([T[0, 3], T[1, 3], T[2, 3]], dtype=np.float64)


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


class RuleGraspAgent:
    PAN_ONLY = 0
    DONE = 1

    def __init__(
        self,
        env,
        stage,
        env_id: int = 0,
        mug_path_tpl: str = "/World/envs/env_{i}/RuntimeMug_proxy",
        # control
        pan_gain: float = 1.2,
        max_pan_step: float = 0.06,
        yaw_tol: float = 0.08,
        align_settle_steps: int = 8,
        # fallback target
        fallback_target_w: Tuple[float, float, float] = (0.20, 0.10, 0.0),
        invalid_mug_norm_eps: float = 1e-4,
        # reachability / saturation
        pan_soft_limit: float = 1.92,      # observed reachable bound
        pan_limit_margin: float = 0.03,    # stop pushing further near limit
        # calibration
        calibrate_once: bool = True,
        # debug
        debug_print_steps: int = 400,
        # compatibility (kept)
        ee_prim_path_rel: str = "",
        pregrasp_dz: float = 0.10,
        grasp_dz: float = 0.02,
        lift_dz: float = 0.15,
        xy_gain: float = 2.0,
        z_gain: float = 2.0,
        yaw_gain: float = 0.0,
        close_steps: int = 25,
        settle_steps: int = 10,
        success_on_lift: bool = False,
        grip_open_val: float = 0.0,
        grip_close_val: float = 0.0,
    ):
        self.env = env
        self.stage = stage
        self.i = int(env_id)

        self.mug_path_tpl = str(mug_path_tpl)

        self.pan_gain = float(pan_gain)
        self.max_pan_step = float(max_pan_step)
        self.yaw_tol = float(yaw_tol)
        self.align_settle_steps = int(align_settle_steps)

        self.fallback_target_w = np.array(fallback_target_w, dtype=np.float64)
        self.invalid_mug_norm_eps = float(invalid_mug_norm_eps)

        self.pan_soft_limit = float(pan_soft_limit)
        self.pan_limit_margin = float(pan_limit_margin)

        self.calibrate_once = bool(calibrate_once)

        self.debug_print_steps = int(debug_print_steps)
        self._dbg_n = 0

        # cached robot handles
        self._robot = None
        self._pan_jidx: Optional[int] = None
        self._gripper_bidx: Optional[int] = None

        # calibrated mapping: only pan_sign (+1 or -1)
        self._pan_sign: Optional[int] = None   # +1 or -1

        self.reset()

    def reset(self):
        self.state = self.PAN_ONLY
        self._align_settle = 0
        self._dbg_n = 0
        self._pan_sign = None
        self.request_pose_sync_off = False
        self.request_success = False

    # -------------------------
    # Robot access
    # -------------------------
    def _get_robot(self):
        if self._robot is not None:
            return self._robot
        scene = getattr(self.env, "scene", None)
        if scene is None:
            return None
        arts = getattr(scene, "articulations", None)
        if isinstance(arts, dict) and "robot" in arts:
            self._robot = arts["robot"]
            return self._robot
        try:
            self._robot = scene["robot"]
            return self._robot
        except Exception:
            return None

    def _get_pan_joint_index(self, robot) -> Optional[int]:
        if robot is None:
            return None
        if self._pan_jidx is not None:
            return self._pan_jidx
        try:
            names = list(getattr(robot.data, "joint_names", []) or [])
            self._pan_jidx = int(names.index("shoulder_pan"))
            return self._pan_jidx
        except Exception:
            self._pan_jidx = None
            return None

    def _get_base_pos_w(self, robot) -> Optional[np.ndarray]:
        if robot is None:
            return None
        try:
            if hasattr(robot.data, "root_pos_w"):
                p = robot.data.root_pos_w[self.i]
                p = p.detach().cpu().numpy() if isinstance(p, torch.Tensor) else np.asarray(p)
                return np.array(p, dtype=np.float64)
            if hasattr(robot.data, "root_state_w"):
                rs = robot.data.root_state_w[self.i]
                rs = rs.detach().cpu().numpy() if isinstance(rs, torch.Tensor) else np.asarray(rs)
                return np.array(rs[0:3], dtype=np.float64)
        except Exception:
            return None
        return None

    def _get_pan_q(self, robot, jidx: Optional[int]) -> Optional[float]:
        if robot is None or jidx is None:
            return None
        try:
            q = robot.data.joint_pos[self.i, jidx]
            return float(q.detach().cpu().item()) if isinstance(q, torch.Tensor) else float(q)
        except Exception:
            return None

    def _get_gripper_body_index(self, robot) -> Optional[int]:
        if robot is None:
            return None
        if self._gripper_bidx is not None:
            return self._gripper_bidx
        try:
            body_idxs, _ = robot.find_bodies("gripper")
            if len(body_idxs) > 0:
                self._gripper_bidx = int(body_idxs[0])
                return self._gripper_bidx
        except Exception:
            pass
        try:
            body_names = list(getattr(robot.data, "body_names", []) or [])
            if "gripper" in body_names:
                self._gripper_bidx = int(body_names.index("gripper"))
                return self._gripper_bidx
        except Exception:
            pass
        self._gripper_bidx = None
        return None

    def _get_gripper_pos_w(self, robot, bidx: Optional[int]) -> Optional[np.ndarray]:
        if robot is None or bidx is None:
            return None
        try:
            p = getattr(robot.data, "body_pos_w", None)
            if p is None:
                return None
            p0 = p[self.i, bidx]
            p0 = p0.detach().cpu().numpy() if isinstance(p0, torch.Tensor) else np.asarray(p0)
            return np.array(p0, dtype=np.float64)
        except Exception:
            return None

    # -------------------------
    # Object pose
    # -------------------------
    def _get_mug_world_pos_best_effort(self) -> Tuple[np.ndarray, str]:
        base = self.mug_path_tpl.format(i=self.i)
        candidates = [f"{base}/visual", base]
        for path in candidates:
            try:
                prim = self.stage.GetPrimAtPath(path)
                if prim.IsValid():
                    T = get_world_xf(self.stage, path)
                    return _mat_to_pos(T), f"usd_prim:{path}"
            except Exception:
                continue
        return np.zeros(3, dtype=np.float64), f"missing:{base}"

    # -------------------------
    # Action packing
    # -------------------------
    def _pack_action(self, pan_inc: float, grip_inc: float = 0.0) -> torch.Tensor:
        try:
            if hasattr(self.env, "action_manager") and hasattr(self.env.action_manager, "total_action_dim"):
                d = int(self.env.action_manager.total_action_dim)
            else:
                d = int(np.prod(self.env.action_space.shape))
        except Exception:
            d = 8

        a = torch.zeros(
            (getattr(self.env, "num_envs", 1), d),
            device=getattr(self.env, "device", "cpu"),
            dtype=torch.float32,
        )
        # Your env: index 6 -> shoulder_pan increment, index 7 -> gripper increment
        a[:, 6] = float(np.clip(pan_inc, -0.2, 0.2))
        a[:, 7] = float(np.clip(grip_inc, -0.2, 0.2))
        return a

    # -------------------------
    # Saturation guard
    # -------------------------
    def _stop_if_pushing_into_limit(self, pan_q: float, pan_inc: float) -> Tuple[float, bool]:
        lim = self.pan_soft_limit
        m = self.pan_limit_margin
        if pan_q > (lim - m) and pan_inc > 0.0:
            return 0.0, True
        if pan_q < (-lim + m) and pan_inc < 0.0:
            return 0.0, True
        return pan_inc, False

    # -------------------------
    # One-time calibration: ONLY pan_sign
    # -------------------------
    def _maybe_calibrate_pan_sign(self, base_p: np.ndarray, pan_q: float, gripper_p: Optional[np.ndarray]) -> bool:
        """
        Determine pan_sign (+1 or -1) once.

        We estimate current facing direction yaw_dir from base->gripper (XY).
        Compare which sign gives smaller residual in:
            yaw_dir ≈ pan_sign * pan_q + offset
        offset is eliminated by best-fit wrap, so we only keep pan_sign.
        """
        if not self.calibrate_once:
            return False
        if self._pan_sign is not None:
            return False
        if gripper_p is None:
            return False

        rel = gripper_p - base_p
        if (abs(rel[0]) + abs(rel[1])) < 1e-6:
            return False

        yaw_dir = math.atan2(float(rel[1]), float(rel[0]))

        # test sign +1
        off_pos = _wrap_pi(yaw_dir - (+1.0) * pan_q)
        err_pos = abs(_wrap_pi((+1.0) * pan_q + off_pos - yaw_dir))

        # test sign -1
        off_neg = _wrap_pi(yaw_dir - (-1.0) * pan_q)
        err_neg = abs(_wrap_pi((-1.0) * pan_q + off_neg - yaw_dir))

        self._pan_sign = -1 if err_neg < err_pos else +1

        if self._dbg_n < self.debug_print_steps:
            print(
                f"[CALIB_SIGN] yaw_dir={yaw_dir:.3f} pan_q={pan_q:.3f} "
                f"choose_sign={self._pan_sign:+d} (err+={err_pos:.3e}, err-={err_neg:.3e})"
            )
        return True

    # -------------------------
    # Main policy
    # -------------------------
    def act(self):
        self.request_pose_sync_off = False
        self.request_success = False

        if self.state == self.DONE:
            return self._pack_action(0.0, 0.0), False, False

        robot = self._get_robot()
        pan_jidx = self._get_pan_joint_index(robot)
        base_p = self._get_base_pos_w(robot)
        pan_q = self._get_pan_q(robot, pan_jidx)

        if base_p is None or pan_q is None:
            if self._dbg_n < self.debug_print_steps:
                print(f"[PAN_ONLY][NOOP] missing base_p/pan_q. base_p={base_p} pan_q={pan_q}")
            self._dbg_n += 1
            return self._pack_action(0.0, 0.0), False, False

        mug_p, mug_src = self._get_mug_world_pos_best_effort()
        mug_norm = float(np.linalg.norm(mug_p))

        if mug_norm < self.invalid_mug_norm_eps:
            target_p = self.fallback_target_w
            target_src = f"fallback:{tuple(self.fallback_target_w.tolist())}"
        else:
            target_p = mug_p
            target_src = mug_src

        # gripper pose (used to infer current arm-facing direction)
        grip_bidx = self._get_gripper_body_index(robot)
        gripper_p = self._get_gripper_pos_w(robot, grip_bidx)

        # one-time sign calibration (handles reverse joint sign)
        self._maybe_calibrate_pan_sign(base_p, float(pan_q), gripper_p)
        pan_sign = int(self._pan_sign) if self._pan_sign is not None else +1

        # Need gripper to infer yaw_arm
        if gripper_p is None:
            if self._dbg_n < self.debug_print_steps:
                print("[PAN_ONLY][NOOP] missing gripper_p for yaw_arm inference.")
            self._dbg_n += 1
            return self._pack_action(0.0, 0.0), False, False

        # (A) current arm facing direction: base -> gripper (XY)
        v_arm = gripper_p - base_p
        if (abs(v_arm[0]) + abs(v_arm[1])) < 1e-6:
            if self._dbg_n < self.debug_print_steps:
                print("[PAN_ONLY][NOOP] gripper too close to base in XY, cannot infer yaw_arm.")
            self._dbg_n += 1
            return self._pack_action(0.0, 0.0), False, False
        yaw_arm = math.atan2(float(v_arm[1]), float(v_arm[0]))

        # (B) target direction: base -> target (XY)
        v_t = target_p - base_p
        yaw_t = math.atan2(float(v_t[1]), float(v_t[0]))

        # (C) signed angle from current arm direction to target
        delta_yaw = _wrap_pi(yaw_t - yaw_arm)

        # (D) convert to joint-coord error using sign
        err_pan = _wrap_pi(pan_sign * delta_yaw)

        # (E) incremental command
        pan_inc = float(np.clip(err_pan * self.pan_gain, -self.max_pan_step, self.max_pan_step))

        # (F) protect against pushing into soft limits
        pan_inc2, saturated = self._stop_if_pushing_into_limit(float(pan_q), pan_inc)

        # settle using world-space delta (sign-invariant)
        if abs(delta_yaw) < self.yaw_tol:
            self._align_settle += 1
            if self._align_settle >= self.align_settle_steps:
                self.state = self.DONE
        else:
            self._align_settle = 0

        if self._dbg_n < self.debug_print_steps:
            sat_str = " SATURATED->0" if saturated else ""
            print(
                f"[PAN_ONLY] base_p=({base_p[0]:.3f},{base_p[1]:.3f}) "
                f"target_p=({target_p[0]:.3f},{target_p[1]:.3f}) src={target_src} "
                f"pan_q={float(pan_q):.3f} sign={pan_sign:+d} "
                f"yaw_arm={yaw_arm:.3f} yaw_t={yaw_t:.3f} delta_yaw={delta_yaw:.3f} "
                f"err_pan={err_pan:.3f} pan_inc={pan_inc:.3f}->{pan_inc2:.3f}{sat_str} "
                f"settle={self._align_settle}/{self.align_settle_steps}"
            )
        self._dbg_n += 1

        return self._pack_action(pan_inc2, 0.0), False, False
