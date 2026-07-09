# -*- coding: utf-8 -*-
"""
leisaac/agents/rule_grasp_agent.py

Pan-align only: compute desired shoulder_pan joint angle from relative position
between robot base and object, then rotate shoulder_pan towards it.

Key idea:
- We want desired_pan in JOINT COORD (same space as pan_q), so yaw_err = desired_pan - pan_q is valid.
- However, "atan2(target_rel_y, target_rel_x)" is a BASE-FRAME yaw, not a joint angle.
  To map base-yaw <-> joint-angle, we calibrate an offset once:
      yaw_from_pan ≈ pan_q + offset
  where yaw_from_pan is measured from current gripper direction in base plane.

Then:
  desired_yaw = atan2(obj_y - base_y, obj_x - base_x)
  desired_pan = wrap_pi(desired_yaw - offset)
  err = wrap_pi(desired_pan - pan_q)
  pan_inc = clip(err * gain, -max_step, +max_step)

Action layout (your env):
  arm_action: 6 dims (DiffIK) -> all zeros
  gripper_action: 2 dims (RelativeJointPositionAction) with joints ["shoulder_pan","gripper"]
    action[6] -> shoulder_pan relative increment (rad)
    action[7] -> gripper increment (kept 0)
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


class RuleGraspAgent:
    ALIGN = 0
    DONE = 1

    def __init__(
        self,
        env,
        stage,
        env_id: int = 0,
        # object prim
        mug_path_tpl: str = "/World/envs/env_{i}/RuntimeMug_proxy",
        # control
        pan_gain: float = 1.2,
        max_pan_step: float = 0.06,      # rad/step
        yaw_tol: float = 0.08,           # rad
        settle_steps: int = 8,
        # debug
        debug_print_steps: int = 300,
        # compatibility args (unused but kept)
        ee_prim_path_rel: str = "",
        pregrasp_dz: float = 0.10,
        grasp_dz: float = 0.02,
        lift_dz: float = 0.15,
        xy_gain: float = 2.0,
        z_gain: float = 2.0,
        yaw_gain: float = 0.0,
        close_steps: int = 25,
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
        self.settle_steps = int(settle_steps)

        self.debug_print_steps = int(debug_print_steps)
        self._dbg_n = 0

        # cached robot handles
        self._robot = None
        self._pan_jidx: Optional[int] = None
        self._gripper_bidx: Optional[int] = None

        # base-yaw <-> pan joint mapping offset
        # yaw_in_base ≈ pan_q + offset
        self._pan_yaw_offset: Optional[float] = None

        self.reset()

    def reset(self):
        self.state = self.ALIGN
        self._settle = 0
        self._dbg_n = 0
        self._pan_yaw_offset = None  # recalibrate each reset
        self.request_pose_sync_off = False
        self.request_success = False

    # -------------------------
    # Robot access (IsaacLab articulation)
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
            self._pan_jidx = names.index("shoulder_pan")
            return self._pan_jidx
        except Exception:
            self._pan_jidx = None
            return None

    def _get_gripper_body_index(self, robot) -> Optional[int]:
        """
        We measure current yaw direction from base->gripper vector in the base plane.
        This is robust and does not depend on quaternion convention.
        """
        if robot is None:
            return None
        if self._gripper_bidx is not None:
            return self._gripper_bidx

        # Prefer find_bodies API if available (as in your keyboard class)
        try:
            body_idxs, _ = robot.find_bodies("gripper")
            if len(body_idxs) > 0:
                self._gripper_bidx = int(body_idxs[0])
                return self._gripper_bidx
        except Exception:
            pass

        # Fallback to body_names list
        try:
            body_names = list(getattr(robot.data, "body_names", []) or [])
            if "gripper" in body_names:
                self._gripper_bidx = int(body_names.index("gripper"))
                return self._gripper_bidx
        except Exception:
            pass

        self._gripper_bidx = None
        return None

    def _get_base_pos_w(self, robot) -> Optional[np.ndarray]:
        if robot is None:
            return None
        try:
            if hasattr(robot.data, "root_pos_w"):
                p = robot.data.root_pos_w[self.i]
                return np.array(p.detach().cpu().numpy(), dtype=np.float64) if isinstance(p, torch.Tensor) else np.array(p, dtype=np.float64)
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

    def _get_gripper_pos_w(self, robot, bidx: Optional[int]) -> Optional[np.ndarray]:
        if robot is None or bidx is None:
            return None
        try:
            if hasattr(robot.data, "body_pos_w"):
                p = robot.data.body_pos_w[self.i, bidx]
                p = p.detach().cpu().numpy() if isinstance(p, torch.Tensor) else np.asarray(p)
                return np.array(p, dtype=np.float64)
        except Exception:
            return None
        return None

    # -------------------------
    # Object pose
    # -------------------------
    def _get_mug_world_pos_best_effort(self) -> Tuple[np.ndarray, str]:
        """
        Robustly fetch mug position:
        - try /RuntimeMug_proxy/visual first (often where pose is applied)
        - then /RuntimeMug_proxy
        """
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
        # fallback to origin (but label it)
        return np.zeros(3, dtype=np.float64), f"missing:{base}"

    # -------------------------
    # Action packing
    # -------------------------
    def _pack_action(self, pan_inc: float) -> torch.Tensor:
        # total dim
        try:
            if hasattr(self.env, "action_manager") and hasattr(self.env.action_manager, "total_action_dim"):
                d = int(self.env.action_manager.total_action_dim)
            else:
                d = int(np.prod(self.env.action_space.shape))
        except Exception:
            d = 8

        a = torch.zeros((getattr(self.env, "num_envs", 1), d), device=getattr(self.env, "device", "cpu"), dtype=torch.float32)
        # arm_action zeros
        # gripper_action: [shoulder_pan, gripper]
        a[:, 6] = float(np.clip(pan_inc, -0.2, 0.2))
        a[:, 7] = 0.0
        return a

    # -------------------------
    # Calibration: map pan_q -> base-plane yaw
    # -------------------------
    def _maybe_calibrate_offset(self, base_p: np.ndarray, gripper_p: np.ndarray, pan_q: float) -> Optional[float]:
        """
        Compute yaw direction of (base->gripper) in base plane, and derive offset:
            yaw_dir ≈ pan_q + offset  => offset = wrap(yaw_dir - pan_q)
        """
        rel = gripper_p - base_p
        # project to xy plane
        if (abs(rel[0]) + abs(rel[1])) < 1e-6:
            return None
        yaw_dir = math.atan2(float(rel[1]), float(rel[0]))
        return _wrap_pi(float(yaw_dir) - float(pan_q))

    # -------------------------
    # Main policy
    # -------------------------
    def act(self):
        self.request_pose_sync_off = False
        self.request_success = False

        if self.state == self.DONE:
            return self._pack_action(0.0), False, False

        robot = self._get_robot()
        pan_jidx = self._get_pan_joint_index(robot)
        grip_bidx = self._get_gripper_body_index(robot)

        base_p = self._get_base_pos_w(robot)
        pan_q = self._get_pan_q(robot, pan_jidx)
        gripper_p = self._get_gripper_pos_w(robot, grip_bidx)

        mug_p, mug_src = self._get_mug_world_pos_best_effort()

        # Safety NO-OP if missing critical data
        if base_p is None or pan_q is None:
            if self._dbg_n < self.debug_print_steps:
                print(f"[PAN_JOINT_ALIGN][NOOP] missing base_p/pan_q. base_p={base_p} pan_q={pan_q}")
            self._dbg_n += 1
            return self._pack_action(0.0), False, False

        # Calibrate offset once if possible
        if self._pan_yaw_offset is None and (gripper_p is not None):
            off = self._maybe_calibrate_offset(base_p, gripper_p, pan_q)
            if off is not None:
                self._pan_yaw_offset = off

        # Desired yaw in base plane from base->mug
        rel_t = mug_p - base_p
        desired_yaw = math.atan2(float(rel_t[1]), float(rel_t[0]))

        # If offset not available, fall back to 0 (still deterministic; may be mirrored depending on robot zero)
        offset = float(self._pan_yaw_offset) if self._pan_yaw_offset is not None else 0.0

        # Convert base-plane yaw to desired joint angle
        desired_pan = _wrap_pi(desired_yaw - offset)

        # Joint-space error (valid!)
        err = _wrap_pi(desired_pan - float(pan_q))

        pan_inc = float(np.clip(err * self.pan_gain, -self.max_pan_step, self.max_pan_step))

        # settle logic
        if abs(err) < self.yaw_tol:
            self._settle += 1
            if self._settle >= self.settle_steps:
                self.state = self.DONE
        else:
            self._settle = 0

        if self._dbg_n < self.debug_print_steps:
            off_str = "None" if self._pan_yaw_offset is None else f"{self._pan_yaw_offset:.3f}"
            print(
                f"[PAN_JOINT_ALIGN] base_p=({base_p[0]:.3f},{base_p[1]:.3f}) "
                f"mug_p=({mug_p[0]:.3f},{mug_p[1]:.3f}) src={mug_src} "
                f"pan_q={float(pan_q):.3f} offset={off_str} "
                f"desired_yaw={desired_yaw:.3f} desired_pan={desired_pan:.3f} err={err:.3f} "
                f"pan_inc={pan_inc:.3f} settle={self._settle}/{self.settle_steps}"
            )
        self._dbg_n += 1

        return self._pack_action(pan_inc), False, False



