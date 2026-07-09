# -*- coding: utf-8 -*-
"""
leisaac/agents/rule_grasp_agent.py

Rule-based grasp agent (FSM) used by teleop_se3_agent.py.

This version is tailored to your exact env action terms (from your debug output):
  - arm_action: DifferentialInverseKinematicsAction, dim = 6
      controller.command_type='pose'
      controller.use_relative_mode=True   <-- IMPORTANT
  - gripper_action: RelativeJointPositionAction, dim = 2
  - total_action_dim = 8

Core fix for "一直反向":
For DifferentialInverseKinematicsAction with use_relative_mode=True, the translational command
(dx, dy, dz) is interpreted in the END-EFFECTOR LOCAL frame (EE-local), not world.
Therefore we must convert world position error into EE-local coordinates:
    err_ee = R_w_ee^T * (target_w - ee_w)

This code:
- Packs actions as [arm(6), gripper(2)] correctly
- Uses small gripper increments (RelativeJointPositionAction)
- Converts position error to EE-local before commanding
- Disables dyaw by default (set yaw_gain>0 if you later want yaw)
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch

from leisaac.utils.physics_prims import get_world_xf


def _mat_to_pos_yaw(T: np.ndarray) -> Tuple[np.ndarray, float]:
    """Extract position (xyz) and yaw (rad) from a 4x4 world transform."""
    px, py, pz = T[0, 3], T[1, 3], T[2, 3]
    yaw = math.atan2(T[1, 0], T[0, 0])
    return np.array([px, py, pz], dtype=np.float64), float(yaw)


def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


class RuleGraspAgent:
    """Simple finite-state machine grasp policy producing DiffIK delta-pose + gripper deltas."""
    APPROACH_XY = 0
    DESCEND = 1
    CLOSE = 2
    LIFT = 3
    DONE = 4

    def __init__(
        self,
        env,
        stage,
        env_id: int = 0,
        ee_prim_path_rel: str = "",
        mug_path_tpl: str = "/World/envs/env_{i}/RuntimeMug_proxy",
        pregrasp_dz: float = 0.10,
        grasp_dz: float = 0.02,
        lift_dz: float = 0.15,
        # gains (start conservative)
        xy_gain: float = 2.0,
        z_gain: float = 2.0,
        yaw_gain: float = 0.0,  # keep 0 first; you can enable later
        close_steps: int = 25,
        settle_steps: int = 10,
        success_on_lift: bool = False,
        # RelativeJointPositionAction expects small increments per step
        grip_open_val: float = 0.02,
        grip_close_val: float = -0.02,
        # debug
        debug_print_steps: int = 0,
    ):
        self.env = env
        self.stage = stage
        self.i = int(env_id)

        self.ee_prim_path_rel = str(ee_prim_path_rel or "").strip()
        self.mug_path_tpl = mug_path_tpl

        self.pregrasp_dz = float(pregrasp_dz)
        self.grasp_dz = float(grasp_dz)
        self.lift_dz = float(lift_dz)

        self.xy_gain = float(xy_gain)
        self.z_gain = float(z_gain)
        self.yaw_gain = float(yaw_gain)

        self.close_steps = int(close_steps)
        self.settle_steps = int(settle_steps)
        self.success_on_lift = bool(success_on_lift)

        self.grip_open_val = float(grip_open_val)
        self.grip_close_val = float(grip_close_val)

        self.debug_print_steps = int(debug_print_steps)
        self._dbg_n = 0

        self.reset()

    def reset(self):
        self.state = self.APPROACH_XY
        self._close_count = 0
        self._settle_count = 0
        self.request_pose_sync_off = False
        self.request_success = False
        self._dbg_n = 0

    # ---------------------------------------------------------
    # Pose getters
    # ---------------------------------------------------------
    def _get_mug_world_T(self) -> np.ndarray:
        mug_path = self.mug_path_tpl.format(i=self.i)
        return get_world_xf(self.stage, mug_path)

    def _get_ee_world_T_best_effort(self) -> np.ndarray:
        """
        For your env cfg: DifferentialIK body_name='gripper'
        So ee_prim_path_rel should be 'Robot/gripper' (as you already use).
        """
        if self.ee_prim_path_rel:
            if self.ee_prim_path_rel.startswith("/"):
                ee_path = self.ee_prim_path_rel
            else:
                ee_path = f"/World/envs/env_{self.i}/{self.ee_prim_path_rel}"
            prim = self.stage.GetPrimAtPath(ee_path)
            if prim.IsValid():
                return get_world_xf(self.stage, ee_path)

        # fallback
        guess = f"/World/envs/env_{self.i}/Robot/gripper"
        prim = self.stage.GetPrimAtPath(guess)
        if prim.IsValid():
            return get_world_xf(self.stage, guess)

        raise RuntimeError("Cannot get EE world pose. Set --ee_prim_path (e.g. 'Robot/gripper').")

    # ---------------------------------------------------------
    # Frame conversion: WORLD error -> EE-local command
    # ---------------------------------------------------------
    @staticmethod
    def _world_err_to_ee_local(T_ee: np.ndarray, err_w: np.ndarray) -> np.ndarray:
        """
        Convert world position error into EE-local coordinates.
        DifferentialInverseKinematicsAction(use_relative_mode=True) expects EE-local.
        """
        R_w_ee = T_ee[:3, :3]      # world-from-ee rotation (USD Xform)
        return R_w_ee.T @ err_w    # world -> ee

    # ---------------------------------------------------------
    # Action packing: arm(6) + gripper(2) = 8
    # ---------------------------------------------------------
    def _pack_action(self, dx: float, dy: float, dz: float, dyaw: float, grip: float) -> torch.Tensor:
        """
        Pack for your env:
          arm_action: dim=6  (DifferentialInverseKinematicsAction)
          gripper_action: dim=2 (RelativeJointPositionAction)
          total: 8

        arm = [dx, dy, dz, droll, dpitch, dyaw]
        gripper = [g, g]
        """
        if hasattr(self.env, "action_manager") and hasattr(self.env.action_manager, "total_action_dim"):
            d = int(self.env.action_manager.total_action_dim)
        else:
            d = int(np.prod(self.env.action_space.shape))

        a = torch.zeros((self.env.num_envs, d), device=self.env.device)

        # arm_action (6)
        a[:, 0] = float(dx)
        a[:, 1] = float(dy)
        a[:, 2] = float(dz)
        a[:, 3] = 0.0
        a[:, 4] = 0.0
        a[:, 5] = float(dyaw)

        # gripper_action (2): relative joint increments (small)
        g = float(np.clip(grip, -0.03, 0.03))
        a[:, 6] = g
        a[:, 7] = g
        return a

    def _maybe_dbg(self, ee_p, tgt_p, err_w, err_ee, cmd):
        if self.debug_print_steps <= 0:
            return
        if self._dbg_n < self.debug_print_steps:
            print(
                f"[dbg] state={self.state} "
                f"ee_w={ee_p} tgt_w={tgt_p} "
                f"err_w={err_w} err_ee={err_ee} cmd={cmd}"
            )
        self._dbg_n += 1

    # ---------------------------------------------------------
    # Policy
    # ---------------------------------------------------------
    def act(self):
        """
        Returns:
          (actions_tensor, request_pose_sync_off: bool, request_success: bool)
        """
        self.request_pose_sync_off = False
        self.request_success = False

        # read poses
        T_mug = self._get_mug_world_T()
        T_ee = self._get_ee_world_T_best_effort()

        mug_p, mug_yaw = _mat_to_pos_yaw(T_mug)
        ee_p, ee_yaw = _mat_to_pos_yaw(T_ee)

        # targets in world
        pregrasp_p = mug_p.copy()
        pregrasp_p[2] = mug_p[2] + self.pregrasp_dz

        grasp_p = mug_p.copy()
        grasp_p[2] = mug_p[2] + self.grasp_dz

        lift_p = mug_p.copy()
        lift_p[2] = mug_p[2] + self.lift_dz

        # -------------------------
        # APPROACH_XY
        # -------------------------
        if self.state == self.APPROACH_XY:
            self.request_pose_sync_off = False

            err_w = pregrasp_p - ee_p
            err_ee = self._world_err_to_ee_local(T_ee, err_w)

            dx = float(np.clip(err_ee[0] * self.xy_gain, -0.05, 0.05))
            dy = float(np.clip(err_ee[1] * self.xy_gain, -0.05, 0.05))
            dz = float(np.clip(err_ee[2] * self.z_gain, -0.05, 0.05))

            # Keep rotation off for bring-up. If you want, set yaw_gain>0 and implement correctly later.
            dyaw = 0.0
            grip = self.grip_open_val

            self._maybe_dbg(ee_p, pregrasp_p, err_w, err_ee, (dx, dy, dz, dyaw, grip))

            # transition check uses world error for clarity
            if (abs(err_w[0]) < 0.01) and (abs(err_w[1]) < 0.01) and (abs(err_w[2]) < 0.02):
                self._settle_count += 1
                if self._settle_count >= self.settle_steps:
                    self._settle_count = 0
                    self.state = self.DESCEND

            return self._pack_action(dx, dy, dz, dyaw, grip), self.request_pose_sync_off, self.request_success

        # -------------------------
        # DESCEND
        # -------------------------
        if self.state == self.DESCEND:
            self.request_pose_sync_off = True

            err_w = grasp_p - ee_p
            err_ee = self._world_err_to_ee_local(T_ee, err_w)

            dx = float(np.clip(err_ee[0] * self.xy_gain, -0.03, 0.03))
            dy = float(np.clip(err_ee[1] * self.xy_gain, -0.03, 0.03))
            dz = float(np.clip(err_ee[2] * self.z_gain, -0.03, 0.03))

            dyaw = 0.0
            grip = self.grip_open_val

            self._maybe_dbg(ee_p, grasp_p, err_w, err_ee, (dx, dy, dz, dyaw, grip))

            if (abs(err_w[2]) < 0.005) and (abs(err_w[0]) < 0.01) and (abs(err_w[1]) < 0.01):
                self.state = self.CLOSE
                self._close_count = 0

            return self._pack_action(dx, dy, dz, dyaw, grip), self.request_pose_sync_off, self.request_success

        # -------------------------
        # CLOSE
        # -------------------------
        if self.state == self.CLOSE:
            self.request_pose_sync_off = True
            self._close_count += 1
            if self._close_count >= self.close_steps:
                self.state = self.LIFT
            return self._pack_action(0.0, 0.0, 0.0, 0.0, self.grip_close_val), self.request_pose_sync_off, self.request_success

        # -------------------------
        # LIFT
        # -------------------------
        if self.state == self.LIFT:
            self.request_pose_sync_off = True

            err_w = lift_p - ee_p
            err_ee = self._world_err_to_ee_local(T_ee, err_w)

            dx = float(np.clip(err_ee[0] * self.xy_gain, -0.03, 0.03))
            dy = float(np.clip(err_ee[1] * self.xy_gain, -0.03, 0.03))
            dz = float(np.clip(err_ee[2] * self.z_gain, -0.05, 0.05))

            dyaw = 0.0
            grip = self.grip_close_val

            self._maybe_dbg(ee_p, lift_p, err_w, err_ee, (dx, dy, dz, dyaw, grip))

            if abs(err_w[2]) < 0.01:
                self.state = self.DONE
                if self.success_on_lift:
                    self.request_success = True

            return self._pack_action(dx, dy, dz, dyaw, grip), self.request_pose_sync_off, self.request_success

        # DONE
        self.request_pose_sync_off = True
        return self._pack_action(0.0, 0.0, 0.0, 0.0, self.grip_close_val), self.request_pose_sync_off, self.request_success
