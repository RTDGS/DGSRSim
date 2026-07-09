# -*- coding: utf-8 -*-
"""
leisaac/agents/rule_grasp_agent.py

NO-OP agent: the robot does nothing.
- Arm delta = 0
- Gripper delta = 0
- Pose-sync/success requests = False
"""

from __future__ import annotations

import numpy as np
import torch


class RuleGraspAgent:
    """Always output zero actions (do nothing)."""

    def __init__(
        self,
        env,
        stage=None,
        env_id: int = 0,
        ee_prim_path_rel: str = "",
        mug_path_tpl: str = "/World/envs/env_{i}/RuntimeMug_proxy",
        # keep compatibility with your factory arguments
        pregrasp_dz: float = 0.10,
        grasp_dz: float = 0.02,
        lift_dz: float = 0.15,
        xy_gain: float = 6.0,
        z_gain: float = 6.0,
        yaw_gain: float = 0.0,
        close_steps: int = 25,
        settle_steps: int = 10,
        success_on_lift: bool = False,
        grip_open_val: float = 0.02,
        grip_close_val: float = -0.02,
        step_xy: float = 0.02,
        step_z: float = 0.02,
        max_xy: float = 0.03,
        max_z: float = 0.03,
        debug_print_steps: int = 0,
        reach_tol_xy: float = 0.015,
        reach_tol_z: float = 0.02,
        target_offset_w=(0.0, 0.0, 0.0),
        command_frame: str = "ee",
        fallback_to_world_on_bad_ee: bool = True,
    ):
        self.env = env
        self.stage = stage
        self.i = int(env_id)

        self.request_pose_sync_off = False
        self.request_success = False

        # cache action dim
        if hasattr(self.env, "action_manager") and hasattr(self.env.action_manager, "total_action_dim"):
            self._d = int(self.env.action_manager.total_action_dim)
        else:
            self._d = int(np.prod(self.env.action_space.shape))

    def reset(self):
        self.request_pose_sync_off = False
        self.request_success = False

    def act(self):
        """
        Returns:
          (actions_tensor, request_pose_sync_off: bool, request_success: bool)
        """
        self.request_pose_sync_off = False
        self.request_success = False
        a = torch.zeros((self.env.num_envs, self._d), device=self.env.device, dtype=torch.float32)
        return a, self.request_pose_sync_off, self.request_success
