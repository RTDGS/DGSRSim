# -*- coding: utf-8 -*-
"""
leisaac/utils/pose_sync_pipeline.py

Pose Sync pipeline split-out:
- Poll .npy 4x4 pose file (T_scene_mug)
- Maintain pose sync state:
    - pose_sync_enabled (default True)
    - freeze_when_sync_off
    - manual_override (None/True/False) for rule-grasp priority
- Apply pose into USD each step:
    - compute T_world_mug = T_world_scene @ T_scene_mug
    - set RuntimeMug_proxy prim world transform
    - optional freeze when sync is OFF

Expected external deps:
- leisaac.utils.physics_prims.get_world_xf
- leisaac.utils.physics_prims.set_prim_world_matrix
"""

from __future__ import annotations

import os
import time
import threading
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from leisaac.utils.physics_prims import get_world_xf, set_prim_world_matrix


class NpyPoseFileStream:
    """Background polling of a .npy 4x4 pose file."""
    def __init__(self, npy_path: str, poll_hz: float = 60.0):
        self.npy_path = str(npy_path)
        self.poll_hz = float(poll_hz)

        self._lock = threading.Lock()
        self._latest_T: Optional[np.ndarray] = None
        self._latest_mtime = -1.0
        self._running = False
        self._th: Optional[threading.Thread] = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def stop(self):
        self._running = False
        if self._th is not None:
            self._th.join(timeout=1.0)
            self._th = None

    def get_latest(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._latest_T is None else self._latest_T.copy()

    def _loop(self):
        dt = 1.0 / max(self.poll_hz, 1e-6)
        while self._running:
            try:
                if os.path.exists(self.npy_path):
                    m = os.path.getmtime(self.npy_path)
                    if m > self._latest_mtime:
                        try:
                            T = np.load(self.npy_path).astype(np.float64).reshape(4, 4)
                            if np.isfinite(T).all():
                                with self._lock:
                                    self._latest_T = T
                                    self._latest_mtime = m
                        except Exception:
                            pass
            except Exception:
                pass
            time.sleep(dt)


@dataclass
class PoseSyncConfig:
    pose_npy: str
    pose_poll_hz: float = 60.0

    pose_sync_key: str = "M"
    pose_sync_freeze: bool = False

    # where to apply
    scene_path_tpl: str = "/World/envs/env_{i}/Scene"
    mug_path_tpl: str = "/World/envs/env_{i}/RuntimeMug_proxy"


class PoseSyncPipeline:
    """
    Owns:
    - pose file stream
    - sync state (enabled/freeze)
    - manual override for rule-grasp
    - last world pose cache
    """
    def __init__(self, cfg: PoseSyncConfig, num_envs: int, rule_grasp_mode: bool):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.rule_grasp_mode = bool(rule_grasp_mode)

        self.pose_sync_enabled: bool = True
        self.freeze_when_sync_off: bool = bool(cfg.pose_sync_freeze)

        # rule-grasp priority: None -> follow agent request; True/False -> force
        self.manual_override: Optional[bool] = None

        self._last_world_mug_pose_by_env: Dict[int, np.ndarray] = {}

        self._stream = NpyPoseFileStream(cfg.pose_npy, poll_hz=cfg.pose_poll_hz)

    # -------------------------
    # lifecycle
    # -------------------------
    def start(self):
        self._stream.start()
        print(f"[pose] streaming absolute pose from: {os.path.abspath(self.cfg.pose_npy)}")
        print(
            f"[pose-sync] initial=ON. toggle key='{str(self.cfg.pose_sync_key).upper()}'. "
            f"freeze_off={self.freeze_when_sync_off}"
        )
        if self.rule_grasp_mode:
            print("[pose-sync] rule-grasp: manual override via key has priority over agent requests.")

    def stop(self):
        self._stream.stop()

    def reset(self):
        """Call on env.reset() so cache doesn't carry across episodes."""
        self._last_world_mug_pose_by_env.clear()
        # Do not forcibly clear manual_override; keep it as user intent

    # -------------------------
    # state transitions
    # -------------------------
    def toggle_by_keypress(self):
        """
        Key behavior:
        - rule-grasp: toggles manual override state (priority over agent)
        - others: toggles pose_sync_enabled directly
        """
        if self.rule_grasp_mode:
            if self.manual_override is None:
                self.manual_override = (not self.pose_sync_enabled)
            else:
                self.manual_override = (not self.manual_override)
            self.pose_sync_enabled = bool(self.manual_override)
            print(
                f"[pose-sync][MANUAL] {'ON' if self.pose_sync_enabled else 'OFF'} "
                f"(override={'ON' if self.manual_override else 'OFF'}) "
                f"(freeze_off={self.freeze_when_sync_off})"
            )
        else:
            self.pose_sync_enabled = not self.pose_sync_enabled
            print(f"[pose-sync] {'ON' if self.pose_sync_enabled else 'OFF'} (freeze_off={self.freeze_when_sync_off})")

    def apply_agent_request(self, req_pose_sync_off: bool):
        """
        Agent request should only apply when there is no manual override.
        """
        if self.rule_grasp_mode and (self.manual_override is not None):
            return
        self.pose_sync_enabled = (not bool(req_pose_sync_off))

    # -------------------------
    # per-step apply
    # -------------------------
    def step(self, stage):
        """
        Apply pose sync to all env_i:
          if enabled -> read T_scene_mug and write RuntimeMug_proxy world matrix
          else if freeze -> keep at last pose
        """
        if self.pose_sync_enabled:
            T_scene_mug = self._stream.get_latest()
            if T_scene_mug is None:
                return

            for i in range(self.num_envs):
                scene_path = self.cfg.scene_path_tpl.format(i=i)
                mug_path = self.cfg.mug_path_tpl.format(i=i)
                try:
                    T_world_scene = get_world_xf(stage, scene_path)
                    T_world_mug = T_world_scene @ np.asarray(T_scene_mug, dtype=np.float64)
                    set_prim_world_matrix(stage, mug_path, T_world_mug)
                    self._last_world_mug_pose_by_env[i] = T_world_mug
                except Exception:
                    pass
            return

        # sync OFF
        if self.freeze_when_sync_off:
            for i, T_world_mug in list(self._last_world_mug_pose_by_env.items()):
                mug_path = self.cfg.mug_path_tpl.format(i=i)
                try:
                    set_prim_world_matrix(stage, mug_path, T_world_mug)
                except Exception:
                    pass
