# -*- coding: utf-8 -*-
"""
leisaac/utils/pose_sync_pipeline.py

Pose Sync pipeline split-out:
- Poll the JSON object-state packet, with a legacy .npy pose fallback
- Maintain pose sync state:
    - pose_sync_enabled (default True)
    - freeze_when_sync_off
    - manual_override (None/True/False) for rule-grasp priority
- Apply state into USD each step:
    - separate rigid scene placement from background-asset scale
    - compute A_world_asset = T_world_scene_placement @ A_scene_asset_raw
    - set RuntimeMug_proxy prim world transform
    - optional freeze when sync is OFF

Expected external deps:
- leisaac.utils.physics_prims.get_world_xf
- leisaac.utils.physics_prims.set_prim_world_matrix
"""

from __future__ import annotations

import json
import os
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from leisaac.utils.physics_prims import get_world_xf, set_prim_world_matrix
from leisaac.utils.state_transform import rigid_scene_placement, validate_similarity_matrix


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


class ObjectStateFileStream:
    """Poll a scale-preserving JSON state packet with a legacy NPY fallback."""

    def __init__(self, state_json: str, pose_npy: str, poll_hz: float = 60.0):
        self.state_json = str(state_json)
        self.pose_npy = str(pose_npy)
        self.poll_hz = float(poll_hz)
        self._lock = threading.Lock()
        self._latest_A: Optional[np.ndarray] = None
        self._latest_mtime = -1.0
        self._latest_source = ""
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
            return None if self._latest_A is None else self._latest_A.copy()

    def _read_candidate(self) -> tuple[Optional[np.ndarray], float, str]:
        if self.state_json and os.path.exists(self.state_json):
            mtime = os.path.getmtime(self.state_json)
            payload = json.loads(Path(self.state_json).read_text(encoding="utf-8"))
            if payload.get("schema") != "dgsrsim.object_state.v1":
                raise ValueError("unsupported DGSRSim object-state schema")
            matrix = validate_similarity_matrix(
                payload["A_scene_from_asset_raw"],
                "A_scene_from_asset_raw",
            )
            return matrix, mtime, self.state_json

        if self.pose_npy and os.path.exists(self.pose_npy):
            mtime = os.path.getmtime(self.pose_npy)
            matrix = validate_similarity_matrix(
                np.load(self.pose_npy, allow_pickle=False).astype(np.float64).reshape(4, 4),
                "legacy_T_scene_from_target",
            )
            return matrix, mtime, self.pose_npy
        return None, -1.0, ""

    def _loop(self):
        dt = 1.0 / max(self.poll_hz, 1e-6)
        while self._running:
            try:
                matrix, mtime, source = self._read_candidate()
                if matrix is not None and (
                    mtime > self._latest_mtime or source != self._latest_source
                ):
                    with self._lock:
                        self._latest_A = matrix
                        self._latest_mtime = mtime
                        self._latest_source = source
            except Exception:
                pass
            time.sleep(dt)


@dataclass
class PoseSyncConfig:
    state_json: str = ""
    pose_npy: str = ""
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

        self._stream = ObjectStateFileStream(
            cfg.state_json,
            cfg.pose_npy,
            poll_hz=cfg.pose_poll_hz,
        )

    # -------------------------
    # lifecycle
    # -------------------------
    def start(self):
        self._stream.start()
        if self.cfg.state_json:
            print(f"[pose] streaming scale-preserving state from: {os.path.abspath(self.cfg.state_json)}")
        if self.cfg.pose_npy:
            print(f"[pose] legacy rigid-pose fallback: {os.path.abspath(self.cfg.pose_npy)}")
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
          if enabled -> read A_scene_asset_raw and write RuntimeMug_proxy world matrix
          else if freeze -> keep at last pose
        """
        if self.pose_sync_enabled:
            A_scene_asset_raw = self._stream.get_latest()
            if A_scene_asset_raw is None:
                return

            for i in range(self.num_envs):
                scene_path = self.cfg.scene_path_tpl.format(i=i)
                mug_path = self.cfg.mug_path_tpl.format(i=i)
                try:
                    T_world_scene = get_world_xf(stage, scene_path)
                    T_world_scene_placement = rigid_scene_placement(T_world_scene)
                    A_world_asset_raw = T_world_scene_placement @ np.asarray(
                        A_scene_asset_raw,
                        dtype=np.float64,
                    )
                    set_prim_world_matrix(stage, mug_path, A_world_asset_raw)
                    self._last_world_mug_pose_by_env[i] = A_world_asset_raw
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
