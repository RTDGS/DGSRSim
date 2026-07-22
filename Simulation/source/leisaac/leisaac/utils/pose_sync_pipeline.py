# -*- coding: utf-8 -*-
"""
leisaac/utils/pose_sync_pipeline.py

Pose Sync pipeline split-out:
- Poll a single-object or multi-object JSON state packet, with a legacy .npy pose fallback
- Maintain pose sync state:
    - pose_sync_enabled (default True)
    - freeze_when_sync_off
    - manual_override (None/True/False) for rule-grasp priority
- Apply every bound object state into USD each step:
    - separate rigid scene placement from background-asset scale
    - compute A_world_asset = T_world_scene_placement @ A_scene_asset_raw
    - resolve object IDs to USD prim paths through an explicit binding map
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional

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


SINGLE_STATE_SCHEMA = "dgsrsim.object_state.v1"
MULTI_STATE_SCHEMA = "dgsrsim.object_states.v1"
BINDINGS_SCHEMA = "dgsrsim.simulation_object_bindings.v1"


def _state_matrix(payload: Mapping[str, object]) -> np.ndarray:
    return validate_similarity_matrix(
        payload["A_scene_from_asset_raw"],
        "A_scene_from_asset_raw",
    )


def load_object_path_templates(path: str) -> Dict[str, str]:
    """Load object-ID to USD-prim-path-template bindings from JSON."""
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != BINDINGS_SCHEMA:
        raise ValueError(f"unsupported DGSRSim object-binding schema: {payload.get('schema')!r}")
    objects = payload.get("objects")
    if not isinstance(objects, dict):
        raise ValueError("object binding file must contain an 'objects' mapping")

    bindings: Dict[str, str] = {}
    for object_id, record in objects.items():
        if isinstance(record, str):
            path_template = record
        elif isinstance(record, dict):
            path_template = record.get("prim_path_template", "")
        else:
            path_template = ""
        object_id = str(object_id).strip()
        path_template = str(path_template).strip()
        if not object_id or not path_template:
            raise ValueError(f"invalid object binding for {object_id!r}")
        bindings[object_id] = path_template
    return bindings


class ObjectStateFileStream:
    """Poll scale-preserving single- or multi-object state packets."""

    def __init__(
        self,
        state_json: str,
        pose_npy: str,
        poll_hz: float = 60.0,
        default_object_id: str = "object",
    ):
        self.state_json = str(state_json)
        self.pose_npy = str(pose_npy)
        self.poll_hz = float(poll_hz)
        self.default_object_id = str(default_object_id)
        self._lock = threading.Lock()
        self._latest_states: Dict[str, np.ndarray] = {}
        self._latest_inactive_ids: set[str] = set()
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
            if self.default_object_id in self._latest_states:
                return self._latest_states[self.default_object_id].copy()
            if len(self._latest_states) == 1:
                return next(iter(self._latest_states.values())).copy()
            return None

    def get_latest_all(self) -> Dict[str, np.ndarray]:
        with self._lock:
            return {key: value.copy() for key, value in self._latest_states.items()}

    def get_latest_packet(self) -> tuple[Dict[str, np.ndarray], set[str]]:
        with self._lock:
            return (
                {key: value.copy() for key, value in self._latest_states.items()},
                set(self._latest_inactive_ids),
            )

    def _read_candidate(self) -> tuple[Dict[str, np.ndarray], set[str], float, str]:
        if self.state_json and os.path.exists(self.state_json):
            mtime = os.path.getmtime(self.state_json)
            payload = json.loads(Path(self.state_json).read_text(encoding="utf-8"))
            schema = payload.get("schema")
            if schema == SINGLE_STATE_SCHEMA:
                object_id = str(payload.get("object_id") or self.default_object_id)
                return {object_id: _state_matrix(payload)}, set(), mtime, self.state_json
            if schema == MULTI_STATE_SCHEMA:
                records = payload.get("objects")
                if not isinstance(records, dict):
                    raise ValueError("multi-object state packet requires an 'objects' mapping")
                states: Dict[str, np.ndarray] = {}
                inactive_ids: set[str] = set()
                for object_id, record in records.items():
                    if not isinstance(record, dict):
                        continue
                    normalized_id = str(object_id)
                    if bool(record.get("active", True)):
                        states[normalized_id] = _state_matrix(record)
                    else:
                        inactive_ids.add(normalized_id)
                return states, inactive_ids, mtime, self.state_json
            raise ValueError(f"unsupported DGSRSim object-state schema: {schema!r}")

        # A multi-object consumer can still read the legacy sibling packet.
        if self.state_json:
            legacy_json = str(Path(self.state_json).with_name("object_state.json"))
            if legacy_json != self.state_json and os.path.exists(legacy_json):
                mtime = os.path.getmtime(legacy_json)
                payload = json.loads(Path(legacy_json).read_text(encoding="utf-8"))
                if payload.get("schema") != SINGLE_STATE_SCHEMA:
                    raise ValueError("unsupported DGSRSim legacy object-state schema")
                object_id = str(payload.get("object_id") or self.default_object_id)
                return {object_id: _state_matrix(payload)}, set(), mtime, legacy_json

        if self.pose_npy and os.path.exists(self.pose_npy):
            mtime = os.path.getmtime(self.pose_npy)
            matrix = validate_similarity_matrix(
                np.load(self.pose_npy, allow_pickle=False).astype(np.float64).reshape(4, 4),
                "legacy_T_scene_from_target",
            )
            return {self.default_object_id: matrix}, set(), mtime, self.pose_npy
        return {}, set(), -1.0, ""

    def _store_candidate(
        self,
        states: Dict[str, np.ndarray],
        inactive_ids: set[str],
        mtime: float,
        source: str,
    ) -> bool:
        """Publish a newer valid packet, including a packet with no active objects."""
        if not source or not (
            mtime > self._latest_mtime or source != self._latest_source
        ):
            return False
        with self._lock:
            self._latest_states = states
            self._latest_inactive_ids = set(inactive_ids)
            self._latest_mtime = mtime
            self._latest_source = source
        return True

    def _loop(self):
        dt = 1.0 / max(self.poll_hz, 1e-6)
        while self._running:
            try:
                states, inactive_ids, mtime, source = self._read_candidate()
                self._store_candidate(states, inactive_ids, mtime, source)
            except Exception:
                pass
            time.sleep(dt)


@dataclass
class PoseSyncConfig:
    state_json: str = ""
    pose_npy: str = ""
    pose_poll_hz: float = 60.0
    default_object_id: str = "object"
    object_path_templates: Dict[str, str] = field(default_factory=dict)

    pose_sync_key: str = "M"
    pose_sync_freeze: bool = False

    # Scene and legacy single-object bindings.
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

        self._last_world_pose_by_env_object: Dict[tuple[int, str], np.ndarray] = {}
        self._unbound_object_ids: set[str] = set()
        self._active_object_ids: set[str] = set()
        self._lifecycle_manager = None

        self._object_path_templates = dict(cfg.object_path_templates)
        if not self._object_path_templates:
            self._object_path_templates[cfg.default_object_id] = cfg.mug_path_tpl

        self._stream = ObjectStateFileStream(
            cfg.state_json,
            cfg.pose_npy,
            poll_hz=cfg.pose_poll_hz,
            default_object_id=cfg.default_object_id,
        )

    # -------------------------
    # lifecycle
    # -------------------------
    def start(self):
        self._stream.start()
        if self.cfg.state_json:
            print(f"[pose] streaming scale-preserving states from: {os.path.abspath(self.cfg.state_json)}")
        if self.cfg.pose_npy:
            print(f"[pose] legacy rigid-pose fallback: {os.path.abspath(self.cfg.pose_npy)}")
        print(
            f"[pose-sync] initial=ON. toggle key='{str(self.cfg.pose_sync_key).upper()}'. "
            f"freeze_off={self.freeze_when_sync_off}"
        )
        print(f"[pose-sync] object bindings: {sorted(self._object_path_templates)}")
        if self.rule_grasp_mode:
            print("[pose-sync] rule-grasp: manual override via key has priority over agent requests.")

    def stop(self):
        self._stream.stop()

    def reset(self):
        """Call on env.reset() so cache doesn't carry across episodes."""
        self._last_world_pose_by_env_object.clear()
        # Do not forcibly clear manual_override; keep it as user intent

    def set_lifecycle_manager(self, manager) -> None:
        self._lifecycle_manager = manager

    def set_object_path_templates(self, bindings: Mapping[str, str]) -> None:
        self._object_path_templates = dict(bindings)
        self._active_object_ids.clear()
        self._last_world_pose_by_env_object.clear()
        self._unbound_object_ids.clear()

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
        Apply pose sync to all bound objects in every env_i:
          if enabled -> read each A_scene_asset_raw and write its bound prim
          else if freeze -> keep at last pose
        """
        if self.pose_sync_enabled:
            states, inactive_ids = self._stream.get_latest_packet()
            for object_id in sorted(inactive_ids):
                if self._lifecycle_manager is not None:
                    try:
                        self._lifecycle_manager.deactivate(stage, object_id)
                    except Exception as exc:
                        print(f"[pose-sync] failed to deactivate {object_id!r}: {exc}")
                self._active_object_ids.discard(object_id)
                for key in [key for key in self._last_world_pose_by_env_object if key[1] == object_id]:
                    self._last_world_pose_by_env_object.pop(key, None)
            if not states:
                return

            for i in range(self.num_envs):
                scene_path = self.cfg.scene_path_tpl.format(i=i)
                for object_id, A_scene_asset_raw in states.items():
                    path_template = self._object_path_templates.get(object_id)
                    if path_template is None:
                        if object_id not in self._unbound_object_ids:
                            print(f"[pose-sync] no simulation binding for object_id={object_id!r}; skipping")
                            self._unbound_object_ids.add(object_id)
                        continue
                    if object_id not in self._active_object_ids and self._lifecycle_manager is not None:
                        try:
                            self._lifecycle_manager.activate(stage, object_id)
                        except Exception as exc:
                            print(f"[pose-sync] failed to activate {object_id!r}: {exc}")
                            continue
                    self._active_object_ids.add(object_id)
                    object_path = path_template.format(i=i, object_id=object_id)
                    try:
                        T_world_scene = get_world_xf(stage, scene_path)
                        T_world_scene_placement = rigid_scene_placement(T_world_scene)
                        A_world_asset_raw = T_world_scene_placement @ np.asarray(
                            A_scene_asset_raw,
                            dtype=np.float64,
                        )
                        key = (i, object_id)
                        previous = self._last_world_pose_by_env_object.get(key)
                        if previous is not None and np.allclose(
                            previous, A_world_asset_raw, rtol=0.0, atol=1e-10
                        ):
                            continue
                        set_prim_world_matrix(stage, object_path, A_world_asset_raw)
                        self._last_world_pose_by_env_object[key] = A_world_asset_raw
                    except Exception:
                        pass
            return

        # sync OFF
        if self.freeze_when_sync_off:
            for (i, object_id), T_world_object in list(self._last_world_pose_by_env_object.items()):
                path_template = self._object_path_templates.get(object_id)
                if path_template is None:
                    continue
                object_path = path_template.format(i=i, object_id=object_id)
                try:
                    set_prim_world_matrix(stage, object_path, T_world_object)
                except Exception:
                    pass
