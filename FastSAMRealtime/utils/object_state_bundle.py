"""Atomic multi-object state-bundle updates for DGSRSim producers."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


MULTI_STATE_SCHEMA = "dgsrsim.object_states.v1"


def _atomic_save_json(path: str, payload: Dict[str, Any]) -> None:
    target_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    tmp_path = target_path + ".tmp"
    Path(tmp_path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, target_path)


@contextmanager
def exclusive_file_lock(path: str, timeout_s: float = 3.0) -> Iterator[None]:
    """Serialize read-modify-write updates from concurrent object producers."""
    target_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    lock_path = target_path + ".lock"
    deadline = time.time() + timeout_s
    fd: Optional[int] = None
    while fd is None:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.time() >= deadline:
                raise TimeoutError(f"timed out waiting for state-bundle lock: {lock_path}")
            time.sleep(0.02)
    try:
        os.write(fd, f"pid={os.getpid()}\n".encode("ascii"))
        yield
    finally:
        os.close(fd)
        try:
            os.unlink(lock_path)
        except FileNotFoundError:
            pass


def update_object_state_bundle(
    path: str,
    object_id: str,
    state_payload: Dict[str, Any],
    *,
    active: bool = True,
) -> None:
    """Atomically replace one object entry while retaining the other objects."""
    target_path = os.path.abspath(path)
    with exclusive_file_lock(target_path):
        if os.path.exists(target_path):
            bundle = json.loads(Path(target_path).read_text(encoding="utf-8"))
            if bundle.get("schema") != MULTI_STATE_SCHEMA:
                raise ValueError(f"unsupported object-state bundle schema in {target_path}")
        else:
            bundle = {
                "schema": MULTI_STATE_SCHEMA,
                "timestamp_unix": 0.0,
                "objects": {},
            }
        objects = bundle.setdefault("objects", {})
        if not isinstance(objects, dict):
            raise ValueError("object-state bundle 'objects' field must be a mapping")
        objects[object_id] = dict(state_payload, active=bool(active))
        bundle["timestamp_unix"] = float(state_payload["timestamp_unix"])
        _atomic_save_json(target_path, bundle)


def set_object_active(
    path: str,
    object_id: str,
    active: bool,
    *,
    timestamp_unix: float | None = None,
) -> None:
    """Activate or deactivate one retained object record atomically.

    Deactivation may create a matrix-free tombstone. Reactivation requires a
    previously retained state matrix so that the consumer never invents a pose.
    """
    target_path = os.path.abspath(path)
    timestamp = time.time() if timestamp_unix is None else float(timestamp_unix)
    with exclusive_file_lock(target_path):
        if os.path.exists(target_path):
            bundle = json.loads(Path(target_path).read_text(encoding="utf-8"))
            if bundle.get("schema") != MULTI_STATE_SCHEMA:
                raise ValueError(f"unsupported object-state bundle schema in {target_path}")
        else:
            bundle = {"schema": MULTI_STATE_SCHEMA, "timestamp_unix": 0.0, "objects": {}}
        objects = bundle.setdefault("objects", {})
        if not isinstance(objects, dict):
            raise ValueError("object-state bundle 'objects' field must be a mapping")
        record = dict(objects.get(object_id, {}))
        if active and "A_scene_from_asset_raw" not in record:
            raise ValueError(
                f"cannot activate {object_id!r} without a retained A_scene_from_asset_raw state"
            )
        record.update(
            {
                "object_id": object_id,
                "active": bool(active),
                "timestamp_unix": timestamp,
            }
        )
        objects[object_id] = record
        bundle["timestamp_unix"] = timestamp
        _atomic_save_json(target_path, bundle)


def deactivate_object(path: str, object_id: str, *, timestamp_unix: float | None = None) -> None:
    """Publish an inactive tombstone that removes the object from the live scene."""
    set_object_active(path, object_id, False, timestamp_unix=timestamp_unix)
