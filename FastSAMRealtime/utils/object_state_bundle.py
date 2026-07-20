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
        objects[object_id] = dict(state_payload, active=True)
        bundle["timestamp_unix"] = float(state_payload["timestamp_unix"])
        _atomic_save_json(target_path, bundle)
