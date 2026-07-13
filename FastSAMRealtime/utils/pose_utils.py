# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple
import numpy as np


def validate_transform_4x4(T: np.ndarray, name: str = "T") -> np.ndarray:
    """Return a finite homogeneous transform after checking its rigid rotation."""
    value = np.asarray(T, dtype=np.float64)
    if value.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} must be a homogeneous transform with last row [0, 0, 0, 1]")

    R = value[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(R), 1.0, atol=1e-5):
        raise ValueError(f"{name} rotation determinant must be +1")
    return value


def validate_similarity_4x4(T: np.ndarray, name: str = "A") -> np.ndarray:
    """Return a finite, orientation-preserving uniform-scale transform."""
    value = np.asarray(T, dtype=np.float64)
    if value.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} must be homogeneous")

    linear = value[:3, :3]
    scale = float(np.cbrt(np.linalg.det(linear)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"{name} must have a positive uniform scale")
    rotation = linear / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} linear part is not a uniform-scale rotation")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"{name} rotation determinant must be +1")
    return value


def similarity_scale(T: np.ndarray, name: str = "A") -> float:
    """Extract the positive uniform scale from a validated similarity transform."""
    value = validate_similarity_4x4(T, name)
    return float(np.cbrt(np.linalg.det(value[:3, :3])))


def make_target_normalization(
    scale: float,
    source_center: np.ndarray,
    target_center: np.ndarray,
) -> np.ndarray:
    """Build q_normalized = c_source + scale * (q_raw - c_target)."""
    s = float(scale)
    if not np.isfinite(s) or s <= 0.0:
        raise ValueError("scale must be finite and positive")
    c_source = np.asarray(source_center, dtype=np.float64).reshape(3)
    c_target = np.asarray(target_center, dtype=np.float64).reshape(3)
    if not np.isfinite(c_source).all() or not np.isfinite(c_target).all():
        raise ValueError("normalization centers must be finite")

    value = np.eye(4, dtype=np.float64)
    value[:3, :3] *= s
    value[:3, 3] = c_source - s * c_target
    return validate_similarity_4x4(value, "A_normalized_target_from_asset_raw")


def load_transform_4x4(path: str | Path, key: str = "T_scene_from_camera") -> np.ndarray:
    """Load a calibrated transform from NPY, JSON, or whitespace-delimited text."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Transform file not found: {source}")

    suffix = source.suffix.lower()
    if suffix == ".npy":
        value = np.load(source, allow_pickle=False)
    elif suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        value = payload[key] if isinstance(payload, dict) and key in payload else payload
    else:
        value = np.loadtxt(source, dtype=np.float64)
    return validate_transform_4x4(value, key)


def compose_target_to_scene(
    T_scene_from_camera: np.ndarray,
    T_target_from_camera: np.ndarray,
) -> np.ndarray:
    """Compose target-to-scene pose from camera calibration and registration output."""
    T_sc = validate_transform_4x4(T_scene_from_camera, "T_scene_from_camera")
    T_tc = validate_transform_4x4(T_target_from_camera, "T_target_from_camera")
    T_camera_from_target = np.linalg.inv(T_tc)
    return validate_transform_4x4(T_sc @ T_camera_from_target, "T_scene_from_target")


def compose_asset_to_scene(
    T_scene_from_camera: np.ndarray,
    T_normalized_target_from_camera: np.ndarray,
    A_normalized_target_from_asset_raw: np.ndarray,
) -> np.ndarray:
    """Compose the raw-asset-to-scene similarity without dropping scale."""
    T_sc = validate_transform_4x4(T_scene_from_camera, "T_scene_from_camera")
    T_nc = validate_transform_4x4(
        T_normalized_target_from_camera,
        "T_normalized_target_from_camera",
    )
    A_na = validate_similarity_4x4(
        A_normalized_target_from_asset_raw,
        "A_normalized_target_from_asset_raw",
    )
    return validate_similarity_4x4(
        T_sc @ np.linalg.inv(T_nc) @ A_na,
        "A_scene_from_asset_raw",
    )


def rot_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    tr = float(np.trace(R))
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S
    q = np.array([w, x, y, z], dtype=np.float64)
    n = np.linalg.norm(q)
    if n > 1e-12:
        q /= n
    return q


def rot_to_yaw_pitch_roll(R: np.ndarray) -> Tuple[float, float, float]:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    sy = -R[2, 0]
    sy = float(np.clip(sy, -1.0, 1.0))
    pitch = np.arcsin(sy)
    if abs(np.cos(pitch)) < 1e-9:
        yaw = np.arctan2(-R[0, 1], R[1, 1])
        roll = 0.0
    else:
        yaw = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
    return float(yaw), float(pitch), float(roll)


def pretty_pose(T: np.ndarray) -> str:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]
    q = rot_to_quat_wxyz(R)
    yaw, pitch, roll = rot_to_yaw_pitch_roll(R)
    s = []
    s.append(f"t(m) = [{t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f}]")
    s.append("R =\n" + np.array2string(R, precision=4, suppress_small=True))
    s.append(f"quat(wxyz) = [{q[0]:+.6f}, {q[1]:+.6f}, {q[2]:+.6f}, {q[3]:+.6f}]")
    s.append(f"ypr(rad)   = [yaw={yaw:+.4f}, pitch={pitch:+.4f}, roll={roll:+.4f}]")
    return "\n".join(s)


def pretty_similarity(T: np.ndarray) -> str:
    """Format a similarity transform as scale plus its rigid component."""
    value = validate_similarity_4x4(T)
    scale = similarity_scale(value)
    rigid = value.copy()
    rigid[:3, :3] /= scale
    return f"uniform_scale = {scale:.9f}\n" + pretty_pose(rigid)
