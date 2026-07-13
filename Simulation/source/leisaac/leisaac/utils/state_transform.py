"""Pure NumPy validation and composition for scale-preserving object states."""

from __future__ import annotations

import numpy as np


def validate_similarity_matrix(value: np.ndarray, name: str = "A") -> np.ndarray:
    """Validate a homogeneous transform with positive uniform scale."""
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} must be homogeneous")
    scale = float(np.cbrt(np.linalg.det(matrix[:3, :3])))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"{name} must have positive scale")
    rotation = matrix[:3, :3] / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} must contain a uniform-scale rotation")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"{name} rotation determinant must be +1")
    return matrix


def rigid_scene_placement(T_world_scene: np.ndarray) -> np.ndarray:
    """Remove the background asset's uniform scale from scene placement."""
    matrix = validate_similarity_matrix(T_world_scene, "T_world_scene")
    scale = float(np.cbrt(np.linalg.det(matrix[:3, :3])))
    placement = matrix.copy()
    placement[:3, :3] /= scale
    return placement
