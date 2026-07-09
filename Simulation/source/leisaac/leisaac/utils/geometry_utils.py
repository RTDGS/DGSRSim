# -*- coding: utf-8 -*-
"""
leisaac/utils/geometry_utils.py

Geometry / pose helper utilities used across teleop scripts.
"""

from __future__ import annotations

import math
from typing import Tuple


def quat_wxyz_from_euler_deg(roll_deg: float, pitch_deg: float, yaw_deg: float) -> Tuple[float, float, float, float]:
    """Convert Euler angles in degrees (roll, pitch, yaw) to quaternion (w, x, y, z)."""
    cr = math.cos(math.radians(roll_deg) * 0.5)
    sr = math.sin(math.radians(roll_deg) * 0.5)
    cp = math.cos(math.radians(pitch_deg) * 0.5)
    sp = math.sin(math.radians(pitch_deg) * 0.5)
    cy = math.cos(math.radians(yaw_deg) * 0.5)
    sy = math.sin(math.radians(yaw_deg) * 0.5)

    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = cy * sp * cr + sy * cp * sr
    z = sy * cp * cr - cy * sp * sr
    return (w, x, y, z)
