# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Tuple
import numpy as np


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
