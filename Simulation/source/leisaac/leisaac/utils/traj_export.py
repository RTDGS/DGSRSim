# -*- coding: utf-8 -*-
import os
import time
import numpy as np


def export_joint_traj_npz(
    out_dir: str,
    joint_names: list[str],
    q_traj: np.ndarray,  # [T, 6] rad
    dt: float,
    task_name: str = "macro",
    success: bool = True,
) -> str:
    """Export joint trajectory to an atomic npz file."""
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tmp_path = os.path.join(out_dir, f".tmp_{stamp}_{task_name}.npz")
    out_path = os.path.join(out_dir, f"{stamp}_{task_name}.npz")

    np.savez_compressed(
        tmp_path,
        joint_names=np.array(joint_names, dtype=object),
        q=np.asarray(q_traj, dtype=np.float32),
        dt=np.float32(dt),
        success=np.bool_(success),
        task=np.array(task_name, dtype=object),
        created_at=np.array(stamp, dtype=object),
    )
    os.replace(tmp_path, out_path)
    return out_path
