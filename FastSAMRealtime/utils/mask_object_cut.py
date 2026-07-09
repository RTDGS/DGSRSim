# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Any, Dict, Tuple

import numpy as np


def cut_object_pcd_by_selected_mask(
    xyz_m: np.ndarray,
    rgb_u8: np.ndarray,
    pix_color: np.ndarray,
    masks_bool: Optional[np.ndarray],
    selected_idx: Optional[int],
    sample_round: str = "round",
    require_in_bounds: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    默认 sample_round="round"（比 floor 更贴近像素归属，减少边界偏差）
    """
    stats: Dict[str, Any] = {
        "ok": False,
        "reason": "",
        "points_in": int(xyz_m.shape[0]),
        "points_out": 0,
    }

    if selected_idx is None or masks_bool is None or masks_bool.shape[0] == 0:
        stats["reason"] = "no_selection_or_no_masks"
        return (
            np.zeros((0, 3), np.float32),
            np.zeros((0, 3), np.uint8),
            np.zeros((0,), np.int64),
            stats,
        )

    si = int(selected_idx)
    if not (0 <= si < masks_bool.shape[0]):
        stats["reason"] = "selected_idx_out_of_range"
        return (
            np.zeros((0, 3), np.float32),
            np.zeros((0, 3), np.uint8),
            np.zeros((0,), np.int64),
            stats,
        )

    mask = masks_bool[si]
    Hm, Wm = mask.shape[:2]

    u = pix_color[:, 0]
    v = pix_color[:, 1]

    if sample_round == "floor":
        ui = np.floor(u).astype(np.int32)
        vi = np.floor(v).astype(np.int32)
    else:
        ui = np.rint(u).astype(np.int32)
        vi = np.rint(v).astype(np.int32)

    finite = np.isfinite(u) & np.isfinite(v)
    inb = (ui >= 0) & (vi >= 0) & (ui < Wm) & (vi < Hm)

    if require_in_bounds:
        valid = finite & inb
    else:
        ui = np.clip(ui, 0, Wm - 1)
        vi = np.clip(vi, 0, Hm - 1)
        valid = finite

    hit = np.zeros((pix_color.shape[0],), dtype=bool)
    if np.any(valid):
        hit[valid] = mask[vi[valid], ui[valid]]

    idx = np.where(hit)[0].astype(np.int64)

    stats.update({
        "mask_hw": (int(Hm), int(Wm)),
        "finite": int(np.count_nonzero(finite)),
        "in_bounds": int(np.count_nonzero(inb)),
        "valid": int(np.count_nonzero(valid)),
        "hit": int(idx.size),
        "mask_fg_pixels": int(np.count_nonzero(mask)),
    })

    if idx.size == 0:
        stats["reason"] = "no_points_hit_mask"
        return (
            np.zeros((0, 3), np.float32),
            np.zeros((0, 3), np.uint8),
            idx,
            stats,
        )

    xyz_obj = xyz_m[idx]
    rgb_obj = rgb_u8[idx]
    stats["ok"] = True
    stats["points_out"] = int(idx.size)
    stats["reason"] = "ok"
    return xyz_obj, rgb_obj, idx, stats
