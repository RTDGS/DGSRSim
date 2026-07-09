# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import cv2


def pick_mask_by_click(masks_bool: np.ndarray, x: int, y: int) -> Optional[int]:
    if masks_bool is None or masks_bool.shape[0] == 0:
        return None
    hit = masks_bool[:, y, x]
    idxs = np.where(hit)[0]
    if idxs.size == 0:
        return None
    if idxs.size == 1:
        return int(idxs[0])
    areas = masks_bool[idxs].reshape(idxs.size, -1).sum(axis=1)
    return int(idxs[np.argmin(areas)])


def build_instance_color_image(masks_bool: np.ndarray, seed: int = 12345) -> np.ndarray:
    if masks_bool is None or masks_bool.ndim != 3:
        raise ValueError("masks_bool must be (N,H,W)")
    N, H, W = masks_bool.shape
    inst = np.zeros((H, W, 3), dtype=np.uint8)
    rng = np.random.default_rng(int(seed))
    colors = rng.integers(0, 255, size=(N, 3), dtype=np.uint8)
    for i in range(N):
        inst[masks_bool[i]] = colors[i]
    return inst


def mask_bbox(mask_bool: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask_bool)
    if ys.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1, y1


def ensure_masks_match_color(masks_bool: Optional[np.ndarray], color_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    强制把 masks 变成与 color 同分辨率 (H,W)。
    - 若已经一致：原样返回
    - 若不一致：对每个实例 mask 用 nearest resize 到 (W,H)
    """
    if masks_bool is None:
        return None
    if masks_bool.ndim != 3 or masks_bool.shape[0] == 0:
        return masks_bool
    Hc, Wc = color_bgr.shape[:2]
    N, Hm, Wm = masks_bool.shape
    if (Hm, Wm) == (Hc, Wc):
        return masks_bool

    out = np.zeros((N, Hc, Wc), dtype=bool)
    for i in range(N):
        m = masks_bool[i].astype(np.uint8) * 255
        m2 = cv2.resize(m, (Wc, Hc), interpolation=cv2.INTER_NEAREST)
        out[i] = m2 > 0
    return out
