# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from rgb_segment_click_select_lib import overlay_mask
from utils.seg_selection_helpers import (
    pick_mask_by_click,
    build_instance_color_image,
    mask_bbox,
)


class RealTimeSegUI:
    def __init__(self, win: str = "RT Seg: click select | R reset | P save OBJ | G save GLOBAL | M match | Q quit"):
        self.win = win
        self._selected_idx: Optional[int] = None
        self._selected_point: Optional[Tuple[int, int]] = None
        self._W = None
        self._H = None
        self._last_key: int = 0

        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.win, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self._W is None or self._H is None:
            return

        W = int(self._W)
        H = int(self._H)

        if x < W:
            xx, yy = x, y
        else:
            xx, yy = x - W, y

        if 0 <= xx < W and 0 <= yy < H:
            self._selected_point = (int(xx), int(yy))
            print(f"[UI] click=({xx},{yy})")

    def reset(self):
        self._selected_idx = None
        self._selected_point = None

    def get_selected_idx(self) -> Optional[int]:
        return None if self._selected_idx is None else int(self._selected_idx)

    def pop_last_key(self) -> int:
        k = int(self._last_key)
        self._last_key = 0
        return k

    def render(self, color_bgr: np.ndarray, masks_bool: Optional[np.ndarray], inst_seed: int = 12345) -> str:
        H, W = color_bgr.shape[:2]
        self._H, self._W = H, W

        left = color_bgr.copy()
        right = np.zeros_like(left)
        mi = None

        if masks_bool is not None and masks_bool.shape[0] > 0:
            right = build_instance_color_image(masks_bool, seed=inst_seed)

            if self._selected_point is not None:
                px, py = self._selected_point
                if 0 <= px < W and 0 <= py < H:
                    mi = pick_mask_by_click(masks_bool, px, py)
                    self._selected_idx = mi

            if self._selected_idx is not None:
                mi = int(self._selected_idx)
                if 0 <= mi < masks_bool.shape[0]:
                    left = overlay_mask(left, masks_bool[mi], alpha=0.45)
                    bb = mask_bbox(masks_bool[mi])
                    if bb is not None:
                        x0, y0, x1, y1 = bb
                        cv2.rectangle(left, (x0, y0), (x1, y1), (0, 255, 0), 2)
                        cv2.putText(
                            left, f"Selected #{mi}", (x0, max(0, y0 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA
                        )

        show = np.hstack([left, right])
        cv2.putText(
            show,
            f"masks={(0 if masks_bool is None else masks_bool.shape[0])}  selected={mi}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(self.win, show)

        k = cv2.waitKey(1) & 0xFF
        self._last_key = k

        if k in (ord("q"), 27):
            return "quit"
        if k == ord("r"):
            self.reset()
        return "continue"
