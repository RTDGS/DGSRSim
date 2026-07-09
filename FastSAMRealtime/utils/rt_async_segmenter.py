# -*- coding: utf-8 -*-
from __future__ import annotations

import time
import threading
from typing import Optional, Tuple

import numpy as np

from rgb_segment_click_select_lib import (
    load_fastsam_model,
    unload_fastsam_model,
    infer_instance_masks,
)


class AsyncSegmenter:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        img_size: int = 1024,
        conf: float = 0.85,
        iou: float = 0.05,
        min_area: int = 460,
    ):
        self.model_path = model_path
        self.device = device
        self.img_size = img_size
        self.conf = conf
        self.iou = iou
        self.min_area = min_area

        self._lock = threading.Lock()
        self._latest_color: Optional[np.ndarray] = None
        self._latest_ts: float = 0.0

        self._res_lock = threading.Lock()
        self._latest_masks: Optional[np.ndarray] = None
        self._latest_res_ts: float = 0.0

        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        load_fastsam_model(self.model_path, device=self.device)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        unload_fastsam_model()

    def submit(self, color_bgr: np.ndarray, ts: float):
        with self._lock:
            self._latest_color = color_bgr
            self._latest_ts = ts

    def get_latest_result(self) -> Tuple[Optional[np.ndarray], float]:
        with self._res_lock:
            return self._latest_masks, float(self._latest_res_ts)

    def _loop(self):
        last_used_ts = -1.0
        while self._running:
            with self._lock:
                color = self._latest_color
                ts = self._latest_ts

            if color is None or ts <= last_used_ts:
                time.sleep(0.002)
                continue

            last_used_ts = ts
            try:
                masks = infer_instance_masks(
                    color,
                    img_size=self.img_size,
                    conf=self.conf,
                    iou=self.iou,
                    min_area=self.min_area,
                )
            except Exception as e:
                print("[Seg] infer error:", repr(e))
                time.sleep(0.01)
                continue

            with self._res_lock:
                self._latest_masks = masks
                self._latest_res_ts = ts
