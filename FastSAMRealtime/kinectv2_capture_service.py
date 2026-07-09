# -*- coding: utf-8 -*-
"""
kinectv2_capture_service.py

Goal
----
A small, import-safe Kinect v2 RGBD capture "service" module that exposes:
  - start_capture(...)  : open Kinect and begin capturing (no side effects on import)
  - stop_capture()      : close Kinect and release resources
  - get_latest_frame()  : obtain latest (color_bgr, depth_u16) for external callers
  - show_preview(...)   : optional preview helper (OpenCV)

Design notes
------------
- Kinect runtime must be initialized ONCE (not per frame).
- Capture runs in a background thread; external code can pull latest frames at any time.
- Depth is uint16 (mm), shape (424, 512).
- Color is BGR uint8, shape (1080, 1920, 3).

Dependencies (live)
-------------------
  pip install comtypes pykinect2 opencv-python numpy
Requires Kinect for Windows SDK 2.0 installed and Kinect v2 connected.

Usage
-----
from kinectv2_capture_service import start_capture, stop_capture, get_latest_frame, show_preview

start_capture(preview=True, fps=10)
while True:
    color, depth = get_latest_frame(copy=True)
    if color is None: continue
    # do something ...
    if show_preview(color, depth, win="Kinect v2", wait=1) in ("q", "esc"):
        break
stop_capture()
"""

from __future__ import annotations

import sys
import time
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2


# =========================
# Errors / dependency check
# =========================

def _live_dep_error(e: Exception) -> RuntimeError:
    exe = sys.executable
    msg = (
        "Kinect v2 live capture requires 'pykinect2' and 'comtypes' installed in the CURRENT Python environment.\n"
        f"Current python: {exe}\n\n"
        "Install into THIS env:\n"
        f'  "{exe}" -m pip install --upgrade pip\n'
        f'  "{exe}" -m pip install comtypes\n'
        f'  "{exe}" -m pip install pykinect2\n'
        f'  "{exe}" -m pip install opencv-python numpy\n\n'
        "Also ensure:\n"
        "  - Kinect for Windows SDK 2.0 installed\n"
        "  - Kinect v2 connected and works in Kinect Configuration Verifier / Kinect Studio\n\n"
        f"Import error: {repr(e)}"
    )
    return RuntimeError(msg)


def is_kinectv2_available() -> bool:
    try:
        import comtypes  # noqa: F401
        from pykinect2 import PyKinectV2  # noqa: F401
        from pykinect2.PyKinectRuntime import PyKinectRuntime  # noqa: F401
        return True
    except Exception:
        return False


# =========================
# Preview helper (optional)
# =========================

def _depth_to_vis_bgr(
    depth_u16: np.ndarray,
    depth_trunc_mm: int = 4000,
    percentile: Tuple[float, float] = (1.0, 99.0),
    use_jet: bool = True,
) -> np.ndarray:
    d = depth_u16.astype(np.uint16, copy=False)

    valid = (d > 0)
    if depth_trunc_mm and depth_trunc_mm > 0:
        valid = valid & (d <= int(depth_trunc_mm))

    depth8 = np.zeros_like(d, dtype=np.uint8)
    if np.any(valid):
        dv = d[valid].astype(np.float32)
        vmin = float(np.percentile(dv, float(percentile[0])))
        vmax = float(np.percentile(dv, float(percentile[1])))
        if vmax <= vmin + 1e-6:
            depth8[valid] = 255
        else:
            x = (dv - vmin) / (vmax - vmin)
            x = np.clip(x, 0.0, 1.0)
            depth8[valid] = (x * 255.0).astype(np.uint8)

    if use_jet:
        vis = cv2.applyColorMap(depth8, cv2.COLORMAP_JET)
        vis[~valid] = (0, 0, 0)
        return vis
    return cv2.cvtColor(depth8, cv2.COLOR_GRAY2BGR)


def show_preview(
    color_bgr: np.ndarray,
    depth_u16: np.ndarray,
    win: str = "Kinect v2 RGBD (Color | Depth)",
    resize_to_depth: bool = True,
    depth_trunc_mm: int = 4000,
    percentile: Tuple[float, float] = (1.0, 99.0),
    use_jet: bool = True,
    wait: int = 1,
) -> Optional[str]:
    """
    Returns:
      "q" or "esc" if user requested exit, else None.
    """
    depth_show = _depth_to_vis_bgr(depth_u16, depth_trunc_mm=depth_trunc_mm, percentile=percentile, use_jet=use_jet)

    if resize_to_depth:
        dh, dw = depth_u16.shape
        color_show = cv2.resize(color_bgr, (dw, dh), interpolation=cv2.INTER_AREA)
    else:
        color_show = color_bgr

    if color_show.shape[0] != depth_show.shape[0]:
        depth_show = cv2.resize(depth_show, (color_show.shape[1], color_show.shape[0]), interpolation=cv2.INTER_NEAREST)

    show = np.hstack([color_show, depth_show])
    cv2.putText(show, "Color", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(show, "Depth(JET)", (color_show.shape[1] + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow(win, show)
    k = cv2.waitKey(int(wait)) & 0xFF
    if k in (ord("q"), 27):
        return "q" if k == ord("q") else "esc"
    return None


# =========================
# Capture service
# =========================

@dataclass
class KinectV2Config:
    fps: float = 0.0           # 0 => no throttle
    preview: bool = False      # preview handled in thread if True
    preview_win: str = "Kinect v2 RGBD (Color | Depth)"
    depth_trunc_mm: int = 4000
    percentile: Tuple[float, float] = (1.0, 99.0)
    use_jet: bool = True
    resize_color_to_depth: bool = True


class KinectV2CaptureService:
    """
    Threaded capture service. External users should call module-level
    start_capture / stop_capture / get_latest_frame.
    """
    def __init__(self, cfg: KinectV2Config):
        self.cfg = cfg

        try:
            from pykinect2 import PyKinectV2  # type: ignore
            from pykinect2.PyKinectRuntime import PyKinectRuntime  # type: ignore
        except Exception as e:
            raise _live_dep_error(e)

        self.PyKinectV2 = PyKinectV2
        self.PyKinectRuntime = PyKinectRuntime

        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._kinect = None
        self._color_h = 0
        self._color_w = 0
        self._depth_h = 0
        self._depth_w = 0

        self._last_color_bgr: Optional[np.ndarray] = None
        self._last_depth_u16: Optional[np.ndarray] = None
        self._last_ts: float = 0.0

    def start(self):
        if self._running:
            return

        # init runtime ONCE
        self._kinect = self.PyKinectRuntime(
            self.PyKinectV2.FrameSourceTypes_Color |
            self.PyKinectV2.FrameSourceTypes_Depth
        )
        self._depth_h = int(self._kinect.depth_frame_desc.Height)
        self._depth_w = int(self._kinect.depth_frame_desc.Width)
        self._color_h = int(self._kinect.color_frame_desc.Height)
        self._color_w = int(self._kinect.color_frame_desc.Width)

        self._running = True
        self._thread = threading.Thread(target=self._loop, name="KinectV2CaptureThread", daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        try:
            if self._kinect is not None:
                self._kinect.close()
        except Exception:
            pass
        finally:
            self._kinect = None

        if self.cfg.preview:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    def get_latest(self, copy: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        with self._lock:
            c = self._last_color_bgr
            d = self._last_depth_u16
            ts = float(self._last_ts)

            if not copy:
                return c, d, ts

            c2 = None if c is None else c.copy()
            d2 = None if d is None else d.copy()
            return c2, d2, ts

    def _loop(self):
        period = (1.0 / self.cfg.fps) if (self.cfg.fps and self.cfg.fps > 0) else 0.0
        next_t = time.perf_counter()

        while self._running:
            if period > 0:
                now = time.perf_counter()
                if now < next_t:
                    time.sleep(min(0.002, next_t - now))
                    continue
                next_t += period

            updated = False

            try:
                if self._kinect is None:
                    time.sleep(0.01)
                    continue

                if self._kinect.has_new_color_frame():
                    cf = self._kinect.get_last_color_frame()  # BGRA flat
                    color_bgra = np.asarray(cf).reshape((self._color_h, self._color_w, 4)).astype(np.uint8, copy=False)
                    color_bgr = cv2.cvtColor(color_bgra, cv2.COLOR_BGRA2BGR)
                    with self._lock:
                        self._last_color_bgr = color_bgr
                        updated = True

                if self._kinect.has_new_depth_frame():
                    df = self._kinect.get_last_depth_frame()
                    depth = np.frombuffer(df, dtype=np.uint16, count=self._depth_h * self._depth_w).reshape((self._depth_h, self._depth_w))
                    depth_u16 = np.ascontiguousarray(depth)
                    with self._lock:
                        self._last_depth_u16 = depth_u16
                        updated = True

                if updated:
                    with self._lock:
                        self._last_ts = time.time()
                        c = self._last_color_bgr
                        d = self._last_depth_u16

                    if self.cfg.preview and (c is not None) and (d is not None):
                        key = show_preview(
                            c, d,
                            win=self.cfg.preview_win,
                            resize_to_depth=self.cfg.resize_color_to_depth,
                            depth_trunc_mm=self.cfg.depth_trunc_mm,
                            percentile=self.cfg.percentile,
                            use_jet=self.cfg.use_jet,
                            wait=1,
                        )
                        if key in ("q", "esc"):
                            self._running = False
                            break
                else:
                    # still pump UI if preview is on
                    if self.cfg.preview:
                        k = cv2.waitKey(1) & 0xFF
                        if k in (ord("q"), 27):
                            self._running = False
                            break
                    else:
                        time.sleep(0.001)

            except Exception:
                # avoid crashing the thread; brief backoff
                time.sleep(0.01)


# =========================
# Module-level API (what you requested)
# =========================

_service: Optional[KinectV2CaptureService] = None


def start_capture(
    fps: float = 0.0,
    preview: bool = False,
    preview_win: str = "Kinect v2 RGBD (Color | Depth)",
    depth_trunc_mm: int = 4000,
    percentile: Tuple[float, float] = (1.0, 99.0),
    use_jet: bool = True,
    resize_color_to_depth: bool = True,
) -> None:
    """
    Open Kinect v2 and start capturing frames in a background thread.
    Safe to call multiple times; subsequent calls are no-ops if already running.
    """
    global _service
    if _service is not None:
        # already created; ensure running
        _service.start()
        return

    cfg = KinectV2Config(
        fps=float(fps),
        preview=bool(preview),
        preview_win=str(preview_win),
        depth_trunc_mm=int(depth_trunc_mm),
        percentile=percentile,
        use_jet=bool(use_jet),
        resize_color_to_depth=bool(resize_color_to_depth),
    )
    _service = KinectV2CaptureService(cfg)
    _service.start()


def stop_capture() -> None:
    """
    Stop capturing and release Kinect resources.
    Safe to call even if not started.
    """
    global _service
    if _service is None:
        return
    try:
        _service.stop()
    finally:
        _service = None


def get_latest_frame(copy: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    External callers use this to fetch the latest frame.

    Returns:
      (color_bgr, depth_u16)
      - color_bgr: uint8 BGR (H=1080,W=1920,3)
      - depth_u16: uint16 depth in mm (H=424,W=512)

    If no frame is available yet, returns (None, None).

    copy=True is safer if you will keep the arrays beyond the current iteration.
    """
    if _service is None:
        return None, None
    c, d, _ts = _service.get_latest(copy=bool(copy))
    return c, d


def get_latest_frame_with_timestamp(copy: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    """
    Same as get_latest_frame, but also returns timestamp_unix (float).
    """
    if _service is None:
        return None, None, 0.0
    return _service.get_latest(copy=bool(copy))


# =========================
# Demo main (optional)
# =========================

if __name__ == "__main__":
    # Adjustable parameters
    PREVIEW = True
    FPS = 10.0

    if not is_kinectv2_available():
        raise RuntimeError(
            "Kinect v2 dependencies not available in this Python env. "
            "Call is_kinectv2_available() / install pykinect2+comtypes."
        )

    start_capture(preview=PREVIEW, fps=FPS)

    try:
        t0 = time.time()
        n = 0
        while True:
            color, depth = get_latest_frame(copy=False)
            if color is None or depth is None:
                time.sleep(0.005)
                continue

            # place your external processing here
            n += 1
            if n % 60 == 0:
                dt = max(time.time() - t0, 1e-6)
                print(f"[RUN] frames={n} fps~{n/dt:.2f} color={tuple(color.shape)} depth={tuple(depth.shape)}")

            # If preview=False, allow exit via keyboard here (optional)
            if not PREVIEW:
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break

    finally:
        stop_capture()
