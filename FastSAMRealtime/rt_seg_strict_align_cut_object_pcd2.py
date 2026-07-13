# -*- coding: utf-8 -*-
"""
Fix goals (no change to your high-level logic):
1) 保证 masks 的分辨率与 color 完全一致（FastSAM/resize/letterbox 造成的 mismatch 会被强制修正）
2) 兼容两种 depth 输入：
   A) 原生 depth (424x512, uint16) -> 使用 MapDepthFrameToCameraSpace + MapDepthFrameToColorSpace（你原来的路径）
   B) depth 已经对齐到 color 分辨率 (与 color 同 HxW, uint16) -> 使用 MapColorFrameToCameraSpace（正确的路径）
      - 如果当前 PyKinect2 fork 不支持 MapColorFrameToCameraSpace，会给出清晰报错
3) PyKinectRuntime 初始化为 Color|Depth，避免 mapper 缺 stream 导致行为不一致
4) 增加“投影一致性”保护：强制 clip / finite / in-bounds，避免异常点扰动

你只需要用这个脚本替换你当前脚本即可。
"""

import os
import time
import threading
import ctypes
import copy
from dataclasses import dataclass
from typing import Optional, Tuple, Any, Dict

import cv2
import numpy as np
import open3d as o3d

from kinectv2_capture_service import start_capture, stop_capture, get_latest_frame
from rgb_segment_click_select_lib import (
    load_fastsam_model,
    unload_fastsam_model,
    infer_instance_masks,
    overlay_mask,
)

# ============================================================
# 0) Pose utilities
# ============================================================

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


# ============================================================
# 1) Seg selection helpers
# ============================================================

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


# ============================================================
# 2) Async segmenter
# ============================================================

class AsyncSegmenter:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        img_size: int = 1024,
        conf: float = 0.4,
        iou: float = 0.9,
        min_area: int = 80,
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


# ============================================================
# 3) UI
# ============================================================

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


# ============================================================
# 4) Strict alignment (CoordinateMapper)
# ============================================================

def get_coordinate_mapper(runtime) -> Any:
    for name in ("coordinate_mapper", "_coordinate_mapper"):
        if hasattr(runtime, name):
            cm = getattr(runtime, name)
            if cm is not None:
                return cm

    sensor_candidates = (
        "_sensor", "_kinect", "_kinect_sensor",
        "_KinectRuntime__kinect", "_PyKinectRuntime__kinect"
    )
    sensor = None
    for n in sensor_candidates:
        if hasattr(runtime, n):
            sensor = getattr(runtime, n)
            if sensor is not None:
                break

    if sensor is None:
        raise RuntimeError("Cannot find underlying Kinect sensor COM object from PyKinectRuntime.")
    if hasattr(sensor, "CoordinateMapper"):
        cm = sensor.CoordinateMapper
        if cm is not None:
            return cm
    raise RuntimeError("Found sensor object but no CoordinateMapper property.")


def bilinear_sample_bgr(color_bgr: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    h, w = color_bgr.shape[:2]

    xs = xs.astype(np.float32, copy=False)
    ys = ys.astype(np.float32, copy=False)

    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    valid = (
        (x0 >= 0) & (y0 >= 0) &
        (x1 < w) & (y1 < h) &
        np.isfinite(xs) & np.isfinite(ys)
    )

    out = np.zeros((xs.shape[0], 3), dtype=np.uint8)
    if not np.any(valid):
        return out

    xv = xs[valid]
    yv = ys[valid]
    x0v, y0v, x1v, y1v = x0[valid], y0[valid], x1[valid], y1[valid]

    wa = (x1v - xv) * (y1v - yv)
    wb = (xv - x0v) * (y1v - yv)
    wc = (x1v - xv) * (yv - y0v)
    wd = (xv - x0v) * (yv - y0v)

    Ia = color_bgr[y0v, x0v].astype(np.float32)
    Ib = color_bgr[y0v, x1v].astype(np.float32)
    Ic = color_bgr[y1v, x0v].astype(np.float32)
    Id = color_bgr[y1v, x1v].astype(np.float32)

    col = Ia * wa[:, None] + Ib * wb[:, None] + Ic * wc[:, None] + Id * wd[:, None]
    out[valid] = np.clip(col, 0, 255).astype(np.uint8)
    return out


def _ctypes_as_array_of_structs(ctypes_arr, field_names: Optional[Tuple[str, ...]], elem_dim: int) -> np.ndarray:
    """
    ctypes array -> numpy array
    - 若 dtype.names 存在，用字段取值；否则按 float32 view reshape
    """
    a = np.ctypeslib.as_array(ctypes_arr)
    if a.dtype.names and field_names is not None:
        cols = [a[n].astype(np.float32, copy=False) for n in field_names]
        return np.stack(cols, axis=1).astype(np.float32, copy=False)
    return a.view(np.float32).reshape(-1, elem_dim).astype(np.float32, copy=False)


def align_and_build_pointcloud_strict_with_pixcolor(
    color_bgr: np.ndarray,
    depth_u16: np.ndarray,
    cm: Any,
    PyKinectV2: Any,
    depth_trunc_m: float = 4.0,
    flip_y: bool = False,
    flip_z: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    两种输入深度的正确处理：

    (A) 原生 depth (通常 424x512)：
        MapDepthFrameToCameraSpace + MapDepthFrameToColorSpace -> (xyz, u,v)

    (B) depth 已对齐到 color 分辨率 (Hc x Wc)：
        使用 MapColorFrameToCameraSpace，把每个 color 像素+depth 映射到 CameraSpace
        这时 u,v 直接就是像素网格。
    """
    if depth_u16.dtype != np.uint16 or depth_u16.ndim != 2:
        raise ValueError(f"depth_u16 must be uint16 (H,W). Got {depth_u16.dtype}, {depth_u16.shape}")

    Hc, Wc = color_bgr.shape[:2]
    Hd, Wd = depth_u16.shape[:2]

    # -------------------------
    # Case B: depth already aligned to color
    # -------------------------
    if (Hd, Wd) == (Hc, Wc):
        if not hasattr(cm, "MapColorFrameToCameraSpace"):
            raise RuntimeError(
                "Depth shape equals Color shape => treating as 'depth aligned to color'. "
                "But CoordinateMapper has no MapColorFrameToCameraSpace. "
                "Fix by returning native depth (424x512) from kinectv2_capture_service, "
                "or use a PyKinect2 fork that exposes MapColorFrameToCameraSpace."
            )

        n_color = Hc * Wc

        # ctypes arrays
        if not hasattr(PyKinectV2, "_CameraSpacePoint"):
            raise RuntimeError("PyKinectV2 missing _CameraSpacePoint in this fork.")
        CameraSpacePointArray = PyKinectV2._CameraSpacePoint * n_color
        cam_points = CameraSpacePointArray()

        depth_flat = np.ascontiguousarray(depth_u16.reshape(-1))
        depth_ptr = depth_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))

        # MapColorFrameToCameraSpace expects depth buffer at color resolution
        cm.MapColorFrameToCameraSpace(n_color, depth_ptr, n_color, cam_points)

        cam_xyz = _ctypes_as_array_of_structs(cam_points, ("x", "y", "z"), 3)  # (N,3)
        x, y, z = cam_xyz[:, 0], cam_xyz[:, 1], cam_xyz[:, 2]

        # u,v are pixel grid
        uu, vv = np.meshgrid(np.arange(Wc, dtype=np.float32), np.arange(Hc, dtype=np.float32))
        u = uu.reshape(-1)
        v = vv.reshape(-1)

        z_valid = np.isfinite(z) & (z > 0.0) & (z < float(depth_trunc_m))
        valid = z_valid

        if not np.any(valid):
            return (
                np.zeros((0, 3), np.float32),
                np.zeros((0, 3), np.uint8),
                np.zeros((0, 2), np.float32),
            )

        xyz = np.stack([x[valid], y[valid], z[valid]], axis=1).astype(np.float32, copy=False)
        u_v = u[valid]
        v_v = v[valid]
        cols_bgr = bilinear_sample_bgr(color_bgr, u_v, v_v)
        rgb = cols_bgr[:, ::-1].astype(np.uint8, copy=False)
        pix_color = np.stack([u_v, v_v], axis=1).astype(np.float32, copy=False)

        if flip_y:
            xyz = xyz.copy()
            xyz[:, 1] *= -1.0
        if flip_z:
            xyz = xyz.copy()
            xyz[:, 2] *= -1.0
        return xyz, rgb, pix_color

    # -------------------------
    # Case A: native depth
    # -------------------------
    depth_h, depth_w = Hd, Wd
    n_depth = depth_h * depth_w

    if not hasattr(PyKinectV2, "_CameraSpacePoint") or not hasattr(PyKinectV2, "_ColorSpacePoint"):
        raise RuntimeError("PyKinectV2 missing _CameraSpacePoint/_ColorSpacePoint in this fork.")

    CameraSpacePointArray = PyKinectV2._CameraSpacePoint * n_depth
    ColorSpacePointArray = PyKinectV2._ColorSpacePoint * n_depth

    cam_points = CameraSpacePointArray()
    col_points = ColorSpacePointArray()

    depth_flat = np.ascontiguousarray(depth_u16.reshape(-1))
    depth_ptr = depth_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))

    cm.MapDepthFrameToCameraSpace(n_depth, depth_ptr, n_depth, cam_points)
    cm.MapDepthFrameToColorSpace(n_depth, depth_ptr, n_depth, col_points)

    cam_xyz = _ctypes_as_array_of_structs(cam_points, ("x", "y", "z"), 3)  # (N,3)
    col_uv = _ctypes_as_array_of_structs(col_points, ("x", "y"), 2)        # (N,2)

    x, y, z = cam_xyz[:, 0], cam_xyz[:, 1], cam_xyz[:, 2]
    u, v = col_uv[:, 0], col_uv[:, 1]

    depth_valid = depth_flat > 0
    z_valid = np.isfinite(z) & (z > 0.0) & (z < float(depth_trunc_m))
    uv_valid = np.isfinite(u) & np.isfinite(v)
    valid = depth_valid & z_valid & uv_valid

    if not np.any(valid):
        return (
            np.zeros((0, 3), np.float32),
            np.zeros((0, 3), np.uint8),
            np.zeros((0, 2), np.float32),
        )

    xyz = np.stack([x[valid], y[valid], z[valid]], axis=1).astype(np.float32, copy=False)

    u_v = u[valid]
    v_v = v[valid]
    cols_bgr = bilinear_sample_bgr(color_bgr, u_v, v_v)
    rgb = cols_bgr[:, ::-1].astype(np.uint8, copy=False)
    pix_color = np.stack([u_v, v_v], axis=1).astype(np.float32, copy=False)

    if flip_y:
        xyz = xyz.copy()
        xyz[:, 1] *= -1.0
    if flip_z:
        xyz = xyz.copy()
        xyz[:, 2] *= -1.0

    return xyz, rgb, pix_color


# ============================================================
# 5) Mask -> object cut (lookup by pix_color)
# ============================================================

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
    小修：默认 sample_round="round"（比 floor 更贴近像素归属，减少边界偏差）
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


# ============================================================
# 6) Open3D realtime viewer
# ============================================================

class RealtimePCDViewer:
    def __init__(self, title: str, width: int = 960, height: int = 720, point_size: float = 3.0):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=title, width=width, height=height, visible=True)

        self.pcd = o3d.geometry.PointCloud()
        self._added = False

        opt = self.vis.get_render_option()
        if opt is not None:
            opt.point_size = float(point_size)

        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        self.vis.add_geometry(axis, reset_bounding_box=True)

    def update(self, xyz_m: np.ndarray, rgb_u8: np.ndarray, reset_view: bool = False):
        if xyz_m is None or xyz_m.shape[0] == 0:
            self.vis.poll_events()
            self.vis.update_renderer()
            return

        self.pcd.points = o3d.utility.Vector3dVector(xyz_m.astype(np.float64, copy=False))
        self.pcd.colors = o3d.utility.Vector3dVector((rgb_u8.astype(np.float32) / 255.0).astype(np.float64, copy=False))

        if not self._added:
            self.vis.add_geometry(self.pcd, reset_bounding_box=True)
            self._added = True
            self.vis.reset_view_point(True)
        else:
            self.vis.update_geometry(self.pcd)
            if reset_view:
                self.vis.reset_view_point(True)

        self.vis.poll_events()
        self.vis.update_renderer()

    def save_ply(self, out_ply: str):
        if len(self.pcd.points) == 0:
            print("[PLY] Empty point cloud, skip saving.")
            return
        os.makedirs(os.path.dirname(out_ply), exist_ok=True)
        ok = o3d.io.write_point_cloud(out_ply, self.pcd, write_ascii=False, compressed=False)
        if not ok:
            raise RuntimeError(f"Failed to write PLY: {out_ply}")
        print(f"[PLY Saved] {out_ply}")

    def tick(self):
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        try:
            self.vis.destroy_window()
        except Exception:
            pass


# ============================================================
# 7) Registration utilities (unchanged from your version)
# ============================================================

def load_pcd(p: str) -> o3d.geometry.PointCloud:
    q = o3d.io.read_point_cloud(p)
    if q.is_empty():
        raise RuntimeError(f"Empty/invalid point cloud: {p}")
    return q


def make_pcd_from_xyzrgb(xyz: np.ndarray, rgb_u8: np.ndarray) -> o3d.geometry.PointCloud:
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(xyz.astype(np.float64, copy=False))
    if rgb_u8 is not None and rgb_u8.shape[0] == xyz.shape[0]:
        p.colors = o3d.utility.Vector3dVector((rgb_u8.astype(np.float32) / 255.0).astype(np.float64, copy=False))
    return p


def aabb_diag(pcd: o3d.geometry.PointCloud) -> float:
    aabb = pcd.get_axis_aligned_bounding_box()
    ext = np.asarray(aabb.get_extent())
    return float(np.linalg.norm(ext))


def scale_about_center(pcd: o3d.geometry.PointCloud, s: float) -> o3d.geometry.PointCloud:
    q = copy.deepcopy(pcd)
    q.scale(float(s), center=q.get_center())
    return q


def remove_outliers_stat(pcd: o3d.geometry.PointCloud, nb_neighbors=20, std_ratio=2.0) -> o3d.geometry.PointCloud:
    if pcd.is_empty():
        return pcd
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=int(nb_neighbors), std_ratio=float(std_ratio))
    return pcd.select_by_index(ind)


def estimate_normals(pcd: o3d.geometry.PointCloud, radius: float, max_nn: int = 120) -> o3d.geometry.PointCloud:
    if pcd.is_empty():
        return pcd
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=float(radius), max_nn=int(max_nn)))
    pcd.normalize_normals()
    return pcd


def voxel_down(pcd: o3d.geometry.PointCloud, voxel_size: float) -> o3d.geometry.PointCloud:
    if pcd.is_empty():
        return pcd
    if voxel_size is None or voxel_size <= 0:
        return pcd
    out = pcd.voxel_down_sample(float(voxel_size))
    if out.is_empty():
        return pcd
    return out


def preprocess_fpfh(pcd: o3d.geometry.PointCloud, voxel: float):
    p_down = voxel_down(pcd, voxel)
    estimate_normals(p_down, radius=voxel * 2.5, max_nn=200)
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        p_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5.0, max_nn=300)
    )
    return p_down, fpfh


def global_ransac(src_down, tgt_down, src_fpfh, tgt_fpfh, voxel: float):
    dist_th = voxel * 1.5
    res = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down,
        src_fpfh, tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=dist_th,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_th),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )
    return res


def refine_gicp(src: o3d.geometry.PointCloud,
                tgt: o3d.geometry.PointCloud,
                init_T: np.ndarray,
                max_corr: float,
                max_iter: int = 80):
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(max_iter))
    res = o3d.pipelines.registration.registration_generalized_icp(
        src, tgt,
        max_correspondence_distance=float(max_corr),
        init=init_T,
        estimation_method=o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
        criteria=criteria
    )
    return res


@dataclass
class RegistrationParams:
    enable_scale_prealign: bool = True
    clean_tgt_interior: bool = False
    do_outlier_remove: bool = True
    out_nb: int = 20
    out_std: float = 2.0
    fpfh_voxel_div: float = 80.0
    gicp_max_iter: int = 80
    verbose: bool = True


class AsyncRegistrar:
    def __init__(self, tgt_ply_path: str, params: RegistrationParams):
        self.params = params
        self.tgt_raw = load_pcd(tgt_ply_path)

        self._src_lock = threading.Lock()
        self._latest_src: Optional[o3d.geometry.PointCloud] = None
        self._latest_src_ts: float = 0.0

        self._job_lock = threading.Lock()
        self._trigger = False

        self._res_lock = threading.Lock()
        self._latest_T: Optional[np.ndarray] = None  # T_src_to_tgt
        self._latest_metrics: Dict[str, Any] = {}
        self._latest_ts: float = 0.0
        self._latest_tgt_used: Optional[o3d.geometry.PointCloud] = None
        self._latest_src_used: Optional[o3d.geometry.PointCloud] = None
        self._latest_src_aligned: Optional[o3d.geometry.PointCloud] = None

        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def submit_src(self, src_pcd: o3d.geometry.PointCloud, ts: float):
        with self._src_lock:
            self._latest_src = copy.deepcopy(src_pcd)
            self._latest_src_ts = float(ts)

    def trigger_once(self):
        with self._job_lock:
            self._trigger = True

    def get_latest(self):
        with self._res_lock:
            return (
                None if self._latest_T is None else np.asarray(self._latest_T),
                dict(self._latest_metrics),
                float(self._latest_ts),
                None if self._latest_tgt_used is None else copy.deepcopy(self._latest_tgt_used),
                None if self._latest_src_used is None else copy.deepcopy(self._latest_src_used),
                None if self._latest_src_aligned is None else copy.deepcopy(self._latest_src_aligned),
            )

    @staticmethod
    def _clean_src_for_reg(src: o3d.geometry.PointCloud, nb=20, std=2.0) -> o3d.geometry.PointCloud:
        src2 = copy.deepcopy(src)
        src2.remove_non_finite_points()
        if len(src2.points) > 50:
            src2 = remove_outliers_stat(src2, nb_neighbors=nb, std_ratio=std)
        return src2

    @staticmethod
    def _crop_by_tgt_aabb(src_aligned: o3d.geometry.PointCloud,
                          tgt_ref: o3d.geometry.PointCloud,
                          expand_ratio: float = 0.20) -> o3d.geometry.PointCloud:
        if src_aligned.is_empty() or tgt_ref.is_empty():
            return src_aligned
        aabb = tgt_ref.get_axis_aligned_bounding_box()
        ext = aabb.get_extent()
        aabb = o3d.geometry.AxisAlignedBoundingBox(
            aabb.get_min_bound() - expand_ratio * ext,
            aabb.get_max_bound() + expand_ratio * ext,
        )
        return src_aligned.crop(aabb)

    def _register_once(self, src_used: o3d.geometry.PointCloud) -> Tuple[np.ndarray, Dict[str, Any], o3d.geometry.PointCloud]:
        if src_used is None or src_used.is_empty():
            raise RuntimeError("src_used empty")

        t0 = time.perf_counter()

        tgt_used = copy.deepcopy(self.tgt_raw)
        tgt_used.remove_non_finite_points()

        if self.params.do_outlier_remove:
            tgt_used = remove_outliers_stat(tgt_used, nb_neighbors=self.params.out_nb, std_ratio=self.params.out_std)

        t_prep = time.perf_counter()

        scale_applied = 1.0
        if self.params.enable_scale_prealign:
            ds = aabb_diag(src_used)
            dt = aabb_diag(tgt_used)
            if ds > 1e-9 and dt > 1e-9:
                scale_applied = ds / dt
                tgt_used = scale_about_center(tgt_used, scale_applied)

        diag_src = aabb_diag(src_used)
        voxel = max(diag_src / float(self.params.fpfh_voxel_div), 1e-4)
        max_corr = voxel * 2.0

        t_scale = time.perf_counter()

        src_down, src_fpfh = preprocess_fpfh(src_used, voxel)
        tgt_down, tgt_fpfh = preprocess_fpfh(tgt_used, voxel)

        if src_down.is_empty() or tgt_down.is_empty():
            raise RuntimeError(f"downsample empty: src_down={len(src_down.points)} tgt_down={len(tgt_down.points)} voxel={voxel}")

        t_fpfh = time.perf_counter()

        res_ransac = global_ransac(src_down, tgt_down, src_fpfh, tgt_fpfh, voxel)
        t_ransac = time.perf_counter()

        src_ref = estimate_normals(copy.deepcopy(src_used), radius=voxel * 2.5, max_nn=200)
        tgt_ref = estimate_normals(copy.deepcopy(tgt_used), radius=voxel * 2.5, max_nn=200)
        t_normals = time.perf_counter()

        res_gicp = refine_gicp(
            src_ref, tgt_ref,
            init_T=res_ransac.transformation,
            max_corr=max_corr,
            max_iter=self.params.gicp_max_iter
        )
        t_gicp = time.perf_counter()

        T = np.asarray(res_gicp.transformation)

        metrics = {
            "ok": True,
            "scale_applied_to_target": float(scale_applied),
            "diag_src": float(diag_src),
            "voxel": float(voxel),
            "max_corr": float(max_corr),
            "ransac_fitness": float(res_ransac.fitness),
            "ransac_rmse": float(res_ransac.inlier_rmse),
            "gicp_fitness": float(res_gicp.fitness),
            "gicp_rmse": float(res_gicp.inlier_rmse),
            "t_prepare_sec": float(t_prep - t0),
            "t_scale_sec": float(t_scale - t_prep),
            "t_fpfh_sec": float(t_fpfh - t_scale),
            "t_ransac_sec": float(t_ransac - t_fpfh),
            "t_normals_sec": float(t_normals - t_ransac),
            "t_gicp_sec": float(t_gicp - t_normals),
            "t_total_sec": float(t_gicp - t0),
        }

        if res_gicp.fitness < 0.05:
            metrics["ok"] = False
            metrics["reason"] = "low_fitness"

        if self.params.verbose:
            print(
                f"[Reg] total={metrics['t_total_sec'] * 1000:.1f}ms "
                f"(prep={metrics['t_prepare_sec'] * 1000:.1f} "
                f"fpfh={metrics['t_fpfh_sec'] * 1000:.1f} "
                f"ransac={metrics['t_ransac_sec'] * 1000:.1f} "
                f"norm={metrics['t_normals_sec'] * 1000:.1f} "
                f"gicp={metrics['t_gicp_sec'] * 1000:.1f}) "
                f"gicp_fit={res_gicp.fitness:.3f} rmse={res_gicp.inlier_rmse:.4f}"
            )

        return T, metrics, tgt_used

    def _loop(self):
        while self._running:
            do = False
            with self._job_lock:
                if self._trigger:
                    self._trigger = False
                    do = True
            if not do:
                time.sleep(0.005)
                continue

            with self._src_lock:
                src0 = None if self._latest_src is None else copy.deepcopy(self._latest_src)
                src_ts = float(self._latest_src_ts)

            if src0 is None or src0.is_empty():
                with self._res_lock:
                    self._latest_T = None
                    self._latest_metrics = {"ok": False, "reason": "no_src"}
                    self._latest_ts = time.time()
                    self._latest_tgt_used = None
                    self._latest_src_used = None
                    self._latest_src_aligned = None
                continue

            try:
                src_used = self._clean_src_for_reg(src0, nb=self.params.out_nb, std=self.params.out_std)
                T, metrics, tgt_used = self._register_once(src_used)

                if not metrics.get("ok", False):
                    with self._res_lock:
                        self._latest_T = T
                        self._latest_metrics = metrics
                        self._latest_ts = src_ts
                        self._latest_tgt_used = tgt_used
                        self._latest_src_used = src_used
                        self._latest_src_aligned = None
                    continue

                src_aligned = copy.deepcopy(src_used)
                src_aligned.transform(T)
                src_aligned = self._crop_by_tgt_aabb(src_aligned, tgt_used, expand_ratio=0.25)

                with self._res_lock:
                    self._latest_T = T
                    self._latest_metrics = metrics
                    self._latest_ts = src_ts
                    self._latest_tgt_used = tgt_used
                    self._latest_src_used = src_used
                    self._latest_src_aligned = src_aligned

            except Exception as e:
                with self._res_lock:
                    self._latest_T = None
                    self._latest_metrics = {"ok": False, "reason": "exception", "err": repr(e)}
                    self._latest_ts = src_ts
                    self._latest_tgt_used = None
                    self._latest_src_used = None
                    self._latest_src_aligned = None


class RealtimeMatchViewer:
    def __init__(self, title: str = "Match View (Green=TGT used, Blue=SRC aligned)", point_size: float = 3.0):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=title, width=960, height=720, visible=True)

        opt = self.vis.get_render_option()
        if opt is not None:
            opt.point_size = float(point_size)

        self.axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        self.vis.add_geometry(self.axis, reset_bounding_box=True)

        self.tgt_geom = o3d.geometry.PointCloud()
        self.src_geom = o3d.geometry.PointCloud()
        self._added_tgt = False
        self._added_src = False

    @staticmethod
    def _set_pcd_inplace(dst: o3d.geometry.PointCloud,
                         src: o3d.geometry.PointCloud,
                         color_rgb: Tuple[float, float, float]):
        dst.points = src.points
        if len(dst.points) > 0:
            dst.colors = o3d.utility.Vector3dVector(
                np.tile(np.array(color_rgb, dtype=np.float64)[None, :], (len(dst.points), 1))
            )
        else:
            dst.colors = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))

    def set_target(self, tgt_pcd: o3d.geometry.PointCloud, reset_view: bool = False):
        if tgt_pcd is None or tgt_pcd.is_empty():
            return
        self._set_pcd_inplace(self.tgt_geom, tgt_pcd, (0.0, 1.0, 0.0))
        if not self._added_tgt:
            self.vis.add_geometry(self.tgt_geom, reset_bounding_box=True)
            self._added_tgt = True
            self.vis.reset_view_point(True)
        else:
            self.vis.update_geometry(self.tgt_geom)
            if reset_view:
                self.vis.reset_view_point(True)

    def set_aligned_src(self, src_aligned: o3d.geometry.PointCloud, reset_view: bool = False):
        if src_aligned is None or src_aligned.is_empty():
            return
        self._set_pcd_inplace(self.src_geom, src_aligned, (0.0, 0.0, 1.0))
        if not self._added_src:
            self.vis.add_geometry(self.src_geom, reset_bounding_box=True)
            self._added_src = True
            if reset_view:
                self.vis.reset_view_point(True)
        else:
            self.vis.update_geometry(self.src_geom)
            if reset_view:
                self.vis.reset_view_point(True)

    def tick(self):
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        try:
            self.vis.destroy_window()
        except Exception:
            pass


# ============================================================
# 8) Main processor
# ============================================================

class MyProcessor:
    def __init__(self, fastsam_model_path: str, thresh_ply: str, flip_y: bool = False, flip_z: bool = False):
        # Kinect capture (your service)
        start_capture(preview=True, fps=10)

        # segmentation background
        self.seg = AsyncSegmenter(
            model_path=fastsam_model_path,
            device="cuda",
            img_size=1024,
            conf=0.4,
            iou=0.9,
            min_area=80,
        )
        self.seg.start()

        # UI
        self.ui = RealTimeSegUI()

        # Open3D viewers
        self.viewer_global = RealtimePCDViewer(title="Global PCD (strict aligned)", point_size=2.0)
        self.viewer_object = RealtimePCDViewer(title="Object PCD (cut by selected mask)", point_size=4.0)

        # strict align mapper (FIX: init runtime with Color|Depth)
        self.strict_ok = False
        self.cm = None
        self.PyKinectV2 = None
        self._map_runtime = None
        try:
            from pykinect2 import PyKinectV2  # type: ignore
            from pykinect2.PyKinectRuntime import PyKinectRuntime  # type: ignore

            # FIX: request both streams so CoordinateMapper has consistent calibration context
            flags = PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth
            self._map_runtime = PyKinectRuntime(flags)

            self.cm = get_coordinate_mapper(self._map_runtime)
            self.PyKinectV2 = PyKinectV2
            self.strict_ok = True
            print("[Align] CoordinateMapper ready. (runtime Color|Depth)")
        except Exception as e:
            print("[Align] init failed -> strict alignment disabled:", repr(e))
            self.strict_ok = False

        self.flip_y = bool(flip_y)
        self.flip_z = bool(flip_z)
        print(f"[Align] flip_y={self.flip_y} flip_z={self.flip_z}")

        # match registrar + match viewer
        self.match_viewer = RealtimeMatchViewer()
        reg_params = RegistrationParams(
            enable_scale_prealign=True,
            clean_tgt_interior=False,
            do_outlier_remove=True,
            out_nb=20,
            out_std=2.0,
            fpfh_voxel_div=80.0,
            gicp_max_iter=80,
            verbose=True,
        )
        self.reg = AsyncRegistrar(thresh_ply, params=reg_params)
        self.reg.start()

        try:
            self.match_viewer.set_target(self.reg.tgt_raw, reset_view=True)
        except Exception as e:
            print("[MatchView] initial target failed:", repr(e))

        self._match_first_show = True

        self.PCD_UPDATE_HZ = 5.0
        self._pcd_next_t = 0.0

        self._last_xyz: Optional[np.ndarray] = None
        self._last_rgb: Optional[np.ndarray] = None
        self._last_pixc: Optional[np.ndarray] = None
        self._last_obj_xyz: Optional[np.ndarray] = None
        self._last_obj_rgb: Optional[np.ndarray] = None

        self.OUT_DIR = os.environ.get(
            "DGSRSIM_RT_PLY_OUT",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "rt_ply_out"),
        )
        os.makedirs(self.OUT_DIR, exist_ok=True)

        self._last_pose_ts_printed: float = -1.0
        self._t0 = time.time()
        self._n = 0

        self._printed_shapes = False

    def _save_pose(self, T_tgt_to_scene: np.ndarray):
        npy_path = os.path.join(self.OUT_DIR, "T_tgt_to_scene.npy")
        np.save(npy_path, T_tgt_to_scene.astype(np.float64))
        txt_path = os.path.join(self.OUT_DIR, "latest_pose.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(pretty_pose(T_tgt_to_scene) + "\n")
        print(f"[PoseSaved] {npy_path}")
        print(f"[PoseSaved] {txt_path}")

    def run_once(self) -> bool:
        color, depth = get_latest_frame(copy=True)
        if color is None or depth is None:
            time.sleep(0.002)
            self.viewer_global.tick()
            self.viewer_object.tick()
            self.match_viewer.tick()
            return True

        if not self._printed_shapes:
            self._printed_shapes = True
            print("[SHAPES] color=", color.shape, color.dtype, "depth=", depth.shape, depth.dtype)

        ts = time.time()

        # async seg
        self.seg.submit(color, ts)
        masks, _ = self.seg.get_latest_result()

        # FIX: force masks match color resolution
        masks = ensure_masks_match_color(masks, color)

        # UI render
        status = self.ui.render(color, masks)
        if status == "quit":
            return False

        sel = self.ui.get_selected_idx()

        # strict align + cut (throttled)
        now = time.time()
        if self.strict_ok and now >= self._pcd_next_t:
            self._pcd_next_t = now + 1.0 / max(self.PCD_UPDATE_HZ, 1e-6)

            try:
                xyz, rgb, pixc = align_and_build_pointcloud_strict_with_pixcolor(
                    color_bgr=color,
                    depth_u16=depth,
                    cm=self.cm,
                    PyKinectV2=self.PyKinectV2,
                    depth_trunc_m=4.0,
                    flip_y=self.flip_y,
                    flip_z=self.flip_z,
                )
                self._last_xyz, self._last_rgb, self._last_pixc = xyz, rgb, pixc
                self.viewer_global.update(xyz, rgb, reset_view=False)

                xyz_obj, rgb_obj, idx, stats = cut_object_pcd_by_selected_mask(
                    xyz_m=xyz,
                    rgb_u8=rgb,
                    pix_color=pixc,
                    masks_bool=masks,
                    selected_idx=sel,
                    sample_round="round",
                    require_in_bounds=True,
                )
                self._last_obj_xyz, self._last_obj_rgb = xyz_obj, rgb_obj
                self.viewer_object.update(xyz_obj, rgb_obj, reset_view=False)

                # optional: print cut stats occasionally to verify consistency
                if sel is not None and (self._n % 30 == 0):
                    ratio = stats.get("hit", 0) / max(stats.get("mask_fg_pixels", 1), 1)
                    print("[CUT]", "ok=", stats.get("ok"), "hit=", stats.get("hit"),
                          "mask_fg=", stats.get("mask_fg_pixels"), "ratio=", f"{ratio:.3f}",
                          "mask_hw=", stats.get("mask_hw"))

                # cache src for registrar
                if xyz_obj.shape[0] > 1000:
                    src_pcd = make_pcd_from_xyzrgb(xyz_obj, rgb_obj)
                    self.reg.submit_src(src_pcd, ts)

            except Exception as e:
                print("[Align] error:", repr(e))

        # hotkeys
        k = self.ui.pop_last_key()

        if k == ord("p"):
            if self._last_obj_xyz is None or self._last_obj_xyz.shape[0] == 0:
                print("[PLY] object empty; select a mask and wait alignment update.")
            else:
                out_ply = os.path.join(self.OUT_DIR, f"object_{int(time.time()*1000)}.ply")
                self.viewer_object.save_ply(out_ply)

        if k == ord("g"):
            if self._last_xyz is None or self._last_xyz.shape[0] == 0:
                print("[PLY] global empty; wait alignment update.")
            else:
                out_ply = os.path.join(self.OUT_DIR, f"global_{int(time.time()*1000)}.ply")
                self.viewer_global.save_ply(out_ply)

        if k == ord("m"):
            if self._last_obj_xyz is None or self._last_obj_xyz.shape[0] < 1000:
                print("[Reg] object too small/empty; select a mask and wait.")
            else:
                print("[Reg] trigger once... (make sure focus is on OpenCV UI window)")
                self.reg.trigger_once()

        # pull reg result and show
        T_src_to_tgt, metrics, mts, tgt_used, src_used, src_aligned = self.reg.get_latest()
        if metrics.get("ok", False) and T_src_to_tgt is not None:
            if tgt_used is not None and (not tgt_used.is_empty()):
                self.match_viewer.set_target(tgt_used, reset_view=self._match_first_show)

            if src_aligned is not None and (not src_aligned.is_empty()):
                self.match_viewer.set_aligned_src(src_aligned, reset_view=self._match_first_show)
                self._match_first_show = False

            T_tgt_to_scene = np.linalg.inv(np.asarray(T_src_to_tgt, dtype=np.float64))

            if abs(float(mts) - float(self._last_pose_ts_printed)) > 1e-6:
                self._last_pose_ts_printed = float(mts)
                print("[RegMetrics] scale=", metrics.get("scale_applied_to_target"),
                      "diag_src=", metrics.get("diag_src"),
                      "voxel=", metrics.get("voxel"),
                      "gicp_fit=", metrics.get("gicp_fitness"))
                print("[Pose] T_tgt_to_scene (= inv(T_src_to_tgt))")
                print(pretty_pose(T_tgt_to_scene))
                self._save_pose(T_tgt_to_scene)
        else:
            if metrics and metrics.get("reason") not in (None, "") and (self._n % 60 == 0):
                print("[Reg] last status:", metrics)

        # tick viewers
        self.viewer_global.tick()
        self.viewer_object.tick()
        self.match_viewer.tick()

        # fps
        self._n += 1
        if self._n % 60 == 0:
            dt = max(time.time() - self._t0, 1e-6)
            print(f"[Main] fps~{self._n/dt:.2f}  strict={self.strict_ok}  sel={sel}")

        return True

    def close(self):
        try:
            self.seg.stop()
        finally:
            try:
                self.reg.stop()
            except Exception:
                pass
            try:
                self.viewer_global.close()
            except Exception:
                pass
            try:
                self.viewer_object.close()
            except Exception:
                pass
            try:
                self.match_viewer.close()
            except Exception:
                pass
            try:
                if self._map_runtime is not None:
                    self._map_runtime.close()
            except Exception:
                pass
            stop_capture()
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


if __name__ == "__main__":
    _HERE = os.path.dirname(os.path.abspath(__file__))
    FASTSAM_MODEL = os.environ.get(
        "DGSRSIM_FASTSAM_MODEL",
        os.path.join(_HERE, "weights", "FastSAM-x.pt"),
    )
    THRESH_PLY = os.environ.get(
        "DGSRSIM_TARGET_PLY",
        os.path.join(_HERE, "rt_ply_out", "astronaut.ply"),
    )

    FLIP_Y = False
    FLIP_Z = False

    p = MyProcessor(fastsam_model_path=FASTSAM_MODEL, thresh_ply=THRESH_PLY, flip_y=FLIP_Y, flip_z=FLIP_Z)
    try:
        while True:
            if not p.run_once():
                break
    finally:
        p.close()
