# -*- coding: utf-8 -*-
from __future__ import annotations

import ctypes
from typing import Optional, Tuple, Any

import numpy as np


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

        if not hasattr(PyKinectV2, "_CameraSpacePoint"):
            raise RuntimeError("PyKinectV2 missing _CameraSpacePoint in this fork.")
        CameraSpacePointArray = PyKinectV2._CameraSpacePoint * n_color
        cam_points = CameraSpacePointArray()

        depth_flat = np.ascontiguousarray(depth_u16.reshape(-1))
        depth_ptr = depth_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))

        cm.MapColorFrameToCameraSpace(n_color, depth_ptr, n_color, cam_points)

        cam_xyz = _ctypes_as_array_of_structs(cam_points, ("x", "y", "z"), 3)
        x, y, z = cam_xyz[:, 0], cam_xyz[:, 1], cam_xyz[:, 2]

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
    n_depth = Hd * Wd

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

    cam_xyz = _ctypes_as_array_of_structs(cam_points, ("x", "y", "z"), 3)
    col_uv = _ctypes_as_array_of_structs(col_points, ("x", "y"), 2)

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
