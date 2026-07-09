# -*- coding: utf-8 -*-

import os
import time
from typing import Optional

import cv2
import numpy as np

from kinectv2_capture_service import start_capture, stop_capture, get_latest_frame

# ============================================================
# Externalized helpers (already externalized)
# ============================================================
from utils.pose_utils import pretty_pose
from utils.seg_selection_helpers import ensure_masks_match_color

# ============================================================
# Externalized (2)(3)(4)(5)
# ============================================================
from utils.rt_async_segmenter import AsyncSegmenter
from utils.rt_seg_ui import RealTimeSegUI
from utils.kinect_strict_align import (
    get_coordinate_mapper,
    align_and_build_pointcloud_strict_with_pixcolor,
)
from utils.mask_object_cut import cut_object_pcd_by_selected_mask

# ============================================================
# Externalized (6)(7)
# ============================================================
from utils.o3d_realtime_viewer import RealtimePCDViewer
from utils.registration_async import (
    make_pcd_from_xyzrgb,
    RegistrationParams,
    AsyncRegistrar,
    RealtimeMatchViewer,
)

def atomic_save_npy(path: str, arr: np.ndarray):
    """Write .npy atomically: save to temp file then os.replace()."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # ensure final path endswith .npy
    final_path = path if path.lower().endswith(".npy") else (path + ".npy")

    tmp_path = final_path + ".tmp"
    # np.save will append ".npy" if not present
    np.save(tmp_path, arr)
    if not tmp_path.lower().endswith(".npy"):
        tmp_path = tmp_path + ".npy"

    os.replace(tmp_path, final_path)


def atomic_save_text(path: str, text: str):
    """Write text atomically."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp_path, path)

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
            iou=0.6,
            min_area=280,
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

        self.OUT_DIR = os.path.join(".", "rt_ply_out")
        os.makedirs(self.OUT_DIR, exist_ok=True)

        self._last_pose_ts_printed: float = -1.0
        self._t0 = time.time()
        self._n = 0

        self._printed_shapes = False

    # def _save_pose(self, T_tgt_to_scene: np.ndarray):
    #     npy_path = os.path.join(self.OUT_DIR, "T_tgt_to_scene.npy")
    #     np.save(npy_path, T_tgt_to_scene.astype(np.float64))
    #     txt_path = os.path.join(self.OUT_DIR, "latest_pose.txt")
    #     with open(txt_path, "w", encoding="utf-8") as f:
    #         f.write(pretty_pose(T_tgt_to_scene) + "\n")
    #     print(f"[PoseSaved] {npy_path}")
    #     print(f"[PoseSaved] {txt_path}")
    def _save_pose(self, T_tgt_to_scene: np.ndarray):
        # 建议：把 OUT_DIR 改成绝对路径，避免不同启动目录导致另一个进程找不到文件
        npy_path = os.path.join(self.OUT_DIR, "T_tgt_to_scene.npy")
        txt_path = os.path.join(self.OUT_DIR, "latest_pose.txt")

        T64 = np.asarray(T_tgt_to_scene, dtype=np.float64)

        # 原子写：避免读端读到半截文件
        atomic_save_npy(npy_path, T64)
        atomic_save_text(txt_path, pretty_pose(T64) + "\n")

        # 可选：写一个时间戳文件，读端可以用它做“是否更新”判断（比 mtime 更直观）
        ts_path = os.path.join(self.OUT_DIR, "pose_ts.txt")
        atomic_save_text(ts_path, f"{time.time():.6f}\n")

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

        # force masks match color resolution
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
    FASTSAM_MODEL = r"E:\code\FastSAM\weights\FastSAM-x.pt"
    THRESH_PLY = r"C:\Users\quyuanjin\Downloads\2.ply"

    FLIP_Y = False
    FLIP_Z = False

    p = MyProcessor(fastsam_model_path=FASTSAM_MODEL, thresh_ply=THRESH_PLY, flip_y=FLIP_Y, flip_z=FLIP_Z)
    try:
        while True:
            if not p.run_once():
                break
    finally:
        p.close()
