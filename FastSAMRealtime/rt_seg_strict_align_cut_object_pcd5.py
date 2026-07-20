# -*- coding: utf-8 -*-

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from kinectv2_capture_service import start_capture, stop_capture, get_latest_frame

# ============================================================
# Externalized helpers
# ============================================================
from utils.pose_utils import (
    compose_asset_to_scene,
    compose_target_to_scene,
    load_transform_4x4,
    pretty_pose,
    pretty_similarity,
)
from utils.seg_selection_helpers import ensure_masks_match_color
from utils.object_state_bundle import update_object_state_bundle

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

    final_path = path if path.lower().endswith(".npy") else (path + ".npy")
    tmp_path = final_path + ".tmp"

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


def atomic_save_json(path: str, payload: Dict[str, Any]):
    """Write JSON atomically."""
    atomic_save_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ============================================================
# Main processor
# ============================================================

class MyProcessor:
    def __init__(
        self,
        fastsam_model_path: str,
        thresh_ply: str,
        T_scene_from_camera: np.ndarray,
        calibration_info: Optional[Dict[str, Any]] = None,
        object_id: Optional[str] = None,
        flip_y: bool = False,
        flip_z: bool = False,
    ):
        # Kinect capture
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
        self.viewer_global = RealtimePCDViewer(
            title="Global PCD (strict aligned)",
            point_size=2.0
        )
        self.viewer_object = RealtimePCDViewer(
            title="Object PCD (cut by selected mask)",
            point_size=4.0
        )

        # strict align mapper
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
        self.T_scene_from_camera = np.asarray(T_scene_from_camera, dtype=np.float64)
        self.calibration_info = dict(calibration_info or {})
        self.target_asset_name = Path(thresh_ply).name
        self.target_asset_sha256 = sha256_file(thresh_ply)
        default_object_id = Path(thresh_ply).stem
        self.object_id = re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            str(object_id or default_object_id).strip(),
        ).strip("_")
        if not self.object_id:
            raise ValueError("DGSRSim object_id must contain at least one valid character")
        print(f"[Align] flip_y={self.flip_y} flip_z={self.flip_z}")
        print("[Frames] calibrated T_scene_from_camera")
        print(pretty_pose(self.T_scene_from_camera))

        # ============================================================
        # Registration params
        # 重点：
        # - 不再使用旧版 center-cluster 参数
        # - 使用新版主轴单侧长尾裁剪参数
        # - 注册参数比“显示调参”更温和，避免源点云被裁得太瘦
        # ============================================================
        reg_params = RegistrationParams(
            enable_scale_prealign=True,
            do_outlier_remove=True,
            out_nb=20,
            out_std=1.5,
            fpfh_voxel_div=80.0,
            gicp_max_iter=80,
            verbose=True,

            # 基础预处理
            # 注册比显示略大一点的 voxel，更稳
            src_pre_voxel=0.007,
            tgt_pre_voxel=0.003,
            src_use_radius=False,
            src_radius_nb=10,
            src_radius=0.01,

            # ========= 源点云：温和版长尾裁剪（用于注册） =========
            # 注意：这里不要用你离线显示时那组很狠的参数
            src_do_axial_tail_remove=True,
            src_axial_keep_main_ratio=0.82,
            src_axial_tail_gap_ratio=2.5,
            src_axial_min_points=30,
            src_axial_side_hard_keep_ratio=0.55,

            # 可选后备：一般先关掉，避免把有效几何再切碎
            src_do_cluster_fallback=False,
            src_dbscan_eps=0.02,
            src_dbscan_min_points=12,

            # ========= 目标模板：一般不做长尾裁剪 =========
            tgt_do_axial_tail_remove=False,
            tgt_axial_keep_main_ratio=0.90,
            tgt_axial_tail_gap_ratio=3.0,
            tgt_axial_min_points=30,
            tgt_axial_side_hard_keep_ratio=0.50,

            tgt_do_cluster_fallback=False,
            tgt_dbscan_eps=0.02,
            tgt_dbscan_min_points=12,
        )

        # match registrar + match viewer
        self.match_viewer = RealtimeMatchViewer()
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

    def _save_pose(
        self,
        T_normalized_tgt_to_scene: np.ndarray,
        A_asset_raw_to_scene: np.ndarray,
        metrics: Dict[str, Any],
    ):
        npy_path = os.path.join(self.OUT_DIR, "T_tgt_to_scene.npy")
        similarity_path = os.path.join(self.OUT_DIR, "A_asset_raw_to_scene.npy")
        state_path = os.path.join(self.OUT_DIR, "object_state.json")
        states_path = os.path.join(self.OUT_DIR, "object_states.json")
        txt_path = os.path.join(self.OUT_DIR, "latest_pose.txt")

        T64 = np.asarray(T_normalized_tgt_to_scene, dtype=np.float64)
        A64 = np.asarray(A_asset_raw_to_scene, dtype=np.float64)
        timestamp = time.time()

        atomic_save_npy(npy_path, T64)
        atomic_save_npy(similarity_path, A64)
        state_payload = {
            "schema": "dgsrsim.object_state.v1",
            "object_id": self.object_id,
            "timestamp_unix": timestamp,
            "frames": {
                "observation": "Kinect CameraSpace (meters)",
                "normalized_target": "metric registration target",
                "asset_raw": "shared Gaussian asset coordinates",
                "scene": "metric simulation scene frame",
            },
            "T_scene_from_normalized_target": T64.tolist(),
            "A_scene_from_asset_raw": A64.tolist(),
            "normalization": {
                "scale_raw_to_normalized": float(metrics["scale_applied_to_target"]),
                "source_center_camera_m": metrics["source_center_camera_m"],
                "target_center_asset_raw": metrics["target_center_asset_raw"],
                "target_extent_asset_raw": metrics["target_extent_asset_raw"],
                "target_extent_normalized_m": metrics["target_extent_normalized_m"],
                "A_normalized_target_from_asset_raw": metrics[
                    "A_normalized_target_from_asset_raw"
                ],
            },
            "calibration": self.calibration_info,
            "target_asset": {
                "file_name": self.target_asset_name,
                "sha256": self.target_asset_sha256,
            },
            "registration": {
                key: metrics.get(key)
                for key in (
                    "ransac_fitness",
                    "ransac_rmse",
                    "gicp_fitness",
                    "gicp_rmse",
                    "src_n",
                    "tgt_n",
                    "voxel",
                    "max_corr",
                )
            },
        }
        atomic_save_json(state_path, state_payload)
        update_object_state_bundle(states_path, self.object_id, state_payload)
        atomic_save_text(
            txt_path,
            "T_scene_from_normalized_target\n"
            + pretty_pose(T64)
            + "\n\nA_scene_from_asset_raw\n"
            + pretty_similarity(A64)
            + "\n",
        )

        ts_path = os.path.join(self.OUT_DIR, "pose_ts.txt")
        atomic_save_text(ts_path, f"{timestamp:.6f}\n")

        print(f"[PoseSaved] {npy_path}")
        print(f"[PoseSaved] {similarity_path}")
        print(f"[PoseSaved] {states_path} object_id={self.object_id}")
        print(f"[PoseSaved] {state_path}")
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
                    print(
                        "[CUT]",
                        "ok=", stats.get("ok"),
                        "hit=", stats.get("hit"),
                        "mask_fg=", stats.get("mask_fg_pixels"),
                        "ratio=", f"{ratio:.3f}",
                        "mask_hw=", stats.get("mask_hw"),
                    )

                # 提交原始裁剪点云给注册器
                # 注册器内部会做“温和版”预处理
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

            # Registration maps the observed CameraSpace source into the target
            # asset frame. Its inverse maps the target asset into CameraSpace;
            # the calibrated camera-to-scene extrinsic closes the frame chain.
            T_tgt_to_scene = compose_target_to_scene(
                self.T_scene_from_camera,
                np.asarray(T_src_to_tgt, dtype=np.float64),
            )
            A_asset_raw_to_scene = compose_asset_to_scene(
                self.T_scene_from_camera,
                np.asarray(T_src_to_tgt, dtype=np.float64),
                np.asarray(metrics["A_normalized_target_from_asset_raw"], dtype=np.float64),
            )

            if abs(float(mts) - float(self._last_pose_ts_printed)) > 1e-6:
                self._last_pose_ts_printed = float(mts)
                print(
                    "[RegMetrics] scale=", metrics.get("scale_applied_to_target"),
                    "diag_src=", metrics.get("diag_src"),
                    "voxel=", metrics.get("voxel"),
                    "ransac_fit=", metrics.get("ransac_fitness"),
                    "gicp_fit=", metrics.get("gicp_fitness"),
                )
                if "src_prep" in metrics:
                    print("[RegMetrics] src_prep=", metrics.get("src_prep"))
                if "tgt_prep" in metrics:
                    print("[RegMetrics] tgt_prep=", metrics.get("tgt_prep"))
                print("[Pose] T_tgt_to_scene (= T_scene_from_camera @ inv(T_src_to_tgt))")
                print(pretty_pose(T_tgt_to_scene))
                print("[State] A_asset_raw_to_scene retains the registration scale")
                print(pretty_similarity(A_asset_raw_to_scene))
                self._save_pose(T_tgt_to_scene, A_asset_raw_to_scene, metrics)
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

    default_calibration_path = os.path.join(
        _HERE,
        "configs",
        "calibration",
        "kinect_camera_space_to_scene.json",
    )
    camera_to_scene_path = os.environ.get(
        "DGSRSIM_T_SCENE_FROM_CAMERA",
        default_calibration_path,
    ).strip()
    camera_is_scene_frame = os.environ.get("DGSRSIM_CAMERA_IS_SCENE_FRAME", "0").strip() == "1"
    if camera_to_scene_path:
        T_SCENE_FROM_CAMERA = load_transform_4x4(camera_to_scene_path)
        calibration_info = {
            "file_name": Path(camera_to_scene_path).name,
            "sha256": sha256_file(camera_to_scene_path),
        }
        if Path(camera_to_scene_path).suffix.lower() == ".json":
            calibration_payload = json.loads(Path(camera_to_scene_path).read_text(encoding="utf-8"))
            if isinstance(calibration_payload, dict):
                for key in (
                    "calibration_id",
                    "calibration_type",
                    "source_frame",
                    "target_frame",
                    "units",
                    "provenance",
                ):
                    if key in calibration_payload:
                        calibration_info[key] = calibration_payload[key]
    elif camera_is_scene_frame:
        T_SCENE_FROM_CAMERA = np.eye(4, dtype=np.float64)
        calibration_info = {
            "calibration_type": "explicit_runtime_frame_coincidence",
            "source_frame": "Kinect CameraSpace",
            "target_frame": "metric simulation scene frame",
        }
        print("[Frames] DGSRSIM_CAMERA_IS_SCENE_FRAME=1; using an explicitly declared identity extrinsic.")
    else:
        raise RuntimeError(
            "Set DGSRSIM_T_SCENE_FROM_CAMERA to a calibrated 4x4 NPY/JSON/text transform. "
            "Use DGSRSIM_CAMERA_IS_SCENE_FRAME=1 only when the simulation scene frame is explicitly defined as CameraSpace."
        )

    FLIP_Y = False
    FLIP_Z = False

    p = MyProcessor(
        fastsam_model_path=FASTSAM_MODEL,
        thresh_ply=THRESH_PLY,
        T_scene_from_camera=T_SCENE_FROM_CAMERA,
        calibration_info=calibration_info,
        object_id=os.environ.get("DGSRSIM_OBJECT_ID"),
        flip_y=FLIP_Y,
        flip_z=FLIP_Z
    )
    try:
        while True:
            if not p.run_once():
                break
    finally:
        p.close()
