# -*- coding: utf-8 -*-
raise SystemExit("Legacy RGB-D snapshot disabled. Use rt_seg_strict_align_cut_object_pcd5.py.")

import os
import time
import threading
import copy
from dataclasses import dataclass
from typing import Optional, Tuple, Any, Dict

import cv2
import numpy as np
import open3d as o3d

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
        self.pcd.colors = o3d.utility.Vector3dVector(
            (rgb_u8.astype(np.float32) / 255.0).astype(np.float64, copy=False)
        )

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
            raise RuntimeError(
                f"downsample empty: src_down={len(src_down.points)} tgt_down={len(tgt_down.points)} voxel={voxel}"
            )

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
