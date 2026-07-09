# -*- coding: utf-8 -*-

import time
import threading
import copy
from dataclasses import dataclass
from typing import Optional, Tuple, Any, Dict

import numpy as np
import open3d as o3d


# ============================================================
# Basic utilities
# ============================================================

def load_pcd(p: str, voxel_size: float = 0.0) -> o3d.geometry.PointCloud:
    q = o3d.io.read_point_cloud(p)
    if q.is_empty():
        raise RuntimeError(f"Empty/invalid point cloud: {p}")

    q.remove_non_finite_points()

    if voxel_size is not None and voxel_size > 0:
        q_down = q.voxel_down_sample(float(voxel_size))
        if not q_down.is_empty():
            q = q_down

    return q


def make_pcd_from_xyzrgb(xyz: np.ndarray, rgb_u8: Optional[np.ndarray]) -> o3d.geometry.PointCloud:
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(xyz.astype(np.float64, copy=False))
    if rgb_u8 is not None and rgb_u8.shape[0] == xyz.shape[0]:
        p.colors = o3d.utility.Vector3dVector(
            (rgb_u8.astype(np.float32) / 255.0).astype(np.float64, copy=False)
        )
    return p


def pcd_to_xyzrgb(pcd: o3d.geometry.PointCloud) -> Tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(pcd.points).astype(np.float32)
    cols = np.asarray(pcd.colors)
    if cols.shape[0] == xyz.shape[0] and cols.shape[0] > 0:
        rgb = np.clip(cols * 255.0, 0, 255).astype(np.uint8)
    else:
        rgb = np.zeros((xyz.shape[0], 3), dtype=np.uint8)
    return xyz, rgb


def aabb_diag(pcd: o3d.geometry.PointCloud) -> float:
    if pcd.is_empty():
        return 0.0
    aabb = pcd.get_axis_aligned_bounding_box()
    ext = np.asarray(aabb.get_extent(), dtype=np.float64)
    return float(np.linalg.norm(ext))


def scale_about_center(pcd: o3d.geometry.PointCloud, s: float) -> o3d.geometry.PointCloud:
    q = copy.deepcopy(pcd)
    q.scale(float(s), center=q.get_center())
    return q


def remove_outliers_stat(
    pcd: o3d.geometry.PointCloud,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0
) -> o3d.geometry.PointCloud:
    if pcd.is_empty():
        return pcd
    _, ind = pcd.remove_statistical_outlier(
        nb_neighbors=int(nb_neighbors),
        std_ratio=float(std_ratio)
    )
    if len(ind) == 0:
        return pcd
    return pcd.select_by_index(ind)


def remove_outliers_radius(
    pcd: o3d.geometry.PointCloud,
    nb_points: int = 10,
    radius: float = 0.01
) -> o3d.geometry.PointCloud:
    if pcd.is_empty():
        return pcd
    _, ind = pcd.remove_radius_outlier(
        nb_points=int(nb_points),
        radius=float(radius)
    )
    if len(ind) == 0:
        return pcd
    return pcd.select_by_index(ind)


def estimate_normals(
    pcd: o3d.geometry.PointCloud,
    radius: float,
    max_nn: int = 120
) -> o3d.geometry.PointCloud:
    if pcd.is_empty():
        return pcd
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=float(radius),
            max_nn=int(max_nn)
        )
    )
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


# ============================================================
# Optional DBSCAN fallback
# ============================================================

def keep_largest_cluster_dbscan(
    pcd: o3d.geometry.PointCloud,
    eps: float = 0.02,
    min_points: int = 12,
) -> Tuple[o3d.geometry.PointCloud, Dict[str, Any]]:
    q = copy.deepcopy(pcd)
    stats: Dict[str, Any] = {
        "n_in": int(len(q.points)),
        "n_out": int(len(q.points)),
        "ok": False,
        "reason": "",
    }

    if q.is_empty() or len(q.points) < max(10, min_points):
        stats["ok"] = True
        stats["reason"] = "too_few_points_skip"
        return q, stats

    labels = np.array(q.cluster_dbscan(
        eps=float(eps),
        min_points=int(min_points),
        print_progress=False
    ))

    valid_labels = np.unique(labels[labels >= 0])
    if valid_labels.size == 0:
        stats["ok"] = False
        stats["reason"] = "no_cluster_found"
        return pcd, stats

    best_label = None
    best_size = -1
    cluster_info = []

    for lb in valid_labels:
        ind = np.where(labels == lb)[0]
        sz = int(len(ind))
        cluster_info.append({
            "label": int(lb),
            "size": sz,
        })
        if sz > best_size:
            best_size = sz
            best_label = int(lb)

    keep_idx = np.where(labels == best_label)[0]
    q = q.select_by_index(keep_idx.tolist())

    stats.update({
        "ok": True,
        "reason": "ok",
        "n_clusters": int(len(valid_labels)),
        "selected_label": int(best_label),
        "selected_cluster_size": int(len(keep_idx)),
        "cluster_info": cluster_info,
        "n_out": int(len(q.points)),
    })
    return q, stats


# ============================================================
# Strong one-sided axial tail removal
# ============================================================

def remove_axial_tail_one_sided(
    pcd: o3d.geometry.PointCloud,
    keep_main_ratio: float = 0.80,
    tail_gap_ratio: float = 2.5,
    min_points: int = 30,
    use_robust_center: bool = True,
    side_hard_keep_ratio: float = 0.30,
) -> Tuple[o3d.geometry.PointCloud, Dict[str, Any]]:
    q = copy.deepcopy(pcd)
    stats: Dict[str, Any] = {
        "n_in": int(len(q.points)),
        "n_out": int(len(q.points)),
        "ok": False,
        "reason": "",
    }

    if q.is_empty() or len(q.points) < int(min_points):
        stats["ok"] = True
        stats["reason"] = "too_few_points_skip"
        return q, stats

    pts = np.asarray(q.points, dtype=np.float64)
    n = pts.shape[0]

    center = np.median(pts, axis=0) if use_robust_center else pts.mean(axis=0)

    X = pts - center[None, :]
    cov = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal_axis = eigvecs[:, np.argmax(eigvals)]
    principal_axis = principal_axis / max(np.linalg.norm(principal_axis), 1e-12)

    s = X @ principal_axis

    keep_main_ratio = float(np.clip(keep_main_ratio, 0.50, 0.98))
    alpha = (1.0 - keep_main_ratio) * 50.0
    ql = float(np.percentile(s, alpha))
    qh = float(np.percentile(s, 100.0 - alpha))

    pos_excess = float(np.max(s) - qh)
    neg_excess = float(ql - np.min(s))

    if pos_excess >= neg_excess:
        tail_side = +1
        side_name = "positive"
        side_idx = np.where(s > qh)[0]
        boundary = qh
    else:
        tail_side = -1
        side_name = "negative"
        side_idx = np.where(s < ql)[0]
        boundary = ql

    if side_idx.size < 5:
        stats["ok"] = True
        stats["reason"] = "tail_side_too_small_skip"
        stats["tail_side"] = side_name
        stats["n_out"] = int(len(q.points))
        return q, stats

    s_side = s[side_idx]
    if tail_side > 0:
        order = np.argsort(s_side)
    else:
        order = np.argsort(-s_side)

    side_idx_sorted = side_idx[order]
    s_sorted = s[side_idx_sorted]
    s_mono = -s_sorted if tail_side < 0 else s_sorted

    gaps = np.diff(s_mono)

    remove_idx_stage1 = np.array([], dtype=np.int64)
    cut_value_1 = None
    reason_1 = "skip_gap"
    med_gap = 0.0
    gap_th = 0.0

    if gaps.size > 0:
        med_gap = float(np.median(gaps))
        if med_gap <= 1e-9:
            med_gap = float(np.mean(gaps) + 1e-9)

        gap_th = float(tail_gap_ratio) * med_gap
        gap_idx = np.where(gaps > gap_th)[0]

        if gap_idx.size > 0:
            first_gap = int(gap_idx[0])
            remove_idx_stage1 = side_idx_sorted[first_gap + 1:]
            cut_value_1 = float(s_mono[first_gap + 1])
            reason_1 = "cut_by_gap"

    keep_mask = np.ones((n,), dtype=bool)
    if remove_idx_stage1.size > 0:
        keep_mask[remove_idx_stage1] = False

    # 第二阶段：对保留下来的“尾巴方向半边”做更强的单侧硬截断
    remain_idx = np.where(keep_mask)[0]
    pts2 = pts[remain_idx]
    X2 = pts2 - center[None, :]
    s2 = X2 @ principal_axis

    if tail_side > 0:
        side2_local = np.where(s2 > 0.0)[0]
        order2 = np.argsort(s2[side2_local])
    else:
        side2_local = np.where(s2 < 0.0)[0]
        order2 = np.argsort(-s2[side2_local])

    remove_idx_stage2 = np.array([], dtype=np.int64)
    cut_value_2 = None

    if side2_local.size >= 4:
        side2_local_sorted = side2_local[order2]
        side_hard_keep_ratio = float(np.clip(side_hard_keep_ratio, 0.20, 0.95))
        k = max(1, int(np.ceil(side2_local_sorted.size * side_hard_keep_ratio)))

        if k < side2_local_sorted.size:
            remove_local = side2_local_sorted[k:]
            remove_idx_stage2 = remain_idx[remove_local]

            s2_side_sorted = s2[side2_local_sorted]
            if tail_side < 0:
                s2_side_sorted = -s2_side_sorted
            cut_value_2 = float(s2_side_sorted[k])

    if remove_idx_stage2.size > 0:
        keep_mask[remove_idx_stage2] = False

    keep_idx = np.where(keep_mask)[0]
    if keep_idx.size == 0:
        stats["ok"] = False
        stats["reason"] = "all_removed_unexpected"
        return pcd, stats

    q = q.select_by_index(keep_idx.tolist())

    stats.update({
        "ok": True,
        "reason": f"{reason_1}+side_hard_cut",
        "n_out": int(len(q.points)),
        "tail_side": side_name,
        "boundary": float(boundary),
        "pos_excess": float(pos_excess),
        "neg_excess": float(neg_excess),
        "tail_side_count": int(side_idx.size),
        "removed_count_stage1": int(remove_idx_stage1.size),
        "removed_count_stage2": int(remove_idx_stage2.size),
        "median_gap": float(med_gap),
        "gap_threshold": float(gap_th),
        "cut_value_stage1": None if cut_value_1 is None else float(cut_value_1),
        "cut_value_stage2": None if cut_value_2 is None else float(cut_value_2),
        "principal_axis": principal_axis.astype(np.float64).tolist(),
        "center": center.astype(np.float64).tolist(),
    })
    return q, stats


# ============================================================
# Observation preprocessing
# ============================================================

def preprocess_observation_pcd(
    pcd: o3d.geometry.PointCloud,
    do_statistical: bool = True,
    stat_nb: int = 20,
    stat_std: float = 2.0,
    do_radius: bool = False,
    radius_nb: int = 10,
    radius: float = 0.01,
    voxel_size: float = 0.005,

    do_axial_tail_remove: bool = True,
    axial_keep_main_ratio: float = 0.80,
    axial_tail_gap_ratio: float = 2.5,
    axial_min_points: int = 30,
    axial_side_hard_keep_ratio: float = 0.30,

    do_cluster_fallback: bool = False,
    dbscan_eps: float = 0.02,
    dbscan_min_points: int = 12,
) -> Tuple[o3d.geometry.PointCloud, Dict[str, Any]]:
    q = copy.deepcopy(pcd)
    q.remove_non_finite_points()

    stats: Dict[str, Any] = {
        "n_in": int(len(q.points)),
        "n_after_stat": int(len(q.points)),
        "n_after_radius": int(len(q.points)),
        "n_after_axial_tail": int(len(q.points)),
        "n_after_cluster_fallback": int(len(q.points)),
        "n_after_voxel": int(len(q.points)),
        "axial_tail": {},
        "cluster_fallback": {},
    }

    if q.is_empty():
        stats["ok"] = False
        stats["reason"] = "empty_input"
        return q, stats

    if do_statistical and len(q.points) >= max(5, stat_nb):
        q = remove_outliers_stat(q, nb_neighbors=stat_nb, std_ratio=stat_std)
    stats["n_after_stat"] = int(len(q.points))

    if q.is_empty():
        stats["ok"] = False
        stats["reason"] = "empty_after_stat"
        return q, stats

    if do_radius and len(q.points) >= max(5, radius_nb):
        q = remove_outliers_radius(q, nb_points=radius_nb, radius=radius)
    stats["n_after_radius"] = int(len(q.points))

    if q.is_empty():
        stats["ok"] = False
        stats["reason"] = "empty_after_radius"
        return q, stats

    if do_axial_tail_remove and len(q.points) >= max(10, axial_min_points):
        q, tail_stats = remove_axial_tail_one_sided(
            q,
            keep_main_ratio=axial_keep_main_ratio,
            tail_gap_ratio=axial_tail_gap_ratio,
            min_points=axial_min_points,
            use_robust_center=True,
            side_hard_keep_ratio=axial_side_hard_keep_ratio,
        )
        stats["axial_tail"] = tail_stats
    stats["n_after_axial_tail"] = int(len(q.points))

    if q.is_empty():
        stats["ok"] = False
        stats["reason"] = "empty_after_axial_tail"
        return q, stats

    if do_cluster_fallback and len(q.points) >= max(10, dbscan_min_points):
        q, cc_stats = keep_largest_cluster_dbscan(
            q,
            eps=dbscan_eps,
            min_points=dbscan_min_points,
        )
        stats["cluster_fallback"] = cc_stats
    stats["n_after_cluster_fallback"] = int(len(q.points))

    if q.is_empty():
        stats["ok"] = False
        stats["reason"] = "empty_after_cluster_fallback"
        return q, stats

    if voxel_size is not None and voxel_size > 0 and len(q.points) > 0:
        q = voxel_down(q, voxel_size=voxel_size)
    stats["n_after_voxel"] = int(len(q.points))

    stats["ok"] = True
    stats["reason"] = "ok"
    return q, stats


# ============================================================
# Registration
# ============================================================

def preprocess_fpfh(pcd: o3d.geometry.PointCloud, voxel: float):
    p_down = voxel_down(pcd, voxel)
    estimate_normals(p_down, radius=voxel * 2.5, max_nn=200)
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        p_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5.0, max_nn=300)
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


def refine_gicp(
    src: o3d.geometry.PointCloud,
    tgt: o3d.geometry.PointCloud,
    init_T: np.ndarray,
    max_corr: float,
    max_iter: int = 80
):
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(max_iter))
    res = o3d.pipelines.registration.registration_generalized_icp(
        src, tgt,
        max_correspondence_distance=float(max_corr),
        init=init_T,
        estimation_method=o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
        criteria=criteria
    )
    return res


# ============================================================
# Params
# ============================================================

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

    src_pre_voxel: float = 0.005
    tgt_pre_voxel: float = 0.003
    src_use_radius: bool = False
    src_radius_nb: int = 10
    src_radius: float = 0.01

    src_do_axial_tail_remove: bool = True
    src_axial_keep_main_ratio: float = 0.75
    src_axial_tail_gap_ratio: float = 2.0
    src_axial_min_points: int = 30
    src_axial_side_hard_keep_ratio: float = 0.30

    src_do_cluster_fallback: bool = False
    src_dbscan_eps: float = 0.02
    src_dbscan_min_points: int = 12

    tgt_do_axial_tail_remove: bool = False
    tgt_axial_keep_main_ratio: float = 0.90
    tgt_axial_tail_gap_ratio: float = 3.0
    tgt_axial_min_points: int = 30
    tgt_axial_side_hard_keep_ratio: float = 0.50

    tgt_do_cluster_fallback: bool = False
    tgt_dbscan_eps: float = 0.02
    tgt_dbscan_min_points: int = 12


# ============================================================
# Async registrar
# ============================================================

class AsyncRegistrar:
    def __init__(self, tgt_ply_path: str, params: RegistrationParams):
        self.params = params
        self.tgt_raw = load_pcd(tgt_ply_path, voxel_size=self.params.tgt_pre_voxel)

        self._src_lock = threading.Lock()
        self._latest_src: Optional[o3d.geometry.PointCloud] = None
        self._latest_src_ts: float = 0.0

        self._job_lock = threading.Lock()
        self._trigger = False

        self._res_lock = threading.Lock()
        self._latest_T: Optional[np.ndarray] = None
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

    def _clean_src_for_reg(self, src: o3d.geometry.PointCloud) -> Tuple[o3d.geometry.PointCloud, Dict[str, Any]]:
        src2 = copy.deepcopy(src)
        src2.remove_non_finite_points()

        src2, prep_stats = preprocess_observation_pcd(
            src2,
            do_statistical=self.params.do_outlier_remove,
            stat_nb=self.params.out_nb,
            stat_std=self.params.out_std,
            do_radius=self.params.src_use_radius,
            radius_nb=self.params.src_radius_nb,
            radius=self.params.src_radius,
            voxel_size=self.params.src_pre_voxel,

            do_axial_tail_remove=self.params.src_do_axial_tail_remove,
            axial_keep_main_ratio=self.params.src_axial_keep_main_ratio,
            axial_tail_gap_ratio=self.params.src_axial_tail_gap_ratio,
            axial_min_points=self.params.src_axial_min_points,
            axial_side_hard_keep_ratio=self.params.src_axial_side_hard_keep_ratio,

            do_cluster_fallback=self.params.src_do_cluster_fallback,
            dbscan_eps=self.params.src_dbscan_eps,
            dbscan_min_points=self.params.src_dbscan_min_points,
        )
        return src2, prep_stats

    @staticmethod
    def _crop_by_tgt_aabb(
        src_aligned: o3d.geometry.PointCloud,
        tgt_ref: o3d.geometry.PointCloud,
        expand_ratio: float = 0.20
    ) -> o3d.geometry.PointCloud:
        if src_aligned.is_empty() or tgt_ref.is_empty():
            return src_aligned
        aabb = tgt_ref.get_axis_aligned_bounding_box()
        ext = aabb.get_extent()
        aabb = o3d.geometry.AxisAlignedBoundingBox(
            aabb.get_min_bound() - expand_ratio * ext,
            aabb.get_max_bound() + expand_ratio * ext,
        )
        return src_aligned.crop(aabb)

    def _register_once(
        self,
        src_used: o3d.geometry.PointCloud
    ) -> Tuple[np.ndarray, Dict[str, Any], o3d.geometry.PointCloud]:
        if src_used is None or src_used.is_empty():
            raise RuntimeError("src_used empty")

        t0 = time.perf_counter()

        tgt_used = copy.deepcopy(self.tgt_raw)
        tgt_used.remove_non_finite_points()

        tgt_used, tgt_prep_stats = preprocess_observation_pcd(
            tgt_used,
            do_statistical=self.params.do_outlier_remove,
            stat_nb=self.params.out_nb,
            stat_std=self.params.out_std,
            do_radius=False,
            voxel_size=self.params.tgt_pre_voxel,

            do_axial_tail_remove=self.params.tgt_do_axial_tail_remove,
            axial_keep_main_ratio=self.params.tgt_axial_keep_main_ratio,
            axial_tail_gap_ratio=self.params.tgt_axial_tail_gap_ratio,
            axial_min_points=self.params.tgt_axial_min_points,
            axial_side_hard_keep_ratio=self.params.tgt_axial_side_hard_keep_ratio,

            do_cluster_fallback=self.params.tgt_do_cluster_fallback,
            dbscan_eps=self.params.tgt_dbscan_eps,
            dbscan_min_points=self.params.tgt_dbscan_min_points,
        )

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
                f"downsample empty: src_down={len(src_down.points)} "
                f"tgt_down={len(tgt_down.points)} voxel={voxel}"
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

        metrics: Dict[str, Any] = {
            "ok": True,
            "scale_applied_to_target": float(scale_applied),
            "diag_src": float(diag_src),
            "voxel": float(voxel),
            "max_corr": float(max_corr),

            "ransac_fitness": float(res_ransac.fitness),
            "ransac_rmse": float(res_ransac.inlier_rmse),
            "gicp_fitness": float(res_gicp.fitness),
            "gicp_rmse": float(res_gicp.inlier_rmse),

            "src_n": int(len(src_used.points)),
            "tgt_n": int(len(tgt_used.points)),
            "tgt_prep": tgt_prep_stats,

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
                f"gicp_fit={res_gicp.fitness:.3f} rmse={res_gicp.inlier_rmse:.4f} "
                f"src_n={metrics['src_n']} tgt_n={metrics['tgt_n']}"
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
                src_used, src_prep_stats = self._clean_src_for_reg(src0)

                if src_used.is_empty():
                    with self._res_lock:
                        self._latest_T = None
                        self._latest_metrics = {
                            "ok": False,
                            "reason": "src_empty_after_preprocess",
                            "src_prep": src_prep_stats,
                        }
                        self._latest_ts = src_ts
                        self._latest_tgt_used = None
                        self._latest_src_used = None
                        self._latest_src_aligned = None
                    continue

                T, metrics, tgt_used = self._register_once(src_used)
                metrics["src_prep"] = src_prep_stats

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


# ============================================================
# Viewer
# ============================================================

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
    def _set_pcd_inplace(
        dst: o3d.geometry.PointCloud,
        src: o3d.geometry.PointCloud,
        color_rgb: Tuple[float, float, float] = None,
        keep_src_color: bool = True
    ):
        dst.points = src.points

        if len(dst.points) == 0:
            dst.colors = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
            return

        if keep_src_color:
            src_colors = np.asarray(src.colors)
            src_points = np.asarray(src.points)
            if src_colors.shape[0] == src_points.shape[0] and src_colors.shape[0] > 0:
                dst.colors = o3d.utility.Vector3dVector(src_colors.astype(np.float64, copy=False))
                return

        if color_rgb is None:
            dst.colors = o3d.utility.Vector3dVector(np.zeros((len(dst.points), 3), dtype=np.float64))
        else:
            dst.colors = o3d.utility.Vector3dVector(
                np.tile(np.array(color_rgb, dtype=np.float64)[None, :], (len(dst.points), 1))
            )

    def set_target(self, tgt_pcd: o3d.geometry.PointCloud, reset_view: bool = False):
        if tgt_pcd is None or tgt_pcd.is_empty():
            return
        self._set_pcd_inplace(self.tgt_geom, tgt_pcd, (0.0, 1.0, 0.0), keep_src_color=True)
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
        self._set_pcd_inplace(self.src_geom, src_aligned, (0.0, 0.0, 1.0), keep_src_color=True)
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