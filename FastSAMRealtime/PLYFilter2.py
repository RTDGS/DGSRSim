# -*- coding: utf-8 -*-
import os
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from typing import Dict, Optional, List, Tuple


def load_ply_props(path: str) -> Dict[str, np.ndarray]:
    ply = PlyData.read(path)
    v = ply["vertex"].data
    names = v.dtype.names
    props = {n: np.asarray(v[n]) for n in names}
    for req in ["x", "y", "z"]:
        if req not in props:
            raise ValueError(f"PLY missing '{req}'")
    return props


def save_ply_props(path: str, props: Dict[str, np.ndarray], comments: Optional[List[str]] = None) -> None:
    keys = list(props.keys())
    if len(keys) == 0:
        raise ValueError("props is empty")

    N = len(props[keys[0]])
    for k in keys:
        if len(props[k]) != N:
            raise ValueError(f"Property '{k}' length mismatch")

    dtype = [(k, props[k].dtype) for k in keys]
    data = np.empty(N, dtype=dtype)
    for k in keys:
        data[k] = props[k]

    el = PlyElement.describe(data, "vertex")
    ply = PlyData([el], text=False)
    if comments:
        ply.comments = comments
    ply.write(path)


def obb_from_points_pca(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        center: (3,)
        axes:   (3,3), columns are orthonormal axes
        half:   (3,), half extents
    """
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("pts must be (N,3)")
    if pts.shape[0] < 3:
        center = pts.mean(axis=0)
        axes = np.eye(3, dtype=np.float64)
        half = np.zeros(3, dtype=np.float64)
        return center, axes, half

    center = pts.mean(axis=0)
    X = pts - center
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    axes = Vt.T  # columns are principal axes

    proj = X @ axes
    mn = proj.min(axis=0)
    mx = proj.max(axis=0)

    half = (mx - mn) / 2.0
    box_center_local = (mx + mn) / 2.0
    box_center = center + axes @ box_center_local
    return box_center, axes, half


def points_in_obb(points: np.ndarray, center: np.ndarray, axes: np.ndarray, half: np.ndarray) -> np.ndarray:
    local = (points - center) @ axes
    return np.all(np.abs(local) <= half, axis=1)


def points_in_aabb(points: np.ndarray, mn: np.ndarray, mx: np.ndarray) -> np.ndarray:
    return np.all((points >= mn) & (points <= mx), axis=1)


def post_statistical_clean_mask(
    pts: np.ndarray,
    k: int = 16,
    std_mul: float = 2.0,
) -> np.ndarray:
    """
    在 crop 后做一次二次统计清理：
    - 对每个点计算其 k 近邻平均距离
    - 平均距离显著偏大的点视为孤立点并剔除

    返回:
        keep_mask: (N,) bool
    """
    N = pts.shape[0]
    if N < max(20, k + 1):
        return np.ones(N, dtype=bool)

    kk = min(max(k + 1, 2), N)  # +1 because query includes self
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=kk)

    # 第 0 列通常是自己到自己的距离 0
    if dists.ndim == 1:
        # 极端小样本兜底
        mean_d = dists.astype(np.float64)
    else:
        mean_d = dists[:, 1:].mean(axis=1).astype(np.float64)

    mu = float(mean_d.mean())
    sigma = float(mean_d.std())
    thr = mu + float(std_mul) * (sigma + 1e-12)
    keep = mean_d <= thr
    return keep


def dense_core_box_crop(
    props: Dict[str, np.ndarray],
    mode: str = "obb",                 # "obb" or "aabb"
    k: int = 24,                       # kNN for density estimate
    core_ratio: float = 0.05,          # densest 5% as core
    margin_ratio: float = 0.12,        # expand box by 12%
    max_points_for_knn: Optional[int] = 8000,
    post_stat_k: int = 16,             # 二次统计清理用
    post_stat_std: float = 2.0,        # 二次统计清理阈值
    sparsity_ratio: float = 1.0,       # 1.0 = no sparsity
    seed: int = 0,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    Stronger version:
      1) kNN density estimate
      2) take densest core
      3) build OBB/AABB from core
      4) keep points inside box
      5) post statistical clean inside cropped set
      6) optional random sparsify at the end
    """
    for req in ["x", "y", "z"]:
        if req not in props:
            raise ValueError(f"props missing '{req}'")

    x = props["x"].astype(np.float64)
    y = props["y"].astype(np.float64)
    z = props["z"].astype(np.float64)
    P = np.stack([x, y, z], axis=1)
    N = P.shape[0]

    if N < 10:
        return props, {
            "N_in": float(N),
            "N_after_crop": float(N),
            "N_after_post": float(N),
            "N_out": float(N),
            "keep_ratio_crop": 1.0,
            "keep_ratio_post": 1.0,
            "keep_ratio_final": 1.0,
            "k": float(k),
            "core_ratio": float(core_ratio),
            "margin_ratio": float(margin_ratio),
            "post_stat_k": float(post_stat_k),
            "post_stat_std": float(post_stat_std),
            "sparsity_ratio": float(sparsity_ratio),
            "used_downsample": 0.0,
            "knn_points": float(N),
        }

    if not (0.0 < core_ratio <= 1.0):
        raise ValueError("core_ratio must be in (0,1]")
    if margin_ratio < 0.0:
        raise ValueError("margin_ratio must be >= 0")
    if not (0.0 < sparsity_ratio <= 1.0):
        raise ValueError("sparsity_ratio must be in (0,1]")

    rng = np.random.default_rng(seed)

    # -------------------------------------------------
    # 1) Optional downsample for kNN density estimation
    # -------------------------------------------------
    if max_points_for_knn is not None and N > max_points_for_knn:
        idx_sample = rng.choice(N, size=max_points_for_knn, replace=False)
        P_knn = P[idx_sample]
        used_downsample = 1.0
    else:
        idx_sample = None
        P_knn = P
        used_downsample = 0.0

    # -------------------------------------------------
    # 2) Density estimation by kNN distance
    # -------------------------------------------------
    tree = cKDTree(P_knn)
    kk = min(max(int(k), 2), P_knn.shape[0])
    dists, _ = tree.query(P_knn, k=kk)
    if dists.ndim == 1:
        dk = dists.astype(np.float64)
    else:
        dk = dists[:, -1].astype(np.float64)  # k-th neighbor distance

    # Core = densest points => smallest dk
    m = max(16, int(core_ratio * P_knn.shape[0]))
    m = min(m, P_knn.shape[0])
    core_ids_local = np.argpartition(dk, m - 1)[:m]
    core_pts = P_knn[core_ids_local]

    # -------------------------------------------------
    # 3) Build box on core
    # -------------------------------------------------
    if mode == "aabb":
        mn = core_pts.min(axis=0)
        mx = core_pts.max(axis=0)
        size = mx - mn
        mn2 = mn - margin_ratio * size
        mx2 = mx + margin_ratio * size
        keep_crop = points_in_aabb(P, mn2, mx2)
    elif mode == "obb":
        center, axes, half = obb_from_points_pca(core_pts)
        half2 = half * (1.0 + margin_ratio)
        keep_crop = points_in_obb(P, center, axes, half2)
    else:
        raise ValueError("mode must be 'obb' or 'aabb'")

    N_after_crop = int(np.count_nonzero(keep_crop))
    if N_after_crop == 0:
        # 极端情况下返回 core 对应盒内为空，直接回退到 core 最近的点集不现实，这里简单回退原 props
        return props, {
            "N_in": float(N),
            "N_after_crop": 0.0,
            "N_after_post": 0.0,
            "N_out": float(N),
            "keep_ratio_crop": 0.0,
            "keep_ratio_post": 0.0,
            "keep_ratio_final": 1.0,
            "k": float(k),
            "core_ratio": float(core_ratio),
            "margin_ratio": float(margin_ratio),
            "post_stat_k": float(post_stat_k),
            "post_stat_std": float(post_stat_std),
            "sparsity_ratio": float(sparsity_ratio),
            "used_downsample": float(used_downsample),
            "knn_points": float(P_knn.shape[0]),
        }

    P_crop = P[keep_crop]

    # -------------------------------------------------
    # 4) Post statistical clean inside cropped points
    # -------------------------------------------------
    keep_post_local = post_statistical_clean_mask(
        P_crop,
        k=post_stat_k,
        std_mul=post_stat_std,
    )
    N_after_post = int(np.count_nonzero(keep_post_local))

    crop_indices = np.where(keep_crop)[0]
    post_indices = crop_indices[keep_post_local]

    out_props = {k0: np.asarray(v)[post_indices] for k0, v in props.items()}

    # -------------------------------------------------
    # 5) Optional sparsify at the end
    # -------------------------------------------------
    if 0.0 < sparsity_ratio < 1.0 and len(out_props["x"]) > 0:
        n_sparse = max(1, int(len(out_props["x"]) * sparsity_ratio))
        sparse_idx = rng.choice(len(out_props["x"]), size=n_sparse, replace=False)
        out_props = {k0: v[sparse_idx] for k0, v in out_props.items()}

    N_out = int(len(out_props["x"]))

    stats = {
        "N_in": float(N),
        "N_after_crop": float(N_after_crop),
        "N_after_post": float(N_after_post),
        "N_out": float(N_out),
        "keep_ratio_crop": float(N_after_crop / N),
        "keep_ratio_post": float(N_after_post / N),
        "keep_ratio_final": float(N_out / N),
        "k": float(k),
        "core_ratio": float(core_ratio),
        "margin_ratio": float(margin_ratio),
        "post_stat_k": float(post_stat_k),
        "post_stat_std": float(post_stat_std),
        "sparsity_ratio": float(sparsity_ratio),
        "used_downsample": float(used_downsample),
        "knn_points": float(P_knn.shape[0]),
    }
    return out_props, stats


if __name__ == "__main__":
    import argparse
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _RT_OUT = os.environ.get("DGSRSIM_RT_PLY_OUT", os.path.join(_HERE, "rt_ply_out"))

    ap = argparse.ArgumentParser()
    ap.add_argument("--in_ply", default=os.path.join(_RT_OUT, "yumi.ply"))
    ap.add_argument("--out_ply", default=os.path.join(_RT_OUT, "Yumipoint_cloud.ply"))

    ap.add_argument("--mode", choices=["obb", "aabb"], default="obb")

    # 改成更合理默认值
    ap.add_argument("--k", type=int, default=24,
                    help="kNN for density estimate; recommended 16~32, do NOT use 1")
    ap.add_argument("--core_ratio", type=float, default=0.05,
                    help="Take densest core_ratio points as core")
    ap.add_argument("--margin_ratio", type=float, default=0.12,
                    help="Expand OBB/AABB by this ratio")

    ap.add_argument("--max_points_for_knn", type=int, default=8000,
                    help="Downsample only for density estimation when N is huge")

    # 新增：crop 后二次统计清理
    ap.add_argument("--post_stat_k", type=int, default=16,
                    help="k for post statistical clean")
    ap.add_argument("--post_stat_std", type=float, default=2.0,
                    help="std multiplier for post statistical clean")

    # 保留稀疏化，但默认不做
    ap.add_argument("--sparsity_ratio", type=float, default=1.0,
                    help="Final random sparsify ratio: 1.0 = no sparsity, 0.1 = keep 10%%")

    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    props = load_ply_props(args.in_ply)

    out_props, stats = dense_core_box_crop(
        props,
        mode=args.mode,
        k=args.k,
        core_ratio=args.core_ratio,
        margin_ratio=args.margin_ratio,
        max_points_for_knn=args.max_points_for_knn,
        post_stat_k=args.post_stat_k,
        post_stat_std=args.post_stat_std,
        sparsity_ratio=args.sparsity_ratio,
        seed=args.seed,
    )

    comments = [
        "dense_core_box_crop stronger: kNN density core -> OBB/AABB crop -> post statistical clean -> optional sparsify",
        f"mode={args.mode}, k={args.k}, core_ratio={args.core_ratio}, margin_ratio={args.margin_ratio}",
        f"post_stat_k={args.post_stat_k}, post_stat_std={args.post_stat_std}",
        f"sparsity_ratio={args.sparsity_ratio}",
        f"keep_ratio_crop={stats['keep_ratio_crop']:.6f}",
        f"keep_ratio_post={stats['keep_ratio_post']:.6f}",
        f"keep_ratio_final={stats['keep_ratio_final']:.6f}",
    ]

    save_ply_props(args.out_ply, out_props, comments=comments)

    print("Done. Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
