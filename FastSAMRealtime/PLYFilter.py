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
    N = len(props[keys[0]])
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
    Returns (center, axes, half_extents).
    axes: (3,3) columns are orthonormal axes.
    """
    center = pts.mean(axis=0)
    X = pts - center
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    axes = Vt.T
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


def dense_core_box_crop(
    props: Dict[str, np.ndarray],
    mode: str = "obb",          # "obb" or "aabb"
    k: int = 32,                # kNN for density estimate
    core_ratio: float = 0.05,   # take densest 5% as core
    margin_ratio: float = 0.08, # expand box by 8%
    max_points_for_knn: Optional[int] = None,  # optional downsample for kNN if N huge
    seed: int = 0,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    Fast crop: kNN density -> densest core -> (OBB/AABB) -> keep box-in points.
    """
    x = props["x"].astype(np.float64)
    y = props["y"].astype(np.float64)
    z = props["z"].astype(np.float64)
    P = np.stack([x, y, z], axis=1)
    N = P.shape[0]
    if N < 50:
        return props, {"N_in": float(N), "N_out": float(N), "keep_ratio": 1.0}

    rng = np.random.default_rng(seed)

    # Optional downsample for kNN density estimation (keeps speed for very large N)
    if max_points_for_knn is not None and N > max_points_for_knn:
        idx_sample = rng.choice(N, size=max_points_for_knn, replace=False)
        P_knn = P[idx_sample]
    else:
        idx_sample = None
        P_knn = P

    tree = cKDTree(P_knn)
    kk = min(max(k, 2), P_knn.shape[0])
    dists, _ = tree.query(P_knn, k=kk)  # includes self at [:,0]
    dk = dists[:, -1]  # k-th neighbor distance

    # Core = densest points (small dk)
    m = max(16, int(core_ratio * P_knn.shape[0]))
    core_ids_local = np.argpartition(dk, m)[:m]
    core_pts = P_knn[core_ids_local]

    # Build box on core
    if mode == "aabb":
        mn = core_pts.min(axis=0)
        mx = core_pts.max(axis=0)
        size = mx - mn
        mn2 = mn - margin_ratio * size
        mx2 = mx + margin_ratio * size
        keep = points_in_aabb(P, mn2, mx2)
    elif mode == "obb":
        center, axes, half = obb_from_points_pca(core_pts)
        half2 = half * (1.0 + margin_ratio)
        keep = points_in_obb(P, center, axes, half2)
    else:
        raise ValueError("mode must be 'obb' or 'aabb'")

    out_props = {k: np.asarray(v)[keep] for k, v in props.items()}
    stats = {
        "N_in": float(N),
        "N_out": float(keep.sum()),
        "keep_ratio": float(keep.mean()),
        "k": float(k),
        "core_ratio": float(core_ratio),
        "margin_ratio": float(margin_ratio),
        "used_downsample": float(0.0 if idx_sample is None else 1.0),
        "knn_points": float(P_knn.shape[0]),
    }
    return out_props, stats


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    # ap.add_argument("--in_ply", default='rt_ply_out/yumi.ply')
    # ap.add_argument("--out_ply", default='rt_ply_out/Yumipoint_cloud.ply')
    ap.add_argument("--in_ply", default=r'E:\code\FastSAM\rt_ply_out\2.ply')
    ap.add_argument("--out_ply", default='rt_ply_out/astronaut.ply')
    ap.add_argument("--mode", choices=["obb", "aabb"], default="obb")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--core_ratio", type=float, default=0.1)
    ap.add_argument("--margin_ratio", type=float, default=0.1)
    ap.add_argument("--max_points_for_knn", type=int, default=None)  # e.g. 200000
    args = ap.parse_args()

    props = load_ply_props(args.in_ply)
    out_props, stats = dense_core_box_crop(
        props,
        mode=args.mode,
        k=args.k,
        core_ratio=args.core_ratio,
        margin_ratio=args.margin_ratio,
        max_points_for_knn=args.max_points_for_knn,
    )

    comments = [
        "dense_core_box_crop: kNN density core -> box crop",
        f"mode={args.mode}, k={args.k}, core_ratio={args.core_ratio}, margin_ratio={args.margin_ratio}",
        f"keep_ratio={stats['keep_ratio']:.6f}",
    ]
    save_ply_props(args.out_ply, out_props, comments=comments)

    print("Done. Stats:", stats)





