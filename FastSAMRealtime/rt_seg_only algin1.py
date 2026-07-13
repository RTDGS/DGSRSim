# -*- coding: utf-8 -*-
import os
import time
import copy
import numpy as np
import open3d as o3d

from utils.registration_async import (
    load_pcd,
    RegistrationParams,
    AsyncRegistrar,
    RealtimeMatchViewer,
    preprocess_observation_pcd,
    remove_outliers_stat,
    remove_outliers_radius,
    remove_axial_tail_one_sided,
    voxel_down,
)


def pretty_pose(T: np.ndarray) -> str:
    return np.array2string(T, precision=5, suppress_small=True)


def show_single_pcd(
    pcd: o3d.geometry.PointCloud,
    title: str,
    point_size: float = 3.0,
    color=None
):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=960, height=720, visible=True)

    opt = vis.get_render_option()
    if opt is not None:
        opt.point_size = float(point_size)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis.add_geometry(axis)

    show_pcd = copy.deepcopy(pcd)
    if color is not None and (not show_pcd.is_empty()):
        show_pcd.paint_uniform_color(color)

    vis.add_geometry(show_pcd)
    vis.run()
    vis.destroy_window()


def show_stagewise_preprocess(
    raw_pcd: o3d.geometry.PointCloud,
    reg_params: RegistrationParams,
    point_size: float = 3.0,
):
    """
    按阶段显示:
    0) 原始点云
    1) 统计/半径滤波后
    2) 主轴单侧长尾硬截断后
    3) 最终体素化后
    """
    q = copy.deepcopy(raw_pcd)

    # Stage 0: Raw
    show_single_pcd(
        q,
        "Stage 0 - Raw Object PCD",
        point_size=point_size,
        color=[1.0, 0.0, 0.0],
    )

    # Stage 1: statistical / radius
    q1 = copy.deepcopy(q)
    if reg_params.do_outlier_remove and len(q1.points) >= max(5, reg_params.out_nb):
        q1 = remove_outliers_stat(
            q1,
            nb_neighbors=reg_params.out_nb,
            std_ratio=reg_params.out_std,
        )

    if reg_params.src_use_radius and len(q1.points) >= max(5, reg_params.src_radius_nb):
        q1 = remove_outliers_radius(
            q1,
            nb_points=reg_params.src_radius_nb,
            radius=reg_params.src_radius,
        )

    show_single_pcd(
        q1,
        "Stage 1 - After Statistical/Radius Filtering",
        point_size=point_size,
        color=[1.0, 0.6, 0.0],
    )

    # Stage 2: stronger axial one-sided tail removal
    q2 = copy.deepcopy(q1)
    if reg_params.src_do_axial_tail_remove:
        q2, tail_stats = remove_axial_tail_one_sided(
            q2,
            keep_main_ratio=reg_params.src_axial_keep_main_ratio,
            tail_gap_ratio=reg_params.src_axial_tail_gap_ratio,
            min_points=reg_params.src_axial_min_points,
            use_robust_center=True,
            side_hard_keep_ratio=reg_params.src_axial_side_hard_keep_ratio,
        )
    else:
        tail_stats = {"ok": True, "reason": "skip"}

    show_single_pcd(
        q2,
        "Stage 2 - After One-Sided Axial Tail Hard Cut",
        point_size=point_size,
        color=[0.0, 1.0, 0.0],
    )

    # Stage 3: voxel
    q3 = copy.deepcopy(q2)
    if reg_params.src_pre_voxel is not None and reg_params.src_pre_voxel > 0:
        q3 = voxel_down(q3, reg_params.src_pre_voxel)

    show_single_pcd(
        q3,
        "Stage 3 - Final Voxelized Object PCD",
        point_size=point_size,
        color=[0.0, 0.3, 1.0],
    )

    print("\n[Stagewise Stats]")
    print("raw:", np.asarray(q.points).shape)
    print("after_stat_radius:", np.asarray(q1.points).shape)
    print("after_axial_tail_remove:", np.asarray(q2.points).shape, tail_stats)
    print("after_voxel:", np.asarray(q3.points).shape)


if __name__ == "__main__":
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _RT_OUT = os.environ.get("DGSRSIM_RT_PLY_OUT", os.path.join(_HERE, "rt_ply_out"))
    # Override these defaults with DGSRSIM_SOURCE_PLY and DGSRSIM_TARGET_PLY.
    SRC_PLY = os.environ.get("DGSRSIM_SOURCE_PLY", os.path.join(_RT_OUT, "object_latest.ply"))
    TGT_PLY = os.environ.get("DGSRSIM_TARGET_PLY", os.path.join(_RT_OUT, "Yumipoint_cloud.ply"))



    reg_params = RegistrationParams(
        enable_scale_prealign=False,
        do_outlier_remove=True,
        out_nb=20,
        out_std=1.5,
        fpfh_voxel_div=80.0,
        gicp_max_iter=80,
        verbose=True,

        # 基础预处理
        src_pre_voxel=0.005,
        tgt_pre_voxel=0.003,
        src_use_radius=False,
        src_radius_nb=10,
        src_radius=0.01,

        # 更强版：主轴单侧长尾硬截断
        src_do_axial_tail_remove=True,
        src_axial_keep_main_ratio=0.82,
        src_axial_tail_gap_ratio=2.8,
        src_axial_min_points=30,
        src_axial_side_hard_keep_ratio=0.55,

        # 可选后备：DBSCAN 最大簇保留
        src_do_cluster_fallback=False,
        src_dbscan_eps=0.02,
        src_dbscan_min_points=12,

        # target 一般不做长尾裁剪
        tgt_do_axial_tail_remove=False,
        tgt_axial_keep_main_ratio=0.90,
        tgt_axial_tail_gap_ratio=3.0,
        tgt_axial_min_points=30,
        tgt_axial_side_hard_keep_ratio=0.50,

        tgt_do_cluster_fallback=False,
        tgt_dbscan_eps=0.02,
        tgt_dbscan_min_points=12,
    )

    # =========================
    # 1) 先单独可视化各阶段过滤结果
    # =========================
    src_raw = load_pcd(SRC_PLY, voxel_size=0.0)

    # 一次性最终预处理结果（和注册器内部保持一致）
    src_filtered, src_prep_stats = preprocess_observation_pcd(
        src_raw,
        do_statistical=reg_params.do_outlier_remove,
        stat_nb=reg_params.out_nb,
        stat_std=reg_params.out_std,
        do_radius=reg_params.src_use_radius,
        radius_nb=reg_params.src_radius_nb,
        radius=reg_params.src_radius,
        voxel_size=reg_params.src_pre_voxel,

        do_axial_tail_remove=reg_params.src_do_axial_tail_remove,
        axial_keep_main_ratio=reg_params.src_axial_keep_main_ratio,
        axial_tail_gap_ratio=reg_params.src_axial_tail_gap_ratio,
        axial_min_points=reg_params.src_axial_min_points,
        axial_side_hard_keep_ratio=reg_params.src_axial_side_hard_keep_ratio,

        do_cluster_fallback=reg_params.src_do_cluster_fallback,
        dbscan_eps=reg_params.src_dbscan_eps,
        dbscan_min_points=reg_params.src_dbscan_min_points,
    )

    print("\n[Source Raw]")
    print("points:", np.asarray(src_raw.points).shape)

    print("\n[Source Filtered]")
    print("points:", np.asarray(src_filtered.points).shape)

    print("\n[Source Preprocess Stats]")
    print(src_prep_stats)

    # 多阶段可视化
    show_stagewise_preprocess(src_raw, reg_params, point_size=3.0)

    # =========================
    # 2) 再进入配准流程
    # =========================
    reg = AsyncRegistrar(TGT_PLY, params=reg_params)
    reg.start()

    viewer = RealtimeMatchViewer(
        title="Offline Registration View",
        point_size=3.0,
    )

    try:
        viewer.set_target(reg.tgt_raw, reset_view=True)

        reg.submit_src(src_raw, time.time())
        reg.trigger_once()

        got = False
        last_status = None

        while True:
            T_src_to_tgt, metrics, mts, tgt_used, src_used, src_aligned = reg.get_latest()

            if metrics != last_status:
                last_status = dict(metrics)

                if tgt_used is not None and not tgt_used.is_empty():
                    viewer.set_target(tgt_used, reset_view=False)

                if src_aligned is not None and not src_aligned.is_empty():
                    viewer.set_aligned_src(src_aligned, reset_view=False)

                if metrics.get("ok", False) and (not got):
                    got = True

                    print("\n[RegMetrics]")
                    print(metrics)

                    if "src_prep" in metrics:
                        print("\n[Source Preprocess Stats from Registrar]")
                        print(metrics["src_prep"])

                    if "tgt_prep" in metrics:
                        print("\n[Target Preprocess Stats from Registrar]")
                        print(metrics["tgt_prep"])

                    print("\n[Pose] T_src_to_tgt")
                    print(pretty_pose(np.asarray(T_src_to_tgt, dtype=np.float64)))

                    T_tgt_to_scene = np.linalg.inv(np.asarray(T_src_to_tgt, dtype=np.float64))
                    print("\n[Pose] T_tgt_to_scene = inv(T_src_to_tgt)")
                    print(pretty_pose(T_tgt_to_scene))

                elif metrics and (not metrics.get("ok", False)):
                    print("\n[Reg Failed]")
                    print(metrics)

            viewer.tick()
            time.sleep(0.01)

    finally:
        reg.stop()
        viewer.close()
