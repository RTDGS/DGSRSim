# -*- coding: utf-8 -*-
import os
import open3d as o3d
import numpy as np


def inspect_ply(path: str):
    print(f"\n[INFO] Loading: {path}")

    pcd = o3d.io.read_point_cloud(path)

    if pcd.is_empty():
        print("❌ 点云为空！")
        return

    # =============================================
    # ✅ 强制把点云变成纯绿色（解决所有黑色问题）
    # =============================================
    # 1. 清空原有颜色
    pcd.colors = o3d.utility.Vector3dVector(np.zeros((len(pcd.points), 3)))
    # 2. 统一设置绿色
    pcd.paint_uniform_color([0, 1, 0])

    print("\n[INFO] 可视化中...")
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    # =============================================
    # ✅ 关键：关闭光照 + 加大点大小（彻底告别黑色）
    # =============================================
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(coord)

    # 关闭灯光影响 → 颜色完全显示
    opt = vis.get_render_option()
    opt.light_on = False       # 关闭光照
    opt.point_size = 3.0       # 加大点，更容易看清颜色

    vis.run()
    vis.destroy_window()



# Select another asset with DGSRSIM_TARGET_PLY; no machine-specific path is required.
_HERE = os.path.dirname(os.path.abspath(__file__))
_RT_OUT = os.environ.get("DGSRSIM_RT_PLY_OUT", os.path.join(_HERE, "rt_ply_out"))
PLY_PATH = os.environ.get("DGSRSIM_TARGET_PLY", os.path.join(_RT_OUT, "Yumipoint_cloud.ply"))
if __name__ == "__main__":
    inspect_ply(PLY_PATH)
