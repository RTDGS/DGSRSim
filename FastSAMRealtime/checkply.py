# -*- coding: utf-8 -*-
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



# 👉 直接在这里改路径
# PLY_PATH = r"E:\code\FastSAM\rt_ply_out\object_1774005719631.ply"   # 宇航员
# PLY_PATH = r"E:\code\FastSAM\rt_ply_out\object_1774005743584.ply"   # 宇航员
#
# PLY_PATH = r"E:\code\FastSAM\rt_ply_out\object_1774005754857.ply"   # 宇航员
# PLY_PATH = r"E:\code\FastSAM\rt_ply_out\object_1774005763032.ply"   # 宇航员
# PLY_PATH = r"E:\code\FastSAM\rt_ply_out\object_1774005771207.ply"   # 宇航员
#
# PLY_PATH = r"E:\code\FastSAM\rt_ply_out\object_1774006297998.ply"   # 乒乓球
# PLY_PATH = r"E:\code\FastSAM\rt_ply_out\object_1774006306352.ply"   # 乒乓球
#
#
# PLY_PATH = r"E:\code\FastSAM\rt_ply_out\object_1774007087764.ply"   # 锅
# PLY_PATH = r"E:\code\FastSAM\rt_ply_out\object_1774007096586.ply"   # 锅
# PLY_PATH = r"E:\code\FastSAM\rt_ply_out\object_1774007131639.ply"   # 锅
# PLY_PATH = r"E:\code\FastSAM\rt_ply_out\BALL_point_cloud1.ply"
# PLY_PATH = r"E:\code\FastSAM\rt_ply_out\BALL_point_cloud.ply"
# PLY_PATH = r"E:\code\FastSAM\rt_ply_out\guo_point_cloud.ply"
PLY_PATH = r"E:\code\FastSAM\rt_ply_out\Yumipoint_cloud.ply"
#PLY_PATH = r"E:\code\FastSAM\rt_ply_out\object_1774280247462.ply"
#PLY_PATH = r"E:\code\FastSAM\rt_ply_out\plant.ply"
if __name__ == "__main__":
    inspect_ply(PLY_PATH)