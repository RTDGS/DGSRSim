# -*- coding: utf-8 -*-

import os
import copy
from typing import Optional

import numpy as np
import open3d as o3d


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
        self.pcd.colors = o3d.utility.Vector3dVector((rgb_u8.astype(np.float32) / 255.0).astype(np.float64, copy=False))

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
