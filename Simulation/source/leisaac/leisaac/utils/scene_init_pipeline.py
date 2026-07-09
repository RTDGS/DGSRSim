# -*- coding: utf-8 -*-
"""
leisaac/utils/scene_init_pipeline.py

Scene initialization pipeline:
- Set /World/envs/env_i/Scene transform (TRS)
- Build static proxy colliders under /World/envs/env_i/SceneProxy

Dependencies:
- leisaac.utils.physics_prims.set_xform_trs
- leisaac.utils.physics_prims.build_scene_proxy_collisions_for_env
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import math


@dataclass
class SceneInitConfig:
    # which prim to set under each env root
    scene_name: str = "Scene"

    # TRS
    pos: Tuple[float, float, float] = (1.2, -0.095, 0.06)
    # Euler degrees -> quat(wxyz)
    rot_euler_deg: Tuple[float, float, float] = (35.0, -1.0, 5.0)  # (roll, pitch, yaw)
    scale: float = 0.08

    # proxy collisions
    enable_scene_proxy_collisions: bool = True

    # logging
    verbose: bool = True


def _quat_wxyz_from_euler_deg(roll_deg: float, pitch_deg: float, yaw_deg: float):
    cr = math.cos(math.radians(roll_deg) * 0.5)
    sr = math.sin(math.radians(roll_deg) * 0.5)
    cp = math.cos(math.radians(pitch_deg) * 0.5)
    sp = math.sin(math.radians(pitch_deg) * 0.5)
    cy = math.cos(math.radians(yaw_deg) * 0.5)
    sy = math.sin(math.radians(yaw_deg) * 0.5)
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = cy * sp * cr + sy * cp * sr
    z = sy * cp * cr - cy * sp * sr
    return (w, x, y, z)


class SceneInitPipeline:
    def __init__(self, cfg: SceneInitConfig):
        self.cfg = cfg

    def apply(self, env) -> None:
        """
        Apply scene initialization for all envs:
        - set /World/envs/env_i/{scene_name} TRS
        - build /World/envs/env_i/SceneProxy static colliders
        """
        from leisaac.utils.physics_prims import set_xform_trs, build_scene_proxy_collisions_for_env
        import omni.usd

        stage = omni.usd.get_context().get_stage()

        # 1) set scene root TRS for each env
        q = _quat_wxyz_from_euler_deg(*self.cfg.rot_euler_deg)
        for i in range(env.num_envs):
            scene_path = f"/World/envs/env_{i}/{self.cfg.scene_name}"
            prim = stage.GetPrimAtPath(scene_path)
            if not prim.IsValid():
                if self.cfg.verbose:
                    print(f"[scene-init] WARN: prim not found: {scene_path}")
                continue

            set_xform_trs(prim, pos=self.cfg.pos, quat_wxyz=q, scale=self.cfg.scale)

            if self.cfg.verbose:
                print(
                    f"[scene-init] set TRS for {scene_path}: "
                    f"pos={self.cfg.pos}, euler_deg={self.cfg.rot_euler_deg}, scale={self.cfg.scale}"
                )

        # 2) build static proxy colliders
        if self.cfg.enable_scene_proxy_collisions:
            for i in range(env.num_envs):
                env_root = f"/World/envs/env_{i}"
                build_scene_proxy_collisions_for_env(env_root)

            if self.cfg.verbose:
                print("[scene-init] built static proxy colliders under /World/envs/env_i/SceneProxy")
