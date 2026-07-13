# -*- coding: utf-8 -*-
"""
leisaac/utils/runtime_mug_factory.py

Factory utilities for spawning:
- /World/envs/env_i/RuntimeMug_proxy  (rigid body proxy)
- visual under proxy (USDZ reference)
- collision disable under visual
- either auto-fit visual scale or retain raw asset coordinates for state similarity
- set proxy rigidbody kinematic for absolute pose driving

Depends on:
- leisaac.utils.physics_prims helpers
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence, Tuple, Optional

import numpy as np

from leisaac.utils.physics_prims import (
    set_xform_trs,
    get_prim_world_aabb_size,
    make_rigidbody_kinematic,
    create_proxy_rigid_box,
    spawn_usdz_under_parent,
    disable_collisions_under,
)


AutoFitAxis = Literal["x", "y", "z", "max"]


@dataclass
class RuntimeMugSpec:
    # prim paths
    proxy_name: str = "RuntimeMug_proxy"
    visual_child_name: str = "visual"

    # initial proxy transform
    proxy_pos: Tuple[float, float, float] = (0.72, -0.73, -0.277)
    proxy_quat_wxyz: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    # proxy physical size (also used as "target size" for auto-fit by default)
    target_visual_size_m: Tuple[float, float, float] = (0.08, 0.08, 0.12)
    proxy_size_asset_units: Optional[Tuple[float, float, float]] = None

    density: float = 300.0
    proxy_visible: bool = True

    # visual
    usdz_path: str = ""  # absolute path recommended
    visual_base_scale: float = 1.0  # initial local scale before auto-fit

    # auto-fit
    auto_fit_axis: AutoFitAxis = "z"
    extra_visual_scale: float = 1.0  # optional multiplier after auto-fit
    consume_state_similarity: bool = False

    # rigidbody
    kinematic: bool = True
    disable_gravity: bool = True

    # NEW: 手动把 visual / geom 都搬到“父节点中心”
    visual_local_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    visual_local_quat_wxyz: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    geom_local_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    geom_local_quat_wxyz: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

def _compute_fit_scale(
    current_size_xyz: Sequence[float],
    target_size_xyz: Sequence[float],
    axis: AutoFitAxis,
) -> float:
    cx, cy, cz = float(current_size_xyz[0]), float(current_size_xyz[1]), float(current_size_xyz[2])
    tx, ty, tz = float(target_size_xyz[0]), float(target_size_xyz[1]), float(target_size_xyz[2])

    eps = 1e-9
    if axis == "x":
        return tx / max(cx, eps)
    if axis == "y":
        return ty / max(cy, eps)
    if axis == "z":
        return tz / max(cz, eps)
    # "max": ensure at least one axis matches, keeps within target in a loose sense
    return max(tx / max(cx, eps), ty / max(cy, eps), tz / max(cz, eps))


def spawn_runtime_mug_for_env(stage, env_root: str, spec: RuntimeMugSpec) -> dict:
    if not spec.usdz_path:
        raise ValueError("RuntimeMugSpec.usdz_path is empty.")

    env_root = str(env_root).rstrip("/")
    proxy_path = f"{env_root}/{spec.proxy_name}"
    usdz_abs = str(Path(spec.usdz_path))

    # 1) proxy rigid box（geom 子节点也做手动 TR）
    proxy_size = (
        tuple(spec.proxy_size_asset_units)
        if spec.consume_state_similarity and spec.proxy_size_asset_units is not None
        else tuple(spec.target_visual_size_m)
    )
    create_proxy_rigid_box(
        prim_path=proxy_path,
        pos=tuple(spec.proxy_pos),
        quat_wxyz=tuple(spec.proxy_quat_wxyz),
        size_xyz=proxy_size,
        density=float(spec.density),
        visible=bool(spec.proxy_visible),
        geom_local_pos=tuple(spec.geom_local_pos),
        geom_local_quat_wxyz=tuple(spec.geom_local_quat_wxyz),
    )

    # 2) attach visual
    visual_path = spawn_usdz_under_parent(
        parent_xform_path=proxy_path,
        usdz_path=usdz_abs,
        child_name=spec.visual_child_name,
        scale=float(spec.visual_base_scale),
    )

    # 3) disable collisions under visual
    disable_collisions_under(visual_path)

    # 4) keep raw asset coordinates for a scale-preserving state packet, or auto-fit
    cur = get_prim_world_aabb_size(stage, visual_path)
    if spec.consume_state_similarity:
        s_fit = float(spec.visual_base_scale)
    else:
        s_fit = _compute_fit_scale(cur, spec.target_visual_size_m, spec.auto_fit_axis)
    s_final = float(s_fit) * float(spec.extra_visual_scale)

    # 4.1) 一次性写入 visual 的 local TRS（关键：避免覆盖）
    vis_prim = stage.GetPrimAtPath(visual_path)
    set_xform_trs(
        vis_prim,
        pos=tuple(spec.visual_local_pos),
        quat_wxyz=tuple(spec.visual_local_quat_wxyz),
        scale=float(s_final),
    )

    new_size = get_prim_world_aabb_size(stage, visual_path)

    # 5) kinematic
    if spec.kinematic:
        make_rigidbody_kinematic(stage, proxy_path, disable_gravity=bool(spec.disable_gravity))

    return {
        "proxy_path": proxy_path,
        "visual_path": visual_path,
        "fit_scale": float(s_final),
        "scale_mode": "state_similarity" if spec.consume_state_similarity else "auto_fit",
        "cur_size": tuple(float(x) for x in cur),
        "new_size": tuple(float(x) for x in new_size),
    }



def spawn_runtime_mugs_for_all_envs(
    stage,
    num_envs: int,
    env_root_tpl: str,
    spec: RuntimeMugSpec,
    verbose: bool = True,
) -> list[dict]:
    """
    Spawn RuntimeMug_proxy + visual for all envs.

    Args:
        env_root_tpl: e.g. "/World/envs/env_{i}"
    """
    out: list[dict] = []
    for i in range(int(num_envs)):
        env_root = env_root_tpl.format(i=i)
        info = spawn_runtime_mug_for_env(stage, env_root, spec)
        out.append(info)

        if verbose:
            print(
                f"[runtime-mug] env{i}: proxy={info['proxy_path']} visual={info['visual_path']}\n"
                f"             cur={info['cur_size']}, target={spec.target_visual_size_m}, "
                f"scale={info['fit_scale']:.6f}"
            )
    return out


