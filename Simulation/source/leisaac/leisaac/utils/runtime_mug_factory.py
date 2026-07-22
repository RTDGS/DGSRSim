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
from typing import Callable, Literal, Mapping, Sequence, Tuple, Optional

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

    # Optional local offsets for the visual and collision geometry.
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

    # 1) Create the rigid proxy and its collision geometry.
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

    # 4.1) Write the visual's local transform once after scale selection.
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


# Generic names used by the multi-object entry point. The legacy mug-specific
# names remain available for scripts that import them directly.
RuntimeObjectSpec = RuntimeMugSpec
spawn_runtime_object_for_env = spawn_runtime_mug_for_env


def spawn_runtime_objects_for_all_envs(
    stage,
    num_envs: int,
    env_root_tpl: str,
    specs: dict[str, RuntimeObjectSpec],
    verbose: bool = True,
) -> dict[str, list[dict]]:
    """Spawn every configured runtime object in each simulation environment."""
    spawned: dict[str, list[dict]] = {}
    for object_id, spec in specs.items():
        spawned[object_id] = spawn_runtime_mugs_for_all_envs(
            stage=stage,
            num_envs=num_envs,
            env_root_tpl=env_root_tpl,
            spec=spec,
            verbose=False,
        )
        if verbose:
            paths = [entry["proxy_path"] for entry in spawned[object_id]]
            print(f"[runtime-object] object_id={object_id!r} proxies={paths}")
    return spawned


class RuntimeObjectManager:
    """Reconcile configured assets with the live USD stage.

    Adding a specification spawns the object, removing it unspawns the object,
    and changing a specification replaces its prims. State-bundle activation
    and deactivation reuse the same methods without changing the config file.
    """

    def __init__(
        self,
        stage,
        num_envs: int,
        env_root_tpl: str = "/World/envs/env_{i}",
        *,
        object_path_templates: Optional[Mapping[str, str]] = None,
        spawn_all: Callable = spawn_runtime_mugs_for_all_envs,
    ):
        self.stage = stage
        self.num_envs = int(num_envs)
        self.env_root_tpl = str(env_root_tpl)
        self._spawn_all = spawn_all
        self._specs: dict[str, RuntimeObjectSpec] = {}
        self._object_path_templates = dict(object_path_templates or {})
        self._active: set[str] = set()

    @property
    def active_object_ids(self) -> frozenset[str]:
        return frozenset(self._active)

    def set_object_path_templates(self, bindings: Mapping[str, str]) -> None:
        updated = dict(bindings)
        changed = {
            object_id
            for object_id in set(self._object_path_templates) | set(updated)
            if self._object_path_templates.get(object_id) != updated.get(object_id)
        }
        self._object_path_templates = updated
        self._active.difference_update(changed)

    def _proxy_path(self, object_id: str, env_index: int) -> str:
        spec = self._specs.get(object_id)
        if spec is not None:
            return f"{self.env_root_tpl.format(i=env_index).rstrip('/')}/{spec.proxy_name}"
        template = self._object_path_templates.get(object_id)
        if not template:
            raise KeyError(f"no simulation binding for object {object_id!r}")
        return template.format(i=env_index, object_id=object_id)

    def _bound_prims(self, stage, object_id: str):
        paths = [self._proxy_path(object_id, index) for index in range(self.num_envs)]
        return [(path, stage.GetPrimAtPath(path)) for path in paths]

    @staticmethod
    def _is_valid_prim(prim) -> bool:
        return bool(prim and prim.IsValid())

    def _remove(self, stage, object_id: str) -> None:
        for path, prim in self._bound_prims(stage, object_id):
            if self._is_valid_prim(prim):
                stage.RemovePrim(path)
        self._active.discard(object_id)

    def activate(self, stage, object_id: str) -> None:
        if object_id in self._active:
            return
        spec = self._specs.get(object_id)
        bound_prims = self._bound_prims(stage, object_id)
        valid = [self._is_valid_prim(prim) for _, prim in bound_prims]
        if all(valid):
            for _, prim in bound_prims:
                prim.SetActive(True)
        elif any(valid):
            raise RuntimeError(f"object {object_id!r} is present in only part of the environments")
        elif spec is not None:
            self._spawn_all(
                stage=stage,
                num_envs=self.num_envs,
                env_root_tpl=self.env_root_tpl,
                spec=spec,
                verbose=False,
            )
        else:
            raise KeyError(
                f"object {object_id!r} has a binding but no existing prim or enabled spawn specification"
            )
        self._active.add(object_id)
        print(f"[runtime-object] activated object_id={object_id!r}")

    def deactivate(self, stage, object_id: str) -> None:
        for _, prim in self._bound_prims(stage, object_id):
            if self._is_valid_prim(prim):
                prim.SetActive(False)
        self._active.discard(object_id)
        print(f"[runtime-object] deactivated object_id={object_id!r}")

    def reconcile(self, specs: dict[str, RuntimeObjectSpec]) -> None:
        """Apply add, remove, and replacement changes from a config snapshot."""
        old_ids = set(self._specs)
        new_ids = set(specs)
        changed = {
            object_id
            for object_id in old_ids & new_ids
            if self._specs.get(object_id) != specs[object_id]
        }
        removed = old_ids - new_ids
        for object_id in sorted(removed | changed):
            self._remove(self.stage, object_id)
        self._specs = dict(specs)
        for object_id in sorted((new_ids - old_ids) | changed):
            self.activate(self.stage, object_id)


