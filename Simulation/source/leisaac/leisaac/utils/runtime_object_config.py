"""Validated runtime-object declarations for multi-object DGSRSim scenes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Tuple


BINDINGS_SCHEMA = "dgsrsim.simulation_object_bindings.v1"


def _tuple(values, length: int, field_name: str) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise ValueError(f"{field_name} must contain {length} numeric values")
    return tuple(float(value) for value in values)


def _resolve_path(value: str, base: str, assets_root: Path, project_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if base == "assets_root":
        return assets_root / path
    if base == "project_root":
        return project_root / path
    raise ValueError(f"unsupported path base {base!r}; use 'assets_root' or 'project_root'")


@dataclass(frozen=True)
class RuntimeObjectConfig:
    object_id: str
    prim_path_template: str
    proxy_name: str
    usdz_path: Path
    asset_profile_path: Path
    initial_position_m: Tuple[float, float, float]
    initial_quaternion_wxyz: Tuple[float, float, float, float]
    target_visual_size_m: Tuple[float, float, float]
    density_kg_m3: float
    proxy_visible: bool


def load_runtime_object_configs(
    path: str,
    *,
    assets_root: str | Path,
    project_root: str | Path,
) -> Dict[str, RuntimeObjectConfig]:
    """Load enabled object-spawn records from the shared binding file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != BINDINGS_SCHEMA:
        raise ValueError(f"unsupported DGSRSim object-binding schema: {payload.get('schema')!r}")
    objects = payload.get("objects")
    if not isinstance(objects, Mapping) or not objects:
        raise ValueError("object binding file must contain a non-empty 'objects' mapping")

    assets_root = Path(assets_root)
    project_root = Path(project_root)
    result: Dict[str, RuntimeObjectConfig] = {}
    for raw_object_id, raw_record in objects.items():
        object_id = str(raw_object_id).strip()
        if not object_id or not isinstance(raw_record, Mapping):
            raise ValueError(f"invalid runtime-object record for {raw_object_id!r}")
        spawn = raw_record.get("spawn")
        if not isinstance(spawn, Mapping) or not bool(spawn.get("enabled", False)):
            continue
        prim_path_template = str(raw_record.get("prim_path_template", "")).strip()
        if not prim_path_template:
            raise ValueError(f"missing prim_path_template for enabled object {object_id!r}")
        proxy_name = prim_path_template.rstrip("/").rsplit("/", 1)[-1]
        if not proxy_name or "{" in proxy_name:
            raise ValueError(f"invalid runtime proxy path for object {object_id!r}")

        usdz_path = _resolve_path(
            str(spawn.get("asset_path", "")),
            str(spawn.get("asset_path_base", "assets_root")),
            assets_root,
            project_root,
        )
        profile_path = _resolve_path(
            str(spawn.get("asset_profile_json", "")),
            str(spawn.get("asset_profile_base", "project_root")),
            assets_root,
            project_root,
        )
        if not str(spawn.get("asset_path", "")).strip():
            raise ValueError(f"missing asset_path for enabled object {object_id!r}")
        if not str(spawn.get("asset_profile_json", "")).strip():
            raise ValueError(f"missing asset_profile_json for enabled object {object_id!r}")

        result[object_id] = RuntimeObjectConfig(
            object_id=object_id,
            prim_path_template=prim_path_template,
            proxy_name=proxy_name,
            usdz_path=usdz_path,
            asset_profile_path=profile_path,
            initial_position_m=_tuple(
                spawn.get("initial_position_m", [0.0, 0.0, 0.0]), 3, "initial_position_m"
            ),
            initial_quaternion_wxyz=_tuple(
                spawn.get("initial_quaternion_wxyz", [1.0, 0.0, 0.0, 0.0]),
                4,
                "initial_quaternion_wxyz",
            ),
            target_visual_size_m=_tuple(
                spawn.get("target_visual_size_m", [0.08, 0.08, 0.12]),
                3,
                "target_visual_size_m",
            ),
            density_kg_m3=float(spawn.get("density_kg_m3", 300.0)),
            proxy_visible=bool(spawn.get("proxy_visible", True)),
        )
    return result
