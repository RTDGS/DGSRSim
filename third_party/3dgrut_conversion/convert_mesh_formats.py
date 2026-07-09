#!/usr/bin/env python3
"""
Export the reconstructed point-cloud mesh to DCC/simulator formats.

Defaults:
- FBX for Unreal Engine / general DCC import.
- USD for Isaac Sim native import.

This script uses Blender's Python API through the `bpy` package.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def clear_scene(bpy) -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_mesh(bpy, path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif suffix == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    elif suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    elif suffix == ".ply":
        bpy.ops.wm.ply_import(filepath=str(path))
    else:
        raise ValueError(f"Unsupported input format: {path.suffix}")


def mesh_objects(bpy):
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def select_meshes(bpy) -> list:
    meshes = mesh_objects(bpy)
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    if meshes:
        bpy.context.view_layer.objects.active = meshes[0]
    return meshes


def ensure_names(meshes: list, base_name: str) -> None:
    for index, obj in enumerate(meshes):
        obj.name = base_name if index == 0 else f"{base_name}_{index:02d}"
        obj.data.name = f"{obj.name}_mesh"


def export_fbx(bpy, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.fbx(
        filepath=str(output),
        use_selection=True,
        object_types={"MESH"},
        global_scale=1.0,
        apply_unit_scale=True,
        apply_scale_options="FBX_SCALE_UNITS",
        axis_forward="-Z",
        axis_up="Y",
        use_mesh_modifiers=True,
        mesh_smooth_type="FACE",
        colors_type="SRGB",
        prioritize_active_color=True,
        use_triangles=True,
        bake_anim=False,
        add_leaf_bones=False,
        path_mode="AUTO",
        embed_textures=False,
    )


def export_usd(bpy, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.usd_export(
        filepath=str(output),
        selected_objects_only=True,
        export_animation=False,
        export_meshes=True,
        export_lights=False,
        export_cameras=False,
        export_materials=True,
        export_mesh_colors=True,
        export_normals=True,
        triangulate_meshes=True,
        root_prim_path="/World/PointCloudMesh",
        convert_scene_units="METERS",
        meters_per_unit=1.0,
    )


def export_usd_fallback(input_path: Path, output: Path) -> None:
    from mesh_to_usd import as_mesh, write_usd

    mesh = as_mesh(input_path)
    write_usd(mesh, output, "/World/PointCloudMesh")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a mesh to FBX and USD using Blender/bpy.")
    parser.add_argument("input", nargs="?", default="assets/point_cloud_mesh.glb")
    parser.add_argument("--fbx", default="assets/point_cloud_mesh_ue5.fbx")
    parser.add_argument("--usd", default="assets/point_cloud_mesh_isaac.usda")
    parser.add_argument("--name", default="PointCloudMesh")
    parser.add_argument("--skip-fbx", action="store_true")
    parser.add_argument("--skip-usd", action="store_true")
    return parser.parse_args()


def main() -> None:
    import bpy

    args = parse_args()
    input_path = Path(args.input).resolve()
    fbx_path = Path(args.fbx).resolve()
    usd_path = Path(args.usd).resolve()

    clear_scene(bpy)
    import_mesh(bpy, input_path)
    meshes = select_meshes(bpy)
    if not meshes:
        raise RuntimeError(f"No mesh objects imported from {input_path}")

    ensure_names(meshes, args.name)

    print(f"[load] {input_path}")
    print(f"[mesh] objects={len(meshes)} vertices={sum(len(obj.data.vertices) for obj in meshes)} faces={sum(len(obj.data.polygons) for obj in meshes)}")

    if not args.skip_fbx:
        select_meshes(bpy)
        export_fbx(bpy, fbx_path)
        print(f"[save] FBX {fbx_path}")

    if not args.skip_usd:
        select_meshes(bpy)
        try:
            export_usd(bpy, usd_path)
        except Exception as exc:
            print(f"[warn] Blender USD export failed, using direct USD writer: {exc}")
            export_usd_fallback(input_path, usd_path)
        print(f"[save] USD {usd_path}")


if __name__ == "__main__":
    main()
