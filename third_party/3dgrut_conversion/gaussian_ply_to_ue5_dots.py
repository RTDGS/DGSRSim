#!/usr/bin/env python3
"""
Create a UE5-friendly FBX from a Gaussian-splat PLY.

This version is designed for manual drag-and-drop into UE5:
- It crops the largest high-opacity object cluster by default.
- It renders each Gaussian as a small isotropic icosahedron dot, not a stretched splat.
- It bakes colors into FBX material slots, so UE5 can show color without a custom
  Vertex Color material.

The result is still an approximation, but it behaves much better as a regular
Static Mesh than a Poisson surface or stretched ellipsoid splat mesh.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from gaussian_ply_to_splat_mesh import apply_bounds, largest_cluster_bounds, parse_vec3
from ply_to_mesh import SH_C0, read_vertex_ply, sigmoid


def ico_geometry() -> tuple[np.ndarray, np.ndarray]:
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    vertices = np.array(
        [
            [-1, phi, 0],
            [1, phi, 0],
            [-1, -phi, 0],
            [1, -phi, 0],
            [0, -1, phi],
            [0, 1, phi],
            [0, -1, -phi],
            [0, 1, -phi],
            [phi, 0, -1],
            [phi, 0, 1],
            [-phi, 0, -1],
            [-phi, 0, 1],
        ],
        dtype=np.float32,
    )
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    faces = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=np.int64,
    )
    return vertices, faces


def load_and_filter(path: Path, args: argparse.Namespace):
    vertices, names = read_vertex_ply(path)
    required = {"x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2", "f_dc_0", "f_dc_1", "f_dc_2"}
    missing = sorted(required.difference(names))
    if missing:
        raise ValueError(f"Input PLY is missing Gaussian fields: {missing}")

    points = np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(np.float32)
    opacity = sigmoid(vertices["opacity"].astype(np.float32))
    scales = np.exp(np.column_stack([vertices["scale_0"], vertices["scale_1"], vertices["scale_2"]]).astype(np.float32))
    colors = np.clip(
        0.5 + SH_C0 * np.column_stack([vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]]).astype(np.float32),
        0.0,
        1.0,
    )

    crop_min = args.crop_min
    crop_max = args.crop_max
    auto_bounds = None
    if args.auto_crop_largest_cluster:
        auto_min, auto_max = largest_cluster_bounds(points, opacity, args)
        crop_min = tuple(auto_min.tolist()) if crop_min is None else tuple(np.maximum(auto_min, np.array(crop_min)).tolist())
        crop_max = tuple(auto_max.tolist()) if crop_max is None else tuple(np.minimum(auto_max, np.array(crop_max)).tolist())
        auto_bounds = (np.array(crop_min, dtype=np.float32), np.array(crop_max, dtype=np.float32))

    keep = np.isfinite(points).all(axis=1) & np.isfinite(scales).all(axis=1)
    keep &= opacity >= float(args.min_opacity)
    keep = apply_bounds(points, keep, crop_min, crop_max)
    idx = np.where(keep)[0]
    if args.max_points and len(idx) > args.max_points:
        order = np.argsort(opacity[idx])[::-1]
        idx = idx[order[: args.max_points]]

    return points[idx], colors[idx], opacity[idx], scales[idx], auto_bounds


def quantize_materials(colors: np.ndarray, levels: int) -> tuple[np.ndarray, np.ndarray]:
    bins = np.clip(np.floor(colors * levels), 0, levels - 1).astype(np.int32)
    ids = bins[:, 0] * levels * levels + bins[:, 1] * levels + bins[:, 2]
    unique_ids, inverse = np.unique(ids, return_inverse=True)

    palette = np.zeros((len(unique_ids), 3), dtype=np.float32)
    for i in range(len(unique_ids)):
        palette[i] = colors[inverse == i].mean(axis=0)
    return inverse.astype(np.int32), palette


def build_dot_mesh(points, colors, opacity, scales, args):
    base_vertices, base_faces = ico_geometry()
    n_base_vertices = len(base_vertices)
    n_base_faces = len(base_faces)
    count = len(points)
    if count == 0:
        raise RuntimeError("No points survived filtering.")

    if args.constant_radius > 0:
        radii = np.full(count, float(args.constant_radius), dtype=np.float32)
    else:
        radii = np.cbrt(np.prod(scales, axis=1)) * float(args.radius_scale)
        # Very transparent points get slightly smaller.
        radii *= np.clip(opacity / max(float(args.min_opacity), 1e-4), 0.65, 1.35)
        radii = np.clip(radii, float(args.min_radius), float(args.max_radius)).astype(np.float32)

    mesh_vertices = (base_vertices[None, :, :] * radii[:, None, None] + points[:, None, :]).reshape((-1, 3))
    offsets = (np.arange(count, dtype=np.int64) * n_base_vertices)[:, None, None]
    mesh_faces = (base_faces[None, :, :] + offsets).reshape((-1, 3))

    material_indices, palette = quantize_materials(colors, int(args.color_levels))
    face_material_indices = np.repeat(material_indices, n_base_faces).astype(np.int32)
    return mesh_vertices, mesh_faces, face_material_indices, palette


def export_fbx_with_blender(vertices, faces, face_material_indices, palette, output: Path, name: str) -> None:
    import bpy

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update(calc_edges=False)

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    for i, color in enumerate(palette):
        mat = bpy.data.materials.new(f"M_Dot_{i:03d}")
        mat.diffuse_color = (float(color[0]), float(color[1]), float(color[2]), 1.0)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf is not None:
            try:
                bsdf.inputs["Base Color"].default_value = (float(color[0]), float(color[1]), float(color[2]), 1.0)
                bsdf.inputs["Roughness"].default_value = 0.85
            except Exception:
                pass
        obj.data.materials.append(mat)

    for poly, mat_idx in zip(obj.data.polygons, face_material_indices.tolist()):
        poly.material_index = int(mat_idx)

    bpy.ops.object.shade_smooth()

    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.fbx(
        filepath=str(output.resolve()),
        use_selection=True,
        object_types={"MESH"},
        global_scale=1.0,
        apply_unit_scale=True,
        apply_scale_options="FBX_SCALE_UNITS",
        axis_forward="-Z",
        axis_up="Y",
        use_mesh_modifiers=True,
        mesh_smooth_type="FACE",
        use_triangles=True,
        bake_anim=False,
        add_leaf_bones=False,
        path_mode="COPY",
        embed_textures=False,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Create a drag-and-drop UE5 FBX dot mesh from Gaussian PLY.")
    parser.add_argument("input", nargs="?", default="assets/point_cloud.ply")
    parser.add_argument("--output", "-o", default="assets/ue5_point_cloud_object/PointCloudObjectDots_DragDrop_UE5.fbx")
    parser.add_argument("--name", default="PointCloudObjectDots")
    parser.add_argument("--min-opacity", type=float, default=0.25)
    parser.add_argument("--max-points", type=int, default=14000)
    parser.add_argument("--radius-scale", type=float, default=1.15)
    parser.add_argument("--constant-radius", type=float, default=0.0)
    parser.add_argument("--min-radius", type=float, default=0.006)
    parser.add_argument("--max-radius", type=float, default=0.018)
    parser.add_argument("--color-levels", type=int, default=4, help="4 means up to 64 imported UE materials.")
    parser.add_argument("--crop-min", type=parse_vec3, default=None)
    parser.add_argument("--crop-max", type=parse_vec3, default=None)
    parser.add_argument("--auto-crop-largest-cluster", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cluster-opacity", type=float, default=0.7)
    parser.add_argument("--cluster-eps", type=float, default=0.5)
    parser.add_argument("--cluster-min-points", type=int, default=10)
    parser.add_argument("--cluster-margin", type=float, default=0.08)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points, colors, opacity, scales, auto_bounds = load_and_filter(Path(args.input), args)
    vertices, faces, face_material_indices, palette = build_dot_mesh(points, colors, opacity, scales, args)
    export_fbx_with_blender(vertices, faces, face_material_indices, palette, Path(args.output), args.name)

    print(f"[points] {len(points)}")
    if auto_bounds is not None:
        print(f"[auto-crop] min={auto_bounds[0]} max={auto_bounds[1]}")
    print(f"[mesh] vertices={len(vertices)} faces={len(faces)} materials={len(palette)}")
    print(f"[save] {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
