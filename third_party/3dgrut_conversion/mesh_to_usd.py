#!/usr/bin/env python3
"""
Write a colored triangle mesh to an Isaac Sim friendly USD ASCII file.

This is a direct USD writer fallback for environments where Blender's USD
export operator is unavailable or incomplete.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, Vt


def as_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        geometries = [geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not geometries:
            raise RuntimeError(f"No mesh geometry found in {path}")
        return trimesh.util.concatenate(geometries)
    if not isinstance(loaded, trimesh.Trimesh):
        raise RuntimeError(f"Unsupported mesh type from {path}: {type(loaded).__name__}")
    return loaded


def vertex_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    colors = getattr(mesh.visual, "vertex_colors", None)
    if colors is None or len(colors) != len(mesh.vertices):
        return np.full((len(mesh.vertices), 3), 0.75, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32)
    if colors.max(initial=1.0) > 1.0:
        colors = colors / 255.0
    return np.clip(colors[:, :3], 0.0, 1.0).astype(np.float32)


def write_usd(mesh: trimesh.Trimesh, output: Path, root_path: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    colors = vertex_colors(mesh)

    mesh.fix_normals()
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)

    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, root_path)
    stage.SetDefaultPrim(root.GetPrim())

    usd_mesh = UsdGeom.Mesh.Define(stage, f"{root_path}/Mesh")
    usd_mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)

    points = Vt.Vec3fArray([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in vertices])
    usd_mesh.CreatePointsAttr(points)
    usd_mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
    usd_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(faces.reshape(-1).tolist()))
    usd_mesh.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in normals]))
    usd_mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

    extent = UsdGeom.PointBased.ComputeExtent(points)
    usd_mesh.CreateExtentAttr(extent)

    primvars = UsdGeom.PrimvarsAPI(usd_mesh)
    color_primvar = primvars.CreatePrimvar(
        "displayColor",
        Sdf.ValueTypeNames.Color3fArray,
        UsdGeom.Tokens.vertex,
    )
    color_primvar.Set(Vt.Vec3fArray([Gf.Vec3f(float(r), float(g), float(b)) for r, g, b in colors]))

    stage.GetRootLayer().Save()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert GLB/OBJ/PLY mesh to USD ASCII for Isaac Sim.")
    parser.add_argument("input", nargs="?", default="assets/point_cloud_mesh.glb")
    parser.add_argument("--output", "-o", default="assets/point_cloud_mesh_isaac.usda")
    parser.add_argument("--root-path", default="/World/PointCloudMesh")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    mesh = as_mesh(input_path)
    write_usd(mesh, output_path, args.root_path)
    print(f"[load] {input_path.resolve()}")
    print(f"[mesh] vertices={len(mesh.vertices)} faces={len(mesh.faces)}")
    print(f"[save] {output_path.resolve()}")


if __name__ == "__main__":
    main()
