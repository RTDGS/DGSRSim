# -*- coding: utf-8 -*-
"""
leisaac/utils/physics_prims.py

All USD/PhysX prim utilities that are related to:
- creating static collider boxes
- creating runtime proxy rigid bodies (with mass/density/CCD/solver iterations)
- setting prim world transform / TRS
- disabling collisions under a subtree
- spawning USDZ visuals under a parent prim
"""

from __future__ import annotations

from typing import Optional, Tuple, Union, List

import numpy as np


# -------------------------
# Xform helpers
# -------------------------
def _find_op(xform, op_type):
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == op_type:
            return op
    return None


def set_xform_trs(prim, pos=None, quat_wxyz=None, scale=None):
    from pxr import UsdGeom, Gf

    xform = UsdGeom.Xformable(prim)

    if pos is not None:
        op = _find_op(xform, UsdGeom.XformOp.TypeTranslate)
        if op is None:
            op = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        px, py, pz = pos
        op.Set(Gf.Vec3d(float(px), float(py), float(pz)))

    if quat_wxyz is not None:
        op = _find_op(xform, UsdGeom.XformOp.TypeOrient)
        if op is None:
            op = xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        w, x, y, z = quat_wxyz
        op.Set(Gf.Quatd(float(w), Gf.Vec3d(float(x), float(y), float(z))))

    if scale is not None:
        op = _find_op(xform, UsdGeom.XformOp.TypeScale)
        if op is None:
            op = xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
        if isinstance(scale, (tuple, list)):
            sx, sy, sz = scale
        else:
            sx = sy = sz = scale
        op.Set(Gf.Vec3d(float(sx), float(sy), float(sz)))


# -------------------------
# World transform helpers
# -------------------------
def get_world_xf(stage, prim_path: str) -> np.ndarray:
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Invalid prim: {prim_path}")

    cache = UsdGeom.XformCache()
    M = cache.GetLocalToWorldTransform(prim)
    rows = [M.GetRow(i) for i in range(4)]
    T = np.array([[float(rows[r][c]) for c in range(4)] for r in range(4)], dtype=np.float64)
    return T


def set_prim_world_matrix(stage, prim_path: str, T_world: np.ndarray):
    from pxr import UsdGeom, Gf

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Invalid prim: {prim_path}")

    T = np.asarray(T_world, dtype=np.float64).reshape(4, 4)

    gf = Gf.Matrix4d(
        float(T[0, 0]), float(T[0, 1]), float(T[0, 2]), float(T[0, 3]),
        float(T[1, 0]), float(T[1, 1]), float(T[1, 2]), float(T[1, 3]),
        float(T[2, 0]), float(T[2, 1]), float(T[2, 2]), float(T[2, 3]),
        float(T[3, 0]), float(T[3, 1]), float(T[3, 2]), float(T[3, 3]),
    )

    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    op = xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)
    op.Set(gf)


def get_prim_world_aabb_size(stage, prim_path: str) -> Tuple[float, float, float]:
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Invalid prim for bbox: {prim_path}")

    included = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
    try:
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), included, True)
    except Exception:
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), included)

    world = bbox_cache.ComputeWorldBound(prim)
    box = world.ComputeAlignedBox()
    mn = box.GetMin()
    mx = box.GetMax()
    size = mx - mn
    return (float(size[0]), float(size[1]), float(size[2]))


# -------------------------
# PhysX helpers
# -------------------------
def _set_physx_attr(api, candidates: List[str], value) -> bool:
    for name in candidates:
        get_name = f"Get{name}Attr"
        if hasattr(api, get_name):
            attr = getattr(api, get_name)()
            if attr:
                try:
                    attr.Set(value)
                    return True
                except Exception:
                    pass
        create_name = f"Create{name}Attr"
        if hasattr(api, create_name):
            try:
                getattr(api, create_name)(value)
                return True
            except Exception:
                pass
    return False


def make_rigidbody_kinematic(stage, prim_path: str, disable_gravity: bool = True):
    from pxr import UsdPhysics, PhysxSchema

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Invalid prim: {prim_path}")

    UsdPhysics.RigidBodyAPI.Apply(prim)
    physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)

    for fn in ["GetKinematicEnabledAttr", "CreateKinematicEnabledAttr"]:
        if hasattr(physx_rb, fn):
            try:
                attr = getattr(physx_rb, fn)()
                if attr is not None:
                    attr.Set(True)
            except Exception:
                pass

    if disable_gravity and hasattr(physx_rb, "CreateDisableGravityAttr"):
        try:
            physx_rb.CreateDisableGravityAttr(True)
        except Exception:
            pass


# -------------------------
# Visual / collision utilities
# -------------------------
def disable_collisions_under(root_prim_path: str):
    import omni.usd
    from pxr import UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_prim_path)
    if not root.IsValid():
        return

    def walk(p):
        yield p
        for c in p.GetChildren():
            yield from walk(c)

    for prim in walk(root):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            api = UsdPhysics.CollisionAPI(prim)
            if api.GetCollisionEnabledAttr():
                api.GetCollisionEnabledAttr().Set(False)
            else:
                api.CreateCollisionEnabledAttr(False)


def spawn_usdz_under_parent(parent_xform_path: str, usdz_path: str, child_name: str = "visual", scale=1.0) -> str:
    import omni.usd
    from pxr import UsdGeom, Gf

    stage = omni.usd.get_context().get_stage()
    parent = stage.GetPrimAtPath(parent_xform_path)
    if not parent.IsValid():
        raise RuntimeError(f"Parent prim not found: {parent_xform_path}")

    visual_path = f"{parent_xform_path}/{child_name}"
    xform = UsdGeom.Xform.Define(stage, visual_path)
    prim = xform.GetPrim()
    prim.GetReferences().AddReference(usdz_path)

    xf = UsdGeom.Xformable(prim)
    s = float(scale)
    sop = _find_op(xf, UsdGeom.XformOp.TypeScale) or xf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
    sop.Set(Gf.Vec3d(s, s, s))
    return visual_path


# -------------------------
# Static collider boxes
# -------------------------
def create_static_collider_box(
    prim_path: str,
    pos=(0.0, 0.0, 0.0),
    quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    size_xyz=(1.0, 1.0, 0.1),
    visible: bool = False,
) -> str:
    import omni.usd
    from pxr import UsdGeom, Gf, UsdPhysics

    stage = omni.usd.get_context().get_stage()

    root = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
    xf = UsdGeom.Xformable(root)

    t_op = _find_op(xf, UsdGeom.XformOp.TypeTranslate) or xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    o_op = _find_op(xf, UsdGeom.XformOp.TypeOrient) or xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)

    px, py, pz = pos
    t_op.Set(Gf.Vec3d(float(px), float(py), float(pz)))
    w, x, y, z = quat_wxyz
    o_op.Set(Gf.Quatd(float(w), Gf.Vec3d(float(x), float(y), float(z))))

    geom_path = f"{prim_path}/geom"
    cube = UsdGeom.Cube.Define(stage, geom_path)
    cube.CreateSizeAttr(1.0)
    gprim = cube.GetPrim()
    gxf = UsdGeom.Xformable(gprim)
    sop = _find_op(gxf, UsdGeom.XformOp.TypeScale) or gxf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
    sx, sy, sz = size_xyz
    sop.Set(Gf.Vec3d(float(sx), float(sy), float(sz)))

    UsdPhysics.CollisionAPI.Apply(gprim)

    if not visible:
        try:
            UsdGeom.Imageable(gprim).MakeInvisible()
        except Exception:
            pass

    return prim_path


def build_scene_proxy_collisions_for_env(env_root: str):
    # # table
    # create_static_collider_box(
    #     prim_path=f"{env_root}/SceneProxy/TableTop",
    #     pos=(0.55, 0.0, 1.38),
    #     quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    #     size_xyz=(2.0, 2.0, 0.05),
    #     visible=True,
    # )
    # floor
    create_static_collider_box(
        prim_path=f"{env_root}/SceneProxy/Floor",
        pos=(0.55, 0.0, -0.469),
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        size_xyz=(6.0, 6.0, 0.1),
        visible=False,
    )


# -------------------------
# Runtime proxy rigid box (dynamic body with mass/density)
# -------------------------
def create_proxy_rigid_box(
    prim_path: str,
    pos=(0.55, 0.0, 0.80),
    quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    size_xyz=(0.08, 0.08, 0.12),
    density: float = 300.0,
    visible: bool = False,
    enable_ccd: bool = True,
    solver_pos_iters: int = 12,
    solver_vel_iters: int = 2,
    geom_local_pos=(0.0, 0.0, 0.0),                 # NEW
    geom_local_quat_wxyz=(1.0, 0.0, 0.0, 0.0),      # NEW
) -> str:
    import omni.usd
    from pxr import UsdGeom, Gf, UsdPhysics, PhysxSchema

    stage = omni.usd.get_context().get_stage()

    xform = UsdGeom.Xform.Define(stage, prim_path)
    prim = xform.GetPrim()
    xf = UsdGeom.Xformable(prim)

    t_op = _find_op(xf, UsdGeom.XformOp.TypeTranslate) or xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    o_op = _find_op(xf, UsdGeom.XformOp.TypeOrient) or xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)

    px, py, pz = pos
    t_op.Set(Gf.Vec3d(float(px), float(py), float(pz)))
    w, x, y, z = quat_wxyz
    o_op.Set(Gf.Quatd(float(w), Gf.Vec3d(float(x), float(y), float(z))))

    geom_path = f"{prim_path}/geom"
    cube = UsdGeom.Cube.Define(stage, geom_path)
    cube.CreateSizeAttr(1.0)
    gprim = cube.GetPrim()
    gxf = UsdGeom.Xformable(gprim)

    # NEW: 先给 geom 加 local TR（对齐/旋转用）
    gt_op = _find_op(gxf, UsdGeom.XformOp.TypeTranslate) or gxf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    go_op = _find_op(gxf, UsdGeom.XformOp.TypeOrient) or gxf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)

    gx, gy, gz = geom_local_pos
    gt_op.Set(Gf.Vec3d(float(gx), float(gy), float(gz)))
    gw, gqx, gqy, gqz = geom_local_quat_wxyz
    go_op.Set(Gf.Quatd(float(gw), Gf.Vec3d(float(gqx), float(gqy), float(gqz))))

    # 再设置 scale（size_xyz 是你想要的 box 尺寸）
    gs = _find_op(gxf, UsdGeom.XformOp.TypeScale) or gxf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
    sx, sy, sz = size_xyz
    gs.Set(Gf.Vec3d(float(sx), float(sy), float(sz)))

    UsdPhysics.CollisionAPI.Apply(gprim)
    UsdPhysics.RigidBodyAPI.Apply(prim)

    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateDensityAttr(float(density))

    physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    _set_physx_attr(physx_rb, ["KinematicEnabled", "Kinematic"], False)
    _set_physx_attr(physx_rb, ["DisableGravity"], False)

    if hasattr(physx_rb, "CreateSolverPositionIterationCountAttr"):
        physx_rb.CreateSolverPositionIterationCountAttr(int(solver_pos_iters))
    if hasattr(physx_rb, "CreateSolverVelocityIterationCountAttr"):
        physx_rb.CreateSolverVelocityIterationCountAttr(int(solver_vel_iters))

    if enable_ccd and hasattr(physx_rb, "CreateEnableCCDAttr"):
        try:
            physx_rb.CreateEnableCCDAttr(True)
        except Exception:
            pass

    if not visible:
        try:
            UsdGeom.Imageable(gprim).MakeInvisible()
        except Exception:
            pass

    return prim_path

