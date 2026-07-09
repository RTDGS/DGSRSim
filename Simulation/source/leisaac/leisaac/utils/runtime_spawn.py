# source/leisaac/leisaac/utils/runtime_spawn.py

def spawn_usdz_reference(
    prim_path: str,
    usdz_path: str,
    pos=(0.0, 0.0, 0.0),
    quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    scale=1.0,
):
    import omni.usd
    from pxr import UsdGeom, Gf

    stage = omni.usd.get_context().get_stage()

    xform = UsdGeom.Xform.Define(stage, prim_path)
    prim = xform.GetPrim()

    prim.GetReferences().AddReference(usdz_path)

    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    w, x, y, z = quat_wxyz
    xf.AddOrientOp().Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
    xf.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))

    return prim


def make_rigid_body(prim, density: float = 300.0):
    from pxr import UsdPhysics, PhysxSchema

    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.RigidBodyAPI.Apply(prim)

    mass = UsdPhysics.MassAPI.Apply(prim)
    mass.CreateDensityAttr(density)

    physx = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    physx.CreateSolverPositionIterationCountAttr(8)
