# -*- coding: utf-8 -*-
raise SystemExit("Legacy simulation snapshot disabled. Use teleop_se3_agent.py.")

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a leisaac teleoperation with leisaac manipulation environments."""

"""Launch Isaac Sim Simulator first."""
import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="leisaac teleoperation for leisaac environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    choices=[
        "keyboard",
        "gamepad",
        "keyboard-world",   # 新增
        "macro-keyboard",
        "so101leader",
        "bi-so101leader",
        "lekiwi-keyboard",
        "lekiwi-gamepad",
        "lekiwi-leader",
    ],
    help="Device for interacting with environment",
)
parser.add_argument(
    "--port", type=str, default="/dev/ttyACM0", help="Port for the teleop device:so101leader, default is /dev/ttyACM0"
)
parser.add_argument(
    "--left_arm_port",
    type=str,
    default="/dev/ttyACM0",
    help="Port for the left teleop device:bi-so101leader, default is /dev/ttyACM0",
)
parser.add_argument(
    "--right_arm_port",
    type=str,
    default="/dev/ttyACM1",
    help="Port for the right teleop device:bi-so101leader, default is /dev/ttyACM1",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed for the environment.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")

# recorder_parameter
parser.add_argument("--record", action="store_true", help="whether to enable record function")
parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")
parser.add_argument(
    "--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to export recorded demos."
)
parser.add_argument("--resume", action="store_true", help="whether to resume recording in the existing dataset file")
parser.add_argument(
    "--num_demos", type=int, default=0, help="Number of demonstrations to record. Set to 0 for infinite."
)

parser.add_argument("--recalibrate", action="store_true", help="recalibrate SO101-Leader or Bi-SO101Leader")
parser.add_argument("--quality", action="store_true", help="whether to enable quality render mode.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import os
import time

import gymnasium as gym
import torch
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import DatasetExportMode, TerminationTermCfg
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.enhance.managers import EnhanceDatasetExportMode, StreamingRecorderManager
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        """
        Args:
            hz (int): frequency to enforce
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration

from pathlib import Path
from leisaac.utils.constant import ASSETS_ROOT

def make_zero_actions(env):
    import torch

    # 最可靠：ActionManager 明确给出了总动作维度
    if hasattr(env, "action_manager") and hasattr(env.action_manager, "total_action_dim"):
        d = int(env.action_manager.total_action_dim)
        return torch.zeros((env.num_envs, d), device=env.device)

    # 退化：再尝试 action_space
    space = getattr(env, "action_space", None)
    if space is not None and hasattr(space, "shape") and space.shape is not None:
        return torch.zeros((env.num_envs,) + tuple(space.shape), device=env.device)

    raise RuntimeError("Cannot infer action dimension. Please print env.action_manager and env.action_space.")

def make_dynamic_rigid_body(root_prim_path: str, density: float = 300.0):
    """Apply RigidBody + Mass to a root prim so it responds to gravity."""
    import omni.usd
    from pxr import UsdPhysics, PhysxSchema

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(root_prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Invalid prim path: {root_prim_path}")

    # Dynamic rigid body
    UsdPhysics.RigidBodyAPI.Apply(prim)

    # Mass properties (density-based)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateDensityAttr(float(density))

    # PhysX tuning (optional but helpful)
    physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    physx_rb.CreateSolverPositionIterationCountAttr(8)
    physx_rb.CreateSolverVelocityIterationCountAttr(1)

def spawn_usdz_under_parent(parent_xform_path: str, usdz_path: str, child_name: str = "visual", scale=1.0):
    """在 parent_xform_path 下创建一个子 Xform，并 reference USDZ 作为可视模型。"""
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
    # 只给可视层 scale（如果你已经缩放了场景，建议这里用 1.0）
    s = float(scale)
    # 避免重复 op：找已有 scale op
    scale_op = None
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            scale_op = op
            break
    if scale_op is None:
        scale_op = xf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
    scale_op.Set(Gf.Vec3d(s, s, s))

    return visual_path

def _set_physx_attr(api, candidates: list[str], value):
    """Try to set PhysX attr using multiple possible method names across versions."""
    for name in candidates:
        # getter: GetXxxAttr
        get_name = f"Get{name}Attr"
        if hasattr(api, get_name):
            attr = getattr(api, get_name)()
            if attr:
                attr.Set(value)
                return True

        # creator: CreateXxxAttr
        create_name = f"Create{name}Attr"
        if hasattr(api, create_name):
            getattr(api, create_name)(value)
            return True
    return False

def create_proxy_rigid_box(
    prim_path: str,
    pos=(0.55, 0.0, 0.80),
    quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    size_xyz=(0.08, 0.08, 0.12),   # 代理碰撞体尺寸（米），你按杯子大小调
    density: float = 300.0,
    visible: bool = False,
):
    """创建一个 Box 代理，自己就是 rigid body + collider，必然受重力。"""
    import omni.usd
    from pxr import UsdGeom, Gf, UsdPhysics, PhysxSchema

    stage = omni.usd.get_context().get_stage()

    # 用 Xform 做容器
    xform = UsdGeom.Xform.Define(stage, prim_path)
    prim = xform.GetPrim()

    xf = UsdGeom.Xformable(prim)

    # translate / orient（用 double 更通用）
    # translate
    t_op = None
    o_op = None
    s_op = None
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            t_op = op
        elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            o_op = op
        elif op.GetOpType() == UsdGeom.XformOp.TypeScale:
            s_op = op

    if t_op is None:
        t_op = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    if o_op is None:
        o_op = xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)

    px, py, pz = pos
    t_op.Set(Gf.Vec3d(float(px), float(py), float(pz)))

    w, x, y, z = quat_wxyz
    o_op.Set(Gf.Quatd(float(w), Gf.Vec3d(float(x), float(y), float(z))))

    # 在 proxy 下创建一个 Cube 作为碰撞几何
    geom_path = f"{prim_path}/geom"
    cube = UsdGeom.Cube.Define(stage, geom_path)
    cube.CreateSizeAttr(1.0)  # unit cube

    # 用 scale 把 unit cube 拉伸成目标尺寸
    gprim = cube.GetPrim()
    gxf = UsdGeom.Xformable(gprim)
    if s_op is None:
        # proxy 根的 scale 不用于碰撞，碰撞几何自己缩放
        pass

    sx, sy, sz = size_xyz
    gs = None
    for op in gxf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            gs = op
            break
    if gs is None:
        gs = gxf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
    gs.Set(Gf.Vec3d(float(sx), float(sy), float(sz)))

    # 碰撞只加在 geom 上
    UsdPhysics.CollisionAPI.Apply(gprim)

    # 刚体加在 proxy 根上
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateDensityAttr(float(density))

    physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)

    # 兼容设置：不同版本可能叫 KinematicEnabled / Kinematic 或 DisableGravity / EnableGravity
    _set_physx_attr(physx_rb, ["KinematicEnabled", "Kinematic"], False)
    _set_physx_attr(physx_rb, ["DisableGravity"], False)

    # solver 迭代（这些通常都有 Create）
    if hasattr(physx_rb, "CreateSolverPositionIterationCountAttr"):
        physx_rb.CreateSolverPositionIterationCountAttr(8)
    if hasattr(physx_rb, "CreateSolverVelocityIterationCountAttr"):
        physx_rb.CreateSolverVelocityIterationCountAttr(1)


    # # 代理可见性：默认不可见（只当碰撞体）
    # if not visible:
    #     UsdGeom.Imageable(gprim).MakeInvisible()

    return prim_path


def disable_collisions_under(root_prim_path: str):
    """如果 USDZ 内部自带碰撞（少数情况），这里可统一禁用，避免和 proxy 双重碰撞。"""
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
        # 如果它已经是 collider，则禁用；不强行 Apply 新 collider（避免污染）
        if UsdPhysics.CollisionAPI(prim):
            api = UsdPhysics.CollisionAPI(prim)
            # collisionEnabled 是标准字段
            if api.GetCollisionEnabledAttr():
                api.GetCollisionEnabledAttr().Set(False)
            else:
                api.CreateCollisionEnabledAttr(False)

def apply_collision_approximation_to_usdz(
    root_prim_path: str,
    approximation: str = "convexHull",  # 常用：convexHull / boundingCube / boundingSphere / none
    include_descendants: bool = True,
):
    """
    给 root_prim_path 下的所有 Mesh prim 加碰撞，并设置 PhysX 碰撞近似方式。
    注意：这是 override，不会修改 usdz 文件本体。
    """
    import omni.usd
    from pxr import UsdGeom, UsdPhysics, PhysxSchema

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_prim_path)
    if not root.IsValid():
        raise RuntimeError(f"Invalid prim path: {root_prim_path}")

    # PhysX 支持的近似 token（不同版本会略有差异）
    # 常见可用：convexHull, boundingCube, boundingSphere, none
    approx_token = approximation

    def iter_prims(p):
        yield p
        for c in p.GetChildren():
            yield from iter_prims(c)

    prims = iter_prims(root) if include_descendants else [root]

    count_mesh = 0
    for prim in prims:
        if prim.GetTypeName() == "Mesh":
            # 1) 碰撞 API
            UsdPhysics.CollisionAPI.Apply(prim)

            # 2) PhysX 碰撞细节
            physx_col = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            # 设置近似方式（如果你版本没有这个 attr，会抛异常；见下方 try/except 版本）
            physx_col.CreateCollisionApproximationAttr().Set(approx_token)

            count_mesh += 1

    print(f"[collision] applied '{approx_token}' to {count_mesh} Mesh prim(s) under {root_prim_path}")



def quat_wxyz_from_euler_deg(roll_deg, pitch_deg, yaw_deg):
    """roll/pitch/yaw in degrees, returns (w,x,y,z). Uses intrinsic XYZ order (common)."""
    import math

    cr = math.cos(math.radians(roll_deg) * 0.5)
    sr = math.sin(math.radians(roll_deg) * 0.5)
    cp = math.cos(math.radians(pitch_deg) * 0.5)
    sp = math.sin(math.radians(pitch_deg) * 0.5)
    cy = math.cos(math.radians(yaw_deg) * 0.5)
    sy = math.sin(math.radians(yaw_deg) * 0.5)

    # quaternion (w, x, y, z) for R = Rz(yaw)*Ry(pitch)*Rx(roll)
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = cy * sp * cr + sy * cp * sr
    z = sy * cp * cr - cy * sp * sr
    return (w, x, y, z)

def _get_env_root_paths(env) -> list[str]:
    """Best-effort: infer each env root prim path for multi-env.
    先覆盖最常见的 IsaacLab 路径：/World/envs/env_0, env_1 ...
    如果你工程实际不是这个路径，改这里一行即可。
    """
    return [f"/World/envs/env_{i}" for i in range(env.num_envs)]


def spawn_usdz_reference_once_per_env(
    env,
    usdz_rel_path: str,
    name: str = "RuntimeObject",
    pos=(0.55, 0.0, 0.80),
    quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    scale: float = 0.01,
    density: float = 300.0,
):
    """Spawn a USDZ (reference) under each env namespace with rigid body + collision."""
    # 延迟 import：只在 Isaac Sim 运行时才存在
    import omni.usd
    from pxr import UsdGeom, Gf, UsdPhysics, PhysxSchema

    stage = omni.usd.get_context().get_stage()

    # USDZ 绝对路径（不写死盘符，跟随 ASSETS_ROOT）
    usdz_path = str(Path(ASSETS_ROOT) / usdz_rel_path)

    env_roots = _get_env_root_paths(env)
    spawned = []

    for i, root in enumerate(env_roots):
        prim_path = f"{root}/{name}"

        # 如果已经存在则跳过（避免重复按键生成同名冲突）
        if stage.GetPrimAtPath(prim_path).IsValid():
            spawned.append(prim_path)
            continue

        # 1) 创建一个 Xform 容器
        xform = UsdGeom.Xform.Define(stage, prim_path)
        prim = xform.GetPrim()

        # 2) 引用 USDZ
        prim.GetReferences().AddReference(usdz_path)

        # 3) 设置变换（全部用 float，避免 USD 类型不匹配）
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()

        px, py, pz = pos
        xf.AddTranslateOp().Set(Gf.Vec3f(float(px), float(py), float(pz)))

        w, x, y, z = quat_wxyz
        xf.AddOrientOp().Set(Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z))))

        s = float(scale)
        xf.AddScaleOp().Set(Gf.Vec3f(s, s, s))

        # 4) 加物理：碰撞 + 刚体 + 密度
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateDensityAttr(density)

        physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        physx_rb.CreateSolverPositionIterationCountAttr(8)
        physx_rb.CreateSolverVelocityIterationCountAttr(1)

        spawned.append(prim_path)

    return spawned
def _set_scale_op(xform, scale: float):
    from pxr import UsdGeom, Gf
    s = float(scale)

    # 1) 先找是否已有 scale op
    scale_op = None
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            scale_op = op
            break

    # 2) 没有才创建
    if scale_op is None:
        scale_op = xform.AddScaleOp(UsdGeom.XformOp.PrecisionFloat)

    # 3) 设置值（用 Vec3f）
    scale_op.Set(Gf.Vec3f(s, s, s))

def scale_scene_root(env, scale: float, scene_name: str = "Scene"):
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()

    for i in range(env.num_envs):
        scene_path = f"/World/envs/env_{i}/{scene_name}"
        prim = stage.GetPrimAtPath(scene_path)
        if not prim.IsValid():
            print(f"[scale_scene_root] prim not found: {scene_path}")
            continue

        xform = UsdGeom.Xformable(prim)
        _set_scale_op(xform, scale)
        print(f"[scale_scene_root] scaled {scene_path} by {scale}")
def set_scene_root_trs(env, scene_name="Scene", pos=None, quat_wxyz=None, scale=None):
    import omni.usd

    stage = omni.usd.get_context().get_stage()

    for i in range(env.num_envs):
        scene_path = f"/World/envs/env_{i}/{scene_name}"
        prim = stage.GetPrimAtPath(scene_path)
        if not prim.IsValid():
            print(f"[set_scene_root_trs] prim not found: {scene_path}")
            continue

        set_xform_trs(prim, pos=pos, quat_wxyz=quat_wxyz, scale=scale)
        print(f"[set_scene_root_trs] set TRS for {scene_path}: pos={pos}, quat={quat_wxyz}, scale={scale}")

def _get_or_add_op(xform, op_type, precision_float=True):
    """Find existing xformOp of given type; if none, add one. Use float precision by default."""
    from pxr import UsdGeom

    # Find existing
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == op_type:
            return op

    # Add new
    if precision_float:
        return xform.AddXformOp(op_type, UsdGeom.XformOp.PrecisionFloat)
    else:
        return xform.AddXformOp(op_type, UsdGeom.XformOp.PrecisionDouble)


def _find_op(xform, op_type):
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == op_type:
            return op
    return None


def set_xform_trs(prim, pos=None, quat_wxyz=None, scale=None):
    from pxr import UsdGeom, Gf

    xform = UsdGeom.Xformable(prim)

    # -------------------------
    # Translate
    # -------------------------
    if pos is not None:
        op = _find_op(xform, UsdGeom.XformOp.TypeTranslate)
        if op is None:
            # 默认用 float 创建；如果你更想 double，就改 PrecisionDouble
            op = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        px, py, pz = pos
        if op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
            op.Set(Gf.Vec3d(float(px), float(py), float(pz)))
        else:
            op.Set(Gf.Vec3f(float(px), float(py), float(pz)))

    # -------------------------
    # Orient (Quat)
    # -------------------------
    if quat_wxyz is not None:
        op = _find_op(xform, UsdGeom.XformOp.TypeOrient)
        if op is None:
            op = xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)  # Scene 常见是 double
        w, x, y, z = quat_wxyz
        if op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
            op.Set(Gf.Quatd(float(w), Gf.Vec3d(float(x), float(y), float(z))))
        else:
            op.Set(Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z))))

    # -------------------------
    # Scale
    # -------------------------
    if scale is not None:
        op = _find_op(xform, UsdGeom.XformOp.TypeScale)
        if op is None:
            op = xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
        if isinstance(scale, (tuple, list)):
            sx, sy, sz = scale
        else:
            sx = sy = sz = scale
        if op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
            op.Set(Gf.Vec3d(float(sx), float(sy), float(sz)))
        else:
            op.Set(Gf.Vec3f(float(sx), float(sy), float(sz)))







def manual_terminate(env: ManagerBasedRLEnv | DirectRLEnv, success: bool):
    if hasattr(env, "termination_manager"):
        if success:
            env.termination_manager.set_term_cfg(
                "success",
                TerminationTermCfg(func=lambda env: torch.ones(env.num_envs, dtype=torch.bool, device=env.device)),
            )
        else:
            env.termination_manager.set_term_cfg(
                "success",
                TerminationTermCfg(func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)),
            )
        env.termination_manager.compute()
    elif hasattr(env, "_get_dones"):
        env.cfg.return_success_status = success


def main():  # noqa: C901
    """Running lerobot teleoperation with leisaac manipulation environment."""

    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.use_teleop_device(args_cli.teleop_device)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())
    task_name = args_cli.task

    if args_cli.quality:
        env_cfg.sim.render.antialiasing_mode = "FXAA"
        env_cfg.sim.render.rendering_mode = "quality"

    # precheck task and teleop device
    if "BiArm" in task_name:
        assert args_cli.teleop_device == "bi-so101leader", "only support bi-so101leader for bi-arm task"
    if "LeKiwi" in task_name:
        assert args_cli.teleop_device in [
            "lekiwi-leader",
            "lekiwi-keyboard",
            "lekiwi-gamepad",
        ], "only support lekiwi-leader, lekiwi-keyboard, lekiwi-gamepad for lekiwi task"
    is_direct_env = "Direct" in task_name
    if is_direct_env:
        assert args_cli.teleop_device in [
            "so101leader",
            "bi-so101leader",
        ], "only support so101leader or bi-so101leader for direct task"

    # timeout and terminate preprocess
    if is_direct_env:
        env_cfg.never_time_out = True
        env_cfg.manual_terminate = True
    else:
        # modify configuration
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
        if hasattr(env_cfg.terminations, "success"):
            env_cfg.terminations.success = None
    # recorder preprocess & manual success terminate preprocess
    if args_cli.record:
        if args_cli.resume:
            env_cfg.recorders.dataset_export_mode = EnhanceDatasetExportMode.EXPORT_ALL_RESUME
            assert os.path.exists(
                args_cli.dataset_file
            ), "the dataset file does not exist, please don't use '--resume' if you want to record a new dataset"
        else:
            env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
            assert not os.path.exists(
                args_cli.dataset_file
            ), "the dataset file already exists, please use '--resume' to resume recording"
        env_cfg.recorders.dataset_export_dir_path = output_dir
        env_cfg.recorders.dataset_filename = output_file_name
        if is_direct_env:
            env_cfg.return_success_status = False
        else:
            if not hasattr(env_cfg.terminations, "success"):
                setattr(env_cfg.terminations, "success", None)
            env_cfg.terminations.success = TerminationTermCfg(
                func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            )
    else:
        env_cfg.recorders = None

    # create environment
    env: ManagerBasedRLEnv | DirectRLEnv = gym.make(task_name, cfg=env_cfg).unwrapped
    # replace the original recorder manager with the streaming recorder manager
    print("[action_dim]", getattr(env.action_manager, "total_action_dim", None), "action_space:", env.action_space)


    if args_cli.record:
        del env.recorder_manager
        env.recorder_manager = StreamingRecorderManager(env_cfg.recorders, env)
        env.recorder_manager.flush_steps = 100
        env.recorder_manager.compression = "lzf"

    # create controller
    if args_cli.teleop_device == "keyboard":
        from leisaac.devices import SO101Keyboard

        teleop_interface = SO101Keyboard(env, sensitivity=args_cli.sensitivity)
    elif args_cli.teleop_device == "gamepad":
        from leisaac.devices import SO101Gamepad

        teleop_interface = SO101Gamepad(env, sensitivity=args_cli.sensitivity)


    elif args_cli.teleop_device == "keyboard-world":
        from leisaac.devices import SO101KeyboardWorld

        teleop_interface = SO101KeyboardWorld(env, sensitivity=args_cli.sensitivity)

    elif args_cli.teleop_device == "macro-keyboard":
        from leisaac.devices.so101_macro_keyboard import SO101MacroKeyboard
        teleop_interface = SO101MacroKeyboard(env, sensitivity=args_cli.sensitivity, out_dir="./traj_out")


    elif args_cli.teleop_device == "so101leader":
        from leisaac.devices import SO101Leader

        teleop_interface = SO101Leader(env, port=args_cli.port, recalibrate=args_cli.recalibrate)
    elif args_cli.teleop_device == "bi-so101leader":
        from leisaac.devices import BiSO101Leader

        teleop_interface = BiSO101Leader(
            env, left_port=args_cli.left_arm_port, right_port=args_cli.right_arm_port, recalibrate=args_cli.recalibrate
        )
    elif args_cli.teleop_device == "lekiwi-keyboard":
        from leisaac.devices import LeKiwiKeyboard

        teleop_interface = LeKiwiKeyboard(env, sensitivity=args_cli.sensitivity)
    elif args_cli.teleop_device == "lekiwi-leader":
        from leisaac.devices import LeKiwiLeader

        teleop_interface = LeKiwiLeader(env, port=args_cli.port, recalibrate=args_cli.recalibrate)
    elif args_cli.teleop_device == "lekiwi-gamepad":
        from leisaac.devices import LeKiwiGamepad

        teleop_interface = LeKiwiGamepad(env, sensitivity=args_cli.sensitivity)
    else:
        raise ValueError(
            f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'gamepad', 'so101leader',"
            " 'bi-so101leader', 'lekiwi-keyboard', 'lekiwi-leader', 'lekiwi-gamepad'."
        )

    # add teleoperation key for env reset
    should_reset_recording_instance = False

    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True

    # add teleoperation key for task success
    should_reset_task_success = False

    def reset_task_success():
        nonlocal should_reset_task_success
        should_reset_task_success = True
        reset_recording_instance()

    teleop_interface.add_callback("R", reset_recording_instance)
    teleop_interface.add_callback("N", reset_task_success)
    print(teleop_interface)

    spawn_counter = 0

    def spawn_one():
        nonlocal spawn_counter
        spawn_counter += 1
        name = f"RuntimeObj_{spawn_counter}"
        paths = spawn_usdz_reference_once_per_env(
            env,
            usdz_rel_path="scenes/my_scene/2.usdz",
            name=name,
            pos=(0.55, 0.0, 0.80),
            quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            scale=0.08,
        )
        print("Spawned:", paths)

    teleop_interface.add_callback("T", spawn_one)

    rate_limiter = RateLimiter(args_cli.step_hz)

    # reset environment
    if hasattr(env, "initialize"):
        env.initialize()
    env.reset()
    teleop_interface.reset()

    # 场景太大就缩小（常见 0.01 或 0.001）
    # 例子：缩放 + 平移 + 旋转（四元数 wxyz）
    q = quat_wxyz_from_euler_deg(25, -3, 4)
    set_scene_root_trs(
        env,
        scene_name="Scene",
        pos=(1.195, -0.095, 0),  # 要移动多少就填多少（米）
        quat_wxyz=q,  # 默认不旋转
        scale=0.08,
    )

    # --- spawn one object at startup (optional) ---
    # 例如：把 usdz 放在 leisaac/assets/scenes/my_scene/2.usdz
    q1 = quat_wxyz_from_euler_deg(0, 0, 0)
    from pathlib import Path
    from leisaac.utils.constant import ASSETS_ROOT

    usdz_abs = str(Path(ASSETS_ROOT) / "scenes" / "my_scene" / "2.usdz")

    spawned_proxies = []
    spawned_visuals = []

    for i in range(env.num_envs):
        env_root = f"/World/envs/env_{i}"

        # 1) 创建 proxy 刚体（这是唯一受物理影响的东西）
        proxy_path = f"{env_root}/RuntimeMug_proxy"

        create_proxy_rigid_box(
            prim_path=proxy_path,
            pos=(0.55, 0.0, 0.90),  # z 放高一点，先验证“会掉”
            quat_wxyz=q1,
            size_xyz=(0.08, 0.08, 0.12),  # 代理碰撞体尺寸（米，自己微调）
            density=300.0,
            visible=False,  # 不显示代理
        )
        spawned_proxies.append(proxy_path)

        # 2) 把 USDZ 作为“视觉模型”挂到 proxy 下面
        visual_path = spawn_usdz_under_parent(
            parent_xform_path=proxy_path,
            usdz_path=usdz_abs,
            child_name="visual",
            scale=1.0,  # 强烈建议这里先用 1.0
        )
        spawned_visuals.append(visual_path)

        # 3) 禁用 USDZ 内部可能自带的碰撞（防止双重碰撞）
        disable_collisions_under(visual_path)

    print("Spawned proxies:", spawned_proxies)
    print("Spawned visuals:", spawned_visuals)
    def print_gravity():
        import omni.usd
        from pxr import UsdPhysics

        stage = omni.usd.get_context().get_stage()
        # 常见路径：/World/physicsScene 或 /World/PhysicsScene
        for p in ["/World/physicsScene", "/World/PhysicsScene", "/physicsScene"]:
            prim = stage.GetPrimAtPath(p)
            if prim.IsValid():
                scene = UsdPhysics.Scene(prim)
                g = scene.GetGravityDirectionAttr().Get()
                mag = scene.GetGravityMagnitudeAttr().Get()
                print("[gravity] scene:", p, "dir=", g, "mag=", mag)
                return
        print("[gravity] physics scene not found at common paths")

    print_gravity()

    resume_recorded_demo_count = 0
    if args_cli.record and args_cli.resume:
        resume_recorded_demo_count = env.recorder_manager._dataset_file_handler.get_num_episodes()
        print(f"Resume recording from existing dataset file with {resume_recorded_demo_count} demonstrations.")
    current_recorded_demo_count = resume_recorded_demo_count

    start_record_state = False

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():

            if hasattr(teleop_interface, "update_macro"):
                teleop_interface.update_macro()

            #actions = teleop_interface.advance()

            if env.cfg.dynamic_reset_gripper_effort_limit:
                dynamic_reset_gripper_effort_limit_sim(env, args_cli.teleop_device)
            actions = teleop_interface.advance()
            if should_reset_task_success:
                print("Task Success!!!")
                should_reset_task_success = False
                if args_cli.record:
                    manual_terminate(env, True)
            if should_reset_recording_instance:
                env.reset()
                should_reset_recording_instance = False
                if start_record_state:
                    if args_cli.record:
                        print("Stop Recording!!!")
                    start_record_state = False
                if args_cli.record:
                    manual_terminate(env, False)
                # print out the current demo count if it has changed
                if (
                    args_cli.record
                    and env.recorder_manager.exported_successful_episode_count + resume_recorded_demo_count
                    > current_recorded_demo_count
                ):
                    current_recorded_demo_count = (
                        env.recorder_manager.exported_successful_episode_count + resume_recorded_demo_count
                    )
                    print(f"Recorded {current_recorded_demo_count} successful demonstrations.")
                if (
                    args_cli.record
                    and args_cli.num_demos > 0
                    and env.recorder_manager.exported_successful_episode_count + resume_recorded_demo_count
                    >= args_cli.num_demos
                ):
                    print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
                    break
            else:
                if actions is None:
                    actions = make_zero_actions(env)

                if not start_record_state:
                    if args_cli.record:
                        print("Start Recording!!!")
                    start_record_state = True

                env.step(actions)
            if rate_limiter:
                rate_limiter.sleep(env)


            # elif actions is None:
            #     env.render()
            #
            # # apply actions
            # else:
            #     if not start_record_state:
            #         if args_cli.record:
            #             print("Start Recording!!!")
            #         start_record_state = True
            #     env.step(actions)
            # if rate_limiter:
            #     rate_limiter.sleep(env)

    # close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    # run the main function
    main()
