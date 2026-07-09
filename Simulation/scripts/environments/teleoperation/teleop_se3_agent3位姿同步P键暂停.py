# -*- coding: utf-8 -*-
import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
import os
import time
from pathlib import Path
import threading

from isaaclab.app import AppLauncher

# -------------------------
# CLI
# -------------------------
parser = argparse.ArgumentParser(description="leisaac teleoperation for leisaac environments.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    choices=[
        "keyboard",
        "gamepad",
        "keyboard-world",
        "macro-keyboard",
        "so101leader",
        "bi-so101leader",
        "lekiwi-keyboard",
        "lekiwi-gamepad",
        "lekiwi-leader",
    ],
)
parser.add_argument("--port", type=str, default="/dev/ttyACM0")
parser.add_argument("--left_arm_port", type=str, default="/dev/ttyACM0")
parser.add_argument("--right_arm_port", type=str, default="/dev/ttyACM1")
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--sensitivity", type=float, default=1.0)

parser.add_argument("--record", action="store_true")
parser.add_argument("--step_hz", type=int, default=60)
parser.add_argument("--dataset_file", type=str, default="./datasets/dataset.hdf5")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--num_demos", type=int, default=0)

parser.add_argument("--recalibrate", action="store_true")
parser.add_argument("--quality", action="store_true")

# absolute pose npy (template/mug -> Scene)
parser.add_argument(
    "--pose_npy",
    type=str,
    default="E:/code/FastSAM/rt_ply_out/T_tgt_to_scene.npy",
    help="Path to the saved absolute pose npy (4x4): T_scene_mug (= template->scene).",
)
parser.add_argument(
    "--pose_poll_hz",
    type=float,
    default=60.0,
    help="Polling frequency for pose npy file updates.",
)

# NEW: sync switch / phase control
parser.add_argument(
    "--pose_sync_key",
    type=str,
    default="P",
    help="Key to toggle pose synchronization ON/OFF (default: P).",
)
parser.add_argument(
    "--pose_sync_freeze",
    action="store_true",
    help="If set, when sync is OFF, keep mug frozen at the last synced world pose (recommended).",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

# -------------------------
# Imports that require Isaac Sim context
# -------------------------
import gymnasium as gym
import torch

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import DatasetExportMode, TerminationTermCfg
from isaaclab_tasks.utils import parse_env_cfg

from leisaac.enhance.managers import EnhanceDatasetExportMode, StreamingRecorderManager
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim
from leisaac.utils.constant import ASSETS_ROOT


# ============================================================
# Rate limiter
# ============================================================
class RateLimiter:
    def __init__(self, hz: int):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()
        self.last_time = self.last_time + self.sleep_duration
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


# ============================================================
# Action helper
# ============================================================
def make_zero_actions(env):
    if hasattr(env, "action_manager") and hasattr(env.action_manager, "total_action_dim"):
        d = int(env.action_manager.total_action_dim)
        return torch.zeros((env.num_envs, d), device=env.device)
    space = getattr(env, "action_space", None)
    if space is not None and hasattr(space, "shape") and space.shape is not None:
        return torch.zeros((env.num_envs,) + tuple(space.shape), device=env.device)
    raise RuntimeError("Cannot infer action dimension.")


# ============================================================
# Math helper
# ============================================================
def quat_wxyz_from_euler_deg(roll_deg, pitch_deg, yaw_deg):
    import math

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


# ============================================================
# USD helpers
# ============================================================
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


def get_prim_world_aabb_size(stage, prim_path: str):
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


def get_world_xf(stage, prim_path: str):
    import numpy as np
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Invalid prim: {prim_path}")

    cache = UsdGeom.XformCache()
    M = cache.GetLocalToWorldTransform(prim)
    rows = [M.GetRow(i) for i in range(4)]
    T = np.array([[float(rows[r][c]) for c in range(4)] for r in range(4)], dtype=np.float64)
    return T


def set_prim_world_matrix(stage, prim_path: str, T_world):
    from pxr import UsdGeom, Gf
    import numpy as np

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


# ============================================================
# Recording control (best-effort across versions)
# ============================================================
def _set_recorder_state(env, enabled: bool):
    rm = getattr(env, "recorder_manager", None)
    if rm is None:
        return False

    for m in ["set_recording_enabled", "set_enabled"]:
        if hasattr(rm, m):
            try:
                getattr(rm, m)(enabled)
                return True
            except Exception:
                pass

    for m in (["resume", "start"] if enabled else ["pause", "stop"]):
        if hasattr(rm, m):
            try:
                getattr(rm, m)()
                return True
            except Exception:
                pass

    for a in ["is_recording", "enabled", "recording_enabled"]:
        if hasattr(rm, a):
            try:
                setattr(rm, a, enabled)
                return True
            except Exception:
                pass

    return False


def stop_recording(env):
    ok = _set_recorder_state(env, False)
    print(f"[record] stop -> {'OK' if ok else 'NO-OP (API not found)'}")


def start_recording(env):
    ok = _set_recorder_state(env, True)
    print(f"[record] start -> {'OK' if ok else 'NO-OP (API not found)'}")


# ============================================================
# Proxy rigid body + visual
# ============================================================
def _set_physx_attr(api, candidates: list[str], value):
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


def create_proxy_rigid_box(
    prim_path: str,
    pos=(0.55, 0.0, 0.80),
    quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    size_xyz=(0.08, 0.08, 0.12),
    density: float = 300.0,
    visible: bool = False,
):
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
        physx_rb.CreateSolverPositionIterationCountAttr(12)
    if hasattr(physx_rb, "CreateSolverVelocityIterationCountAttr"):
        physx_rb.CreateSolverVelocityIterationCountAttr(2)

    if hasattr(physx_rb, "CreateEnableCCDAttr"):
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


def spawn_usdz_under_parent(parent_xform_path: str, usdz_path: str, child_name: str = "visual", scale=1.0):
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


# ============================================================
# Scene proxy collisions (STATIC colliders)
# ============================================================
def create_static_collider_box(
    prim_path: str,
    pos=(0.0, 0.0, 0.0),
    quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    size_xyz=(1.0, 1.0, 0.1),
    visible: bool = False,
):
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
    create_static_collider_box(
        prim_path=f"{env_root}/SceneProxy/TableTop",
        pos=(0.55, 0.0, 1.38),
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        size_xyz=(2.0, 2.0, 0.05),
        visible=True,
    )

    create_static_collider_box(
        prim_path=f"{env_root}/SceneProxy/Floor",
        pos=(0.55, 0.0, -0.469),
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        size_xyz=(6.0, 6.0, 0.1),
        visible=True,
    )


# ============================================================
# Termination helper
# ============================================================
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


# ============================================================
# Pose file stream (absolute)
# ============================================================
class NpyPoseFileStream:
    def __init__(self, npy_path: str, poll_hz: float = 60.0):
        self.npy_path = npy_path
        self.poll_hz = float(poll_hz)
        self._lock = threading.Lock()
        self._latest_T = None
        self._latest_mtime = -1.0
        self._running = False
        self._th = None

    def start(self):
        self._running = True
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def stop(self):
        self._running = False
        if self._th is not None:
            self._th.join(timeout=1.0)
            self._th = None

    def get_latest(self):
        with self._lock:
            return None if self._latest_T is None else self._latest_T.copy()

    def _loop(self):
        import numpy as np
        dt = 1.0 / max(self.poll_hz, 1e-6)
        while self._running:
            try:
                if os.path.exists(self.npy_path):
                    m = os.path.getmtime(self.npy_path)
                    if m > self._latest_mtime:
                        try:
                            T = np.load(self.npy_path).astype(np.float64).reshape(4, 4)
                            if np.isfinite(T).all():
                                with self._lock:
                                    self._latest_T = T
                                    self._latest_mtime = m
                        except Exception:
                            pass
            except Exception:
                pass
            time.sleep(dt)


# ============================================================
# Main
# ============================================================
def main():  # noqa: C901
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    if output_dir and (not os.path.exists(output_dir)):
        os.makedirs(output_dir)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.use_teleop_device(args_cli.teleop_device)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())
    task_name = args_cli.task

    if args_cli.quality:
        env_cfg.sim.render.antialiasing_mode = "FXAA"
        env_cfg.sim.render.rendering_mode = "quality"

    is_direct_env = "Direct" in task_name
    if is_direct_env:
        env_cfg.never_time_out = True
        env_cfg.manual_terminate = True
    else:
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
        if hasattr(env_cfg.terminations, "success"):
            env_cfg.terminations.success = None

    if args_cli.record:
        if args_cli.resume:
            env_cfg.recorders.dataset_export_mode = EnhanceDatasetExportMode.EXPORT_ALL_RESUME
            assert os.path.exists(args_cli.dataset_file), "dataset file does not exist for --resume"
        else:
            env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
            assert not os.path.exists(args_cli.dataset_file), "dataset file exists; use --resume"
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

    env: ManagerBasedRLEnv | DirectRLEnv = gym.make(task_name, cfg=env_cfg).unwrapped
    print("[action_dim]", getattr(env.action_manager, "total_action_dim", None), "action_space:", env.action_space)

    if args_cli.record:
        del env.recorder_manager
        env.recorder_manager = StreamingRecorderManager(env_cfg.recorders, env)
        env.recorder_manager.flush_steps = 100
        env.recorder_manager.compression = "lzf"

    # teleop interface
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
        raise ValueError(f"Invalid device interface '{args_cli.teleop_device}'.")

    # state flags
    should_reset = False
    should_success = False
    recording_enabled = bool(args_cli.record)

    # NEW: pose sync phase control
    pose_sync_enabled = True  # 初始同步开启
    freeze_when_sync_off = bool(args_cli.pose_sync_freeze)
    last_world_mug_pose_by_env = {}  # env_i -> last 4x4 world pose

    def on_reset():
        nonlocal should_reset
        should_reset = True

    def on_success():
        nonlocal should_success
        should_success = True

    def toggle_recording():
        nonlocal recording_enabled
        recording_enabled = not recording_enabled
        if recording_enabled:
            start_recording(env)
        else:
            stop_recording(env)

    # NEW: toggle pose sync
    def toggle_pose_sync():
        nonlocal pose_sync_enabled
        pose_sync_enabled = not pose_sync_enabled
        print(f"[pose-sync] {'ON' if pose_sync_enabled else 'OFF'}  (freeze_off={freeze_when_sync_off})")

    teleop_interface.add_callback("R", on_reset)
    teleop_interface.add_callback("N", on_success)
    teleop_interface.add_callback("O", toggle_recording)

    # bind key (default: P)
    teleop_interface.add_callback(str(args_cli.pose_sync_key).upper(), toggle_pose_sync)

    rate_limiter = RateLimiter(args_cli.step_hz)

    # init/reset
    if hasattr(env, "initialize"):
        env.initialize()
    env.reset()
    teleop_interface.reset()

    # Scene TRS
    q_scene = quat_wxyz_from_euler_deg(35.0, -1.0, 5.0)
    set_scene_root_trs(
        env,
        scene_name="Scene",
        pos=(1.2, -0.095, 0.06),
        quat_wxyz=q_scene,
        scale=0.08,
    )

    # SCENE PROXY COLLISIONS (STATIC)
    for i in range(env.num_envs):
        env_root = f"/World/envs/env_{i}"
        build_scene_proxy_collisions_for_env(env_root)
    print("[scene-proxy] built static proxy colliders under /World/envs/env_i/SceneProxy")

    # Spawn RuntimeMug_proxy + visual
    import omni.usd
    stage = omni.usd.get_context().get_stage()

    TARGET_VISUAL_SIZE_M = (0.08, 0.08, 0.12)
    AUTO_FIT_AXIS = "z"

    q_mug = quat_wxyz_from_euler_deg(0, 0, 0)
    usdz_abs = str(Path(ASSETS_ROOT) / "scenes" / "my_scene" / "2.usdz")

    for i in range(env.num_envs):
        env_root = f"/World/envs/env_{i}"
        proxy_path = f"{env_root}/RuntimeMug_proxy"

        create_proxy_rigid_box(
            prim_path=proxy_path,
            pos=(1.05, -0.46, -0.277),
            quat_wxyz=q_mug,
            size_xyz=TARGET_VISUAL_SIZE_M,
            density=300.0,
            visible=True,
        )

        visual_path = spawn_usdz_under_parent(proxy_path, usdz_abs, child_name="visual", scale=1.0)
        disable_collisions_under(visual_path)

        cur = get_prim_world_aabb_size(stage, visual_path)
        tx, ty, tz = TARGET_VISUAL_SIZE_M

        if AUTO_FIT_AXIS == "x":
            s = tx / max(cur[0], 1e-9)
        elif AUTO_FIT_AXIS == "y":
            s = ty / max(cur[1], 1e-9)
        elif AUTO_FIT_AXIS == "z":
            s = tz / max(cur[2], 1e-9)
        else:  # "max"
            s = max(tx / max(cur[0], 1e-9), ty / max(cur[1], 1e-9), tz / max(cur[2], 1e-9))

        vis_prim = stage.GetPrimAtPath(visual_path)
        set_xform_trs(vis_prim, scale=s)
        new_size = get_prim_world_aabb_size(stage, visual_path)
        print(f"[visual-scale] env{i}: cur={cur}, target={TARGET_VISUAL_SIZE_M}, scale={s:.6f}, new={new_size}")

        # kinematic for absolute pose driving
        make_rigidbody_kinematic(stage, proxy_path, disable_gravity=True)

    # Pose stream (absolute)
    pose_stream = NpyPoseFileStream(args_cli.pose_npy, poll_hz=args_cli.pose_poll_hz)
    pose_stream.start()
    print(f"[pose] streaming absolute pose from: {os.path.abspath(args_cli.pose_npy)}")
    print(f"[pose-sync] initial=ON. Toggle with key '{str(args_cli.pose_sync_key).upper()}'")

    # resume recording count
    resume_recorded_demo_count = 0
    if args_cli.record and args_cli.resume:
        resume_recorded_demo_count = env.recorder_manager._dataset_file_handler.get_num_episodes()
        print(f"Resume recording from {resume_recorded_demo_count} demonstrations.")
    current_recorded_demo_count = resume_recorded_demo_count

    # If record enabled at start, try to start recorder explicitly
    if args_cli.record:
        if recording_enabled:
            start_recording(env)
        else:
            stop_recording(env)

    # loop
    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                if hasattr(teleop_interface, "update_macro"):
                    teleop_interface.update_macro()

                if env.cfg.dynamic_reset_gripper_effort_limit:
                    dynamic_reset_gripper_effort_limit_sim(env, args_cli.teleop_device)

                actions = teleop_interface.advance()

                if should_success:
                    print("Task Success!!!")
                    should_success = False
                    if args_cli.record:
                        manual_terminate(env, True)

                if should_reset:
                    should_reset = False

                    # reset -> force stop recording
                    if args_cli.record:
                        recording_enabled = False
                        stop_recording(env)

                    # reset env
                    env.reset()
                    if args_cli.record:
                        manual_terminate(env, False)

                    # IMPORTANT: reset 时不强制改变 pose_sync_enabled（你可自行决定）
                    # 但建议清一下 last_world_mug_pose_by_env，避免 freeze 逻辑拿到旧 pose
                    last_world_mug_pose_by_env.clear()

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
                        print(f"All {args_cli.num_demos} demonstrations recorded. Exiting.")
                        break

                else:
                    if actions is None:
                        actions = make_zero_actions(env)

                    # -----------------------------
                    # Pose sync phase control:
                    #   ON  -> apply external absolute pose
                    #   OFF -> either freeze at last pose (recommended) OR do nothing
                    # -----------------------------
                    if pose_sync_enabled:
                        T_scene_mug = pose_stream.get_latest()
                        if T_scene_mug is not None:
                            import numpy as np
                            for i in range(env.num_envs):
                                scene_path = f"/World/envs/env_{i}/Scene"
                                mug_path = f"/World/envs/env_{i}/RuntimeMug_proxy"
                                try:
                                    T_world_scene = get_world_xf(stage, scene_path)
                                    T_world_mug = T_world_scene @ np.asarray(T_scene_mug, dtype=np.float64)
                                    set_prim_world_matrix(stage, mug_path, T_world_mug)
                                    last_world_mug_pose_by_env[i] = T_world_mug
                                except Exception:
                                    pass
                    else:
                        if freeze_when_sync_off:
                            # keep mug at the last synced pose so it won't jitter during grasp
                            for i in range(env.num_envs):
                                if i not in last_world_mug_pose_by_env:
                                    continue
                                mug_path = f"/World/envs/env_{i}/RuntimeMug_proxy"
                                try:
                                    set_prim_world_matrix(stage, mug_path, last_world_mug_pose_by_env[i])
                                except Exception:
                                    pass
                        else:
                            # release: do nothing (NOTE: object is kinematic in this script,
                            # so "release" means it stays where it was, but still kinematic).
                            pass

                    env.step(actions)

                if rate_limiter:
                    rate_limiter.sleep(env)

    finally:
        try:
            pose_stream.stop()
        except Exception:
            pass

        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
