# -*- coding: utf-8 -*-
"""
teleop_se3_agent.py

Teleoperation entry with scale-preserving DGSRSim object-state synchronization.
"""

import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
import json
import os
from pathlib import Path

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
        "rule-grasp",
    ],
)
parser.add_argument("--port", type=str, default="/dev/ttyACM0")
parser.add_argument("--left_arm_port", type=str, default="/dev/ttyACM0")
parser.add_argument("--right_arm_port", type=str, default="/dev/ttyACM1")
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--sensitivity", type=float, default=1.0)
parser.add_argument(
    "--debug-leader",
    action="store_true",
    help="Print SO101 leader shoulder_pan input and converted follower joint command.",
)
parser.add_argument(
    "--debug-leader-period",
    type=float,
    default=0.25,
    help="Seconds between --debug-leader prints. Use 0 to print every frame.",
)

# recording
parser.add_argument("--record", action="store_true")
parser.add_argument("--step_hz", type=int, default=60)
parser.add_argument("--dataset_file", type=str, default="./datasets/dataset.hdf5")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--num_demos", type=int, default=0)

parser.add_argument("--recalibrate", action="store_true")
parser.add_argument("--quality", action="store_true")

# absolute pose npy (template/mug -> Scene)
parser.add_argument(
    "--state_json",
    type=str,
    default=os.environ.get(
        "DGSRSIM_STATE_JSON",
        str(Path(__file__).resolve().parents[4] / "FastSAMRealtime" / "rt_ply_out" / "object_states.json"),
    ),
    help="Scale-preserving DGSRSim single- or multi-object state JSON.",
)
parser.add_argument(
    "--object_id",
    type=str,
    default=os.environ.get("DGSRSIM_OBJECT_ID", "astronaut"),
    help="Object ID used by legacy single-object packets and pose fallback files.",
)
parser.add_argument(
    "--object_bindings_json",
    type=str,
    default=os.environ.get(
        "DGSRSIM_OBJECT_BINDINGS_JSON",
        str(Path(__file__).resolve().parents[3] / "configs" / "object_bindings.example.json"),
    ),
    help="JSON mapping from DGSRSim object IDs to per-environment USD prim paths.",
)
parser.add_argument(
    "--pose_npy",
    type=str,
    default=os.environ.get(
        "DGSRSIM_POSE_NPY",
        str(Path(__file__).resolve().parents[4] / "FastSAMRealtime" / "rt_ply_out" / "T_tgt_to_scene.npy"),
    ),
    help="Path to the saved absolute pose npy (4x4). Defaults to FastSAMRealtime/rt_ply_out/T_tgt_to_scene.npy.",
)
parser.add_argument(
    "--asset_profile_json",
    type=str,
    default=str(
        Path(__file__).resolve().parents[4]
        / "FastSAMRealtime"
        / "configs"
        / "assets"
        / "astronaut_shared_gaussian_asset.json"
    ),
    help="Legacy single-object asset profile, used only when --object_bindings_json is empty.",
)
parser.add_argument("--pose_poll_hz", type=float, default=60.0, help="Polling frequency for pose npy file updates.")

# pose sync key
parser.add_argument("--pose_sync_key", type=str, default="M", help="Key to toggle pose synchronization ON/OFF (default: M).")
parser.add_argument(
    "--pose_sync_freeze",
    action="store_true",
    default=False,
    help="If set, when sync is OFF, keep mug frozen at the last synced world pose (recommended).",
)

# rule-grasp params
parser.add_argument("--ee_prim_path", type=str, default="", help="EE USD prim relative path (per-env root excluded).")
parser.add_argument("--autograsp_pregrasp_dz", type=float, default=0.10)
parser.add_argument("--autograsp_grasp_dz", type=float, default=0.02)
parser.add_argument("--autograsp_lift_dz", type=float, default=0.15)
parser.add_argument("--autograsp_xy_gain", type=float, default=6.0)
parser.add_argument("--autograsp_z_gain", type=float, default=6.0)
parser.add_argument("--autograsp_yaw_gain", type=float, default=3.0)
parser.add_argument("--autograsp_close_steps", type=int, default=25)
parser.add_argument("--autograsp_settle_steps", type=int, default=10)
parser.add_argument("--autograsp_success_on_lift", action="store_true")
parser.add_argument("--autograsp_open_val", type=float, default=1.0, help="Gripper open command value.")
parser.add_argument("--autograsp_close_val", type=float, default=-1.0, help="Gripper close command value.")



# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

# -------------------------
# Imports that require Isaac Sim context
# -------------------------
import torch

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import TerminationTermCfg

from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim
from leisaac.utils.constant import ASSETS_ROOT

# env/cfg factory
from leisaac.utils.teleop_env_factory import create_env_and_cfg

# Runtime-object factory
from leisaac.utils.runtime_mug_factory import (
    RuntimeObjectSpec,
    spawn_runtime_objects_for_all_envs,
)
from leisaac.utils.runtime_object_config import load_runtime_object_configs

# Recording utils
from leisaac.utils.recording_utils import (
    start_recording,
    stop_recording,
    get_resume_demo_count,
    DemoCounter,
)

# Pose Sync pipeline
from leisaac.utils.pose_sync_pipeline import (
    PoseSyncConfig,
    PoseSyncPipeline,
    load_object_path_templates,
)

# Rate limiter
from leisaac.utils.rate_limiter import RateLimiter

# Scene init pipeline
from leisaac.utils.scene_init_pipeline import SceneInitConfig, SceneInitPipeline

# Teleop interface selection
from leisaac.utils.teleop_device_factory import create_teleop_and_agent

# Geometry helpers
from leisaac.utils.geometry_utils import quat_wxyz_from_euler_deg


# ============================================================
# Emergency switch retained for local diagnosis; the released path keeps sync enabled.
# ============================================================
FORCE_DISABLE_POSE_SYNC = False


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
# Main
# ============================================================
def main():  # noqa: C901
    # env/cfg creation
    built = create_env_and_cfg(args_cli)
    if hasattr(built, "env"):
        env = built.env
        task_name = built.task_name
        is_direct_env = built.is_direct_env
    else:
        env, _, task_name, is_direct_env = built

    print("[action_dim]", getattr(env.action_manager, "total_action_dim", None), "action_space:", env.action_space)
    try:
        print("[action_terms]", list(getattr(env.action_manager, "_terms").keys()))
    except Exception:
        pass

    # init env first
    if hasattr(env, "initialize"):
        env.initialize()
    env.reset()
    print("\n[DEBUG] arm_action cfg fields")
    at = env.action_manager._terms["arm_action"]
    cfg = getattr(at, "cfg", None)
    print("arm_action term =", type(at))
    print("cfg =", cfg)

    # 尽量打印常见字段
    for k in ["body_name", "end_effector_name", "ee_frame", "frame", "base_name", "use_world_frame", "asset_name"]:
        if hasattr(cfg, k):
            print(f"cfg.{k} =", getattr(cfg, k))
    print()

    import omni.usd
    stage = omni.usd.get_context().get_stage()

    # -----------------------------
    # flags / state
    # -----------------------------
    should_reset = False
    should_success = False

    # -----------------------------
    # recording state + counter
    # -----------------------------
    demo_counter = None
    resume_recorded_demo_count = 0
    if args_cli.record:
        if args_cli.resume:
            resume_recorded_demo_count = get_resume_demo_count(env)
            print(f"Resume recording from {resume_recorded_demo_count} demonstrations.")
        demo_counter = DemoCounter(resume_count=resume_recorded_demo_count, num_demos=args_cli.num_demos)
        demo_counter.init_from_env(env)
        start_recording(env)

    # -----------------------------
    # Pose Sync pipeline (created, but FORCE disabled)
    # -----------------------------
    object_path_templates = (
        load_object_path_templates(args_cli.object_bindings_json)
        if args_cli.object_bindings_json
        else {}
    )
    pose_sync_cfg = PoseSyncConfig(
        state_json=args_cli.state_json,
        pose_npy=args_cli.pose_npy,
        pose_poll_hz=args_cli.pose_poll_hz,
        default_object_id=args_cli.object_id,
        object_path_templates=object_path_templates,
        pose_sync_key=str(args_cli.pose_sync_key).upper(),
        pose_sync_freeze=bool(args_cli.pose_sync_freeze),
        scene_path_tpl="/World/envs/env_{i}/Scene",
        mug_path_tpl="/World/envs/env_{i}/RuntimeMug_proxy",
    )
    pose_sync = PoseSyncPipeline(
        cfg=pose_sync_cfg,
        num_envs=env.num_envs,
        rule_grasp_mode=(args_cli.teleop_device == "rule-grasp"),
    )

    if FORCE_DISABLE_POSE_SYNC:
        print("[pose-sync] FORCE_DISABLE_POSE_SYNC=True -> Pose sync will NOT run (no step/apply/reset).")
    else:
        pose_sync.start()

    # -----------------------------
    # callbacks
    # -----------------------------
    def on_reset():
        nonlocal should_reset
        should_reset = True

    # -----------------------------
    # teleop interface + rule agent (via factory)
    # -----------------------------
    built_teleop = create_teleop_and_agent(env, stage, args_cli)
    teleop_interface = built_teleop.teleop_interface
    rule_agent = built_teleop.rule_agent

    teleop_interface.reset()
    teleop_interface.add_callback("R", on_reset)

    # 注册 M
    teleop_interface.add_callback(str(args_cli.pose_sync_key).upper(), pose_sync.toggle_by_keypress)

    # Rate limiter
    rate_limiter = RateLimiter(args_cli.step_hz)

    # -----------------------------
    # Scene init pipeline
    # -----------------------------
    scene_init_cfg = SceneInitConfig(
        scene_name="Scene",
        pos=(1.2, -0.095, 0.06),
        rot_euler_deg=(35.0, -1.0, 5.0),
        scale=0.08,
        enable_scene_proxy_collisions=True,
        verbose=True,
    )
    SceneInitPipeline(scene_init_cfg).apply(env)

    # -----------------------------
    # Spawn every runtime object declared by the ID-to-prim binding map.
    # -----------------------------
    project_root = Path(__file__).resolve().parents[4]
    configured_objects = (
        load_runtime_object_configs(
            args_cli.object_bindings_json,
            assets_root=ASSETS_ROOT,
            project_root=project_root,
        )
        if args_cli.object_bindings_json
        else {}
    )
    runtime_specs = {}
    for object_id, object_cfg in configured_objects.items():
        asset_profile = json.loads(
            object_cfg.asset_profile_path.read_text(encoding="utf-8")
        )
        raw_extent = tuple(
            float(value) for value in asset_profile["source_ply"]["aabb_extent"]
        )
        runtime_specs[object_id] = RuntimeObjectSpec(
            proxy_name=object_cfg.proxy_name,
            visual_child_name="visual",
            proxy_pos=object_cfg.initial_position_m,
            proxy_quat_wxyz=object_cfg.initial_quaternion_wxyz,
            target_visual_size_m=object_cfg.target_visual_size_m,
            proxy_size_asset_units=raw_extent,
            density=object_cfg.density_kg_m3,
            proxy_visible=object_cfg.proxy_visible,
            usdz_path=str(object_cfg.usdz_path),
            visual_base_scale=1.0,
            auto_fit_axis="z",
            extra_visual_scale=1.0,
            consume_state_similarity=True,
            kinematic=True,
            disable_gravity=True,
            visual_local_pos=(0.0, 0.0, 0.0),
            visual_local_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            geom_local_pos=(0.0, 0.0, 0.0),
            geom_local_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        )

    if not args_cli.object_bindings_json:
        asset_profile = json.loads(
            Path(args_cli.asset_profile_json).read_text(encoding="utf-8")
        )
        raw_extent = tuple(
            float(value) for value in asset_profile["source_ply"]["aabb_extent"]
        )
        runtime_specs[args_cli.object_id] = RuntimeObjectSpec(
            proxy_name="RuntimeMug_proxy",
            visual_child_name="visual",
            proxy_pos=(1.05, -0.46, -0.277),
            proxy_quat_wxyz=quat_wxyz_from_euler_deg(0.0, 0.0, 0.0),
            target_visual_size_m=(0.08, 0.08, 0.12),
            proxy_size_asset_units=raw_extent,
            density=300.0,
            proxy_visible=True,
            usdz_path=str(Path(ASSETS_ROOT) / "scenes" / "my_scene" / "2.usdz"),
            consume_state_similarity=True,
            kinematic=True,
            disable_gravity=True,
        )

    if runtime_specs:
        spawn_runtime_objects_for_all_envs(
            stage=stage,
            num_envs=env.num_envs,
            env_root_tpl="/World/envs/env_{i}",
            specs=runtime_specs,
            verbose=True,
        )
    else:
        print("[runtime-object] no enabled spawn records; bound prims must already exist in the task scene")

    # -----------------------------
    # main loop
    # -----------------------------
    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                if hasattr(teleop_interface, "update_macro"):
                    teleop_interface.update_macro()

                if env.cfg.dynamic_reset_gripper_effort_limit:
                    dynamic_reset_gripper_effort_limit_sim(env, args_cli.teleop_device)

                # actions source
                if args_cli.teleop_device == "rule-grasp":
                    actions, req_pose_sync_off, req_success = rule_agent.act()

                    # 强制同步：向 pose_sync 发任何请求
                    if not FORCE_DISABLE_POSE_SYNC:
                        pose_sync.apply_agent_request(req_pose_sync_off)

                    if req_success:
                        should_success = True
                else:
                    actions = teleop_interface.advance()

                if should_success:
                    print("Task Success!!!")
                    should_success = False
                    if args_cli.record:
                        manual_terminate(env, True)

                if should_reset:
                    should_reset = False

                    if args_cli.record:
                        stop_recording(env)

                    env.reset()

                    # 强制同步：reset pose_sync
                    if not FORCE_DISABLE_POSE_SYNC:
                        pose_sync.reset()

                    rate_limiter.reset()

                    if args_cli.record:
                        manual_terminate(env, False)

                    if args_cli.teleop_device == "rule-grasp":
                        rule_agent.reset()

                    if args_cli.record:
                        start_recording(env)

                if actions is None:
                    actions = make_zero_actions(env)

                # Apply every active object state to its configured prim.
                if not FORCE_DISABLE_POSE_SYNC:
                    pose_sync.step(stage)

                env.step(actions)

                if args_cli.record and (demo_counter is not None):
                    demo_counter.maybe_print_update(env)
                    if demo_counter.reached_limit(env):
                        print(f"All {args_cli.num_demos} demonstrations recorded. Exiting.")
                        break

                rate_limiter.sleep(env)

    finally:
        try:
            if not FORCE_DISABLE_POSE_SYNC:
                pose_sync.stop()
        except Exception:
            pass

        try:
            if args_cli.record:
                stop_recording(env)
        except Exception:
            pass

        try:
            if hasattr(teleop_interface, "close"):
                teleop_interface.close()
        except Exception:
            pass

        env.close()

        simulation_app.close()


if __name__ == "__main__":
    main()
