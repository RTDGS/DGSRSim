# -*- coding: utf-8 -*-
"""
leisaac/utils/teleop_env_factory.py

Environment creation + env_cfg preprocessing for teleop scripts.

This module encapsulates:
- parse_env_cfg + use_teleop_device + seed
- quality render settings
- Direct vs non-Direct preprocessing (timeouts/manual terminate / terminations)
- recorder preprocessing (dataset_file checks, export modes, success term setup)
- environment creation via gym.make
- replacing recorder_manager with StreamingRecorderManager if recording enabled

NOTE:
- Import this module ONLY after Isaac Sim is launched (AppLauncher),
  because it relies on IsaacLab/LeIsaac runtime imports.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import gymnasium as gym
import torch

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import DatasetExportMode, TerminationTermCfg
from isaaclab_tasks.utils import parse_env_cfg

from leisaac.enhance.managers import EnhanceDatasetExportMode, StreamingRecorderManager


@dataclass
class TeleopEnvBuildResult:
    env: ManagerBasedRLEnv | DirectRLEnv
    env_cfg: object
    task_name: str
    is_direct_env: bool
    output_dir: str
    output_file_stem: str


def _prepare_output_paths(dataset_file: str) -> Tuple[str, str]:
    output_dir = os.path.dirname(dataset_file)
    output_file_stem = os.path.splitext(os.path.basename(dataset_file))[0]
    if output_dir and (not os.path.exists(output_dir)):
        os.makedirs(output_dir, exist_ok=True)
    return output_dir, output_file_stem


def _apply_quality_render(env_cfg, quality: bool):
    if not quality:
        return
    # keep same behavior as your script
    env_cfg.sim.render.antialiasing_mode = "FXAA"
    env_cfg.sim.render.rendering_mode = "quality"


def _apply_direct_or_manager_preprocess(env_cfg, task_name: str) -> bool:
    """
    Returns is_direct_env.
    """
    is_direct_env = "Direct" in str(task_name)

    if is_direct_env:
        # Direct env uses internal flags
        env_cfg.never_time_out = True
        env_cfg.manual_terminate = True
    else:
        # manager-based env: remove built-in time_out/success termination
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
        if hasattr(env_cfg.terminations, "success"):
            env_cfg.terminations.success = None

    return is_direct_env


def _apply_recorder_preprocess(env_cfg, args_cli, is_direct_env: bool, output_dir: str, output_file_stem: str):
    """
    Configure env_cfg.recorders based on args_cli.record/args_cli.resume/args_cli.dataset_file
    and setup manual success term for manager-based env when recording.
    """
    if not bool(getattr(args_cli, "record", False)):
        env_cfg.recorders = None
        return

    dataset_file = str(getattr(args_cli, "dataset_file"))
    resume = bool(getattr(args_cli, "resume", False))

    if resume:
        env_cfg.recorders.dataset_export_mode = EnhanceDatasetExportMode.EXPORT_ALL_RESUME
        assert os.path.exists(dataset_file), "dataset file does not exist for --resume"
    else:
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
        assert not os.path.exists(dataset_file), "dataset file exists; use --resume"

    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_stem

    # When recording, align success handling with your previous script:
    if is_direct_env:
        env_cfg.return_success_status = False
    else:
        if not hasattr(env_cfg.terminations, "success"):
            setattr(env_cfg.terminations, "success", None)
        env_cfg.terminations.success = TerminationTermCfg(
            func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        )


def _replace_recorder_manager_if_needed(env, env_cfg, args_cli):
    """
    Replace env.recorder_manager with StreamingRecorderManager when recording enabled.
    """
    if not bool(getattr(args_cli, "record", False)):
        return

    # keep your existing behavior
    if hasattr(env, "recorder_manager"):
        del env.recorder_manager

    env.recorder_manager = StreamingRecorderManager(env_cfg.recorders, env)
    env.recorder_manager.flush_steps = 100
    env.recorder_manager.compression = "lzf"


def build_env_cfg(args_cli):
    """
    Build and preprocess env_cfg (no env instantiation).
    """
    task_name = str(getattr(args_cli, "task"))
    env_cfg = parse_env_cfg(task_name, device=args_cli.device, num_envs=args_cli.num_envs)

    env_cfg.use_teleop_device(args_cli.teleop_device)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())

    _apply_quality_render(env_cfg, bool(getattr(args_cli, "quality", False)))

    is_direct_env = _apply_direct_or_manager_preprocess(env_cfg, task_name)

    output_dir, output_file_stem = _prepare_output_paths(str(getattr(args_cli, "dataset_file")))
    _apply_recorder_preprocess(env_cfg, args_cli, is_direct_env, output_dir, output_file_stem)

    return env_cfg, task_name, is_direct_env, output_dir, output_file_stem


def create_env_and_cfg(args_cli) -> TeleopEnvBuildResult:
    """
    Full factory: env_cfg preprocess + env creation + recorder_manager replacement.
    """
    env_cfg, task_name, is_direct_env, output_dir, output_file_stem = build_env_cfg(args_cli)

    env: ManagerBasedRLEnv | DirectRLEnv = gym.make(task_name, cfg=env_cfg).unwrapped

    _replace_recorder_manager_if_needed(env, env_cfg, args_cli)

    return TeleopEnvBuildResult(
        env=env,
        env_cfg=env_cfg,
        task_name=task_name,
        is_direct_env=is_direct_env,
        output_dir=output_dir,
        output_file_stem=output_file_stem,
    )
