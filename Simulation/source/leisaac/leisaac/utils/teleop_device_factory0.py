# -*- coding: utf-8 -*-
"""
leisaac/utils/teleop_device_factory.py

Factory for:
- teleop_interface (device)
- optional rule_agent (RuleGraspAgent) when teleop_device == "rule-grasp"

Design notes:
- Import concrete classes (not modules) to avoid:
    TypeError: 'module' object is not callable
- Keep all device selection logic in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Any


@dataclass
class TeleopBuildResult:
    teleop_interface: Any
    rule_agent: Optional[Any] = None


def create_teleop_and_agent(env, stage, args_cli) -> TeleopBuildResult:
    """
    Create the teleop_interface and (optional) RuleGraspAgent based on args_cli.teleop_device.
    """
    device = str(args_cli.teleop_device)

    rule_agent = None

    if device == "rule-grasp":
        # Ensure leisaac/devices/__init__.py exports the CLASS:
        #   from .rule_grasp_keyboard import RuleGraspKeyboard
        from leisaac.devices import RuleGraspKeyboard
        from leisaac.agents import RuleGraspAgent

        teleop_interface = RuleGraspKeyboard(env)

        rule_agent = RuleGraspAgent(
            env=env,
            stage=stage,
            env_id=0,
            ee_prim_path_rel=args_cli.ee_prim_path,
            pregrasp_dz=args_cli.autograsp_pregrasp_dz,
            grasp_dz=args_cli.autograsp_grasp_dz,
            lift_dz=args_cli.autograsp_lift_dz,
            xy_gain=args_cli.autograsp_xy_gain,
            z_gain=args_cli.autograsp_z_gain,
            yaw_gain=args_cli.autograsp_yaw_gain,
            close_steps=args_cli.autograsp_close_steps,
            settle_steps=args_cli.autograsp_settle_steps,
            success_on_lift=args_cli.autograsp_success_on_lift,
            grip_open_val=args_cli.autograsp_open_val,
            grip_close_val=args_cli.autograsp_close_val,



        )

        return TeleopBuildResult(teleop_interface=teleop_interface, rule_agent=rule_agent)

    if device == "keyboard":
        from leisaac.devices import SO101Keyboard
        teleop_interface = SO101Keyboard(env, sensitivity=args_cli.sensitivity)
        return TeleopBuildResult(teleop_interface=teleop_interface)

    if device == "gamepad":
        from leisaac.devices import SO101Gamepad
        teleop_interface = SO101Gamepad(env, sensitivity=args_cli.sensitivity)
        return TeleopBuildResult(teleop_interface=teleop_interface)

    if device == "keyboard-world":
        from leisaac.devices import SO101KeyboardWorld
        teleop_interface = SO101KeyboardWorld(env, sensitivity=args_cli.sensitivity)
        return TeleopBuildResult(teleop_interface=teleop_interface)

    if device == "macro-keyboard":
        from leisaac.devices.so101_macro_keyboard import SO101MacroKeyboard
        teleop_interface = SO101MacroKeyboard(env, sensitivity=args_cli.sensitivity, out_dir="./traj_out")
        return TeleopBuildResult(teleop_interface=teleop_interface)

    if device == "so101leader":
        from leisaac.devices import SO101Leader
        teleop_interface = SO101Leader(env, port=args_cli.port, recalibrate=args_cli.recalibrate)
        return TeleopBuildResult(teleop_interface=teleop_interface)

    if device == "bi-so101leader":
        from leisaac.devices import BiSO101Leader
        teleop_interface = BiSO101Leader(
            env,
            left_port=args_cli.left_arm_port,
            right_port=args_cli.right_arm_port,
            recalibrate=args_cli.recalibrate,
        )
        return TeleopBuildResult(teleop_interface=teleop_interface)

    if device == "lekiwi-keyboard":
        from leisaac.devices import LeKiwiKeyboard
        teleop_interface = LeKiwiKeyboard(env, sensitivity=args_cli.sensitivity)
        return TeleopBuildResult(teleop_interface=teleop_interface)

    if device == "lekiwi-leader":
        from leisaac.devices import LeKiwiLeader
        teleop_interface = LeKiwiLeader(env, port=args_cli.port, recalibrate=args_cli.recalibrate)
        return TeleopBuildResult(teleop_interface=teleop_interface)

    if device == "lekiwi-gamepad":
        from leisaac.devices import LeKiwiGamepad
        teleop_interface = LeKiwiGamepad(env, sensitivity=args_cli.sensitivity)
        return TeleopBuildResult(teleop_interface=teleop_interface)

    raise ValueError(f"Invalid device interface '{device}'.")
