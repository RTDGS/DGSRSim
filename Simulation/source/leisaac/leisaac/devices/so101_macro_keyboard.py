# -*- coding: utf-8 -*-
import carb
import numpy as np
import torch

from .device_base import Device
from leisaac.utils.traj_export import export_joint_traj_npz


class SO101MacroKeyboard(Device):
    """Keyboard macro runner for SO101: press '1' to run a predefined pick-like macro and export joint_pos trajectory."""

    JOINT_NAMES_6 = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    def __init__(self, env, sensitivity: float = 1.0, out_dir: str = "./traj_out"):
        super().__init__(env, "keyboard")

        self.out_dir = out_dir
        self.pos_sensitivity = 0.01 * sensitivity
        self.rot_sensitivity = 0.15 * sensitivity
        self.joint_sensitivity = 0.15 * sensitivity

        # 8-dim delta action: (dx, dy, dz, droll, dpitch, dyaw, d_shoulder_pan, d_gripper)
        self._delta_action = np.zeros(8, dtype=np.float32)

        # macro state
        self._running = False
        self._state = "IDLE"
        self._state_step = 0
        self._max_steps_total = 600  # safety timeout

        # log buffers
        self._q_log: list[np.ndarray] = []
        self._dt_log: list[float] = []
        self._last_sim_time: float | None = None

        # robot handle
        self.robot = self.env.scene["robot"]

        # joint indices for export
        self._joint_ids = self._find_joint_ids(self.JOINT_NAMES_6)

        # key mapping (only macro trigger + optional abort)
        self._INPUT_KEY_MAPPING = {
            "1": "run_macro",
            "ESCAPE": "abort",
        }

    def __str__(self) -> str:
        msg = "SO101 Macro Keyboard\n"
        msg += f"\tKeyboard name: {self._input.get_keyboard_name(self._keyboard)}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tPress 1: run pick macro (demo) + export joint_pos traj\n"
        msg += "\tPress ESC: abort macro\n"
        msg += "\t----------------------------------------------\n"
        return msg

    def reset(self):
        self._delta_action[:] = 0.0
        self._running = False
        self._state = "IDLE"
        self._state_step = 0
        self._q_log.clear()
        self._dt_log.clear()
        self._last_sim_time = None

    def get_device_state(self):
        # macro mode: always return current delta action (8,)
        return self._delta_action

    def _on_keyboard_event(self, event, *args, **kwargs):
        super()._on_keyboard_event(event, *args, **kwargs)

        # event.input may be either an object with .name or a plain string
        key = event.input.name if hasattr(event.input, "name") else str(event.input)

        # print debug
        print("[KEY EVENT]", key, event.type)

        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return

        if key in ["1", "KEY_1", "NUMPAD_1"]:
            if not self._running:
                self._start_macro()
        elif key in ["ESCAPE"]:
            self._abort_macro()

    def advance(self):
        """Override Device.advance behavior? Usually Device.advance calls get_device_state internally.
        If your Device base already implements advance, do NOT override this.
        If it doesn't, you can ignore this method.

        In most LeIsaac devices, you only implement get_device_state and keyboard callbacks.
        Macro progression is done by updating self._delta_action per sim step from outside.
        """
        # If base class calls get_device_state() every step, we can update macro here by hooking into it.
        # However, if your base Device already has advance(), you can implement a per-step update method
        # and call it from get_device_state(). We'll do it in get_device_state-like flow by exposing update().
        return super().advance()

    # -----------------------------
    # Per-step update hook
    # -----------------------------
    def update_macro(self):
        """Call this once per sim step before env.step(actions)."""
        if not self._running:
            self._delta_action[:] = 0.0
            return

        # safety timeout
        if len(self._q_log) > self._max_steps_total:
            self._finish_macro(success=False, reason="timeout")
            return

        # record joint_pos (first env only)
        q6 = self._read_joint_pos_6()
        self._q_log.append(q6)

        # dt estimation (optional)
        dt = self._estimate_dt()
        self._dt_log.append(dt)

        # --- DEMO macro sequence (replace with your real pick planner later) ---
        # State machine uses fixed step counts; you can replace with pose error thresholds.
        self._delta_action[:] = 0.0

        if self._state == "LIFT_UP":
            # dz +
            self._delta_action[2] = +self.pos_sensitivity
            self._state_step += 1
            if self._state_step >= 60:
                self._goto("DOWN")

        elif self._state == "DOWN":
            # dz -
            self._delta_action[2] = -self.pos_sensitivity
            self._state_step += 1
            if self._state_step >= 60:
                self._goto("CLOSE")

        elif self._state == "CLOSE":
            # close gripper (d_gripper is last dim)
            self._delta_action[7] = -self.joint_sensitivity
            self._state_step += 1
            if self._state_step >= 40:
                self._goto("LIFT_WITH_OBJECT")

        elif self._state == "LIFT_WITH_OBJECT":
            self._delta_action[2] = +self.pos_sensitivity
            self._state_step += 1
            if self._state_step >= 80:
                self._finish_macro(success=True, reason="done")

        else:
            self._finish_macro(success=False, reason=f"unknown_state:{self._state}")

    # -----------------------------
    # Internal
    # -----------------------------
    def _start_macro(self):
        self._running = True
        self._q_log.clear()
        self._dt_log.clear()
        self._last_sim_time = None
        self._goto("LIFT_UP")
        print("[MACRO] start: demo pick macro")

    def _abort_macro(self):
        if self._running:
            self._finish_macro(success=False, reason="aborted")

    def _goto(self, state: str):
        self._state = state
        self._state_step = 0
        print(f"[MACRO] state -> {state}")

    def _finish_macro(self, success: bool, reason: str):
        self._running = False
        self._delta_action[:] = 0.0
        print(f"[MACRO] finish: success={success}, reason={reason}")

        if len(self._q_log) == 0:
            return

        # choose dt: use env cfg step_hz if available; else median of estimated dt
        dt = self._infer_dt()

        q_traj = np.stack(self._q_log, axis=0)  # [T,6]
        path = export_joint_traj_npz(
            out_dir=self.out_dir,
            joint_names=self.JOINT_NAMES_6,
            q_traj=q_traj,
            dt=dt,
            task_name="pick_demo",
            success=success,
        )
        print(f"[MACRO] exported traj -> {path}")

    def _find_joint_ids(self, joint_names: list[str]) -> list[int]:
        # IsaacLab articulation typically provides find_joints
        if hasattr(self.robot, "find_joints"):
            ids, _ = self.robot.find_joints(joint_names)
            return list(ids)

        # fallback: try joint_names property
        names = None
        if hasattr(self.robot, "joint_names"):
            names = list(self.robot.joint_names)
        elif hasattr(self.robot, "data") and hasattr(self.robot.data, "joint_names"):
            names = list(self.robot.data.joint_names)

        if names is None:
            raise RuntimeError("Cannot resolve joint ids. Please check articulation API for joint name access.")

        name_to_id = {n: i for i, n in enumerate(names)}
        return [name_to_id[n] for n in joint_names]

    def _read_joint_pos_6(self) -> np.ndarray:
        # joint_pos: [num_envs, num_joints]
        joint_pos = self.robot.data.joint_pos  # torch tensor
        q = joint_pos[0, self._joint_ids].detach().cpu().numpy().astype(np.float32)
        return q

    def _estimate_dt(self) -> float:
        # best-effort using app time; if not available, return NaN and infer later
        try:
            t = float(self.env.sim.get_physics_dt())  # some envs expose this
            return t
        except Exception:
            return float("nan")

    def _infer_dt(self) -> float:
        # priority: env cfg step_hz
        try:
            step_hz = getattr(self.env.cfg, "step_hz", None)
            if step_hz is not None and step_hz > 0:
                return 1.0 / float(step_hz)
        except Exception:
            pass

        # next: if user runs teleop script with args step_hz, env.cfg may not have it.
        # fallback: median of finite dt_log
        dts = [d for d in self._dt_log if np.isfinite(d) and d > 0]
        if dts:
            return float(np.median(dts))

        # final fallback
        return 1.0 / 60.0
