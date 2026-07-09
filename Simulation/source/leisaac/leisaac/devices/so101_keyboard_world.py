import carb
import numpy as np

from .device_base import Device


class SO101KeyboardWorld(Device):
    """Keyboard controller for SO101 single arm: WORLD-frame translation + numeric rotation/gripper.

    Delta action layout (same as SO101Keyboard):
        (dx, dy, dz, droll, dpitch, dyaw, d_shoulder_pan, d_gripper)

    Key bindings (WORLD frame translation):
        ============================== ================= =================
        Description                    Key               Key
        ============================== ================= =================
        +Y / -Y (Forward/Backward)      W                 S
        -X / +X (Left/Right)            A                 D
        +Z / -Z (Up/Down)               Q                 E

        Roll  + / -                    1                 2
        Pitch + / -                    3                 4
        Yaw   + / -                    5                 6

        Gripper Open / Close           7                 8
        ============================== ================= =================

    Notes:
    - This mode intentionally DOES NOT call _convert_delta_from_frame().
      Because we want the delta pose to be interpreted in WORLD coordinates.
    """

    def __init__(self, env, sensitivity: float = 1.0):
        super().__init__(env, "keyboard")

        # sensitivities (keep consistent with SO101Keyboard default scale)
        self.pos_sensitivity = 0.01 * sensitivity
        self.rot_sensitivity = 0.15 * sensitivity
        self.joint_sensitivity = 0.15 * sensitivity

        self._create_key_bindings()

        # (dx, dy, dz, droll, dpitch, dyaw, d_shoulder_pan, d_gripper)
        self._delta_action = np.zeros(8, dtype=np.float32)

    def __str__(self) -> str:
        msg = "Keyboard Controller for SO101 Single Arm (WORLD-frame Translation + Numeric Rotation).\n"
        msg += f"\tKeyboard name: {self._input.get_keyboard_name(self._keyboard)}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tWORLD +Y / -Y (Forward/Backward):      W / S\n"
        msg += "\tWORLD -X / +X (Left/Right):            A / D\n"
        msg += "\tWORLD +Z / -Z (Up/Down):               Q / E\n"
        msg += "\tRoll  + / -:                           1 / 2\n"
        msg += "\tPitch + / -:                           3 / 4\n"
        msg += "\tYaw   + / -:                           5 / 6\n"
        msg += "\tGripper Open / Close:                  7 / 8\n"
        msg += "\t----------------------------------------------\n"
        return msg

    def reset(self):
        self._delta_action[:] = 0.0

    def get_device_state(self):
        # WORLD mode: return raw delta without frame conversion
        return self._delta_action

    def _on_keyboard_event(self, event, *args, **kwargs):
        super()._on_keyboard_event(event, *args, **kwargs)

        # apply the command when pressed
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._INPUT_KEY_MAPPING:
                self._delta_action += self._ACTION_DELTA_MAPPING[self._INPUT_KEY_MAPPING[event.input.name]]

        # remove the command when released
        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in self._INPUT_KEY_MAPPING:
                self._delta_action -= self._ACTION_DELTA_MAPPING[self._INPUT_KEY_MAPPING[event.input.name]]

    def _create_key_bindings(self):
        # World-frame translation mapping:
        # dx -> world X, dy -> world Y, dz -> world Z
        self._ACTION_DELTA_MAPPING = {
            # translation (WORLD)
            "w_y_pos": np.asarray([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            * self.pos_sensitivity,
            "w_y_neg": np.asarray([0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            * self.pos_sensitivity,
            "w_x_neg": np.asarray([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            * self.pos_sensitivity,
            "w_x_pos": np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            * self.pos_sensitivity,
            "w_z_pos": np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            * self.pos_sensitivity,
            "w_z_neg": np.asarray([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            * self.pos_sensitivity,

            # rotation increments (roll, pitch, yaw)
            "roll_pos": np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            * self.rot_sensitivity,
            "roll_neg": np.asarray([0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            * self.rot_sensitivity,
            "pitch_pos": np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            * self.rot_sensitivity,
            "pitch_neg": np.asarray([0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            * self.rot_sensitivity,
            "yaw_pos": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
            * self.rot_sensitivity,
            "yaw_neg": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0], dtype=np.float32)
            * self.rot_sensitivity,

            # gripper
            "gripper_open": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            * self.joint_sensitivity,
            "gripper_close": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
            * self.joint_sensitivity,
        }

        self._INPUT_KEY_MAPPING = {
            # world translation
            "W": "w_y_pos",
            "S": "w_y_neg",
            "A": "w_x_neg",
            "D": "w_x_pos",
            "Q": "w_z_pos",
            "E": "w_z_neg",
            # numeric rotation
            "1": "roll_pos",
            "2": "roll_neg",
            "3": "pitch_pos",
            "4": "pitch_neg",
            "5": "yaw_pos",
            "6": "yaw_neg",
            # gripper
            "7": "gripper_open",
            "8": "gripper_close",
        }
