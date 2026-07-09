# -*- coding: utf-8 -*-
import torch
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from leisaac.assets.scenes.my_scene import (
    MY_SCENE_CFG,
    MY_SCENE_USD_PATH,
)

from leisaac.utils.domain_randomization import (
    domain_randomization,
    randomize_camera_uniform,
    randomize_object_uniform,
)
from leisaac.utils.general_assets import parse_usd_and_create_subassets

from ..template import (
    SingleArmObservationsCfg,
    SingleArmTaskEnvCfg,
    SingleArmTaskSceneCfg,
    SingleArmTerminationsCfg,
)
from . import mdp

import math

def quat_wxyz_from_euler_deg(roll_deg, pitch_deg, yaw_deg):

    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)

    cr = math.cos(r * 0.5)
    sr = math.sin(r * 0.5)
    cp = math.cos(p * 0.5)
    sp = math.sin(p * 0.5)
    cy = math.cos(y * 0.5)
    sy = math.sin(y * 0.5)

    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = cy * sp * cr + sy * cp * sr
    z = sy * cp * cr - cy * sp * sr
    return (w, x, y, z)

@configclass
class PickOrangeSceneCfg(SingleArmTaskSceneCfg):
    """Scene configuration for the pick orange task."""

    # scene: AssetBaseCfg = KITCHEN_WITH_ORANGE_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")
    scene: AssetBaseCfg = MY_SCENE_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")
    def __post_init__(self):
        super().__post_init__()
        # robot 位置（米）
        self.robot.init_state.pos = (0.85, -0.41, -0.42)

        # robot 旋转：用欧拉角（度）
        self.robot.init_state.rot = quat_wxyz_from_euler_deg(
            roll_deg=0.0,
            pitch_deg=0.0,
            yaw_deg=0.0,
        )


@configclass
class ObservationsCfg(SingleArmObservationsCfg):
    pass
    # @configclass
    # class SubtaskCfg(ObsGroup):
    #     """Observations for subtask group."""
    #
    #
    #     def __post_init__(self):
    #         self.enable_corruption = False
    #         self.concatenate_terms = False
    #
    # subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg(SingleArmTerminationsCfg):
    pass
    # success = DoneTerm(
    #     func=mdp.task_done,
    #     params={
    #
    #     },
    # )

from leisaac.devices.action_process import init_action_cfg
@configclass
class PickOrangeEnvCfg(SingleArmTaskEnvCfg):
    """Configuration for the pick orange environment."""

    scene: PickOrangeSceneCfg = PickOrangeSceneCfg(env_spacing=8.0)

    observations: ObservationsCfg = ObservationsCfg()

    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        #    关键：在 validate 前补齐 actions
        self.actions = init_action_cfg(self.actions, device="keyboard")
