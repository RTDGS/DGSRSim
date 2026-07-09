# -*- coding: utf-8 -*-
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from leisaac.utils.constant import ASSETS_ROOT

"""Configuration for My Custom Scene"""

SCENES_ROOT = Path(ASSETS_ROOT) / "scenes"

# 你的 USD 放在：leisaac/assets/scenes/my_scene/scene.usd
MY_SCENE_USD_PATH = str(SCENES_ROOT / "my_scene" / "test01.usd")

MY_SCENE_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=MY_SCENE_USD_PATH,
    )
)
