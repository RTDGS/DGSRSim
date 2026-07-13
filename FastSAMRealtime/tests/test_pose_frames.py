from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


FASTSAM_ROOT = Path(__file__).resolve().parents[1]
if str(FASTSAM_ROOT) not in sys.path:
    sys.path.insert(0, str(FASTSAM_ROOT))

from utils.pose_utils import (
    compose_asset_to_scene,
    compose_target_to_scene,
    load_transform_4x4,
    make_target_normalization,
    similarity_scale,
    validate_transform_4x4,
)

STATE_TRANSFORM_PATH = (
    FASTSAM_ROOT.parent
    / "Simulation"
    / "source"
    / "leisaac"
    / "leisaac"
    / "utils"
    / "state_transform.py"
)
STATE_TRANSFORM_SPEC = importlib.util.spec_from_file_location(
    "dgsrsim_state_transform",
    STATE_TRANSFORM_PATH,
)
assert STATE_TRANSFORM_SPEC is not None and STATE_TRANSFORM_SPEC.loader is not None
STATE_TRANSFORM_MODULE = importlib.util.module_from_spec(STATE_TRANSFORM_SPEC)
STATE_TRANSFORM_SPEC.loader.exec_module(STATE_TRANSFORM_MODULE)
rigid_scene_placement = STATE_TRANSFORM_MODULE.rigid_scene_placement


class PoseFrameTests(unittest.TestCase):
    def test_composition_closes_camera_to_scene_chain(self) -> None:
        T_scene_from_camera = np.eye(4)
        T_scene_from_camera[:3, 3] = [1.0, 2.0, 3.0]

        T_target_from_camera = np.eye(4)
        T_target_from_camera[:3, 3] = [0.25, -0.5, 1.0]

        actual = compose_target_to_scene(T_scene_from_camera, T_target_from_camera)
        expected = T_scene_from_camera @ np.linalg.inv(T_target_from_camera)
        np.testing.assert_allclose(actual, expected)

    def test_json_loader_accepts_named_matrix(self) -> None:
        path = Path(__file__).with_name(".camera_to_scene_test.json")
        try:
            path.write_text(
                '{"T_scene_from_camera": [[1,0,0,0.1],[0,1,0,0.2],[0,0,1,0.3],[0,0,0,1]]}',
                encoding="utf-8",
            )
            loaded = load_transform_4x4(path)
            np.testing.assert_allclose(loaded[:3, 3], [0.1, 0.2, 0.3])
        finally:
            path.unlink(missing_ok=True)

    def test_non_rigid_matrix_is_rejected(self) -> None:
        invalid = np.eye(4)
        invalid[0, 0] = 2.0
        with self.assertRaises(ValueError):
            validate_transform_4x4(invalid, "invalid")

    def test_asset_similarity_retains_aabb_scale(self) -> None:
        scale = 0.08
        source_center = np.array([0.1, -0.2, 1.2])
        target_center = np.array([0.5, 0.25, 2.0])
        normalization = make_target_normalization(scale, source_center, target_center)

        T_normalized_from_camera = np.eye(4)
        T_normalized_from_camera[:3, 3] = [0.02, -0.03, 0.04]
        A_scene_from_asset = compose_asset_to_scene(
            np.eye(4),
            T_normalized_from_camera,
            normalization,
        )

        q_raw = np.array([0.7, -0.1, 2.4, 1.0])
        expected = np.linalg.inv(T_normalized_from_camera) @ normalization @ q_raw
        actual = A_scene_from_asset @ q_raw
        np.testing.assert_allclose(actual, expected)
        self.assertAlmostEqual(similarity_scale(A_scene_from_asset), scale)

    def test_scene_background_scale_is_not_applied_twice(self) -> None:
        T_world_scene = np.eye(4)
        T_world_scene[:3, :3] *= 0.08
        T_world_scene[:3, 3] = [1.2, -0.095, 0.06]
        placement = rigid_scene_placement(T_world_scene)
        np.testing.assert_allclose(placement[:3, :3], np.eye(3))
        np.testing.assert_allclose(placement[:3, 3], T_world_scene[:3, 3])

    def test_checked_in_calibration_is_loadable(self) -> None:
        path = FASTSAM_ROOT / "configs" / "calibration" / "kinect_camera_space_to_scene.json"
        np.testing.assert_allclose(load_transform_4x4(path), np.eye(4))


if __name__ == "__main__":
    unittest.main()
