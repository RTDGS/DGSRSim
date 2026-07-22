from __future__ import annotations

import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
LEISAAC_UTILS = REPO / "Simulation/source/leisaac/leisaac/utils"
FASTSAM_UTILS = REPO / "FastSAMRealtime/utils"
TEST_TMP_ROOT = Path(__file__).resolve().parent / ".tmp"
TEST_TMP_ROOT.mkdir(exist_ok=True)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


leisaac_pkg = types.ModuleType("leisaac")
leisaac_utils_pkg = types.ModuleType("leisaac.utils")
sys.modules.setdefault("leisaac", leisaac_pkg)
sys.modules.setdefault("leisaac.utils", leisaac_utils_pkg)

physics_stub = types.ModuleType("leisaac.utils.physics_prims")
physics_stub.get_world_xf = lambda stage, path: np.eye(4)
physics_stub.set_prim_world_matrix = lambda stage, path, matrix: None
sys.modules["leisaac.utils.physics_prims"] = physics_stub

load_module(
    "leisaac.utils.state_transform",
    LEISAAC_UTILS / "state_transform.py",
)
pose_sync = load_module(
    "leisaac.utils.pose_sync_pipeline",
    LEISAAC_UTILS / "pose_sync_pipeline.py",
)
bundle_writer = load_module(
    "object_state_bundle",
    FASTSAM_UTILS / "object_state_bundle.py",
)
runtime_object_config = load_module(
    "runtime_object_config",
    LEISAAC_UTILS / "runtime_object_config.py",
)


def similarity(scale: float, translation: tuple[float, float, float]) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] *= scale
    value[:3, 3] = translation
    return value


class MultiObjectStateTests(unittest.TestCase):
    def state_path(self, name: str) -> Path:
        path = TEST_TMP_ROOT / name
        for candidate in (path, Path(str(path) + ".lock"), Path(str(path) + ".tmp")):
            if candidate.exists():
                candidate.unlink()
            self.addCleanup(lambda value=candidate: value.unlink(missing_ok=True))
        return path

    def test_bundle_writer_retains_multiple_objects(self):
        path = self.state_path("writer_object_states.json")
        for object_id, matrix in (
            ("astronaut", similarity(0.8, (0.1, 0.2, 0.3))),
            ("pot", similarity(1.2, (-0.2, 0.0, 0.4))),
        ):
            bundle_writer.update_object_state_bundle(
                str(path),
                object_id,
                {
                    "schema": "dgsrsim.object_state.v1",
                    "object_id": object_id,
                    "timestamp_unix": 1.0,
                    "A_scene_from_asset_raw": matrix.tolist(),
                },
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema"], "dgsrsim.object_states.v1")
        self.assertEqual(set(payload["objects"]), {"astronaut", "pot"})

        bundle_writer.deactivate_object(str(path), "pot", timestamp_unix=2.0)
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertFalse(payload["objects"]["pot"]["active"])
        self.assertIn("A_scene_from_asset_raw", payload["objects"]["pot"])

        bundle_writer.set_object_active(str(path), "pot", True, timestamp_unix=3.0)
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(payload["objects"]["pot"]["active"])

    def test_stream_reads_all_active_objects(self):
        path = self.state_path("reader_object_states.json")
        path.write_text(
            json.dumps(
                {
                    "schema": "dgsrsim.object_states.v1",
                    "objects": {
                        "astronaut": {
                            "active": True,
                            "A_scene_from_asset_raw": similarity(
                                0.8, (0.1, 0.2, 0.3)
                            ).tolist(),
                        },
                        "pot": {
                            "active": True,
                            "A_scene_from_asset_raw": similarity(
                                1.2, (-0.2, 0.0, 0.4)
                            ).tolist(),
                        },
                        "hidden": {
                            "active": False,
                            "A_scene_from_asset_raw": np.eye(4).tolist(),
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        stream = pose_sync.ObjectStateFileStream(str(path), "")
        states, inactive_ids, _, source = stream._read_candidate()
        self.assertEqual(source, str(path))
        self.assertEqual(set(states), {"astronaut", "pot"})
        self.assertEqual(inactive_ids, {"hidden"})

    def test_stream_accepts_packet_with_no_active_objects(self):
        stream = pose_sync.ObjectStateFileStream("", "")
        stream._store_candidate(
            {"astronaut": similarity(0.8, (0.1, 0.2, 0.3))},
            set(),
            1.0,
            "object_states.json",
        )
        self.assertEqual(set(stream.get_latest_all()), {"astronaut"})

        updated = stream._store_candidate({}, {"astronaut"}, 2.0, "object_states.json")
        self.assertTrue(updated)
        self.assertEqual(stream.get_latest_all(), {})
        self.assertEqual(stream.get_latest_packet()[1], {"astronaut"})

    def test_pipeline_applies_each_object_binding(self):
        calls = []
        pose_sync.get_world_xf = lambda stage, path: np.eye(4)
        pose_sync.set_prim_world_matrix = (
            lambda stage, path, matrix: calls.append((path, np.asarray(matrix)))
        )
        cfg = pose_sync.PoseSyncConfig(
            default_object_id="astronaut",
            object_path_templates={
                "astronaut": "/World/envs/env_{i}/Astronaut",
                "pot": "/World/envs/env_{i}/Pot",
            },
        )
        pipeline = pose_sync.PoseSyncPipeline(cfg, num_envs=2, rule_grasp_mode=False)
        pipeline._stream._latest_states = {
            "astronaut": similarity(0.8, (0.1, 0.2, 0.3)),
            "pot": similarity(1.2, (-0.2, 0.0, 0.4)),
        }
        pipeline.step(stage=object())
        self.assertEqual(
            {path for path, _ in calls},
            {
                "/World/envs/env_0/Astronaut",
                "/World/envs/env_0/Pot",
                "/World/envs/env_1/Astronaut",
                "/World/envs/env_1/Pot",
            },
        )
        self.assertEqual(len(calls), 4)

        pipeline.step(stage=object())
        self.assertEqual(len(calls), 4, "unchanged states must not be written twice")

        pipeline._stream._latest_states["pot"] = similarity(1.2, (-0.1, 0.0, 0.4))
        pipeline.step(stage=object())
        self.assertEqual(len(calls), 6, "only the changed object is written in each environment")

    def test_pipeline_forwards_lifecycle_transitions(self):
        class Lifecycle:
            def __init__(self):
                self.activated = []
                self.deactivated = []

            def activate(self, stage, object_id):
                self.activated.append(object_id)

            def deactivate(self, stage, object_id):
                self.deactivated.append(object_id)

        pose_sync.get_world_xf = lambda stage, path: np.eye(4)
        pose_sync.set_prim_world_matrix = lambda stage, path, matrix: None
        lifecycle = Lifecycle()
        cfg = pose_sync.PoseSyncConfig(
            default_object_id="astronaut",
            object_path_templates={"astronaut": "/World/envs/env_{i}/Astronaut"},
        )
        pipeline = pose_sync.PoseSyncPipeline(cfg, num_envs=1, rule_grasp_mode=False)
        pipeline.set_lifecycle_manager(lifecycle)
        pipeline._stream._latest_states = {"astronaut": similarity(1.0, (0.0, 0.0, 0.0))}
        pipeline.step(stage=object())
        self.assertEqual(lifecycle.activated, ["astronaut"])

        pipeline._stream._latest_states = {}
        pipeline._stream._latest_inactive_ids = {"astronaut"}
        pipeline.step(stage=object())
        self.assertEqual(lifecycle.deactivated, ["astronaut"])

    def test_runtime_config_exposes_every_enabled_object(self):
        path = self.state_path("runtime_objects.json")
        path.write_text(
            json.dumps(
                {
                    "schema": "dgsrsim.simulation_object_bindings.v1",
                    "objects": {
                        "astronaut": {
                            "prim_path_template": "/World/envs/env_{i}/Astronaut",
                            "spawn": {
                                "enabled": True,
                                "asset_path": "astronaut.usdz",
                                "asset_path_base": "assets_root",
                                "asset_profile_json": "astronaut.json",
                                "asset_profile_base": "project_root",
                                "initial_position_m": [0.1, 0.2, 0.3],
                            },
                        },
                        "pot": {
                            "prim_path_template": "/World/envs/env_{i}/Pot",
                            "spawn": {
                                "enabled": True,
                                "asset_path": "pot.usdz",
                                "asset_profile_json": "pot.json",
                            },
                        },
                        "background": {
                            "prim_path_template": "/World/envs/env_{i}/Background",
                            "spawn": {"enabled": False},
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        configs = runtime_object_config.load_runtime_object_configs(
            str(path),
            assets_root=TEST_TMP_ROOT / "assets",
            project_root=TEST_TMP_ROOT / "project",
        )
        self.assertEqual(set(configs), {"astronaut", "pot"})
        self.assertEqual(configs["astronaut"].proxy_name, "Astronaut")
        self.assertEqual(configs["pot"].usdz_path, TEST_TMP_ROOT / "assets" / "pot.usdz")

    def test_empty_binding_map_removes_all_explicit_objects(self):
        path = self.state_path("empty_runtime_objects.json")
        path.write_text(
            json.dumps(
                {
                    "schema": "dgsrsim.simulation_object_bindings.v1",
                    "objects": {},
                }
            ),
            encoding="utf-8",
        )
        self.assertEqual(pose_sync.load_object_path_templates(str(path)), {})
        self.assertEqual(
            runtime_object_config.load_runtime_object_configs(
                str(path),
                assets_root=TEST_TMP_ROOT / "assets",
                project_root=TEST_TMP_ROOT / "project",
            ),
            {},
        )

        cfg = pose_sync.PoseSyncConfig(
            object_path_templates={},
            default_object_id="legacy",
            mug_path_tpl="/World/envs/env_{i}/Legacy",
        )
        pipeline = pose_sync.PoseSyncPipeline(cfg, num_envs=1, rule_grasp_mode=False)
        self.assertIn("legacy", pipeline._object_path_templates)
        pipeline.set_object_path_templates({})
        self.assertEqual(pipeline._object_path_templates, {})


if __name__ == "__main__":
    unittest.main()
