from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
UTILS = REPO / "Simulation/source/leisaac/leisaac/utils"


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
for name in (
    "set_xform_trs",
    "get_prim_world_aabb_size",
    "make_rigidbody_kinematic",
    "create_proxy_rigid_box",
    "spawn_usdz_under_parent",
    "disable_collisions_under",
):
    setattr(physics_stub, name, lambda *args, **kwargs: None)
sys.modules["leisaac.utils.physics_prims"] = physics_stub

factory = load_module("leisaac.utils.runtime_mug_factory", UTILS / "runtime_mug_factory.py")


class Prim:
    def __init__(self, valid: bool, active: bool = True):
        self._valid = valid
        self.active = active

    def IsValid(self):
        return self._valid

    def SetActive(self, active):
        self.active = bool(active)


class Stage:
    def __init__(self):
        self.paths = set()
        self.active = {}
        self.removed = []

    def GetPrimAtPath(self, path):
        prim = Prim(path in self.paths, self.active.get(path, True))

        def set_active(active):
            prim.active = bool(active)
            self.active[path] = bool(active)

        prim.SetActive = set_active
        return prim

    def RemovePrim(self, path):
        self.paths.discard(path)
        self.removed.append(path)


def spec(name: str, asset: str):
    return factory.RuntimeObjectSpec(proxy_name=name, usdz_path=asset)


class RuntimeObjectLifecycleTests(unittest.TestCase):
    def test_reconcile_add_remove_and_replace(self):
        stage = Stage()
        spawned = []

        def spawn_all(stage, num_envs, env_root_tpl, spec, verbose):
            spawned.append(spec.usdz_path)
            for index in range(num_envs):
                path = f"{env_root_tpl.format(i=index)}/{spec.proxy_name}"
                stage.paths.add(path)
                stage.active[path] = True

        manager = factory.RuntimeObjectManager(stage, 2, spawn_all=spawn_all)
        manager.reconcile({"a": spec("A", "a.usdz")})
        self.assertEqual(manager.active_object_ids, {"a"})
        self.assertEqual(spawned, ["a.usdz"])

        manager.reconcile({"a": spec("A", "a2.usdz"), "b": spec("B", "b.usdz")})
        self.assertEqual(manager.active_object_ids, {"a", "b"})
        self.assertEqual(spawned, ["a.usdz", "a2.usdz", "b.usdz"])
        self.assertIn("/World/envs/env_0/A", stage.removed)

        manager.reconcile({"b": spec("B", "b.usdz")})
        self.assertEqual(manager.active_object_ids, {"b"})
        self.assertIn("/World/envs/env_1/A", stage.removed)

    def test_state_activation_handles_preexisting_bound_prims(self):
        stage = Stage()
        paths = {f"/World/envs/env_{index}/Existing" for index in range(2)}
        stage.paths.update(paths)
        manager = factory.RuntimeObjectManager(
            stage,
            2,
            object_path_templates={"existing": "/World/envs/env_{i}/Existing"},
        )

        manager.activate(stage, "existing")
        manager.deactivate(stage, "existing")
        self.assertTrue(all(stage.active[path] is False for path in paths))
        self.assertTrue(paths.isdisjoint(stage.removed))

        manager.activate(stage, "existing")
        self.assertTrue(all(stage.active[path] is True for path in paths))


if __name__ == "__main__":
    unittest.main()
