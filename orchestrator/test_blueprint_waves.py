import unittest
from blueprint_compiler import BlueprintError, build_compilation_manifest, compile_execution_waves

class BlueprintWaveTests(unittest.TestCase):
    def test_waves_respect_dependency_levels_and_node_cap(self):
        requirements=[{"id":f"r{i:02d}","summary":f"req {i}"} for i in range(30)]
        manifest=build_compilation_manifest({"blueprint_id":"waves","requirements":requirements}, max_requirements_per_unit=1)
        self.assertGreaterEqual(manifest["graph"]["wave_count"],2)
        self.assertTrue(all(len(w["unit_ids"]) <= 24 for w in manifest["waves"]))
    def test_dependent_wave_can_still_run_units_in_parallel(self):
        manifest = build_compilation_manifest(
            {
                "blueprint_id": "parallel-dependent-wave",
                "requirements": [
                    {"id": "root", "summary": "Root"},
                    {"id": "left", "summary": "Left", "depends_on": ["root"], "workstream": "left"},
                    {"id": "right", "summary": "Right", "depends_on": ["root"], "workstream": "right"},
                ],
            },
            max_requirements_per_unit=1,
        )
        self.assertGreaterEqual(len(manifest["waves"]), 2)
        second_wave = manifest["waves"][1]
        self.assertEqual(len(second_wave["unit_ids"]), 2)
        self.assertTrue(second_wave["depends_on_waves"])
        self.assertTrue(second_wave["parallel_candidate"])

    def test_dependency_wave_order(self):
        manifest=build_compilation_manifest({"blueprint_id":"depwaves","requirements":[
            {"id":"a","summary":"A"},{"id":"b","summary":"B","depends_on":["a"]},
            {"id":"c","summary":"C","depends_on":["a"]}]})
        by={w["wave_id"]:w for w in manifest["waves"]}
        self.assertEqual(manifest["waves"][0]["depends_on_waves"],[])
        self.assertTrue(any(by[x]["unit_ids"] == [manifest["units"][0]["unit_id"]] for x in manifest["waves"][1]["depends_on_waves"]))
