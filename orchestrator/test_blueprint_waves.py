import unittest
from blueprint_compiler import BlueprintError, build_compilation_manifest, compile_execution_waves

class BlueprintWaveTests(unittest.TestCase):
    def test_waves_respect_dependency_levels_and_node_cap(self):
        requirements=[{"id":f"r{i:02d}","summary":f"req {i}"} for i in range(30)]
        manifest=build_compilation_manifest({"blueprint_id":"waves","requirements":requirements})
        self.assertGreaterEqual(manifest["graph"]["wave_count"],2)
        self.assertTrue(all(len(w["unit_ids"]) <= 24 for w in manifest["waves"]))
    def test_dependency_wave_order(self):
        manifest=build_compilation_manifest({"blueprint_id":"depwaves","requirements":[
            {"id":"a","summary":"A"},{"id":"b","summary":"B","depends_on":["a"]},
            {"id":"c","summary":"C","depends_on":["a"]}]})
        by={w["wave_id"]:w for w in manifest["waves"]}
        self.assertEqual(manifest["waves"][0]["depends_on_waves"],[])
        self.assertTrue(any(by[x]["unit_ids"] == [manifest["units"][0]["unit_id"]] for x in manifest["waves"][1]["depends_on_waves"]))
