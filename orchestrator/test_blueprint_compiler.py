import unittest
from blueprint_compiler import BlueprintError, build_compilation_manifest, load_blueprint_file, parse_source_document, render_execution_packet

class BlueprintCompilerTests(unittest.TestCase):
    def test_parse_source_document(self):
        parsed=parse_source_document("# A\nalpha\n## B\nbeta\n")
        self.assertEqual(parsed["source_kind"],"markdown-or-text")
        self.assertEqual(parsed["sections"][1]["line_start"],3)
        self.assertIn("digest",parsed["sections"][0])
    def test_dependencies_parallel_and_packet(self):
        manifest=build_compilation_manifest({"blueprint_id":"demo","requirements":[
            {"id":"a","summary":"A","workstream":"one"},
            {"id":"b","summary":"B","workstream":"two"},
            {"id":"c","summary":"C","depends_on":["a"],"workstream":"three"}]})
        self.assertEqual(len(manifest["units"]),3)
        self.assertEqual(len(manifest["graph"]["parallel_candidate_units"]),2)
        packet=render_execution_packet(manifest,manifest["units"][0]["unit_id"])
        self.assertTrue(packet["execution_contract"]["re_audit_previous_results"])
    def test_fail_closed_duplicate_and_cycle(self):
        with self.assertRaises(BlueprintError):
            build_compilation_manifest({"blueprint_id":"dup","requirements":[{"id":"a","summary":"A"},{"id":"a","summary":"A2"}]})
        with self.assertRaises(BlueprintError):
            build_compilation_manifest({"blueprint_id":"cycle","requirements":[{"id":"a","summary":"A","depends_on":["b"]},{"id":"b","summary":"B","depends_on":["a"]}]})
    def test_file_root_boundary(self):
        with __import__("tempfile").TemporaryDirectory() as root:
            from pathlib import Path
            outside=Path(root).parent/"outside-blueprint.md"
            outside.write_text("outside")
            with self.assertRaises(BlueprintError):
                load_blueprint_file("../outside-blueprint.md",root=root)

if __name__=="__main__": unittest.main()
