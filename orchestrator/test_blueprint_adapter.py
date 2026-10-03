import json
import unittest
from pathlib import Path
from blueprint_compiler import BlueprintError
from orchestrator import Node, execute_blueprint_compiler

class BlueprintAdapterTests(unittest.TestCase):
    def test_structured_compilation(self):
        node=Node(id="compile",capability="blueprint",tool="blueprint_compiler",input={"blueprint":{"blueprint_id":"demo","requirements":[{"id":"a","summary":"A"},{"id":"b","summary":"B","depends_on":["a"]}]}})
        result=execute_blueprint_compiler(node,"ignored")
        self.assertEqual(result["blueprint_id"],"demo")
        self.assertEqual(result["unit_count"],2)
        self.assertIn("manifest_digest",result)
        self.assertIn("wave_count",result)
        self.assertIn("waves",result)
    def test_file_mode_needs_workload_root(self):
        node=Node(id="compile",capability="blueprint",tool="blueprint_compiler",input={"blueprint_files":["blueprint.md"]})
        import os
        old=os.environ.pop("ORCHESTRATOR_WORKLOAD_ROOT",None)
        try:
            with self.assertRaises(BlueprintError): execute_blueprint_compiler(node,"ignored")
        finally:
            if old is not None: os.environ["ORCHESTRATOR_WORKLOAD_ROOT"]=old
    def test_registry(self):
        root=Path(__file__).resolve().parent.parent
        registry=json.loads((root/"orchestrator"/"tools.json").read_text())
        self.assertEqual(registry["capability:blueprint"]["default_tool"],"blueprint_compiler")
        self.assertTrue(registry["blueprint_compiler"]["free_tier"])
