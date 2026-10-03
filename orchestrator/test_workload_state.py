import unittest
from state_schema import migrate_state, CURRENT_WORKFLOW_SCHEMA_VERSION

class WorkloadStateTests(unittest.TestCase):
    def test_workload_defaults_are_present(self):
        state=migrate_state({"version":4,"workflows":{"wf":{"id":"wf","nodes":[]}}})
        wf=state["workflows"]["wf"]
        self.assertEqual(wf["workload"], {})
        self.assertEqual(wf["schema_version"], CURRENT_WORKFLOW_SCHEMA_VERSION)
    def test_workload_digest_validation(self):
        with self.assertRaises(Exception):
            migrate_state({"version":4,"workflows":{"wf":{"id":"wf","workload":{"blueprint_digest":"bad"},"nodes":[]}}})
