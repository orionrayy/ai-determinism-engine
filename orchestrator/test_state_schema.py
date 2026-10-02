import unittest

from state_schema import (
    CURRENT_STATE_VERSION,
    StateSchemaError,
    migrate_state,
)


class StateSchemaTests(unittest.TestCase):
    def test_v2_state_migrates_with_new_runtime_defaults(self):
        state = {
            "version": 2,
            "workflows": {
                "wf-old": {
                    "id": "wf-old",
                    "live": False,
                    "nodes": [
                        {
                            "id": "n01",
                            "capability": "execute",
                            "tool": "noop",
                        }
                    ],
                }
            },
        }
        migrated = migrate_state(state)
        self.assertEqual(migrated["version"], CURRENT_STATE_VERSION)
        workflow = migrated["workflows"]["wf-old"]
        self.assertEqual(workflow["schema_version"], 5)
        self.assertEqual(workflow["reconciliations"], {})
        self.assertEqual(workflow["policy_fingerprint"], None)
        self.assertEqual(workflow["route_snapshot"], {})
        self.assertEqual(workflow["policy_integrity"], "legacy_unverified")
        self.assertEqual(workflow["origin_github_run_id"], None)
        self.assertEqual(workflow["origin_github_run_attempt"], None)
        self.assertEqual(workflow["github_run_attempt"], None)
        self.assertEqual(workflow["plan_integrity"], "legacy_unverified")
        self.assertEqual(workflow["nodes"][0]["status"], "pending")
        self.assertEqual(workflow["max_parallel"], 4)
        self.assertEqual(workflow["execution_budget"], {"max_steps": 96, "used_steps": 0})

    def test_invalid_execution_budget_fails_closed(self):
        with self.assertRaises(StateSchemaError):
            migrate_state({
                "version": 4,
                "workflows": {
                    "wf": {
                        "schema_version": 3,
                        "execution_budget": {"max_steps": 0, "used_steps": 0},
                        "nodes": [],
                    }
                },
            })

    def test_future_state_version_fails_closed(self):
        with self.assertRaises(StateSchemaError):
            migrate_state({
                "version": CURRENT_STATE_VERSION + 1,
                "workflows": {},
            })


    def test_future_workflow_schema_fails_closed(self):
        with self.assertRaises(StateSchemaError):
            migrate_state({
                "version": CURRENT_STATE_VERSION,
                "workflows": {
                    "wf": {
                        "schema_version": 6,
                        "nodes": [],
                    }
                },
            })

    def test_malformed_workflow_fails_closed(self):
        with self.assertRaises(StateSchemaError):
            migrate_state({
                "version": 2,
                "workflows": {"wf": "not-an-object"},
            })


if __name__ == "__main__":
    unittest.main()
