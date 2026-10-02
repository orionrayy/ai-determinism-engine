import unittest

from state_schema import (
    CURRENT_STATE_VERSION,
    CURRENT_WORKFLOW_SCHEMA_VERSION,
    StateSchemaError,
    migrate_state,
)


class StateSchemaTests(unittest.TestCase):
    def test_legacy_state_migrates_with_new_runtime_defaults(self):
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
        self.assertEqual(workflow["schema_version"], CURRENT_WORKFLOW_SCHEMA_VERSION)
        self.assertEqual(workflow["reconciliations"], {})
        self.assertEqual(workflow["agent_team"], {})
        self.assertEqual(workflow["federation"], {})
        self.assertEqual(workflow["nodes"][0]["agent_role"], "")
        self.assertEqual(workflow["plan_integrity"], "legacy_unverified")
        self.assertEqual(workflow["nodes"][0]["status"], "pending")
        self.assertEqual(workflow["max_parallel"], 4)
        self.assertEqual(workflow["attempts_used"], 0)
        self.assertEqual(workflow["max_attempts"], 64)
        self.assertEqual(len(workflow["retry_jitter_seed"]), 32)

    def test_run_attempt_is_preserved_and_validated(self):
        migrated = migrate_state({
            "version": CURRENT_STATE_VERSION,
            "workflows": {
                "wf": {
                    "id": "wf",
                    "github_run_attempt": 2,
                    "origin_github_run_attempt": 1,
                    "nodes": [],
                }
            },
        })
        workflow = migrated["workflows"]["wf"]
        self.assertEqual(workflow["github_run_attempt"], 2)
        self.assertEqual(workflow["origin_github_run_attempt"], 1)

        with self.assertRaises(StateSchemaError):
            migrate_state({
                "version": CURRENT_STATE_VERSION,
                "workflows": {
                    "wf": {
                        "id": "wf",
                        "github_run_attempt": 0,
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
                        "schema_version": CURRENT_WORKFLOW_SCHEMA_VERSION + 1,
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
