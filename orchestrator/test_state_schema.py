import unittest

from state_schema import (
    CURRENT_STATE_VERSION,
    StateSchemaError,
    migrate_state,
)


class StateSchemaTests(unittest.TestCase):
    def test_legacy_execution_identity_enables_pending_callback(self):
        migrated = migrate_state({
            "version": CURRENT_STATE_VERSION,
            "workflows": {
                "wf": {
                    "id": "wf",
                    "execution_id": "a" * 64,
                    "nodes": [],
                }
            },
        })
        self.assertEqual(
            migrated["workflows"]["wf"]["callback"]["status"],
            "pending",
        )

    def test_callback_status_and_attempts_are_bounded(self):
        with self.assertRaises(StateSchemaError):
            migrate_state({
                "version": CURRENT_STATE_VERSION,
                "workflows": {
                    "wf": {
                        "id": "wf",
                        "callback": {"status": "unknown"},
                        "nodes": [],
                    }
                },
            })
        with self.assertRaises(StateSchemaError):
            migrate_state({
                "version": CURRENT_STATE_VERSION,
                "workflows": {
                    "wf": {
                        "id": "wf",
                        "callback": {"status": "pending", "attempts": 13},
                        "nodes": [],
                    }
                },
            })

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
        self.assertEqual(workflow["schema_version"], 6)
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

    def test_boolean_and_float_control_fields_fail_closed(self):
        cases = [("version", True), ("version", 1.5)]
        for field_name, value in cases:
            with self.subTest(field_name=field_name, value=value):
                with self.assertRaises(StateSchemaError):
                    migrate_state({field_name: value, "workflows": {}})

        for field_name, value in (
            ("max_parallel", True),
            ("max_parallel", 2.5),
            ("replan_count", True),
        ):
            with self.subTest(field_name=field_name, value=value):
                with self.assertRaises(StateSchemaError):
                    migrate_state({
                        "version": CURRENT_STATE_VERSION,
                        "workflows": {
                            "wf": {
                                "id": "wf",
                                field_name: value,
                                "nodes": [],
                            }
                        },
                    })

    def test_malformed_state_version_fails_closed(self):
        with self.assertRaises(StateSchemaError):
            migrate_state({
                "version": "not-a-version",
                "workflows": {},
            })

    def test_malformed_workflow_schema_version_fails_closed(self):
        with self.assertRaises(StateSchemaError):
            migrate_state({
                "version": CURRENT_STATE_VERSION,
                "workflows": {
                    "wf": {
                        "id": "wf",
                        "schema_version": "broken",
                        "nodes": [],
                    }
                },
            })

    def test_unsafe_workflow_id_fails_closed(self):
        with self.assertRaises(StateSchemaError):
            migrate_state({
                "version": CURRENT_STATE_VERSION,
                "workflows": {
                    "../escape": {
                        "id": "../escape",
                        "nodes": [],
                    }
                },
            })

    def test_workflow_identity_mismatch_fails_closed(self):
        with self.assertRaises(StateSchemaError):
            migrate_state({
                "version": CURRENT_STATE_VERSION,
                "workflows": {
                    "wf-key": {
                        "id": "wf-other",
                        "nodes": [],
                    }
                },
            })

    def test_workflow_identity_is_repaired_when_legacy_id_is_missing(self):
        migrated = migrate_state({
            "version": 1,
            "workflows": {
                "wf-key": {
                    "nodes": [],
                }
            },
        })
        self.assertEqual(migrated["workflows"]["wf-key"]["id"], "wf-key")

    def test_malformed_workflow_control_field_fails_closed(self):
        with self.assertRaises(StateSchemaError):
            migrate_state({
                "version": CURRENT_STATE_VERSION,
                "workflows": {
                    "wf": {
                        "id": "wf",
                        "max_parallel": "not-an-int",
                        "nodes": [],
                    }
                },
            })

    def test_execution_envelope_identity_is_schema_validated(self):
        base = {
            "id": "wf",
            "execution_id": "a" * 64,
            "intent_fingerprint": "b" * 64,
            "input_digest": "c" * 64,
            "external_workflow_id": "external-wf",
            "parent_execution_id": "parent-1",
            "external_domain": "publisher",
            "external_operation": "chapter.produce",
            "idempotency_key": "evt-1",
            "external_attempt": 2,
            "nodes": [],
        }
        migrated = migrate_state({
            "version": CURRENT_STATE_VERSION,
            "workflows": {"wf": base},
        })
        self.assertEqual(
            migrated["workflows"]["wf"]["execution_id"],
            "a" * 64,
        )
        bad = dict(base)
        bad["intent_fingerprint"] = "not-hex"
        with self.assertRaises(StateSchemaError):
            migrate_state({"version": CURRENT_STATE_VERSION, "workflows": {"wf": bad}})

    def test_invalid_workflow_live_type_fails_closed(self):
        with self.assertRaises(StateSchemaError):
            migrate_state({
                "version": CURRENT_STATE_VERSION,
                "workflows": {
                    "wf": {
                        "id": "wf",
                        "live": "false",
                        "nodes": [],
                    }
                },
            })

    def test_invalid_runtime_container_fails_closed(self):
        with self.assertRaises(StateSchemaError):
            migrate_state({
                "version": CURRENT_STATE_VERSION,
                "workflows": {
                    "wf": {
                        "id": "wf",
                        "executions": [],
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
                        "schema_version": 7,
                        "nodes": [],
                    }
                },
            })

    def test_duplicate_node_id_fails_closed(self):
        with self.assertRaises(StateSchemaError):
            migrate_state({
                "version": CURRENT_STATE_VERSION,
                "workflows": {
                    "wf": {
                        "id": "wf",
                        "nodes": [
                            {"id": "n01", "capability": "execute", "tool": "noop"},
                            {"id": "n01", "capability": "validate", "tool": "noop"},
                        ],
                    }
                },
            })

    def test_unknown_dependency_fails_closed(self):
        with self.assertRaises(StateSchemaError):
            migrate_state({
                "version": CURRENT_STATE_VERSION,
                "workflows": {
                    "wf": {
                        "id": "wf",
                        "nodes": [{
                            "id": "n01",
                            "capability": "execute",
                            "tool": "noop",
                            "depends_on": ["missing"],
                        }],
                    }
                },
            })

    def test_self_dependency_fails_closed(self):
        with self.assertRaises(StateSchemaError):
            migrate_state({
                "version": CURRENT_STATE_VERSION,
                "workflows": {
                    "wf": {
                        "id": "wf",
                        "nodes": [{
                            "id": "n01",
                            "capability": "execute",
                            "tool": "noop",
                            "depends_on": ["n01"],
                        }],
                    }
                },
            })

    def test_oversized_goal_fails_closed(self):
        with self.assertRaises(StateSchemaError):
            migrate_state({
                "version": CURRENT_STATE_VERSION,
                "workflows": {
                    "wf": {
                        "id": "wf",
                        "goal": "x" * 4001,
                        "nodes": [],
                    }
                },
            })

    def test_large_runtime_context_does_not_count_as_intent(self):
        migrated = migrate_state({
            "version": CURRENT_STATE_VERSION,
            "workflows": {
                "wf": {
                    "id": "wf",
                    "goal": "x",
                    "nodes": [{
                        "id": "n01",
                        "capability": "execute",
                        "tool": "noop",
                        "input": {
                            "instruction": "small",
                            "context": {"x": "y" * (40 * 1024)},
                        },
                    }],
                }
            },
        })
        self.assertEqual(
            migrated["workflows"]["wf"]["nodes"][0]["input"]["instruction"],
            "small",
        )

    def test_oversized_node_fails_closed(self):
        with self.assertRaises(StateSchemaError):
            migrate_state({
                "version": CURRENT_STATE_VERSION,
                "workflows": {
                    "wf": {
                        "id": "wf",
                        "goal": "x",
                        "nodes": [{
                            "id": "n01",
                            "capability": "execute",
                            "tool": "noop",
                            "input": {"instruction": "x" * (32 * 1024)},
                        }],
                    }
                },
            })

    def test_replan_count_and_retry_budget_are_bounded(self):
        with self.assertRaises(StateSchemaError):
            migrate_state({
                "version": CURRENT_STATE_VERSION,
                "workflows": {
                    "wf": {
                        "id": "wf",
                        "replan_count": 3,
                        "nodes": [],
                    }
                },
            })
        with self.assertRaises(StateSchemaError):
            migrate_state({
                "version": CURRENT_STATE_VERSION,
                "workflows": {
                    "wf": {
                        "id": "wf",
                        "nodes": [{
                            "id": "n01",
                            "capability": "execute",
                            "tool": "noop",
                            "max_retries": 9,
                        }],
                    }
                },
            })

    def test_malformed_node_state_fails_closed(self):
        with self.assertRaises(StateSchemaError):
            migrate_state({
                "version": CURRENT_STATE_VERSION,
                "workflows": {
                    "wf": {
                        "id": "wf",
                        "nodes": [{
                            "id": "n01",
                            "capability": "execute",
                            "tool": "noop",
                            "input": [],
                        }],
                    }
                },
            })
        with self.assertRaises(StateSchemaError):
            migrate_state({
                "version": CURRENT_STATE_VERSION,
                "workflows": {
                    "wf": {
                        "id": "wf",
                        "nodes": [{
                            "id": "n01",
                            "capability": "execute",
                            "tool": "noop",
                            "status": "unknown",
                        }],
                    }
                },
            })

    def test_malformed_workflow_fails_closed(self):
        with self.assertRaises(StateSchemaError):
            migrate_state({
                "version": 2,
                "workflows": {"wf": "not-an-object"},
            })

    def test_private_input_ref_requires_execution_identity_and_digest(self):
        state = {
            "version": CURRENT_STATE_VERSION,
            "workflows": {
                "wf-private": {
                    "id": "wf-private",
                    "schema_version": 6,
                    "goal": "private",
                    "status": "ready",
                    "private_input_ref": "a" * 64,
                    "nodes": [],
                }
            },
        }
        with self.assertRaisesRegex(
            StateSchemaError,
            "private_input_ref requires execution_id and input_digest",
        ):
            migrate_state(state)



if __name__ == "__main__":
    unittest.main()
