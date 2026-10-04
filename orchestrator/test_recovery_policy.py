import unittest
from datetime import datetime, timezone, timedelta

from recovery_policy import recovery_event_id, recovery_reasons, run_is_active


class RecoveryPolicyTests(unittest.TestCase):
    def test_empty_workflow_has_no_recovery_reason(self):
        self.assertEqual(
            recovery_reasons({}, now=datetime.now(timezone.utc)),
            [],
        )

    def test_waiting_approval_is_not_a_recovery_candidate(self):
        workflow = {
            "id": "wf-1",
            "status": "waiting_approval",
            "updated_at": "2026-10-04T00:00:00+00:00",
        }
        self.assertEqual(
            recovery_reasons(
                workflow,
                now=datetime(2026, 10, 4, 1, tzinfo=timezone.utc),
            ),
            [],
        )

    def test_active_github_run_suppresses_stale_recovery(self):
        workflow = {
            "id": "wf-1",
            "status": "running",
            "updated_at": "2026-10-04T00:00:00+00:00",
            "github_run_id": "123",
            "github_run_attempt": 1,
        }
        self.assertEqual(
            recovery_reasons(
                workflow,
                now=datetime(2026, 10, 4, 1, tzinfo=timezone.utc),
                active_run_status="in_progress",
            ),
            [],
        )

    def test_completed_github_run_allows_stale_recovery(self):
        workflow = {
            "id": "wf-1",
            "status": "running",
            "updated_at": "2026-10-04T00:00:00+00:00",
            "github_run_id": "123",
            "github_run_attempt": 1,
        }
        self.assertEqual(
            recovery_reasons(
                workflow,
                now=datetime(2026, 10, 4, 1, tzinfo=timezone.utc),
                active_run_status="completed",
            ),
            ["stale_running"],
        )

    def test_uncertain_effect_is_recoverable_without_waiting_for_staleness(self):
        workflow = {
            "id": "wf-1",
            "status": "failed",
            "nodes": [
                {
                    "id": "n1",
                    "status": "failed",
                    "tool": "connector_bridge",
                    "error": {"execution_uncertain": True},
                }
            ],
        }
        self.assertEqual(
            recovery_reasons(
                workflow,
                now=datetime.now(timezone.utc),
            ),
            ["uncertain_effect"],
        )

    def test_barrier_failure_is_recoverable(self):
        workflow = {
            "id": "wf-1",
            "status": "failed",
            "nodes": [
                {
                    "id": "n1",
                    "status": "failed",
                }
            ],
            "executions": {
                "abc": {"status": "barrier_failed"},
            },
        }
        # The policy consumes the normalized barrier-failed flag so the
        # scheduler does not need to know checkpoint key derivation.
        workflow["barrier_failed"] = True
        self.assertEqual(
            recovery_reasons(
                workflow,
                now=datetime.now(timezone.utc),
            ),
            ["durability_barrier"],
        )

    def test_recovery_event_id_is_stable_for_same_generation(self):
        workflow = {"id": "wf-1"}
        a = recovery_event_id(workflow, "stale_running", "17")
        b = recovery_event_id(workflow, "stale_running", "17")
        self.assertEqual(a, b)
        self.assertTrue(a.startswith("scheduled-recovery:"))

    def test_recovery_event_id_changes_when_generation_changes(self):
        workflow = {"id": "wf-1"}
        a = recovery_event_id(workflow, "stale_running", "17")
        b = recovery_event_id(workflow, "stale_running", "18")
        self.assertNotEqual(a, b)

    def test_run_is_active_only_for_queued_or_in_progress(self):
        self.assertTrue(run_is_active("queued"))
        self.assertTrue(run_is_active("in_progress"))
        self.assertFalse(run_is_active("completed"))
        self.assertFalse(run_is_active("failure"))
        self.assertFalse(run_is_active(None))


if __name__ == "__main__":
    unittest.main()
