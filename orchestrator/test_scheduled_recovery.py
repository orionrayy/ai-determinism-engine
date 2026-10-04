import unittest
from datetime import datetime, timedelta, timezone

from scheduled_recovery import (
    DEFAULT_STALE_SECONDS,
    ACTIVE_RUN_STATUSES,
    is_recovery_candidate,
    is_stale_running,
    recovery_event_id,
    recovery_generation,
)


class ScheduledRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 4, 10, 0, tzinfo=timezone.utc)

    def test_waiting_approval_is_not_a_scheduled_candidate(self):
        workflow = {
            "id": "wf-approval",
            "status": "waiting_approval",
            "updated_at": "2026-10-04T00:00:00+00:00",
        }
        self.assertFalse(is_recovery_candidate(workflow, self.now))

    def test_running_workflow_becomes_stale_only_after_threshold(self):
        fresh = {
            "id": "wf-running",
            "status": "running",
            "updated_at": (
                self.now - timedelta(seconds=DEFAULT_STALE_SECONDS - 1)
            ).isoformat(),
        }
        stale = {
            **fresh,
            "updated_at": (
                self.now - timedelta(seconds=DEFAULT_STALE_SECONDS)
            ).isoformat(),
        }
        self.assertFalse(is_stale_running(fresh, self.now))
        self.assertTrue(is_stale_running(stale, self.now))

    def test_active_github_run_suppresses_recovery(self):
        workflow = {
            "id": "wf-active",
            "status": "running",
            "updated_at": "2026-10-04T00:00:00+00:00",
        }
        for status in ACTIVE_RUN_STATUSES:
            with self.subTest(status=status):
                self.assertFalse(
                    is_recovery_candidate(
                        workflow,
                        self.now,
                        active_run_status=status,
                    )
                )

    def test_missing_or_unknown_github_run_allows_recovery(self):
        workflow = {
            "id": "wf-missing-run",
            "status": "running",
            "updated_at": "2026-10-04T00:00:00+00:00",
        }
        self.assertTrue(
            is_recovery_candidate(
                workflow,
                self.now,
                active_run_status=None,
            )
        )
        self.assertTrue(
            is_recovery_candidate(
                workflow,
                self.now,
                active_run_status="unknown",
            )
        )

    def test_recovery_event_id_is_stable_when_only_updated_at_changes(self):
        first = {
            "id": "wf-stable",
            "status": "failed",
            "updated_at": "2026-10-04T08:00:00+00:00",
            "failed_node": "n1",
            "nodes": [{"id": "n1", "status": "failed", "error": {"execution_uncertain": True}}],
        }
        second = {
            **first,
            "updated_at": "2026-10-04T09:00:00+00:00",
        }
        self.assertEqual(recovery_event_id(first), recovery_event_id(second))
        self.assertEqual(recovery_generation(first), recovery_generation(second))

    def test_failed_durability_barrier_is_recoverable(self):
        workflow = {
            "id": "wf-barrier",
            "status": "failed",
            "nodes": [{"id": "n1", "status": "failed"}],
            "executions": {
                "placeholder": {"status": "barrier_failed"},
            },
        }
        execution_id = __import__("hashlib").sha256(
            b"wf-barrier:n1"
        ).hexdigest()
        workflow["executions"] = {
            execution_id: {"status": "barrier_failed"},
        }
        self.assertTrue(is_recovery_candidate(workflow, self.now))

    def test_uncertain_non_connector_tool_does_not_trigger_side_effect_recovery(self):
        workflow = {
            "id": "wf-uncertain",
            "status": "failed",
            "nodes": [{
                "id": "n1",
                "status": "failed",
                "tool": "github",
                "error": {"execution_uncertain": True},
            }],
        }
        self.assertFalse(is_recovery_candidate(workflow, self.now))

    def test_recovery_event_id_changes_when_plan_or_effect_intent_changes(self):
        base = {
            "id": "wf-effect",
            "status": "failed",
            "plan_fingerprint": "plan-a",
            "nodes": [{
                "id": "n1",
                "status": "failed",
                "tool": "connector_bridge",
                "error": {"execution_uncertain": True},
            }],
            "executions": {
                "effect-1": {
                    "status": "inflight",
                    "effect_id": "effect-1",
                    "effect_semantic_digest": "digest-a",
                },
            },
        }
        changed_plan = {
            **base,
            "plan_fingerprint": "plan-b",
        }
        changed_effect = {
            **base,
            "executions": {
                "effect-1": {
                    "status": "inflight",
                    "effect_id": "effect-1",
                    "effect_semantic_digest": "digest-b",
                },
            },
        }
        self.assertNotEqual(recovery_event_id(base), recovery_event_id(changed_plan))
        self.assertNotEqual(recovery_event_id(base), recovery_event_id(changed_effect))

    def test_recovery_event_id_changes_when_recovery_state_changes(self):
        first = {
            "id": "wf-change",
            "status": "failed",
            "failed_node": "n1",
            "nodes": [{"id": "n1", "status": "failed", "error": {"execution_uncertain": True}}],
        }
        second = {
            **first,
            "nodes": [{"id": "n1", "status": "failed", "error": {"execution_uncertain": False}}],
        }
        self.assertNotEqual(recovery_event_id(first), recovery_event_id(second))


if __name__ == "__main__":
    unittest.main()
