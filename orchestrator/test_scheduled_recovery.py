import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from scheduled_recovery import (
    DEFAULT_STALE_SECONDS,
    ACTIVE_RUN_STATUSES,
    is_recovery_candidate,
    is_stale_running,
    recovery_event_id,
    recovery_generation,
    main,
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

    def test_recovery_due_false_does_not_override_fresh_running_workflow(self):
        workflow = {
            "id": "wf-2",
            "status": "running",
            "updated_at": self.now.isoformat(),
        }
        self.assertFalse(
            is_recovery_candidate(
                workflow,
                self.now,
                active_run_status="missing",
                stale_seconds=3600,
                recovery_due=False,
            )
        )

    def test_recovery_alarm_overrides_active_run_status_after_staleness(self):
        workflow = {
            "id": "wf-3",
            "status": "running",
            "updated_at": self.now.isoformat(),
        }
        self.assertTrue(
            is_recovery_candidate(
                workflow,
                self.now,
                active_run_status="in_progress",
                stale_seconds=3600,
                recovery_due=True,
            )
        )

    def test_control_plane_alarm_can_mark_running_workflow_due_before_timestamp_threshold(self):
        workflow = {
            "id": "wf-1",
            "status": "running",
            "updated_at": self.now.isoformat(),
        }
        self.assertTrue(
            is_recovery_candidate(
                workflow,
                self.now,
                active_run_status="missing",
                stale_seconds=3600,
                recovery_due=True,
            )
        )

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

    def test_missing_github_run_allows_recovery_but_unknown_status_fails_closed(self):
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
        self.assertFalse(
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

    def test_uncertain_non_reconcilable_tool_does_not_trigger_side_effect_recovery(self):
        workflow = {
            "id": "wf-uncertain",
            "status": "failed",
            "nodes": [{
                "id": "n1",
                "status": "failed",
                "tool": "openai",
                "error": {"execution_uncertain": True},
            }],
        }
        self.assertFalse(is_recovery_candidate(workflow, self.now))

    def test_active_run_suppression_also_covers_waiting_agents(self):
        workflow = {
            "id": "wf-agents",
            "status": "waiting_agents",
            "federation": {"status": "prepared"},
            "updated_at": "2026-10-04T00:00:00+00:00",
        }
        self.assertFalse(
            is_recovery_candidate(
                workflow,
                self.now,
                active_run_status="in_progress",
            )
        )

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


    def test_main_writes_failure_marker_and_defers_exit_decision(self):
        import builtins

        with tempfile.TemporaryDirectory() as tmp:
            marker = os.path.join(tmp, "recovery.json")
            real_open = builtins.open

            def open_side_effect(path, *args, **kwargs):
                if path == "/tmp/orchestrator_recovery_failures.json":
                    path = marker
                return real_open(path, *args, **kwargs)

            with patch("scheduled_recovery.run", return_value=(2, 1, ["wf-1: dispatch failed"])) as run_mock, patch(
                "builtins.open",
                side_effect=open_side_effect,
            ):
                self.assertEqual(main(), 0)
                run_mock.assert_called_once()

            with real_open(marker, encoding="utf-8") as handle:
                self.assertEqual(json.load(handle), ["wf-1: dispatch failed"])

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
