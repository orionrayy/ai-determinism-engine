from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scheduled_recovery import (
    barrier_failed,
    recovery_candidate,
    recovery_event_id,
    stale_running,
    dispatch_continuation,
)


class ScheduledRecoveryRuntimeTests(unittest.TestCase):
    def test_stale_running_workflow_is_candidate(self):
        now = datetime.now(timezone.utc)
        workflow = {
            "id": "wf",
            "status": "running",
            "updated_at": (now - timedelta(seconds=301)).isoformat(),
        }
        self.assertTrue(stale_running(workflow, now=now))
        self.assertTrue(recovery_candidate(workflow, now=now))

    def test_fresh_running_workflow_is_not_candidate(self):
        now = datetime.now(timezone.utc)
        workflow = {
            "id": "wf",
            "status": "running",
            "updated_at": now.isoformat(),
        }
        self.assertFalse(recovery_candidate(workflow, now=now))

    def test_barrier_failed_detects_execution_record(self):
        workflow = {
            "id": "wf",
            "nodes": [{"id": "n1", "status": "failed"}],
            "executions": {
                __import__("hashlib").sha256(b"wf:n1").hexdigest(): {
                    "status": "barrier_failed"
                }
            },
        }
        self.assertTrue(barrier_failed(workflow))

    def test_recovery_event_id_is_stable_for_same_revision(self):
        workflow = {"id": "wf", "updated_at": "2026-10-05T00:00:00+00:00"}
        self.assertEqual(
            recovery_event_id(workflow),
            "scheduled-recovery:wf:2026-10-05T00:00:00+00:00",
        )

    def test_dispatch_uses_repository_dispatch_and_exact_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            captured = {}
            def fake_run(args, **kwargs):
                captured["args"] = args
                captured["payload"] = json.loads(kwargs["input"])
                return None
            with patch.object(__import__("scheduled_recovery").subprocess, "run", side_effect=fake_run):
                dispatch_continuation(
                    repository="orionrayy/ai-determinism-engine",
                    workflow_id="wf-1",
                    event_id="scheduled-recovery:wf-1:r1",
                    schedule_run_id="99",
                )
            self.assertEqual(captured["args"][-2:], ["--input", "-"])
            self.assertEqual(
                captured["payload"]["client_payload"]["workflow_id"],
                "wf-1",
            )


if __name__ == "__main__":
    unittest.main()
