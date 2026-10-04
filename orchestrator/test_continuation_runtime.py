from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from continuation_runtime import decide_continuation, write_github_output


class ContinuationRuntimeTests(unittest.TestCase):
    def test_running_workflow_is_continuable(self):
        state = {
            "workflows": {
                "wf": {
                    "id": "wf",
                    "status": "running",
                    "github_run_id": 123,
                    "github_run_attempt": 2,
                    "trigger_issue": 77,
                }
            }
        }
        self.assertEqual(
            decide_continuation(state, run_id="123", run_attempt="2"),
            {
                "workflow_id": "wf",
                "continue": "true",
                "trigger_issue": "77",
                "status": "running",
            },
        )

    def test_nonmatching_run_is_missing_and_not_continuable(self):
        state = {
            "workflows": {
                "wf": {
                    "id": "wf",
                    "status": "running",
                    "github_run_id": 123,
                    "github_run_attempt": 2,
                }
            }
        }
        self.assertEqual(
            decide_continuation(state, run_id="999", run_attempt="2"),
            {
                "workflow_id": "",
                "continue": "false",
                "trigger_issue": "",
                "status": "missing",
            },
        )

    def test_multiline_github_output_uses_delimiter_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            with patch.dict(os.environ, {"GITHUB_OUTPUT": str(output)}, clear=False):
                write_github_output({"goal": "line 1\nline 2", "status": "running"})
            raw = output.read_text(encoding="utf-8")
            self.assertIn("goal<<ORCHESTRATOR_OUTPUT_", raw)
            self.assertIn("line 1\nline 2\n", raw)
            self.assertIn("status=running\n", raw)


if __name__ == "__main__":
    unittest.main()
