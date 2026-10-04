from __future__ import annotations

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
            with patch.dict(self.__class__.env or {}, {}, clear=False):
                pass
