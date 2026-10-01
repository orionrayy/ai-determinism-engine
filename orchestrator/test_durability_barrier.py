import os
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from durability_barrier import DurabilityBarrierError, commit_side_effect_start


ROOT = Path(__file__).resolve().parent.parent


class DurabilityBarrierTests(unittest.TestCase):
    def test_local_run_is_noop(self):
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "false"}, clear=False):
            self.assertFalse(
                commit_side_effect_start(ROOT, execution_id="a" * 64)
            )

    def test_github_actions_requires_explicit_barrier_enablement(self):
        with patch.dict(
            os.environ,
            {"GITHUB_ACTIONS": "true", "ORCHESTRATOR_DURABILITY_BARRIER": "false"},
            clear=False,
        ):
            with self.assertRaises(DurabilityBarrierError):
                commit_side_effect_start(ROOT, execution_id="b" * 64)

    def test_remote_head_drift_fails_before_git_add(self):
        responses = iter([
            subprocess.CompletedProcess([], 0, "", ""),
            subprocess.CompletedProcess([], 0, "", ""),
            subprocess.CompletedProcess([], 0, "local\n", ""),
            subprocess.CompletedProcess([], 0, "remote\n", ""),
        ])
        with patch.dict(
            os.environ,
            {"GITHUB_ACTIONS": "true", "ORCHESTRATOR_DURABILITY_BARRIER": "true"},
            clear=False,
        ), patch("durability_barrier.subprocess.run", side_effect=lambda *args, **kwargs: next(responses)) as run:
            with self.assertRaisesRegex(DurabilityBarrierError, "main changed"):
                commit_side_effect_start(ROOT, execution_id="c" * 64)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertFalse(any(command[1:3] == ["add", "--"] for command in commands))

    def test_success_commits_and_pushes_main(self):
        responses = iter([
            subprocess.CompletedProcess([], 0, "", ""),
            subprocess.CompletedProcess([], 0, "", ""),
            subprocess.CompletedProcess([], 0, "same\n", ""),
            subprocess.CompletedProcess([], 0, "same\n", ""),
            subprocess.CompletedProcess([], 0, "", ""),
            subprocess.CompletedProcess([], 1, "", ""),
            subprocess.CompletedProcess([], 0, "", ""),
            subprocess.CompletedProcess([], 0, "", ""),
        ])
        with patch.dict(
            os.environ,
            {"GITHUB_ACTIONS": "true", "ORCHESTRATOR_DURABILITY_BARRIER": "true"},
            clear=False,
        ), patch("durability_barrier.subprocess.run", side_effect=lambda *args, **kwargs: next(responses)) as run:
            self.assertTrue(
                commit_side_effect_start(ROOT, execution_id="d" * 64)
            )
        commands = [call.args[0] for call in run.call_args_list]
        self.assertIn(
            ["git", "push", "origin", "HEAD:main"],
            commands,
        )
        self.assertIn(
            ["git", "commit", "-m", "chore(orchestrator): persist execution start " + "d" * 64],
            commands,
        )


if __name__ == "__main__":
    unittest.main()
