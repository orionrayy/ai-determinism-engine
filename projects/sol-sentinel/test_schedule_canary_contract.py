from pathlib import Path
import unittest


class ScheduleCanaryContractTests(unittest.TestCase):
    def setUp(self):
        self.workflow = (
            Path(__file__).resolve().parents[2]
            / ".github"
            / "workflows"
            / "sol-sentinel-schedule-canary.yml"
        ).read_text(encoding="utf-8")

    def test_schedule_only_and_independent_cadence(self):
        self.assertIn('cron: "0,30 * * * *"', self.workflow)
        self.assertNotIn("workflow_dispatch:", self.workflow)
        self.assertNotIn("pull_request:", self.workflow)
        self.assertNotIn("push:", self.workflow)

    def test_read_only_minimum_permissions_and_bounded_runtime(self):
        self.assertIn("permissions:\n  actions: read", self.workflow)
        self.assertIn("timeout-minutes: 2", self.workflow)
        self.assertNotIn("contents: write", self.workflow)
        self.assertNotIn("actions: write", self.workflow)

    def test_logs_schedule_identity_and_provenance(self):
        for field in (
            '"workflow_id": workflow.get("id")',
            '"event": os.environ["EVENT_NAME"]',
            '"branch": os.environ["REF_NAME"]',
            '"head_sha": os.environ["HEAD_SHA"]',
            '"run_id": int(os.environ["RUN_ID"])',
            '"run_attempt": int(os.environ["RUN_ATTEMPT"])',
            '"observed_at_utc":',
            "SOL_SENTINEL_SCHEDULER_CANARY=",
        ):
            self.assertIn(field, self.workflow)

    def test_never_invokes_trading_or_accesses_paper_state(self):
        forbidden = (
            "forward_paper.py",
            "paper_engine.py",
            "sol-sentinel-paper-state",
            "secrets.",
            "trade_api",
            "withdraw",
            "wallet",
        )
        lowered = self.workflow.lower()
        for term in forbidden:
            self.assertNotIn(term.lower(), lowered)
        self.assertIn('"paper_state_accessed": False', self.workflow)
        self.assertIn('"trading_engine_invoked": False', self.workflow)
        self.assertIn('"exchange_credentials_accessed": False', self.workflow)


if __name__ == "__main__":
    unittest.main(verbosity=2)
