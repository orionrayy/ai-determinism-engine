from pathlib import Path
import unittest


class ReadOnlyStateAuditWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.workflow = (
            Path(__file__).resolve().parents[2]
            / ".github" / "workflows" / "sol-sentinel-tests.yml"
        ).read_text(encoding="utf-8")

    def test_state_audit_is_explicit_manual_opt_in_or_historical_pr_185(self):
        self.assertIn("audit_active_paper_state:", self.workflow)
        self.assertIn("default: false", self.workflow)
        self.assertIn("inputs.audit_active_paper_state == true", self.workflow)
        self.assertIn("github.event.pull_request.number == 185", self.workflow)

    def test_state_checkout_has_no_persisted_credentials(self):
        self.assertIn("ref: sol-sentinel-paper-state", self.workflow)
        self.assertIn("persist-credentials: false", self.workflow)

    def test_read_only_sqlite_contract(self):
        self.assertIn("mode=ro", self.workflow)
        self.assertIn("PRAGMA query_only=ON", self.workflow)
        self.assertIn("PRAGMA integrity_check", self.workflow)
        self.assertIn("PRAGMA foreign_key_check", self.workflow)
        self.assertIn("sqlite_git_blob=", self.workflow)

    def test_workflow_permission_is_read_only(self):
        self.assertIn("permissions:\n  contents: read", self.workflow)
        self.assertNotIn("contents: write", self.workflow)
        self.assertNotIn("pull-requests: write", self.workflow)


if __name__ == "__main__":
    unittest.main(verbosity=2)
