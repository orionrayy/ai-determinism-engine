import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


class ActionsConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.orchestrator = (ROOT / '.github' / 'workflows' / 'orchestrator.yml').read_text()
        cls.continuation = (ROOT / '.github' / 'workflows' / 'orchestrator-continuation.yml').read_text()
        cls.tests = (ROOT / '.github' / 'workflows' / 'orchestrator-tests.yml').read_text()

    def test_approval_labels_trigger_worker(self):
        self.assertIn('types: [opened, edited, labeled]', self.orchestrator)
        self.assertIn('orchestrator-approved|orchestrator-rejected', self.orchestrator)
        self.assertIn('Authorize approval actor', self.orchestrator)
        self.assertIn('github.actor', self.orchestrator)

    def test_pending_runs_are_not_replaced(self):
        self.assertIn('queue: max', self.orchestrator)

    def test_workflow_target_is_preferred_for_approval(self):
        self.assertIn('ORCHESTRATOR_TARGET_WORKFLOW_ID', self.orchestrator)
        self.assertIn('args=(--workflow-id "$ORCHESTRATOR_TARGET_WORKFLOW_ID" --step)', self.orchestrator)

    def test_source_issue_is_propagated(self):
        self.assertIn('ORCHESTRATOR_TRIGGER_ISSUE', self.orchestrator)
        self.assertIn('ORCHESTRATOR_GITHUB_RUN_ID', self.orchestrator)

    def test_continuation_binds_to_originating_run(self):
        self.assertIn('workflow_run.id', self.continuation)
        self.assertIn("item.get('github_run_id')", self.continuation)

    def test_ci_watches_orchestrator_workflow(self):
        self.assertIn('orchestrator.yml', self.tests)


if __name__ == '__main__':
    unittest.main()