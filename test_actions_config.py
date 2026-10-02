import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


class ActionsConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.orchestrator = (ROOT / '.github' / 'workflows' / 'orchestrator.yml').read_text()
        cls.continuation = (ROOT / '.github' / 'workflows' / 'orchestrator-continuation.yml').read_text()
        cls.approval = (ROOT / '.github' / 'workflows' / 'orchestrator-approval.yml').read_text()
        cls.tests = (ROOT / '.github' / 'workflows' / 'orchestrator-tests.yml').read_text()
        cls.bridge_deploy = (ROOT / '.github' / 'workflows' / 'bridge-deploy.yml').read_text()
        cls.state = json.loads((ROOT / '.orchestrator' / 'state.json').read_text())

    def test_approval_labels_use_dedicated_dispatcher(self):
        self.assertIn('types: [labeled]', self.approval)
        self.assertIn('orchestrator-approved', self.approval)
        self.assertIn('orchestrator-rejected', self.approval)
        self.assertIn('Authorize approval actor', self.approval)
        self.assertIn('workflow_id', self.approval)
        self.assertIn('orchestrator.continue', self.approval)
        self.assertNotIn('Authorize approval actor', self.orchestrator)


    def test_committed_orchestrator_state_matches_current_schema(self):
        from orchestrator.state_schema import CURRENT_STATE_VERSION
        self.assertEqual(self.state.get('version'), CURRENT_STATE_VERSION)
        self.assertIsInstance(self.state.get('workflows'), dict)

    def test_pending_runs_are_not_replaced(self):
        self.assertIn('queue: max', self.orchestrator)

    def test_orchestrator_concurrency_isolated_by_workflow_identity(self):
        self.assertIn(
            "github.event.client_payload.workflow_id",
            self.orchestrator,
        )
        self.assertIn(
            "github.event_name == 'schedule' && 'scheduled-recovery'",
            self.orchestrator,
        )
        self.assertIn("github.run_id", self.orchestrator)
        self.assertIn("queue: max", self.orchestrator)
        self.assertIn("cancel-in-progress: false", self.orchestrator)

    def test_continuation_carries_exact_workflow_identity(self):
        self.assertIn("WORKFLOW_ID", self.continuation)
        self.assertIn("'workflow_id':os.environ['WORKFLOW_ID']", self.continuation)
        self.assertIn("github.event.client_payload.workflow_id", self.orchestrator)


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


    def test_core_actions_are_pinned_to_node24_releases(self):
        expected = {
            "actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803",
            "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97",
            "actions/setup-node@820762786026740c76f36085b0efc47a31fe5020",
        }
        combined = "\n".join((self.orchestrator, self.continuation, self.tests, self.bridge_deploy))
        for ref in expected:
            self.assertIn(ref, combined)


    def test_bridge_deploy_uses_pinned_vercel_cli_and_node24(self):
        self.assertIn('node-version: "24"', self.bridge_deploy)
        self.assertNotIn('node-version: "20"', self.bridge_deploy)
        self.assertIn('vercel@59.19.1', self.bridge_deploy)
        self.assertNotIn('vercel@latest', self.bridge_deploy)
        self.assertEqual(self.bridge_deploy.count('vercel@59.19.1'), 3)


    def test_worker_enables_durability_barrier(self):
        self.assertIn('ORCHESTRATOR_DURABILITY_BARRIER: "true"', self.orchestrator)

    def test_state_persistence_retries_after_main_advances(self):
        self.assertIn("git fetch origin main", self.orchestrator)
        self.assertIn("git rebase origin/main", self.orchestrator)
        self.assertIn("for attempt in 1 2 3", self.orchestrator)
        self.assertIn("git push origin HEAD:main", self.orchestrator)
        self.assertIn("State persistence rebase conflict; refusing to overwrite concurrent main state.", self.orchestrator)
        self.assertIn("State persistence push failed after bounded retries.", self.orchestrator)

if __name__ == '__main__':
    unittest.main()