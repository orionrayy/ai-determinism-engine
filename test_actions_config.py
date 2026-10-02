import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


class ActionsConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.orchestrator = (ROOT / '.github' / 'workflows' / 'orchestrator.yml').read_text()
        cls.continuation = (ROOT / '.github' / 'workflows' / 'orchestrator-continuation.yml').read_text()
        cls.tests = (ROOT / '.github' / 'workflows' / 'orchestrator-tests.yml').read_text()
        cls.bridge_deploy = (ROOT / '.github' / 'workflows' / 'bridge-deploy.yml').read_text()

    def test_approval_labels_trigger_worker(self):
        self.assertIn('types: [opened, edited, labeled]', self.orchestrator)
        self.assertIn('orchestrator-approved|orchestrator-rejected', self.orchestrator)
        self.assertIn('Authorize approval actor', self.orchestrator)
        self.assertIn('github.actor', self.orchestrator)

    def test_approval_issue_prefix_is_allowed_by_job_filter(self):
        self.assertIn("startsWith(github.event.issue.title, '[ORCHESTRATOR]')", self.orchestrator)
        self.assertIn("startsWith(github.event.issue.title, '[ORCHESTRATOR APPROVAL]')", self.orchestrator)

    def test_approval_authority_excludes_triage_role(self):
        self.assertIn("admin|maintain|push) ;;", self.orchestrator)
        self.assertNotIn("admin|maintain|push|triage)", self.orchestrator)

    def test_approval_worker_requires_authenticated_label_event(self):
        self.assertIn('ORCHESTRATOR_APPROVAL_EVENT:', self.orchestrator)
        self.assertIn("github.event.action == 'labeled'", self.orchestrator)
        self.assertIn("github.event.label.name == 'orchestrator-approved'", self.orchestrator)
        self.assertIn("github.event.label.name == 'orchestrator-rejected'", self.orchestrator)

    def test_execution_envelope_contract_is_versioned(self):
        contract = json.loads(
            (ROOT / "contracts" / "execution-envelope.schema.json").read_text()
        )
        self.assertEqual(contract["properties"]["schema_version"]["const"], 1)
        self.assertIn("execution_id", contract["required"])
        self.assertIn("intent_fingerprint", contract["required"])
        self.assertEqual(contract["properties"]["requested_mode"]["enum"], ["dry-run", "live"])

    def test_execution_callback_contract_is_versioned(self):
        contract = json.loads(
            (ROOT / "contracts" / "execution-callback.schema.json").read_text()
        )
        self.assertEqual(contract["properties"]["schema_version"]["const"], 1)
        self.assertEqual(
            contract["properties"]["status"]["enum"],
            ["completed", "failed", "uncertain"],
        )
        self.assertIn("execution_id", contract["required"])

    def test_execution_budget_input_is_wired_to_worker(self):
        self.assertIn('max_steps:', self.orchestrator)
        self.assertIn('ORCHESTRATOR_MAX_EXECUTION_STEPS:', self.orchestrator)
        self.assertIn("inputs.max_steps || '96'", self.orchestrator)

    def test_pending_runs_are_not_replaced(self):
        self.assertIn('queue: max', self.orchestrator)

    def test_workflow_target_is_preferred_for_approval(self):
        self.assertIn('ORCHESTRATOR_TARGET_WORKFLOW_ID', self.orchestrator)
        self.assertIn('args=(--workflow-id "$ORCHESTRATOR_TARGET_WORKFLOW_ID" --step)', self.orchestrator)

    def test_source_issue_is_propagated(self):
        self.assertIn('ORCHESTRATOR_TRIGGER_ISSUE', self.orchestrator)
        self.assertIn('ORCHESTRATOR_GITHUB_RUN_ID', self.orchestrator)
        self.assertIn('ORCHESTRATOR_GITHUB_RUN_ATTEMPT', self.orchestrator)

    def test_continuation_binds_to_current_worker_run(self):
        self.assertIn('workflow_run.id', self.continuation)
        self.assertIn('workflow_run.run_attempt', self.continuation)
        self.assertIn("item.get('github_run_id')", self.continuation)
        self.assertIn("item.get('github_run_attempt')", self.continuation)
        self.assertIn("WORKFLOW_RUN_ID", self.continuation)
        self.assertIn('EVENT_ID', self.continuation)
        self.assertIn('continuation:', self.continuation)

    def test_continuation_has_single_flight_for_same_run_attempt(self):
        self.assertIn('orchestrator-continuation-', self.continuation)
        self.assertNotIn('queue: single', self.continuation)
        self.assertIn('cancel-in-progress: false', self.continuation)

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

if __name__ == '__main__':
    unittest.main()