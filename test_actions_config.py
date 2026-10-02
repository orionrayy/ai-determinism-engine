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
        cls.federation = (ROOT / '.github' / 'workflows' / 'orchestrator-agent-federation.yml').read_text()
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


    def test_federation_has_unique_artifact_per_matrix_task(self):
        self.assertIn("matrix.task.task_id", self.federation)
        self.assertIn("agent-result-${{ matrix.task.task_id }}", self.federation)
        self.assertIn("merge-multiple: false", self.federation)
        self.assertIn("fail-fast: false", self.federation)
        self.assertIn("max-parallel: 4", self.federation)
    def test_federation_uses_fixed_concurrency_slots(self):
        self.assertIn("agent-federation-", self.federation)
        self.assertIn("github.event.client_payload.slot", self.federation)
        self.assertIn("cancel-in-progress: false", self.federation)
        self.assertIn("queue: max", self.federation)

    def test_federated_matrix_is_bounded_and_read_only(self):
        self.assertIn("types: [orchestrator.federate]", self.federation)
        self.assertIn("max-parallel: 4", self.federation)
        self.assertIn("fail-fast: false", self.federation)
        self.assertIn("contents: read", self.federation)
        self.assertIn("agent-result-${{ matrix.task.task_id }}", self.federation)
        self.assertIn("retention-days: 1", self.federation)
        self.assertIn("actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02", self.federation)
        self.assertIn("actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093", self.federation)

    def test_federated_aggregator_is_the_only_write_permission(self):
        self.assertIn("aggregate:", self.federation)
        self.assertIn("contents: write", self.federation)
        self.assertIn("orchestrator.federation.completed", self.federation)

    def test_supervisor_exposes_opt_in_federation_control(self):
        self.assertIn("federate_safe_agents:", self.orchestrator)
        self.assertIn("ORCHESTRATOR_FEDERATION_ENABLED", self.orchestrator)
        self.assertIn("ORCHESTRATOR_FEDERATION_ARTIFACT_ID", self.orchestrator)


    def test_supervisor_handles_federation_completion_dispatch(self):
        self.assertIn("orchestrator.federation.completed", self.orchestrator)
        self.assertIn("ORCHESTRATOR_FEDERATION_ARTIFACT_ID", self.orchestrator)
        self.assertIn("ORCHESTRATOR_FEDERATION_ARTIFACT_DIGEST", self.orchestrator)
        self.assertIn("actions: read", self.orchestrator)

    def test_committed_orchestrator_state_matches_current_schema(self):
        from orchestrator.state_schema import CURRENT_STATE_VERSION
        self.assertEqual(self.state.get('version'), CURRENT_STATE_VERSION)
        self.assertIsInstance(self.state.get('workflows'), dict)

    def test_pending_runs_use_documented_bounded_queue(self):
        self.assertIn('queue: max', self.orchestrator)
        self.assertIn('unexpected key "queue" for "concurrency" section', (ROOT / '.github' / 'actionlint.yaml').read_text())

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
        self.assertIn("github.run_attempt", self.orchestrator)
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
        self.assertIn('workflow_run.run_attempt', self.continuation)
        self.assertIn("item.get('github_run_id')", self.continuation)
        self.assertIn("item.get('github_run_attempt')", self.continuation)
        self.assertIn("orchestrator-continuation-", self.continuation)

    def test_ci_watches_orchestrator_workflow(self):
        self.assertIn('orchestrator.yml', self.tests)

    def test_scheduled_recovery_only_dispatches_per_workflow(self):
        self.assertIn("schedule-recovery:", self.orchestrator)
        self.assertIn("if: github.event_name == 'schedule'", self.orchestrator)
        self.assertIn("github.event_name != 'schedule'", self.orchestrator)
        self.assertIn('"workflow_id": workflow_id', self.orchestrator)
        self.assertIn('"orchestrator.continue"', self.orchestrator)


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