import json
import re
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
        cls.private_input_deploy = (ROOT / '.github' / 'workflows' / 'private-input-deploy.yml').read_text()
        cls.private_input_tests = (ROOT / '.github' / 'workflows' / 'private-input-tests.yml').read_text()
        cls.state = json.loads((ROOT / '.orchestrator' / 'state.json').read_text())

    def test_free_only_workflow_does_not_expose_paid_adapter_secrets(self):
        for secret in (
            "OPENAI_API_KEY",
            "FIRECRAWL_API_KEY",
            "ORCHESTRATOR_WEBHOOK_SECRET",
        ):
            self.assertNotIn(secret, self.orchestrator)

    def test_ci_avoids_duplicate_push_and_pull_request_runs(self):
        self.assertIn("branches:", self.tests)
        self.assertIn("- main", self.tests)
        self.assertIn("github.event_name == 'pull_request'", self.tests)
        self.assertIn("cancel-in-progress: ${{ github.event_name == 'pull_request' }}", self.tests)
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

    def test_state_uses_sharded_storage_format(self):
        self.assertEqual(self.state.get('storage_format'), 'sharded-v1')
        self.assertIn('from orchestrator.orchestrator import load_state', self.continuation)
        self.assertIn('from orchestrator.orchestrator import load_state', self.orchestrator)
        self.assertNotIn('json.load(open(\'.orchestrator/state.json\'))', self.continuation)
        self.assertNotIn('json.load(open(".orchestrator/state.json"))', self.orchestrator)

    def test_committed_orchestrator_state_matches_current_schema(self):
        from orchestrator.state_schema import CURRENT_STATE_VERSION
        self.assertEqual(self.state.get('version'), CURRENT_STATE_VERSION)
        self.assertIsInstance(self.state.get('workflows'), dict)

    def test_pending_runs_use_documented_bounded_queue(self):
        self.assertIn('queue: max', self.orchestrator)
        self.assertIn('unexpected key "queue" for "concurrency" section', (ROOT / '.github' / 'actionlint.yaml').read_text())

    def test_repository_ingress_concurrency_has_identity_fallbacks(self):
        self.assertIn('github.event.client_payload.workflow_id || github.event.client_payload.event_id || github.event.client_payload.idempotency_key', self.orchestrator)

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

    def test_source_issue_trigger_binds_event_action(self):
        self.assertIn('ISSUE_ACTION: ${{ github.event.action }}', self.orchestrator)

    def test_continuation_dispatch_carries_exact_run_attempt(self):
        self.assertIn('WORKFLOW_RUN_ID: ${{ github.event.workflow_run.id }}', self.continuation)
        self.assertIn('WORKFLOW_RUN_ATTEMPT: ${{ github.event.workflow_run.run_attempt }}', self.continuation)
        self.assertIn("'event_id':f\"continuation:", self.continuation)

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

    def test_scheduled_recovery_compacts_terminal_state(self):
        start = self.orchestrator.index('schedule-recovery:')
        recovery = self.orchestrator[start:]
        self.assertIn('compact_terminal_workflows', recovery)
        self.assertIn('ORCHESTRATOR_TERMINAL_COMPACTION_DAYS', recovery)
        self.assertIn('git add .orchestrator/workflows', recovery)
    def test_scheduled_recovery_only_dispatches_per_workflow(self):
        self.assertIn("schedule-recovery:", self.orchestrator)
        self.assertIn("if: github.event_name == 'schedule'", self.orchestrator)
        self.assertIn("github.event_name != 'schedule'", self.orchestrator)
        self.assertIn('"workflow_id": workflow_id', self.orchestrator)
        self.assertIn('"orchestrator.continue"', self.orchestrator)

    def test_scheduled_recovery_uses_canonical_loader(self):
        start = self.orchestrator.index('schedule-recovery:')
        recovery = self.orchestrator[start:]
        self.assertIn('from orchestrator.orchestrator import (', recovery)
        self.assertIn('load_state', recovery)
        self.assertNotIn('state_path = Path(".orchestrator/state.json")', recovery)


    def test_checkout_action_sha_is_consistent_across_all_workflows(self):
        expected = "d23441a48e516b6c34aea4fa41551a30e30af803"
        workflow_dir = ROOT / ".github" / "workflows"
        pins = []
        for path in sorted(workflow_dir.glob("*.y*ml")):
            content = path.read_text()
            for pin in re.findall(r"actions/checkout@([0-9a-f]{40})", content):
                pins.append((path.name, pin))
        self.assertTrue(pins, "no pinned actions/checkout reference found")
        for filename, pin in pins:
            self.assertEqual(
                pin,
                expected,
                f"{filename} contains a checkout SHA drift",
            )

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


    def test_orchestrator_ci_avoids_unused_node_and_worker_validation(self):
        self.assertIn("timeout-minutes: 10", self.tests)
        self.assertNotIn("actions/setup-node@", self.tests)
        self.assertNotIn("wrangler@4.146.0", self.tests)
        self.assertNotIn("node workers/private-input/test.mjs", self.tests)

    def test_private_input_boundary_is_wired(self):
        self.assertIn("ORCHESTRATOR_PRIVATE_INPUT_URL", self.orchestrator)
        self.assertIn("ORCHESTRATOR_PRIVATE_INPUT_SECRET", self.orchestrator)
        self.assertIn("ORCHESTRATOR_PRIVATE_INPUT_REF", self.orchestrator)
        self.assertIn("ORCHESTRATOR_IDEMPOTENCY_KEY", self.orchestrator)
        self.assertIn("private_input_unavailable", Path("gateway.py").read_text())
        self.assertIn("workers/private-input/**", self.private_input_tests)
        self.assertIn("node workers/private-input/test.mjs", self.private_input_tests)
        self.assertIn("wrangler@4.146.0", self.private_input_tests)
        self.assertNotIn("workers/private-input/**", self.tests)

    def test_gemini_free_model_allowlist_is_registry_pinned(self):
        registry = json.loads((ROOT / "orchestrator" / "tools.json").read_text())
        gemini = registry["gemini"]
        self.assertEqual(gemini["default_model"], "gemini-3.8-flash")
        self.assertIn("gemini-3.8-flash", gemini["free_models"])
        self.assertIn("gemini-3.7-flash", gemini["free_models"])
        self.assertIn("gemini-3.6-flash", gemini["free_models"])
        self.assertIn("gemini-3.5-flash", gemini["free_models"])
        self.assertIn("configured_gemini_model", Path("orchestrator/orchestrator.py").read_text())
        self.assertIn("not allowed by the free-only model registry", Path("orchestrator/llm_planner.py").read_text())

    def test_connector_single_flight_invariant_is_wired(self):
        bridge_runtime = (ROOT / "bridge_runtime.py").read_text()
        self.assertIn("_INFLIGHT", bridge_runtime)
        self.assertIn("IDEMPOTENCY_WAIT_TIMEOUT_SECONDS", bridge_runtime)
        self.assertIn("acquire_idempotency_slot", bridge_runtime)
        self.assertIn("release_idempotency_slot", bridge_runtime)

    def test_connector_bridge_free_hosting_is_not_upstream_certification(self):
        registry = json.loads((ROOT / "orchestrator" / "tools.json").read_text())
        bridge = registry["connector_bridge"]
        self.assertTrue(bridge["free_tier"])
        self.assertIn("cost_model", bridge)
        self.assertIn("upstream vendor costs depend on configuration", bridge["cost_model"])
        connector_bridge = Path("orchestrator/connector_bridge.py").read_text()
        self.assertIn("is not certified for free-only execution", connector_bridge)
        self.assertIn('"free_tier": bool(raw.get("free_tier", default_free_tier))', Path("orchestrator/connector_bridge.py").read_text())

    def test_canonical_memory_is_header_first_and_current(self):
        memory = (ROOT / "ORCHESTRATION_MEMORY.md").read_text()
        self.assertTrue(memory.startswith("# ORCHESTRATION MEMORY — Canonical Control-Plane Context"))
        self.assertIn("v43 connector upstream cost gate", memory)
        self.assertIn("v42 free Gemini model gate", memory)
        self.assertIn("v41 private structured input boundary", memory)
        self.assertIn("v44 — reconciliation cost closure", memory)
        self.assertIn("v65", memory)

    def test_artifact_verifier_uses_pinned_public_https_guard(self):
        orchestrator = (ROOT / "orchestrator" / "orchestrator.py").read_text()
        self.assertIn("def safe_public_https_json(", orchestrator)
        self.assertIn("MAX_ARTIFACT_RESPONSE_BYTES", orchestrator)
        self.assertIn("artifact URL redirects are disabled", orchestrator)
        self.assertIn("artifact URL resolves to a non-public IP", orchestrator)
        self.assertIn("artifact URL must use port 443", orchestrator)
        self.assertIn("result = safe_public_https_json(url, timeout=30)", orchestrator)

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