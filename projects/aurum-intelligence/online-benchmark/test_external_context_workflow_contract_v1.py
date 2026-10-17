import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".github" / "workflows" / "aurum-online-dataset-benchmark.yml"
RUNNER = ROOT / "projects" / "aurum-intelligence" / "online-benchmark" / "run_external_context_benchmark_v1.py"
FRED_ADAPTER = ROOT / "projects" / "aurum-intelligence" / "online-benchmark" / "fred_alfred_adapter_v1.py"


class ExternalContextWorkflowContractTests(unittest.TestCase):
    def test_new_paths_trigger_the_existing_benchmark_workflow(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn('projects/aurum-intelligence/online-benchmark/**', workflow)
        self.assertIn('python -m unittest -v', workflow)

    def test_live_and_price_benchmark_gates_remain_no_trade(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn(">= 700000", workflow)
        self.assertIn("r['decision_output'] == 'NO_TRADE'", workflow)
        self.assertIn("live_signal_enabled'] is False", workflow)
        self.assertIn(">= 20000", workflow)

    def test_fred_credentials_are_runtime_optional_and_not_embedded_in_workflow(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        fred_source = FRED_ADAPTER.read_text(encoding="utf-8")
        self.assertNotIn("FRED_API_KEY=", workflow)
        self.assertIn('os.environ.get("FRED_API_KEY"', fred_source)
        self.assertIn("SKIPPED_CREDENTIAL_REQUIRED", fred_source)
        self.assertNotIn("test-secret-do-not-leak", fred_source)

    def test_real_context_runner_reports_not_met_without_claiming_scale_pass(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertTrue(RUNNER.exists(), "bounded external-context runner is missing")
        self.assertIn("run_external_context_benchmark_v1.py", workflow)
        runner = RUNNER.read_text(encoding="utf-8")
        self.assertIn('"NOT_MET"', runner)
        self.assertIn('"passed": bool(scale_met)', runner)
        self.assertIn('"decision_gate": "NO_TRADE"', runner)

    def test_report_separates_pipeline_scale_acquisition_and_source_completeness(self):
        runner = RUNNER.read_text(encoding="utf-8")
        self.assertIn('"software_execution_status": "PASS"', runner)
        self.assertIn('"status": "PASS" if scale["passed"] and source_details["gdelt_gkg"]["status"] == "PASS" else "PARTIAL"', runner)
        self.assertIn('"data_acceptance_status": scale["status"]', runner)
        self.assertIn('"source_acquisition_status": source_details["gdelt_gkg"]["status"]', runner)
        self.assertIn('"source_completeness_status": source_completeness_status', runner)
        self.assertIn('source_details["gdelt_gkg"]["archive_window_complete"]', runner)
        self.assertIn('"temporal_feature_coverage": temporal_feature_coverage', runner)
        self.assertIn('"status_semantics"', runner)
        self.assertIn('"PASS" if source_details["gdelt_gkg"]["archive_window_complete"]', runner)
        self.assertIn('"PARTIAL" if gdelt_coverage else "NOT_MET"', runner)


if __name__ == "__main__":
    unittest.main()
