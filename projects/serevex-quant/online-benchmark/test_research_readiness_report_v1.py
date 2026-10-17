"""Tests for evidence-gated readiness synthesis."""
import json
import tempfile
import unittest
from pathlib import Path

from run_research_readiness_v1 import build_report, write_report


class ReadinessReportTests(unittest.TestCase):
    def make_fixture(self, directory):
        root = Path(directory)
        (root / "large-scale").mkdir(parents=True)
        (root / "external-context").mkdir(parents=True)
        inputs = {
            "online_dataset_benchmark.json": {"status": "PASS", "source": {"sha256": "a" * 64}, "dataset": {"rows": 35954}},
            "tick_bidask_benchmark.json": {
                "status": "PASS", "decision_output": "NO_TRADE",
                "source_qualification_gates": {
                    key: {"research_qualification_status": "REFERENCE_ONLY",
                          "execution_qualification_status": "BLOCKED", "source_promotion_allowed": False}
                    for key in ("primary_price", "secondary_price", "tick_bidask")
                },
            },
            "large-scale/large_scale_bidask_benchmark.json": {
                "status": "PASS", "decision_output": "NO_TRADE", "validated_model": False,
                "deployment_calibrated": False,
                "dataset": {"paired_valid_bid_ask_candles": 1313280, "months_requested": 30},
                "scenario_contract": {"selected_unique_anchor_timestamps": 20000,
                    "eligible_real_contiguous_windows_with_valid_atr": 875611, "tp_sl_path_scenarios": 120000},
                "prequential_probability_evaluation": {
                    "aggregate": {"brier_skill_vs_global_matured_label_baseline": -0.0006}},
            },
            "external-context/external_context_benchmark.json": {
                "status": "PASS", "decision_gate": "NO_TRADE", "validated": False,
                "deployment_calibrated": False, "scale_gate": {"observations": 141020, "status": "PASS"},
                "source_completeness_status": "PARTIAL",
            },
        }
        for relative, value in inputs.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value), encoding="utf-8")
        contracts = root / "calibration-contracts.json"
        contracts.write_text(json.dumps({"status": "PASS", "scenario_count": 240, "passed": 240,
            "failed": 0, "synthetic_only": True}), encoding="utf-8")
        return root, contracts

    def test_green_pipeline_does_not_pass_data_stats_economics_or_ops(self):
        with tempfile.TemporaryDirectory() as temp:
            root, contracts = self.make_fixture(temp)
            report = build_report(root, contracts, "https://github.com/example/repo/actions/runs/123")
            self.assertEqual(report["checks"]["pipeline_status"]["status"], "PASS")
            self.assertEqual(report["checks"]["data_status"]["status"], "REFERENCE_ONLY")
            self.assertEqual(report["checks"]["statistical_status"]["status"], "INCONCLUSIVE")
            self.assertEqual(report["checks"]["economic_status"]["status"], "INCONCLUSIVE")
            self.assertEqual(report["checks"]["operations_status"]["status"], "NOT_RUN")
            self.assertEqual(report["status"], "BLOCKED")
            self.assertEqual(report["decision_output"], "NO_TRADE")
            self.assertFalse(report["live_signal_enabled"])
            self.assertFalse(report["live_order_execution_enabled"])
            self.assertEqual(report["observed_data_summary"]["large_scale_bid_ask_rows"], 1313280)
            self.assertEqual(report["observed_data_summary"]["external_context_completeness"], "PARTIAL")

    def test_missing_benchmark_input_fails_pipeline_but_stays_no_trade(self):
        with tempfile.TemporaryDirectory() as temp:
            root, contracts = self.make_fixture(temp)
            (root / "online_dataset_benchmark.json").unlink()
            report = build_report(root, contracts, "https://github.com/example/repo/actions/runs/124")
            self.assertEqual(report["checks"]["pipeline_status"]["status"], "FAIL")
            self.assertEqual(report["status"], "BLOCKED")
            self.assertEqual(report["decision_output"], "NO_TRADE")
            self.assertTrue(any(x.startswith("missing_input:") for x in report["input_errors"]))

    def test_review_eligibility_does_not_auto_promote_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root, contracts = self.make_fixture(temp)
            path = root / "tick_bidask_benchmark.json"
            tick = json.loads(path.read_text(encoding="utf-8"))
            for item in tick["source_qualification_gates"].values():
                item["research_qualification_status"] = "QUALIFIED_FOR_RESEARCH_REVIEW"
                item["manual_promotion_review_required"] = True
                item["source_promotion_allowed"] = False
            path.write_text(json.dumps(tick), encoding="utf-8")
            report = build_report(root, contracts, "https://github.com/example/repo/actions/runs/125")
            self.assertEqual(report["checks"]["data_status"]["status"], "REFERENCE_ONLY")
            self.assertEqual(report["promotion_status"], "BLOCKED")

    def test_machine_and_human_reports_are_written(self):
        with tempfile.TemporaryDirectory() as temp:
            root, contracts = self.make_fixture(temp)
            report = build_report(root, contracts, "https://github.com/example/repo/actions/runs/126")
            output = root / "readiness"
            write_report(report, output)
            machine = json.loads((output / "research_readiness.json").read_text(encoding="utf-8"))
            human = (output / "research_readiness.md").read_text(encoding="utf-8")
            self.assertEqual(machine["promotion_status"], "BLOCKED")
            self.assertIn("## Blockers", human)
            self.assertIn("NO_TRADE", human)


if __name__ == "__main__":
    unittest.main()
