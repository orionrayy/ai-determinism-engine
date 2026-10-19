import hashlib
import json
import unittest
from datetime import datetime, timedelta, timezone

import numpy as np

from portfolio_panel_selection_adapter_v1 import (
    REQUIRED_GATES,
    evaluate_synchronized_portfolio_panel,
)

HASH = "a" * 64
REFS = {key: f"repo://evidence/{key}.json" for key in REQUIRED_GATES}
FLAGS = {key: True for key in REQUIRED_GATES}


def make_panel(n=600, effect=0.0005):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rng = np.random.default_rng(2)
    noise = rng.normal(0, 0.001, size=n)
    returns = {
        "trial-a": (noise + effect).tolist(),
        "trial-b": rng.normal(0, 0.001, size=n).tolist(),
        "trial-c": rng.normal(0, 0.001, size=n).tolist(),
    }
    panel = {
        "schema_version": "synchronized-portfolio-return-panel-v1.0.0",
        "time_alignment_status": "PASS",
        "selection_guard_input_eligible": True,
        "panel_status": "EVIDENCE_ELIGIBLE",
        "timestamps": [(start + timedelta(minutes=i)).isoformat() for i in range(n)],
        "session_ids": ["continuous"] * n,
        "strategy_returns": returns,
        "benchmark_returns": [0.0] * n,
        "strategy_ledger_sha256": {name: f"{i+1:064x}" for i, name in enumerate(returns)},
        "row_count": n,
        "strategy_count": len(returns),
        "expected_interval_seconds": 60,
    }
    manifest_keys = (
        "schema_version", "expected_interval_seconds", "timestamps", "session_ids",
        "strategy_ledger_sha256", "strategy_returns", "benchmark_returns",
    )
    canonical = json.dumps(
        {key: panel[key] for key in manifest_keys},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    )
    panel["panel_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return panel


def call(panel=None, **changes):
    kwargs = {
        "panel": panel if panel is not None else make_panel(),
        "registered_strategy_ids": ["trial-a", "trial-b", "trial-c"],
        "expected_interval_seconds": 60,
        "trial_registry_sha256": "b" * 64,
        "source_dataset_sha256": "c" * 64,
        "gate_flags": dict(FLAGS),
        "gate_evidence_refs": dict(REFS),
        "block_lengths": (12, 24),
        "bootstrap_replicates": 200,
        "seed": 17,
        "evidence_kind": "synthetic_contract_test",
    }
    kwargs.update(changes)
    return evaluate_synchronized_portfolio_panel(**kwargs)


class PortfolioPanelSelectionAdapterTests(unittest.TestCase):
    def test_synchronized_full_family_reaches_guard_without_enabling_live(self):
        result = call()
        self.assertEqual(result["adapter_status"], "PASS")
        guard = result["selection_guard"]
        self.assertEqual(guard["evidence_kind"], "synthetic_contract_test")
        self.assertEqual(guard["promotion_status"], "SYNTHETIC_ONLY_NO_MARKET_PROMOTION")
        self.assertEqual(guard["n_aligned_observations"], 600)
        self.assertEqual(guard["strategy_count_in_return_panel"], 3)
        self.assertEqual(result["decision_output"], "NO_TRADE")
        self.assertFalse(result["live_signal_enabled"])
        self.assertFalse(result["live_order_execution_enabled"])

    def test_missing_registered_trial_blocks_extra_or_hidden_strategy(self):
        result = call(registered_strategy_ids=["trial-a", "trial-b"])
        self.assertEqual(result["adapter_status"], "BLOCKED")
        self.assertEqual(result["blocker"], "registered_strategy_family_does_not_match_panel")
        self.assertIsNone(result["selection_guard"])

    def test_diagnostic_panel_blocks_before_statistical_call(self):
        panel = make_panel()
        panel["panel_status"] = "DIAGNOSTIC_ONLY"
        panel["selection_guard_input_eligible"] = False
        result = call(panel)
        self.assertEqual(result["adapter_status"], "BLOCKED")
        self.assertIn("evidence_gates_blocked", result["blocker"])
        self.assertIsNone(result["selection_guard"])

    def test_false_inference_gate_blocks(self):
        gates = dict(FLAGS)
        gates["execution_costs_verified"] = False
        result = call(gate_flags=gates)
        self.assertEqual(result["adapter_status"], "BLOCKED")
        self.assertEqual(result["blocker"], "one_or_more_inference_gate_flags_false")

    def test_missing_gate_reference_blocks(self):
        refs = dict(REFS)
        refs.pop("portfolio_replay_verified")
        result = call(gate_evidence_refs=refs)
        self.assertEqual(result["adapter_status"], "BLOCKED")
        self.assertEqual(result["blocker"], "gate_evidence_refs_must_cover_exact_required_gates")

    def test_panel_hash_family_must_be_exact(self):
        panel = make_panel()
        panel["strategy_ledger_sha256"].pop("trial-c")
        result = call(panel)
        self.assertEqual(result["adapter_status"], "BLOCKED")
        self.assertEqual(result["blocker"], "strategy_ledger_hash_family_mismatch")

    def test_clock_must_have_at_least_500_intervals(self):
        result = call(make_panel(n=499))
        self.assertEqual(result["adapter_status"], "BLOCKED")
        self.assertEqual(result["blocker"], "panel_clock_or_benchmark_length_invalid")

    def test_panel_hash_and_source_dataset_hash_are_distinct_lineage_fields(self):
        panel = make_panel()
        result = call(panel)
        self.assertEqual(result["input_panel_sha256"], panel["panel_sha256"])
        self.assertEqual(result["source_dataset_sha256"], "c" * 64)
        self.assertNotEqual(result["input_panel_sha256"], result["source_dataset_sha256"])

    def test_tampered_return_vector_with_stale_panel_hash_blocks(self):
        panel = make_panel()
        panel["strategy_returns"]["trial-a"][0] += 0.25
        result = call(panel)
        self.assertEqual(result["adapter_status"], "BLOCKED")
        self.assertEqual(result["blocker"], "panel_sha256_mismatch")
        self.assertIsNone(result["selection_guard"])

    def test_registry_and_source_dataset_digest_must_be_distinct(self):
        result = call(trial_registry_sha256="c" * 64, source_dataset_sha256="c" * 64)
        self.assertEqual(result["adapter_status"], "BLOCKED")
        self.assertEqual(result["blocker"], "trial_registry_and_source_dataset_digests_must_be_distinct")

    def test_unusable_evidence_reference_blocks_before_inference(self):
        refs = dict(REFS)
        refs["execution_costs_verified"] = "file:///tmp/unverified.json"
        result = call(gate_evidence_refs=refs)
        self.assertEqual(result["adapter_status"], "BLOCKED")
        self.assertEqual(result["blocker"], "gate_evidence_reference_invalid")

    def test_real_market_with_missing_gate_flags_never_runs(self):
        gates = dict(FLAGS)
        gates["locked_holdout_unseen"] = False
        result = call(evidence_kind="real_market_research", gate_flags=gates)
        self.assertEqual(result["adapter_status"], "BLOCKED")
        self.assertIsNone(result["selection_guard"])

    def test_invalid_digest_blocks(self):
        result = call(source_dataset_sha256="not-a-digest")
        self.assertEqual(result["adapter_status"], "BLOCKED")
        self.assertEqual(result["blocker"], "source_dataset_sha256_invalid")


if __name__ == "__main__":
    unittest.main()
