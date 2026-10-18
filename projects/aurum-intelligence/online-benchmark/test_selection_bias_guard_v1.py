import unittest
from datetime import datetime, timedelta, timezone

import numpy as np

from selection_bias_guard_v1 import evaluate_white_reality_check


class SelectionBiasGuardTests(unittest.TestCase):
    def inputs(self, n=600, effect=0.0, seed=7):
        rng = np.random.default_rng(seed)
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        timestamps = [start + timedelta(minutes=i) for i in range(n)]
        session_ids = ["continuous-session"] * n
        noise = rng.normal(0.0, 0.001, size=n)
        returns = {
            "trial_a": noise + effect,
            "trial_b": rng.normal(0.0, 0.001, size=n),
            "trial_c": rng.normal(0.0, 0.001, size=n),
        }
        return {
            "timestamps": timestamps,
            "session_ids": session_ids,
            "strategy_returns": returns,
            "benchmark_returns": np.zeros(n),
            "expected_interval_seconds": 60,
            "block_lengths": (12, 24),
            "bootstrap_replicates": 200,
            "seed": 42,
            "returns_basis": "net_r",
            "evidence_kind": "synthetic_contract_test",
        }

    def test_null_data_returns_synthetic_only_and_no_live_authority(self):
        report = evaluate_white_reality_check(**self.inputs(effect=0.0))
        self.assertEqual(report["protocol_version"], "white-reality-check-segmented-block-bootstrap-v1")
        self.assertEqual(report["promotion_status"], "SYNTHETIC_ONLY_NO_MARKET_PROMOTION")
        self.assertEqual(report["decision_output"], "NO_TRADE")
        self.assertFalse(report["live_signal_enabled"])
        self.assertFalse(report["live_order_execution_enabled"])
        self.assertEqual(len(report["bootstrap_sensitivity"]), 2)

    def test_strong_synthetic_effect_can_pass_statistic_but_never_market_promote(self):
        report = evaluate_white_reality_check(**self.inputs(effect=0.0005))
        self.assertTrue(report["all_block_lengths_reject_null"])
        self.assertEqual(report["candidate_selection_status"], "PASS_FOR_INDEPENDENT_REPLICATION")
        self.assertEqual(report["promotion_status"], "SYNTHETIC_ONLY_NO_MARKET_PROMOTION")
        self.assertFalse(report["live_signal_enabled"])

    def test_default_market_gate_blocks_when_costs_or_holdout_are_unverified(self):
        args = self.inputs(effect=0.0005)
        args["evidence_kind"] = "real_market_research"
        report = evaluate_white_reality_check(**args)
        self.assertEqual(report["promotion_status"], "BLOCKED_MISSING_OR_UNVERIFIED_EVIDENCE_GATES")

    def test_evidence_gate_requires_a_stable_reference_for_each_claim(self):
        args = self.inputs(effect=0.0005)
        args.update({
            "evidence_kind": "real_market_research",
            "trial_registry_complete": True,
            "locked_holdout_unseen": True,
            "source_data_qualified": True,
            "execution_costs_verified": True,
            "portfolio_replay_verified": True,
            "trial_registry_sha256": "a" * 64,
            "dataset_sha256": "b" * 64,
            "gate_evidence_refs": {
                "trial_registry_complete": "repo://registry/trials.json",
                "locked_holdout_unseen": "repo://holdout/lock.json",
                "source_data_qualified": "repo://source/qualification.json",
                "execution_costs_verified": "repo://costs/observed.json",
            },
        })
        report = evaluate_white_reality_check(**args)
        self.assertEqual(report["promotion_status"], "BLOCKED_MISSING_OR_UNVERIFIED_EVIDENCE_GATES")

    def test_evidence_complete_synthetic_result_still_cannot_market_promote(self):
        args = self.inputs(effect=0.0005)
        args.update({
            "trial_registry_complete": True,
            "locked_holdout_unseen": True,
            "source_data_qualified": True,
            "execution_costs_verified": True,
            "portfolio_replay_verified": True,
            "trial_registry_sha256": "a" * 64,
            "dataset_sha256": "b" * 64,
            "gate_evidence_refs": {
                "trial_registry_complete": "repo://registry/trials.json",
                "locked_holdout_unseen": "repo://holdout/lock.json",
                "source_data_qualified": "repo://source/qualification.json",
                "execution_costs_verified": "repo://costs/observed.json",
                "portfolio_replay_verified": "repo://portfolio/replay.json",
            },
        })
        report = evaluate_white_reality_check(**args)
        self.assertEqual(report["promotion_status"], "SYNTHETIC_ONLY_NO_MARKET_PROMOTION")

    def test_nonmonotonic_or_naive_timestamps_fail_closed(self):
        args = self.inputs()
        args["timestamps"][10] = args["timestamps"][9]
        with self.assertRaisesRegex(ValueError, "strictly_increasing"):
            evaluate_white_reality_check(**args)
        args = self.inputs()
        args["timestamps"][5] = datetime(2026, 1, 1, 0, 5, 0)
        with self.assertRaisesRegex(ValueError, "timezone_aware"):
            evaluate_white_reality_check(**args)

    def test_irregular_within_session_cadence_fails_closed(self):
        args = self.inputs()
        args["timestamps"][10] += timedelta(seconds=1)
        with self.assertRaisesRegex(ValueError, "expected_cadence"):
            evaluate_white_reality_check(**args)

    def test_bad_return_panel_shapes_and_nonfinite_values_fail_closed(self):
        args = self.inputs()
        args["strategy_returns"]["trial_a"] = args["strategy_returns"]["trial_a"][:-1]
        with self.assertRaisesRegex(ValueError, "length_or_shape"):
            evaluate_white_reality_check(**args)
        args = self.inputs()
        args["strategy_returns"]["trial_b"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non_finite"):
            evaluate_white_reality_check(**args)

    def test_session_boundaries_are_explicit_and_cannot_recur(self):
        args = self.inputs()
        args["session_ids"] = ["a"] * 300 + ["b"] * 300
        args["timestamps"][300:] = [value + timedelta(hours=6) for value in args["timestamps"][300:]]
        report = evaluate_white_reality_check(**args)
        self.assertEqual(report["promotion_status"], "SYNTHETIC_ONLY_NO_MARKET_PROMOTION")
        args["session_ids"] = ["a"] * 200 + ["b"] * 200 + ["a"] * 200
        with self.assertRaisesRegex(ValueError, "contiguous"):
            evaluate_white_reality_check(**args)


if __name__ == "__main__":
    unittest.main()
