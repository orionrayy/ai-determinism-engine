import unittest

from delayed_fixed_share_expert_v1 import (
    BASELINE_ID,
    run_delayed_fixed_share_expert_mixer,
)


def make_rows(n=30, *, delayed=1, regime_fn=None, expert_a=None, expert_b=None):
    rows = []
    for i in range(n):
        y = i % 2
        regime = regime_fn(i) if regime_fn else ("gold" if i < n // 2 else "silver")
        pa = expert_a(i, y) if expert_a else (0.1 if y == 0 else 0.9)
        pb = expert_b(i, y) if expert_b else (0.9 if y == 0 else 0.1)
        rows.append({
            "decision_id": f"d-{i:04d}",
            "decision_index": i,
            "label_available_index": i + delayed,
            "label": y,
            "label_mature": True,
            "regime_id": regime,
            "regime_available_index": i,
            "segment_id": "session-a" if i < n // 2 else "session-b",
            "baseline_probability": 0.5,
            "expert_probabilities": {"a": pa, "b": pb},
        })
    return rows


def run(rows, **kwargs):
    args = dict(
        as_of_index=1000,
        registered_experts=["a", "b"],
        registered_regimes=["gold", "silver"],
        eta=3.0,
        fixed_share_rate=0.03,
        min_global_matured=1,
        min_regime_matured=3,
        max_hindsight_switches=2,
        evidence_kind="synthetic_contract_test",
    )
    args.update(kwargs)
    return run_delayed_fixed_share_expert_mixer(rows, **args)


class DelayedFixedShareTests(unittest.TestCase):
    def test_first_maturity_index_is_not_used_until_strictly_before_decision(self):
        rows = make_rows(n=8, delayed=1)
        report = run(rows)
        ledger = report["forecast_ledger"]
        # Outcome at decision 0 is nominally available at index 1, but strict < means
        # it cannot change weights for decision 1; it is first used before decision 2.
        self.assertEqual(ledger[0]["matured_labels_seen_globally_before_decision"], 0)
        self.assertEqual(ledger[1]["matured_labels_seen_globally_before_decision"], 0)
        self.assertEqual(ledger[2]["matured_labels_seen_globally_before_decision"], 1)
        first = [e for e in report["update_events"] if e["label_available_index"] == 1]
        self.assertEqual(first[0]["applied_before_decision_index"], 2)

    def test_future_label_change_cannot_change_pre_maturity_predictions(self):
        rows_a = make_rows(n=8, delayed=6)
        rows_b = [dict(r) for r in rows_a]
        rows_b[0]["label"] = 1 - rows_b[0]["label"]
        a, b = run(rows_a), run(rows_b)
        probs_a = [r["probability"] for r in a["forecast_ledger"] if r["decision_index"] < 6]
        probs_b = [r["probability"] for r in b["forecast_ledger"] if r["decision_index"] < 6]
        self.assertEqual(probs_a, probs_b)

    def test_regime_pool_requires_its_own_matured_support(self):
        rows = make_rows(n=10, regime_fn=lambda i: "gold" if i % 5 == 0 else "silver")
        report = run(rows, min_regime_matured=4, min_global_matured=2)
        first_gold = next(x for x in report["forecast_ledger"] if x["regime_id"] == "gold")
        self.assertEqual(first_gold["forecast_source"], "BASELINE_FALLBACK_INSUFFICIENT_MATURED_HISTORY")
        later_gold = [x for x in report["forecast_ledger"] if x["regime_id"] == "gold"][-1]
        self.assertIn(later_gold["forecast_source"], {"REGIME_FIXED_SHARE", "GLOBAL_FIXED_SHARE", "BASELINE_FALLBACK_INSUFFICIENT_MATURED_HISTORY"})

    def test_global_weight_changes_only_after_feedback_arrives(self):
        rows = make_rows(n=12, delayed=1)
        # Expert A is consistently wrong; its relative loss vs baseline should shrink its weight.
        for row in rows:
            y = row["label"]
            row["expert_probabilities"]["a"] = 0.99 if y == 0 else 0.01
            row["expert_probabilities"]["b"] = 0.5
        report = run(rows, min_global_matured=1, min_regime_matured=100, fixed_share_rate=0.0, eta=2.0)
        ledger = report["forecast_ledger"]
        # At decision 1, decision 0 outcome is not available under strict < yet.
        self.assertEqual(ledger[1]["forecast_source"], "BASELINE_FALLBACK_INSUFFICIENT_MATURED_HISTORY")
        # At decision 2, first matured feedback has reduced expert A's share vs uniform.
        self.assertLess(ledger[2]["weights_used"]["a"], 1/3)
        self.assertGreater(sum(ledger[2]["weights_used"].values()), 0.999999)
        self.assertLess(abs(sum(ledger[2]["weights_used"].values()) - 1.0), 1e-9)

    def test_unmatured_labels_never_update_weights_and_are_excluded_from_scoring(self):
        rows = make_rows(n=10)
        for row in rows[3:]:
            row["label_mature"] = False
            row["label"] = None
        report = run(rows, as_of_index=10)
        self.assertEqual(report["scored_matured_rows"], 3)
        self.assertEqual(report["unmatured_or_unavailable_outcomes_excluded"], 7)
        self.assertEqual(report["decision_output"], "NO_TRADE")

    def test_unknown_regime_and_missing_expert_probability_fail_closed(self):
        rows = make_rows(n=5)
        rows[0]["regime_id"] = "future_regime"
        with self.assertRaisesRegex(ValueError, "unregistered_regime"):
            run(rows)
        rows = make_rows(n=5)
        rows[2]["expert_probabilities"].pop("b")
        with self.assertRaisesRegex(ValueError, "expert_probability_keys_must_match_registry"):
            run(rows)

    def test_nonmonotone_indices_and_future_regimes_fail_closed(self):
        rows = make_rows(n=5)
        rows[3]["decision_index"] = 1
        with self.assertRaisesRegex(ValueError, "decision_indices_must_be_strictly_increasing"):
            run(rows)
        rows = make_rows(n=5)
        rows[2]["regime_available_index"] = 3
        with self.assertRaisesRegex(ValueError, "regime_not_available_as_of_decision"):
            run(rows)

    def test_baseline_remains_in_candidate_pool_and_is_always_no_trade(self):
        report = run(make_rows(n=20))
        self.assertIn(BASELINE_ID, report["final_global_weights"])
        self.assertAlmostEqual(sum(report["final_global_weights"].values()), 1.0)
        self.assertFalse(report["market_promotion_allowed"])
        self.assertFalse(report["live_signal_enabled"])
        self.assertFalse(report["live_order_execution_enabled"])
        self.assertEqual(report["decision_output"], "NO_TRADE")

    def test_hindsight_switching_comparator_is_marked_diagnostic_only(self):
        report = run(make_rows(n=20))
        comp = report["hindsight_comparators"]
        self.assertEqual(comp["status"], "DIAGNOSTIC_HINDSIGHT_ONLY_NOT_USABLE_FOR_DECISIONS")
        self.assertIsNotNone(comp["best_switching_total_brier_loss"])
        self.assertIn("entire scored cohort", comp["caveat"])

    def test_deterministic_replay_has_same_forecast_ledger(self):
        rows = make_rows(n=20)
        a, b = run(rows), run(rows)
        self.assertEqual(a["forecast_ledger"], b["forecast_ledger"])
        self.assertEqual(a["update_events"], b["update_events"])

    def test_regime_adaptive_mix_can_track_regime_specialists_on_synthetic_fixture(self):
        rows = make_rows(n=200, delayed=1)
        for row in rows:
            i, y = row["decision_index"], row["label"]
            good = "a" if i < 100 else "b"
            bad = "b" if good == "a" else "a"
            row["expert_probabilities"][good] = 0.05 if y == 0 else 0.95
            row["expert_probabilities"][bad] = 0.95 if y == 0 else 0.05
            row["baseline_probability"] = 0.5
        report = run(rows, as_of_index=202, eta=3.0, fixed_share_rate=0.015,
                     min_global_matured=10, min_regime_matured=10, max_hindsight_switches=2)
        # This is a software fixture proving the selector can combine complementary
        # experts when the regime signal is informative; it is not market evidence.
        self.assertLess(report["online_mixture"]["brier"], report["causal_baseline"]["brier"])
        self.assertEqual(report["evidence_kind"], "synthetic_contract_test")
        self.assertFalse(report["market_evidence_eligible"])
        self.assertEqual(report["decision_output"], "NO_TRADE")

    def test_missing_provenance_manifest_is_explicitly_ineligible_for_market_claim(self):
        report = run(make_rows(n=10))
        self.assertEqual(report["evidence_manifest_validation"]["status"], "MISSING_MANIFEST_NO_MARKET_CLAIM")
        self.assertFalse(report["market_evidence_eligible"])

    def test_manifest_hash_shape_never_claims_source_authenticity(self):
        manifest = {
            "target_id": "xauusd-tp-before-sl-v1",
            "target_definition_sha256": "a" * 64,
            "dataset_sha256": "b" * 64,
            "source_lineage_sha256": "c" * 64,
            "forecast_ledger_sha256": "d" * 64,
            "model_artifact_sha256_by_expert": {"a": "e" * 64, "b": "f" * 64},
        }
        report = run(make_rows(n=10), evidence_manifest=manifest)
        self.assertEqual(report["evidence_manifest_validation"]["status"], "MANIFEST_SHAPE_VALID_SOURCE_NOT_AUTHENTICATED")
        self.assertFalse(report["evidence_manifest_validation"]["source_authenticity_independently_verified"])
        self.assertFalse(report["market_evidence_eligible"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
