import unittest
from datetime import datetime, timedelta, timezone
from portfolio_return_ledger_v1 import validate_portfolio_ledger, build_synchronized_return_panel

REFS = {
    "source_data_qualified": "repo://evidence/source.json",
    "execution_costs_verified": "repo://evidence/costs.json",
    "portfolio_replay_verified": "repo://evidence/replay.json",
    "locked_holdout_unseen": "repo://evidence/holdout.json",
    "trial_registry_complete": "repo://evidence/registry.json",
}
OBS = lambda name: {"state": "observed", "reference": f"repo://evidence/{name}.json"}
NA = lambda why: {"state": "not_applicable", "reason": why}
ASSUMED = lambda why: {"state": "assumed", "reason": why}
UNKNOWN = lambda why: {"state": "unknown", "reason": why}


def make_rows(n=5, strategy="trial-A"):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    equity, rows = 1000.0, []
    for i in range(n):
        trade, exposed = i in (1, 3), i in (1, 2, 3)
        gross = {1: -1.0, 2: 4.0, 3: 3.0}.get(i, 0.0)
        vals = {
            "spread_cost": 0.3 if trade else 0.0,
            "commission_cost": 0.5 if trade else 0.0,
            "slippage_cost": 0.1 if trade else 0.0,
            "other_fees_cost": 0.0,
            "market_impact_cost": 0.0,
            "financing_pnl": 0.0,
        }
        if not exposed:
            evidence = {k: NA("flat interval has no exposure, fills, or costs") for k in vals}
        elif trade:
            evidence = {
                "spread_cost": OBS(f"spread-{i}"),
                "commission_cost": OBS(f"commission-{i}"),
                "slippage_cost": OBS(f"slippage-{i}"),
                "other_fees_cost": OBS(f"fees-{i}"),
                "market_impact_cost": OBS(f"impact-{i}"),
                "financing_pnl": NA("no financing rollover in interval"),
            }
        else:
            evidence = {k: NA("no transaction or financing event in interval") for k in vals}
        net = gross - vals["commission_cost"] - vals["slippage_cost"] + vals["financing_pnl"]
        after = equity + net
        rows.append({
            "timestamp": start + timedelta(minutes=i), "session_id": "session-a",
            "decision_id": f"decision-{i}", "portfolio_id": "portfolio-1",
            "strategy_id": strategy, "trial_id": strategy, "source_id": "test-xauusd",
            "market_snapshot_sha256": f"{i+1:064x}", "decision_trace_sha256": f"{i+101:064x}",
            "pnl_basis": "quote_side_after_spread",
            "equity_before": equity, "equity_after": after, "gross_pnl_quote_side": gross,
            **vals, "exposure_during_interval": exposed,
            "long_exposure_cash": 100.0 if i in (1, 2) else 0.0, "short_exposure_cash": 0.0,
            "trade_count": 1 if trade else 0, "turnover_cash": 100.0 if trade else 0.0,
            "fill_mode": "observed" if trade else "none",
            "fill_reference": f"repo://fills/{i}.json" if trade else None,
            "fill_sha256": f"{i+201:064x}" if trade else None, "cost_evidence": evidence,
        })
        equity = after
    return rows


def flags():
    return {
        "source_data_qualified": True, "execution_costs_verified": True,
        "portfolio_replay_verified": True, "locked_holdout_unseen": True,
        "trial_registry_complete": True, "gate_evidence_refs": dict(REFS),
    }


class PortfolioReturnLedgerTests(unittest.TestCase):
    def validate(self, rows, **kwargs):
        return validate_portfolio_ledger(rows, expected_interval_seconds=60,
            starting_equity=1000, max_gross_leverage=2, **kwargs)

    def test_spread_is_not_deducted_twice_and_equity_reconciles(self):
        report = self.validate(make_rows(), **flags())
        self.assertEqual(report["contract_status"], "PASS")
        self.assertEqual(report["periods"][1]["net_pnl_quote_side_after_costs"], "-1.6")
        self.assertEqual(report["net_pnl_total"], "4.8")
        self.assertEqual(report["ending_equity"], "1004.8")
        self.assertTrue(report["selection_guard_input_eligible"])
        self.assertEqual(report["decision_output"], "NO_TRADE")

    def test_flat_intervals_remain_zero_return_rows(self):
        report = self.validate(make_rows(), **flags())
        self.assertEqual(report["row_count"], 5)
        self.assertEqual(report["no_exposure_interval_count"], 2)
        self.assertEqual(report["net_returns"][0], 0.0)
        self.assertEqual(report["net_returns"][4], 0.0)

    def test_default_evidence_gates_block_market_inference(self):
        report = self.validate(make_rows())
        self.assertFalse(report["selection_guard_input_eligible"])
        self.assertTrue(all(x == "BLOCKED" for x in report["gate_status"].values()))

    def test_assumed_cost_is_not_market_eligible(self):
        rows = make_rows()
        rows[1]["cost_evidence"]["slippage_cost"] = ASSUMED("simulated cost")
        report = self.validate(rows, **flags())
        self.assertEqual(report["economic_evidence_status"], "ASSUMPTION_ONLY")
        self.assertFalse(report["selection_guard_input_eligible"])

    def test_unknown_cost_is_explicitly_incomplete(self):
        rows = make_rows()
        rows[1]["cost_evidence"]["slippage_cost"] = UNKNOWN("broker slippage unknown")
        report = self.validate(rows, **flags())
        self.assertEqual(report["economic_evidence_status"], "INCOMPLETE")
        self.assertIn("material_cost_evidence_unknown", report["selection_guard_blockers"])

    def test_simulated_fill_never_counts_as_observed(self):
        rows = make_rows()
        rows[1]["fill_mode"] = "simulated"
        report = self.validate(rows, **flags())
        self.assertFalse(report["selection_guard_input_eligible"])
        self.assertIn("simulated_trade_fill_present", report["selection_guard_blockers"])

    def test_irregular_cadence_fails_closed(self):
        rows = make_rows()
        rows[2]["timestamp"] += timedelta(seconds=1)
        with self.assertRaisesRegex(ValueError, "expected_cadence"):
            self.validate(rows)

    def test_timestamps_must_increase(self):
        rows = make_rows()
        rows[2]["timestamp"] = rows[1]["timestamp"]
        with self.assertRaisesRegex(ValueError, "strictly_increasing"):
            self.validate(rows)

    def test_session_ids_cannot_recur(self):
        rows = make_rows(6)
        rows[3]["session_id"] = "session-b"
        rows[4]["session_id"] = "session-b"
        rows[5]["session_id"] = "session-a"
        with self.assertRaisesRegex(ValueError, "contiguous"):
            self.validate(rows)

    def test_equity_chain_cannot_reset(self):
        rows = make_rows()
        rows[2]["equity_before"] += 25
        with self.assertRaisesRegex(ValueError, "chain_discontinuity"):
            self.validate(rows)

    def test_equity_after_must_reconcile(self):
        rows = make_rows()
        rows[1]["equity_after"] += 5
        with self.assertRaisesRegex(ValueError, "reconciliation_failed"):
            self.validate(rows)

    def test_flat_interval_cannot_hide_pnl(self):
        rows = make_rows()
        rows[0]["gross_pnl_quote_side"] = 0.01
        with self.assertRaisesRegex(ValueError, "flat_interval_must_have_zero"):
            self.validate(rows)

    def test_midpoint_pnl_basis_is_rejected(self):
        rows = make_rows()
        rows[0]["pnl_basis"] = "midpoint_pre_spread"
        with self.assertRaisesRegex(ValueError, "quote_side_after_spread"):
            self.validate(rows)

    def test_gross_leverage_cap_is_enforced(self):
        rows = make_rows()
        rows[1]["long_exposure_cash"] = 5000
        with self.assertRaisesRegex(ValueError, "leverage_limit_exceeded"):
            self.validate(rows)

    def test_negative_cost_is_rejected(self):
        rows = make_rows()
        rows[1]["commission_cost"] = -0.5
        with self.assertRaisesRegex(ValueError, "must_be_nonnegative"):
            self.validate(rows)

    def test_observed_fill_needs_reference_and_hash(self):
        rows = make_rows()
        rows[1]["fill_reference"] = None
        with self.assertRaisesRegex(ValueError, "fill_requires_reference"):
            self.validate(rows)

    def test_not_applicable_cost_cannot_have_amount(self):
        rows = make_rows()
        rows[1]["cost_evidence"]["commission_cost"] = NA("invalid")
        with self.assertRaisesRegex(ValueError, "not_applicable_but_amount_nonzero"):
            self.validate(rows)

    def test_trade_needs_turnover_and_fill(self):
        rows = make_rows()
        rows[1]["turnover_cash"] = 0
        with self.assertRaisesRegex(ValueError, "positive_turnover"):
            self.validate(rows)

    def test_nonfinite_amount_rejected(self):
        rows = make_rows()
        rows[1]["gross_pnl_quote_side"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite_decimal"):
            self.validate(rows)

    def test_negative_financing_pnl_is_supported(self):
        rows = make_rows()
        rows[2]["financing_pnl"] = -0.25
        rows[2]["cost_evidence"]["financing_pnl"] = OBS("financing")
        rows[2]["equity_after"] -= 0.25
        rows[3]["equity_before"] -= 0.25
        rows[3]["equity_after"] -= 0.25
        rows[4]["equity_before"] -= 0.25
        rows[4]["equity_after"] -= 0.25
        report = self.validate(rows, **flags())
        self.assertEqual(report["periods"][2]["financing_pnl"], "-0.25")
        self.assertTrue(report["selection_guard_input_eligible"])

    def test_synchronized_panel_requires_exact_clock(self):
        a = self.validate(make_rows(strategy="trial-A"), **flags())
        b = self.validate(make_rows(strategy="trial-B"), **flags())
        panel = build_synchronized_return_panel({"trial-A": a, "trial-B": b},
            expected_interval_seconds=60, benchmark_returns=[0.0] * 5)
        self.assertEqual(panel["time_alignment_status"], "PASS")
        self.assertEqual(panel["strategy_count"], 2)
        self.assertTrue(panel["selection_guard_input_eligible"])

    def test_panel_will_not_drop_benchmark_rows(self):
        report = self.validate(make_rows(), **flags())
        with self.assertRaisesRegex(ValueError, "benchmark_length_mismatch"):
            build_synchronized_return_panel({"trial-A": report},
                expected_interval_seconds=60, benchmark_returns=[0.0])

    def test_panel_builder_rejects_mutated_ledger_periods(self):
        report = self.validate(make_rows(), **flags())
        report["periods"][1]["net_return"] += 0.25
        with self.assertRaisesRegex(ValueError, "content_sha256_mismatch"):
            build_synchronized_return_panel({"trial-A": report},
                expected_interval_seconds=60, benchmark_returns=[0.0] * 5)

    def test_panel_builder_rejects_mutated_return_projection(self):
        report = self.validate(make_rows(), **flags())
        report["net_returns"][1] += 0.25
        with self.assertRaisesRegex(ValueError, "series_projection_mismatch"):
            build_synchronized_return_panel({"trial-A": report},
                expected_interval_seconds=60, benchmark_returns=[0.0] * 5)

    def test_panel_builder_checks_strategy_identity(self):
        report = self.validate(make_rows(), **flags())
        with self.assertRaisesRegex(ValueError, "strategy_identity_mismatch"):
            build_synchronized_return_panel({"different-id": report},
                expected_interval_seconds=60, benchmark_returns=[0.0] * 5)

    def test_ledger_digest_is_deterministic(self):
        a = self.validate(make_rows(), **flags())
        b = self.validate(make_rows(), **flags())
        self.assertEqual(a["ledger_sha256"], b["ledger_sha256"])


if __name__ == "__main__":
    unittest.main()
