from __future__ import annotations

import unittest
from datetime import datetime

from execution_cost_model_v1 import Evidence, ExecutionProfile, QuoteSnapshot, calculate_closed_trade


def ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def profile(**overrides) -> ExecutionProfile:
    values = dict(
        profile_id="fixture-xauusd-v1", instrument_id="XAUUSD.fixture",
        quote_currency="USD", account_currency="USD", contract_units_per_lot=100,
        commission_per_lot_per_side=3.5, entry_slippage_price_units=0.1,
        exit_slippage_price_units=0.2,
        commission_evidence=Evidence("assumed"), slippage_evidence=Evidence("assumed"),
        swap_evidence=Evidence("assumed"), swap_long_per_lot_per_rollover=-4.0,
        swap_short_per_lot_per_rollover=1.0,
    )
    values.update(overrides)
    return ExecutionProfile(**values)


def quote(time, bid, ask, evidence=None):
    return QuoteSnapshot(ts(time), bid, ask, "synthetic-fixture", evidence or Evidence("assumed"))


class ExecutionCostModelTests(unittest.TestCase):
    def test_long_quote_side_pnl_includes_spread_once(self):
        result = calculate_closed_trade(
            side="long", lots=1,
            entry_quote=quote("2026-01-01T00:00:00Z", 2000, 2000.4),
            exit_quote=quote("2026-01-01T00:05:00Z", 2001, 2001.4),
            profile=profile(entry_slippage_price_units=0, exit_slippage_price_units=0,
                            commission_per_lot_per_side=0),
        )
        self.assertEqual(result["gross_quote_side_pnl"], "60.0")
        self.assertEqual(result["pnl_decomposition"]["gross_mid_move_pnl"], "100.0")
        self.assertEqual(result["pnl_decomposition"]["spread_drag_included_in_quote_side_pnl"], "40.0")
        self.assertEqual(result["net_pnl"], "60.0")

    def test_short_quote_side_pnl_correct_sign(self):
        result = calculate_closed_trade(
            side="short", lots=1,
            entry_quote=quote("2026-01-01T00:00:00Z", 2000, 2000.4),
            exit_quote=quote("2026-01-01T00:05:00Z", 1999, 1999.4),
            profile=profile(entry_slippage_price_units=0, exit_slippage_price_units=0,
                            commission_per_lot_per_side=0),
        )
        self.assertEqual(result["gross_quote_side_pnl"], "60.0")
        self.assertEqual(result["net_pnl"], "60.0")

    def test_slippage_commission_swap_and_r_are_decomposed(self):
        result = calculate_closed_trade(
            side="long", lots=2,
            entry_quote=quote("2026-01-01T00:00:00Z", 2000, 2000.4),
            exit_quote=quote("2026-01-01T23:55:00Z", 2001, 2001.4),
            profile=profile(), rollovers=1, initial_risk_cash=100,
        )
        self.assertEqual(result["gross_quote_side_pnl"], "120.0")
        self.assertEqual(result["pnl_decomposition"]["slippage_drag"], "60.0")
        self.assertEqual(result["pnl_decomposition"]["commission_cost"], "14.0")
        self.assertEqual(result["pnl_decomposition"]["swap_pnl"], "-8.0")
        self.assertEqual(result["net_pnl"], "38.0")
        self.assertEqual(result["net_r"], "0.38")
        self.assertEqual(result["execution_evidence_status"], "ASSUMPTION_ONLY")
        self.assertFalse(result["live_execution_authorized"])

    def test_unknown_slippage_fails_closed_for_net_pnl(self):
        result = calculate_closed_trade(
            side="long", lots=1,
            entry_quote=quote("2026-01-01T00:00:00Z", 2000, 2000.4),
            exit_quote=quote("2026-01-01T00:05:00Z", 2001, 2001.4),
            profile=profile(slippage_evidence=Evidence("unknown")),
        )
        self.assertIsNone(result["net_pnl"])
        self.assertEqual(result["pnl_status"], "NET_PNL_NOT_COMPUTABLE_UNKNOWN_COSTS")
        self.assertIn("slippage", result["unknown_material_costs"])

    def test_zero_rollovers_does_not_require_swap_evidence(self):
        result = calculate_closed_trade(
            side="long", lots=1,
            entry_quote=quote("2026-01-01T00:00:00Z", 2000, 2000.4),
            exit_quote=quote("2026-01-01T00:05:00Z", 2001, 2001.4),
            profile=profile(swap_evidence=Evidence("unknown")), rollovers=0,
        )
        self.assertIsNotNone(result["net_pnl"])
        self.assertNotIn("swap", result["unknown_material_costs"])

    def test_observed_evidence_needs_reference_and_never_authorizes_live(self):
        result = calculate_closed_trade(
            side="long", lots=1,
            entry_quote=quote("2026-01-01T00:00:00Z", 2000, 2000.4, Evidence("observed", "https://example.com/quote-in")),
            exit_quote=quote("2026-01-01T00:05:00Z", 2001, 2001.4, Evidence("observed", "https://example.com/quote-out")),
            profile=profile(
                commission_evidence=Evidence("observed", "https://example.com/commission"),
                slippage_evidence=Evidence("observed", "https://example.com/slippage"),
                broker_authorized=True,
                broker_authorization_evidence_ref="https://example.com/broker-authorization",
                entry_slippage_price_units=0, exit_slippage_price_units=0,
                commission_per_lot_per_side=0,
            ),
        )
        self.assertEqual(result["execution_evidence_status"], "ELIGIBLE_FOR_MANUAL_REVIEW")
        self.assertFalse(result["live_execution_authorized"])

    def test_observed_cost_without_reference_is_not_manual_review_eligible(self):
        result = calculate_closed_trade(
            side="long", lots=1,
            entry_quote=quote("2026-01-01T00:00:00Z", 2000, 2000.4, Evidence("observed", "https://example.com/quote-in")),
            exit_quote=quote("2026-01-01T00:05:00Z", 2001, 2001.4, Evidence("observed", "https://example.com/quote-out")),
            profile=profile(commission_evidence=Evidence("observed"), slippage_evidence=Evidence("observed"),
                            broker_authorized=True, broker_authorization_evidence_ref="https://example.com/auth"),
        )
        self.assertEqual(result["execution_evidence_status"], "EVIDENCE_INCOMPLETE")

    def test_rejects_naive_timestamp(self):
        with self.assertRaisesRegex(ValueError, "timezone_aware"):
            quote("2026-01-01T00:00:00", 2000, 2000.4).normalized("entry")

    def test_rejects_crossed_quote(self):
        with self.assertRaisesRegex(ValueError, "crossed_quote"):
            quote("2026-01-01T00:00:00Z", 2000.4, 2000).normalized("entry")

    def test_rejects_non_positive_prices(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            quote("2026-01-01T00:00:00Z", 0, 0.4).normalized("entry")

    def test_rejects_exit_before_entry(self):
        with self.assertRaisesRegex(ValueError, "exit_timestamp_must_follow_entry"):
            calculate_closed_trade(
                side="long", lots=1,
                entry_quote=quote("2026-01-01T00:05:00Z", 2000, 2000.4),
                exit_quote=quote("2026-01-01T00:00:00Z", 2001, 2001.4), profile=profile(),
            )

    def test_rejects_invalid_side_lots_and_rollovers(self):
        base = dict(entry_quote=quote("2026-01-01T00:00:00Z", 2000, 2000.4),
                    exit_quote=quote("2026-01-01T00:05:00Z", 2001, 2001.4), profile=profile())
        with self.assertRaisesRegex(ValueError, "side_must_be"):
            calculate_closed_trade(side="buy", lots=1, **base)
        with self.assertRaisesRegex(ValueError, "lots_must_be_positive"):
            calculate_closed_trade(side="long", lots=0, **base)
        with self.assertRaisesRegex(ValueError, "rollovers_must_be_nonnegative"):
            calculate_closed_trade(side="long", lots=1, rollovers=-1, **base)

    def test_rejects_currency_conversion_instead_of_guessing(self):
        with self.assertRaisesRegex(ValueError, "currency_conversion_not_implemented"):
            profile(account_currency="IDR").normalized()

    def test_rejects_negative_costs_and_bad_contract_multiplier(self):
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            profile(entry_slippage_price_units=-0.1).normalized()
        with self.assertRaisesRegex(ValueError, "must_be_positive"):
            profile(contract_units_per_lot=0).normalized()

    def test_rejects_naive_initial_risk(self):
        with self.assertRaisesRegex(ValueError, "initial_risk_cash_must_be_positive"):
            calculate_closed_trade(
                side="long", lots=1,
                entry_quote=quote("2026-01-01T00:00:00Z", 2000, 2000.4),
                exit_quote=quote("2026-01-01T00:05:00Z", 2001, 2001.4),
                profile=profile(), initial_risk_cash=0,
            )

    def test_exit_quote_bid_ask_spread_does_not_get_charged_twice(self):
        result = calculate_closed_trade(
            side="long", lots=1,
            entry_quote=quote("2026-01-01T00:00:00Z", 2000, 2001),
            exit_quote=quote("2026-01-01T00:05:00Z", 2000, 2001),
            profile=profile(entry_slippage_price_units=0, exit_slippage_price_units=0,
                            commission_per_lot_per_side=0),
        )
        self.assertEqual(result["gross_quote_side_pnl"], "-100")
        self.assertEqual(result["pnl_decomposition"]["spread_drag_included_in_quote_side_pnl"], "100")
        self.assertEqual(result["net_pnl"], "-100.0")


if __name__ == "__main__":
    unittest.main()
