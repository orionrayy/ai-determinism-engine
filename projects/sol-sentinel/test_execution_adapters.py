import unittest
import urllib.error
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import execution_adapters as a


NOW = datetime(2026, 10, 30, 0, 0, tzinfo=timezone.utc)


def market():
    return a.MarketSpec(
        venue="MockX", symbol="SOL/IDR", venue_symbol="SOL_IDR",
        base_asset="SOL", quote_asset="IDR", price_tick=Decimal("1000"),
        quantity_step=Decimal("0.001"), min_notional_quote=Decimal("10000"),
        base_decimals=3, quote_decimals=0,
    )


def quote(*, bid="1800000", ask="1801000", observed_at=NOW):
    return a.MarketQuote("MockX", "SOL/IDR", Decimal(bid), Decimal(ask), observed_at, "fixture")


def intent(key="test-1", side=a.Side.BUY, qty="0.01", price="1801000"):
    return a.OrderIntent(key, "MockX", "SOL/IDR", side, Decimal(qty), Decimal(price), NOW)


class AdapterContractTests(unittest.TestCase):
    def test_market_spec_rejects_zero_tick_and_invalid_decimals(self):
        with self.assertRaisesRegex(a.AdapterContractError, "price_tick"):
            a.MarketSpec("X", "SOL/IDR", "SOL_IDR", "SOL", "IDR", 0, "0.001", 0, 9, 0)
        with self.assertRaisesRegex(a.AdapterContractError, "base_decimals"):
            a.MarketSpec("X", "SOL/IDR", "SOL_IDR", "SOL", "IDR", 1, "0.001", 0, 19, 0)

    def test_quote_rejects_crossed_nonpositive_and_naive_timestamps(self):
        with self.assertRaisesRegex(a.AdapterContractError, "crossed quote"):
            quote(bid="1802000", ask="1801000")
        with self.assertRaisesRegex(a.AdapterContractError, "positive"):
            quote(bid="0")
        with self.assertRaisesRegex(a.AdapterContractError, "timezone-aware"):
            quote(observed_at=datetime(2026, 10, 30))

    def test_order_identity_and_precision_are_checked(self):
        spec = market()
        bad_identity = a.validate_order(intent(), spec, a.MarketQuote("Other", "SOL/IDR", 1800000, 1801000, NOW), now=NOW)
        self.assertFalse(bad_identity.accepted)
        self.assertIn("quote_identity_mismatch", bad_identity.reasons)
        bad_qty = a.validate_order(intent(qty="0.0101"), spec, quote(), now=NOW)
        self.assertIn("quantity_step_mismatch", bad_qty.reasons)
        bad_tick = a.validate_order(intent(price="1801500"), spec, quote(), now=NOW)
        self.assertIn("price_tick_mismatch", bad_tick.reasons)

    def test_order_below_minimum_notional_is_rejected(self):
        check = a.validate_order(intent(qty="0.001", price="1801000"), market(), quote(), now=NOW)
        self.assertFalse(check.accepted)
        self.assertIn("minimum_notional_not_met", check.reasons)

    def test_stale_and_future_quotes_fail_closed(self):
        stale = a.validate_order(intent(), market(), quote(observed_at=NOW - timedelta(seconds=31)), now=NOW)
        future = a.validate_order(intent(), market(), quote(observed_at=NOW + timedelta(seconds=8)), now=NOW)
        self.assertIn("quote_stale", stale.reasons)
        self.assertIn("quote_from_future", future.reasons)

    def test_spread_limit_is_enforced(self):
        check = a.validate_order(intent(), market(), quote(bid="1790000", ask="1810000"), now=NOW, max_spread_bps=Decimal("20"))
        self.assertIn("spread_above_limit", check.reasons)

    def test_paper_adapter_buy_fill_is_simulated_at_ask(self):
        ledger = a.InMemoryIdempotentLedger()
        adapter = a.PaperExecutionAdapter(ledger, fee_bps=Decimal("10"))
        receipt = adapter.execute(intent(), market(), quote(), now=NOW)
        self.assertEqual(receipt.status, a.ReceiptStatus.FILLED_SIMULATED)
        self.assertEqual(receipt.fill_price_quote, Decimal("1801000"))
        self.assertEqual(receipt.filled_quantity_base, Decimal("0.01"))
        self.assertEqual(receipt.fee_quote, Decimal("18.01"))
        self.assertFalse(receipt.execution_enabled)
        self.assertFalse(receipt.live_order_submitted)
        self.assertEqual(ledger.count, 1)

    def test_paper_adapter_sell_fill_is_simulated_at_bid(self):
        adapter = a.PaperExecutionAdapter(a.InMemoryIdempotentLedger())
        receipt = adapter.execute(intent(side=a.Side.SELL, price="1800000"), market(), quote(), now=NOW)
        self.assertEqual(receipt.status, a.ReceiptStatus.FILLED_SIMULATED)
        self.assertEqual(receipt.fill_price_quote, Decimal("1800000"))

    def test_non_marketable_limit_is_no_fill_not_a_fake_trade(self):
        adapter = a.PaperExecutionAdapter(a.InMemoryIdempotentLedger())
        receipt = adapter.execute(intent(price="1799000"), market(), quote(), now=NOW)
        self.assertEqual(receipt.status, a.ReceiptStatus.NO_FILL)
        self.assertEqual(receipt.filled_quantity_base, Decimal("0"))
        self.assertIn("limit_not_marketable_at_observed_quote", receipt.reasons)

    def test_paper_adapter_is_idempotent_on_duplicate_intent(self):
        ledger = a.InMemoryIdempotentLedger()
        adapter = a.PaperExecutionAdapter(ledger)
        first = adapter.execute(intent(), market(), quote(), now=NOW)
        second = adapter.execute(intent(), market(), quote(bid="1700000", ask="1701000", observed_at=NOW + timedelta(minutes=1)), now=NOW + timedelta(minutes=1))
        self.assertEqual(first, second)
        self.assertEqual(ledger.count, 1)

    def test_reused_idempotency_key_with_changed_order_fails_closed(self):
        adapter = a.PaperExecutionAdapter(a.InMemoryIdempotentLedger())
        adapter.execute(intent(), market(), quote(), now=NOW)
        with self.assertRaisesRegex(a.AdapterContractError, "different order intent"):
            adapter.execute(intent(qty="0.02"), market(), quote(), now=NOW)

    def test_mock_market_data_capital_and_custody_adapters_are_separated(self):
        market_spec = market()
        adapter = a.StaticMockMarketDataAdapter([quote()])
        self.assertEqual(adapter.get_quote(market_spec), quote())
        self.assertEqual(a.MockCapitalSourceAdapter().source_id(), "reku-onramp-reference")
        custody = a.MockCustodyAdapter({("SOL", "SOLANA"): Decimal("0.25")})
        self.assertEqual(custody.get_balance("sol", "solana"), Decimal("0.25"))
        with self.assertRaisesRegex(a.AdapterContractError, "not configured"):
            custody.get_balance("USDC", "SOLANA")

    def test_indodax_public_depth_adapter_parses_best_levels_without_credentials(self):
        spec = a.MarketSpec(
            venue="INDODAX", symbol="SOL/IDR", venue_symbol="solidr",
            base_asset="SOL", quote_asset="IDR", price_tick="1000",
            quantity_step="0.000001", min_notional_quote="10000",
            base_decimals=9, quote_decimals=0,
        )
        requests = []
        payload = {
            "buy": [["1809000", "0.1"], ["1810000", "0.2"]],
            "sell": [["1820000", "0.1"], ["1815000", "0.3"]],
        }
        def fake_transport(url, timeout):
            requests.append((url, timeout))
            return payload
        adapter = a.IndodaxPublicMarketDataAdapter(transport=fake_transport, timeout=2.0)
        observed = adapter.get_quote(spec)
        self.assertEqual(observed.bid, Decimal("1810000"))
        self.assertEqual(observed.ask, Decimal("1815000"))
        self.assertEqual(observed.source, "INDODAX public /api/depth (request-start timestamp; conservative age)")
        self.assertEqual(requests, [("https://indodax.com/api/depth/solidr", 2.0)])
        self.assertFalse(hasattr(adapter, "execute"))
        self.assertFalse(hasattr(adapter, "sign"))

    def test_indodax_public_requests_obey_a_minimum_request_interval(self):
        spec = a.MarketSpec("INDODAX", "SOL/IDR", "solidr", "SOL", "IDR", "1000", "0.000001", "10000", 6, 0)
        depth = {"buy": [["1810000", "0.2"]], "sell": [["1815000", "0.3"]]}
        pairs = [{
            "id": "solidr", "symbol": "SOLIDR", "base_currency": "idr",
            "traded_currency": "sol", "traded_currency_unit": "SOL",
            "price_precision": "1000", "quantity_increment": "0.000001",
            "trade_min_base_currency": "10000", "trade_min_traded_currency": "0.001",
        }]
        def transport(url, timeout):
            return pairs if url.endswith("/api/pairs") else depth
        adapter = a.IndodaxPublicMarketDataAdapter(transport=transport, min_request_interval_seconds=0.35)
        from unittest.mock import patch
        with patch.object(a.time, "monotonic", side_effect=[100.0, 100.05, 100.35]) as clock:
            with patch.object(a.time, "sleep") as sleep:
                adapter.get_market_spec("SOL/IDR")
                adapter.get_quote(spec)
        self.assertAlmostEqual(sleep.call_args.args[0], 0.30, places=8)
        self.assertEqual(clock.call_count, 3)

    def test_indodax_pair_catalogue_loads_tick_step_and_both_order_minimums(self):
        pair_row = {
            "id": "solidr", "symbol": "SOLIDR",
            "base_currency": "idr", "traded_currency": "sol",
            "traded_currency_unit": "SOL", "price_precision": "1000",
            "quantity_increment": "0.000001", "trade_min_base_currency": "10000",
            "trade_min_traded_currency": "0.001",
        }
        urls = []
        adapter = a.IndodaxPublicMarketDataAdapter(
            transport=lambda url, timeout: (urls.append(url) or [pair_row]),
            attempts=1,
        )
        spec = adapter.get_market_spec("SOL/IDR")
        self.assertEqual(spec.venue_symbol, "solidr")
        self.assertEqual(spec.symbol, "SOL/IDR")
        self.assertEqual(spec.base_asset, "SOL")
        self.assertEqual(spec.quote_asset, "IDR")
        self.assertEqual(spec.price_tick, Decimal("1000"))
        self.assertEqual(spec.quantity_step, Decimal("0.000001"))
        self.assertEqual(spec.min_notional_quote, Decimal("10000"))
        self.assertEqual(spec.min_quantity_base, Decimal("0.001"))
        self.assertEqual(spec.base_decimals, 6)
        self.assertEqual(urls, ["https://indodax.com/api/pairs"])

    def test_indodax_pair_catalogue_fails_closed_on_ambiguous_or_incomplete_metadata(self):
        duplicate = [
            {"id": "solidr", "symbol": "SOLIDR", "base_currency": "idr", "traded_currency": "sol", "traded_currency_unit": "SOL", "price_precision": 1000, "quantity_increment": "0.000001", "trade_min_base_currency": 10000, "trade_min_traded_currency": "0.001"},
            {"id": "sol_idr", "symbol": "SOLIDR", "base_currency": "idr", "traded_currency": "sol", "traded_currency_unit": "SOL", "price_precision": 1000, "quantity_increment": "0.000001", "trade_min_base_currency": 10000, "trade_min_traded_currency": "0.001"},
        ]
        adapter = a.IndodaxPublicMarketDataAdapter(transport=lambda *_: duplicate, attempts=1)
        with self.assertRaisesRegex(a.VenueDataError, "matched 2 rows"):
            adapter.get_market_spec("SOL/IDR")
        incomplete = [{"id": "solidr", "symbol": "SOLIDR", "base_currency": "idr", "traded_currency": "sol", "traded_currency_unit": "SOL", "quantity_increment": "0.000001", "trade_min_base_currency": 10000, "trade_min_traded_currency": "0.001"}]
        with self.assertRaisesRegex(a.VenueDataError, "omitted a required tick"):
            a.IndodaxPublicMarketDataAdapter(transport=lambda *_: incomplete, attempts=1).get_market_spec("SOL/IDR")

    def test_market_validation_enforces_minimum_base_quantity(self):
        spec = a.MarketSpec(
            venue="X", symbol="SOL/IDR", venue_symbol="solidr",
            base_asset="SOL", quote_asset="IDR", price_tick="1000",
            quantity_step="0.000001", min_notional_quote="10000",
            base_decimals=6, quote_decimals=0, min_quantity_base="0.001",
        )
        small = intent(qty="0.0009", price="1801000")
        check = a.validate_order(small, spec, quote(), now=NOW)
        self.assertIn("minimum_base_quantity_not_met", check.reasons)

    def test_indodax_public_depth_adapter_retries_transient_timeout(self):
        spec = a.MarketSpec("INDODAX", "SOL/IDR", "solidr", "SOL", "IDR", "1000", "0.000001", "10000", 9, 0)
        attempts = []
        payload = {"buy": [["1810000", "0.2"]], "sell": [["1815000", "0.3"]]}
        def fake_transport(url, timeout):
            attempts.append(url)
            if len(attempts) == 1:
                raise TimeoutError("temporary timeout")
            return payload
        from unittest.mock import patch
        with patch.object(a.time, "sleep"):
            q = a.IndodaxPublicMarketDataAdapter(transport=fake_transport, attempts=2).get_quote(spec)
        self.assertEqual(len(attempts), 2)
        self.assertEqual(q.bid, Decimal("1810000"))

    def test_indodax_public_depth_fails_closed_on_wrong_venue_and_invalid_book(self):
        spec = a.MarketSpec("Other", "SOL/IDR", "solidr", "SOL", "IDR", "1000", "0.000001", "10000", 9, 0)
        adapter = a.IndodaxPublicMarketDataAdapter(transport=lambda *_: self.fail("transport must not run"))
        with self.assertRaisesRegex(a.AdapterContractError, "venue=INDODAX"):
            adapter.get_quote(spec)
        good_spec = a.MarketSpec("INDODAX", "SOL/IDR", "solidr", "SOL", "IDR", "1000", "0.000001", "10000", 9, 0)
        with self.assertRaisesRegex(a.VenueDataError, "invalid sell level"):
            a.IndodaxPublicMarketDataAdapter(
                transport=lambda *_: {"buy": [["1810000", "0.2"]], "sell": [["bad", "0.3"]]}
            ).get_quote(good_spec)
        with self.assertRaisesRegex(a.AdapterContractError, "IDR-quoted"):
            a.IndodaxPublicMarketDataAdapter(transport=lambda *_: {}).get_quote(
                a.MarketSpec("INDODAX", "SOL/USDT", "solusdt", "SOL", "USDT", "0.01", "0.000001", "1", 9, 6)
            )

    def test_transfer_plan_rejects_stale_fee_evidence(self):
        route = a.TransferRoute(
            "Reku", "Indodax", "SOL", "Solana", "Solana",
            Decimal("0.01"), Decimal("0.002"), Decimal("0.001"),
            source_network_verified=True, destination_network_verified=True,
            cost_verified_at=NOW - timedelta(days=2),
        )
        decision = a.PlanningOnlyTransferAdapter().plan(route, Decimal("0.1"))
        self.assertIn("route_cost_verification_stale", decision.reasons)
        self.assertFalse(decision.transfer_sent)

    def test_live_mode_is_unimplemented_even_when_flags_are_supplied(self):
        policy = a.ExecutionPolicy("LIVE", True, True, "approval-reference")
        with self.assertRaisesRegex(a.AdapterContractError, "no live execution adapter"):
            policy.validate()
        with self.assertRaisesRegex(a.AdapterContractError, "explicit user approval"):
            a.ExecutionPolicy(a.ExecutionMode.LIVE).validate()

    def test_transfer_route_requires_matching_verified_networks_and_known_costs(self):
        route = a.TransferRoute(
            "Reku", "Indodax", "SOL", "Solana", "Solana",
            Decimal("0.01"), None, None,
            source_network_verified=True, destination_network_verified=True,
        )
        decision = a.assess_transfer_route(route, Decimal("1"), now=NOW)
        self.assertFalse(decision.approved_for_planning)
        self.assertIn("withdrawal_fee_unknown", decision.reasons)
        self.assertIn("minimum_deposit_unknown", decision.reasons)
        self.assertFalse(decision.transfer_sent)

    def test_transfer_route_calculates_net_amount_but_never_sends(self):
        route = a.TransferRoute(
            "Reku", "Indodax", "SOL", "Solana", "Solana",
            Decimal("0.01"), Decimal("0.002"), Decimal("0.001"),
            source_network_verified=True, destination_network_verified=True,
            cost_verified_at=NOW,
        )
        decision = a.assess_transfer_route(route, Decimal("0.1"), now=NOW)
        self.assertTrue(decision.approved_for_planning)
        self.assertEqual(decision.estimated_amount_received, Decimal("0.098"))
        self.assertFalse(decision.transfer_sent)


    def test_indodax_public_depth_rejects_mismatched_pair_id_before_network(self):
        spec = a.MarketSpec("INDODAX", "SOL/IDR", "ethidr", "SOL", "IDR", "1000", "0.000001", "10000", 9, 0)
        adapter = a.IndodaxPublicMarketDataAdapter(transport=lambda *_: self.fail("network call must not happen"))
        with self.assertRaisesRegex(a.AdapterContractError, "does not match"):
            adapter.get_quote(spec)

    def test_transfer_fee_equal_to_amount_is_rejected_as_zero_remainder(self):
        route = a.TransferRoute(
            "Reku", "Indodax", "SOL", "Solana", "Solana",
            Decimal("0.01"), Decimal("0.1"), Decimal("0"),
            source_network_verified=True, destination_network_verified=True,
            cost_verified_at=NOW,
        )
        decision = a.assess_transfer_route(route, Decimal("0.1"), now=NOW)
        self.assertIn("no_transfer_value_remaining", decision.reasons)
        self.assertFalse(decision.approved_for_planning)
        self.assertEqual(decision.estimated_amount_received, Decimal("0"))

    def test_transfer_route_rejects_network_mismatch_and_under_minimum(self):
        route = a.TransferRoute(
            "Reku", "Destination", "SOL", "Solana", "Ethereum",
            Decimal("0.1"), Decimal("0.01"), Decimal("0.05"),
            source_network_verified=True, destination_network_verified=True,
        )
        decision = a.assess_transfer_route(route, Decimal("0.05"), now=NOW)
        self.assertIn("network_mismatch", decision.reasons)
        self.assertIn("below_minimum_withdrawal", decision.reasons)

    def test_quote_and_order_precision_use_decimal_not_float_rounding(self):
        spec = a.MarketSpec("X", "SOL/USDC", "SOLUSDC", "SOL", "USDC", "0.0001", "0.000001", "1", 9, 6)
        q = a.MarketQuote("X", "SOL/USDC", "140.0000", "140.0001", NOW)
        req = a.OrderIntent("precision", "X", "SOL/USDC", "BUY", "0.01", "140.0001", NOW)
        check = a.validate_order(req, spec, q, now=NOW)
        self.assertTrue(check.accepted)

    def test_ledger_detects_conflicting_same_order_id(self):
        ledger = a.InMemoryIdempotentLedger()
        first = a.ExecutionReceipt("order-1", "test-1", a.ReceiptStatus.NO_FILL, "X", "SOL/IDR", a.Side.BUY, Decimal("0.01"), Decimal("0"), None, Decimal("0"), NOW, ("no_fill",))
        ledger.record(first)
        ledger.record(first)
        self.assertEqual(ledger.count, 1)
        different = a.ExecutionReceipt("order-1", "test-1", a.ReceiptStatus.FILLED_SIMULATED, "X", "SOL/IDR", a.Side.BUY, Decimal("0.01"), Decimal("0.01"), Decimal("1801000"), Decimal("0"), NOW, ())
        with self.assertRaisesRegex(a.AdapterContractError, "idempotency conflict"):
            ledger.record(different)


if __name__ == "__main__":
    unittest.main(verbosity=2)
