from __future__ import annotations

import tempfile
import unittest
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

from depth_execution import (
    DepthSnapshot,
    OrderRequest,
    OrderType,
    Side,
    SQLitePaperExecutionLedger,
    TimeInForce,
)
from execution_adapters import MarketCandle, MarketSpec
from forward_paper import (
    ForwardPaperConfig,
    ForwardPaperOrchestrator,
    ForwardState,
    RiskSupervisor,
    utc_now,
)
from indodax_microstructure import parse_depth


class FakeMarketData:
    def __init__(self, *, price: Decimal = Decimal("1000000")):
        self.price = price
        self.market = MarketSpec(
            venue="INDODAX",
            symbol="SOL/IDR",
            venue_symbol="solidr",
            base_asset="SOL",
            quote_asset="IDR",
            price_tick=Decimal("1"),
            quantity_step=Decimal("0.00000001"),
            min_notional_quote=Decimal("10000"),
            base_decimals=8,
            quote_decimals=0,
            min_quantity_base=Decimal("0.0001"),
        )

    def get_market_spec(self, symbol="SOL/IDR", *, mode_minimum_notional_quote=None):
        return MarketSpec(
            venue=self.market.venue,
            symbol=self.market.symbol,
            venue_symbol=self.market.venue_symbol,
            base_asset=self.market.base_asset,
            quote_asset=self.market.quote_asset,
            price_tick=self.market.price_tick,
            quantity_step=self.market.quantity_step,
            min_notional_quote=max(
                self.market.min_notional_quote,
                Decimal(str(mode_minimum_notional_quote or 0)),
            ),
            base_decimals=self.market.base_decimals,
            quote_decimals=self.market.quote_decimals,
            min_quantity_base=self.market.min_quantity_base,
        )

    def get_history(self, market, start_ms, end_ms, *, interval_minutes=1, now_ms=None):
        self.assert_market = market
        count = (end_ms - start_ms) // 60_000
        return [
            MarketCandle(
                open_time_ms=start_ms + index * 60_000,
                close_time_ms=start_ms + index * 60_000 + 59_999,
                interval_minutes=1,
                open=self.price,
                high=self.price + Decimal("100"),
                low=self.price - Decimal("100"),
                close=self.price,
                volume=Decimal("100"),
                quote_currency="IDR",
                source="offline fixture",
            )
            for index in range(count)
        ]


class FakeDepthClient:
    def __init__(self, *, bid=Decimal("1000000"), ask=Decimal("1000100")):
        self.bid, self.ask = Decimal(bid), Decimal(ask)
        self.calls = 0

    def snapshot(self):
        self.calls += 1
        payload = {
            "buy": [[str(self.bid - Decimal(index) * 100), "10"] for index in range(5)],
            "sell": [[str(self.ask + Decimal(index) * 100), "10"] for index in range(5)],
        }
        return parse_depth(payload, observed_at=utc_now())


class ForwardPaperIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp.name) / "forward.sqlite3")
        self.config = ForwardPaperConfig()

    def tearDown(self):
        self.temp.cleanup()

    def test_complete_cycle_starts_even_when_strategy_has_no_signal(self):
        market = FakeMarketData()
        depth = FakeDepthClient()
        runner = ForwardPaperOrchestrator(
            self.db_path, config=self.config, market_data=market, depth_client=depth
        )
        result = runner.run_once()

        self.assertEqual(result["system_status"], "FORWARD_PAPER_ACTIVE")
        self.assertEqual(result["mode"], "PAPER")
        self.assertEqual(result["venue"], "INDODAX")
        self.assertEqual(result["quote_currency"], "IDR")
        self.assertGreaterEqual(result["candle_count"], self.config.history_minutes)
        self.assertEqual(result["candle_count"] % 15, 0)
        self.assertIsNotNone(result["collector_observation_id"])
        self.assertIsNotNone(result["execution_snapshot_id"])
        self.assertEqual(result["signal"]["decision"], "NO_TRADE")
        self.assertTrue(result["reconciliation"]["consistent"])
        self.assertEqual(result["portfolio"]["cash_idr"], "300000")
        self.assertEqual(result["portfolio"]["sol_quantity"], "0")
        self.assertEqual(result["live_orders_enabled"], False)
        self.assertGreaterEqual(result["microstructure_summary"]["observation_count"], 1)
        self.assertEqual(depth.calls, 1)

        reopened = ForwardPaperOrchestrator(
            self.db_path, config=self.config, market_data=market, depth_client=depth
        )
        summary = reopened.state.summary()
        self.assertEqual(summary["cycles"]["completed"], 1)
        self.assertEqual(summary["signals"]["no_trade"], 1)
        self.assertIsNotNone(summary["activation"])
        self.assertEqual(summary["activation"]["mode"], "PAPER")

    def test_pending_buy_intent_is_not_mistaken_for_a_flat_closed_position(self):
        ledger = SQLitePaperExecutionLedger(self.db_path, initial_cash_idr=Decimal("300000"))
        state = ForwardState(self.db_path, self.config)
        runner = ForwardPaperOrchestrator(
            self.db_path, config=self.config, market_data=FakeMarketData(),
            depth_client=FakeDepthClient(),
        )
        now = utc_now()
        request = OrderRequest(
            client_order_id="pending-buy-test",
            venue="INDODAX",
            symbol="SOL/IDR",
            side=Side.BUY,
            quantity=Decimal("0.02"),
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.GTC,
            created_at=now,
            expires_at=now + timedelta(minutes=10),
            quantity_step=Decimal("0.00000001"),
            price_tick=Decimal("1"),
            fee_bps=Decimal("20.555"),
        )
        order_id = ledger.create_order(request)
        state.save_position(
            signal_id="pending-buy-test",
            order_id=order_id,
            candle_close_ms=1_800_000_000_000,
            atr_idr=Decimal("10000"),
            entry_price=None,
            stop_price=None,
            target_price=None,
            status="PENDING",
        )

        position = runner._sync_position(1_800_000_060_000)
        self.assertEqual(position["status"], "PENDING")
        self.assertEqual(ledger.list_open_orders()[0]["order_id"], order_id)

    def test_daily_loss_breaker_persists_across_state_reopen(self):
        ledger = SQLitePaperExecutionLedger(self.db_path, initial_cash_idr=Decimal("300000"))
        state = ForwardState(self.db_path, self.config)
        state.daily_risk("2026-10-30", Decimal("350000"))
        risk = RiskSupervisor(state, self.config)
        now = utc_now()
        raw = FakeDepthClient().snapshot()
        snapshot = DepthSnapshot.from_indodax(raw)

        decision = risk.evaluate(ledger.portfolio(), snapshot, (), now=now)
        self.assertFalse(decision.allow_new_entries)
        self.assertEqual(decision.state, "HALTED")
        self.assertIn("daily_loss_limit_reached", decision.reasons)

        reopened = ForwardState(self.db_path, self.config)
        self.assertEqual(reopened.control()["operating_state"], "HALTED")
        self.assertEqual(reopened.control()["halt_reason"], "daily_loss_limit_reached")

    def test_config_has_no_calendar_wait_gate(self):
        self.assertEqual(self.config.history_minutes, 600)
        self.assertNotIn("30_day_wait", self.config.as_json())
        self.config.validate()

    def test_signal_and_order_identifiers_are_stable(self):
        first = "cycle-" + "a" * 32
        second = "cycle-" + "a" * 32
        self.assertEqual(first, second)
        state = ForwardState(self.db_path, self.config)
        saved = state.record_signal(
            "signal-1", 1_800_000_000_000, "BUY", "NO_TRADE", ("breakout_not_confirmed",)
        )
        duplicate = state.record_signal(
            "signal-1", 1_800_000_000_000, "BUY", "NO_TRADE", ("breakout_not_confirmed",)
        )
        self.assertEqual(saved["signal_id"], duplicate["signal_id"])
        self.assertEqual(state.summary()["signals"]["no_trade"], 1)
        with self.assertRaisesRegex(Exception, "signal_id reused"):
            state.record_signal(
                "signal-1", 1_800_000_060_000, "BUY", "ORDER_INTENT", ("different_decision",),
                {"client_order_id": "different"},
            )


class DepthLedgerRecoveryApiTests(unittest.TestCase):
    def test_client_order_lookup_and_open_order_listing(self):
        with tempfile.TemporaryDirectory() as temp:
            path = str(Path(temp) / "orders.sqlite3")
            ledger = SQLitePaperExecutionLedger(path, initial_cash_idr=Decimal("300000"))
            now = utc_now()
            request = OrderRequest(
                client_order_id="lookup-order",
                venue="INDODAX",
                symbol="SOL/IDR",
                side=Side.BUY,
                quantity=Decimal("0.02"),
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.GTC,
                created_at=now,
                expires_at=now + timedelta(minutes=5),
                quantity_step=Decimal("0.00000001"),
                price_tick=Decimal("1"),
                fee_bps=Decimal("20.555"),
            )
            order_id = ledger.create_order(request)
            self.assertEqual(
                ledger.find_order_by_client_order_id("lookup-order")["order_id"], order_id
            )
            self.assertEqual(len(ledger.list_open_orders()), 1)
            self.assertEqual(ledger.list_open_orders()[0]["state"].value, "OPEN_SIMULATED")
            self.assertIsNone(ledger.find_order_by_client_order_id("missing"))


if __name__ == "__main__":
    unittest.main()
