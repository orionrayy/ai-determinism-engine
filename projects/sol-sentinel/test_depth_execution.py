from __future__ import annotations

import sqlite3
from contextlib import closing
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from pathlib import Path

from depth_execution import (
    BookLevel, DepthSnapshot, ExecutionError, LedgerIntegrityError,
    OrderRequest, OrderState, OrderType, Side, SQLitePaperExecutionLedger,
    TimeInForce,
)

T0 = datetime(2026, 10, 30, 4, 0, tzinfo=timezone.utc)


def snap(offset=1, *, bids=None, asks=None):
    return DepthSnapshot(
        "INDODAX", "SOL/IDR", T0 + timedelta(seconds=offset),
        tuple(BookLevel(*x) for x in (bids or [(100, 2), (99, 3), (98, 4)])),
        tuple(BookLevel(*x) for x in (asks or [(101, 1), (102, 2), (103, 3)])),
    )


def request(client="o-1", *, side=Side.BUY, qty="4", order_type=OrderType.MARKET,
            limit=None, tif=TimeInForce.GTC, created=T0, expires=None, fee="10"):
    return OrderRequest(client, "INDODAX", "SOL/IDR", side, Decimal(qty), order_type,
                        Decimal(limit) if limit is not None else None, tif, created, expires,
                        Decimal("1"), Decimal("1"), Decimal(fee), Decimal("0.01"), Decimal("30"))


class DepthExecutionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "paper.sqlite3"
        self.ledger = SQLitePaperExecutionLedger(self.db_path, initial_cash_idr=Decimal("10000"))

    def tearDown(self): self.tmp.cleanup()

    def test_from_indodax_converts_existing_collector_snapshot_without_import_coupling(self):
        source = SimpleNamespace(
            observed_at=T0,
            exchange_timestamp_ms=1800000000000,
            bids=(SimpleNamespace(price=Decimal("100"), quantity=Decimal("2")),),
            asks=(SimpleNamespace(price=Decimal("101"), quantity=Decimal("3")),),
        )
        converted = DepthSnapshot.from_indodax(source)
        self.assertEqual(converted.venue, "INDODAX")
        self.assertEqual(converted.symbol, "SOL/IDR")
        self.assertEqual(converted.bids[0], BookLevel(Decimal("100"), Decimal("2")))
        self.assertEqual(converted.asks[0], BookLevel(Decimal("101"), Decimal("3")))
        self.assertEqual(converted.exchange_timestamp_ms, 1800000000000)

    def test_order_request_default_created_at_is_current_aware_utc(self):
        before = datetime.now(timezone.utc)
        created = OrderRequest("default-time", "INDODAX", "SOL/IDR", Side.BUY, Decimal("1"))
        after = datetime.now(timezone.utc)
        self.assertGreaterEqual(created.created_at, before)
        self.assertLessEqual(created.created_at, after)
        self.assertEqual(created.created_at.utcoffset(), timezone.utc.utcoffset(created.created_at))

    def test_snapshot_immutable_identity_and_validates_book(self):
        a = snap()
        b = snap()
        self.assertEqual(a.snapshot_id, b.snapshot_id)
        with self.assertRaisesRegex(ExecutionError, "descending"):
            snap(bids=[(99, 1), (100, 1)])
        with self.assertRaisesRegex(ExecutionError, "ascending"):
            snap(asks=[(102, 1), (101, 1)])
        with self.assertRaisesRegex(ExecutionError, "crossed"):
            snap(bids=[(102, 1)], asks=[(101, 1)])

    def test_order_retry_with_same_terms_but_new_creation_timestamp_is_idempotent(self):
        first = request("stable-retry", qty="2", created=T0)
        later = request("stable-retry", qty="2", created=T0 + timedelta(seconds=10))
        order_id = self.ledger.create_order(first)
        self.assertEqual(order_id, self.ledger.create_order(later))
        self.assertEqual(len([e for e in self.ledger.get_events(order_id) if e["event_type"] == "OrderCreated"]), 1)

    def test_order_creation_is_idempotent_and_conflict_fails_closed(self):
        r = request()
        first = self.ledger.create_order(r)
        self.assertEqual(first, self.ledger.create_order(r))
        with self.assertRaisesRegex(LedgerIntegrityError, "reused"):
            self.ledger.create_order(request(qty="5"))

    def test_buy_sweeps_multiple_ask_levels_and_computes_cumulative_vwap(self):
        oid = self.ledger.create_order(request(qty="2", fee="10"))
        result = self.ledger.process_snapshot(oid, snap(), as_of=T0+timedelta(seconds=2))
        self.assertEqual(result.state, OrderState.FILLED_SIMULATED)
        self.assertEqual([f.quantity for f in result.fills], [Decimal("1"), Decimal("1")])
        self.assertEqual([f.price for f in result.fills], [Decimal("101"), Decimal("102")])
        self.assertEqual(result.vwap, Decimal("101.5"))
        self.assertEqual(result.gross_idr, Decimal("203"))
        self.assertEqual(result.fee_idr, Decimal("0.20"))
        self.assertEqual(result.remaining_quantity, Decimal("0"))

    def test_partial_fill_preserves_residual_then_continues_on_new_snapshot(self):
        oid = self.ledger.create_order(request(qty="5"))
        one = self.ledger.process_snapshot(oid, snap(1, asks=[(101, 2)]), as_of=T0+timedelta(seconds=2))
        self.assertEqual(one.state, OrderState.PARTIALLY_FILLED_SIMULATED)
        self.assertEqual(one.cumulative_filled_quantity, Decimal("2"))
        self.assertEqual(one.remaining_quantity, Decimal("3"))
        two = self.ledger.process_snapshot(oid, snap(2, asks=[(100, 1), (101, 2)]), as_of=T0+timedelta(seconds=3))
        self.assertEqual(two.state, OrderState.FILLED_SIMULATED)
        self.assertEqual(two.cumulative_filled_quantity, Decimal("5"))
        self.assertEqual(two.remaining_quantity, Decimal("0"))
        self.assertEqual(two.vwap, (Decimal("202")+Decimal("302"))/Decimal("5"))
        self.assertTrue(self.ledger.reconcile()["consistent"])

    def test_insufficient_depth_does_not_fabricate_fill(self):
        oid = self.ledger.create_order(request(qty="20"))
        result = self.ledger.process_snapshot(oid, snap(), as_of=T0+timedelta(seconds=2))
        self.assertEqual(result.cumulative_filled_quantity, Decimal("6"))
        self.assertEqual(result.remaining_quantity, Decimal("14"))
        self.assertEqual(result.state, OrderState.PARTIALLY_FILLED_SIMULATED)

    def test_buy_limit_stops_before_level_above_limit(self):
        oid = self.ledger.create_order(request(qty="5", order_type=OrderType.LIMIT, limit="102"))
        result = self.ledger.process_snapshot(oid, snap(), as_of=T0+timedelta(seconds=2))
        self.assertEqual(result.cumulative_filled_quantity, Decimal("3"))
        self.assertEqual(result.remaining_quantity, Decimal("2"))
        self.assertIn("limit_price_boundary_reached", result.reasons)

    def test_sell_limit_stops_before_level_below_limit(self):
        # Seed inventory from a previous purchase with explicit fills.
        buy_id = self.ledger.create_order(request("seed-buy", qty="3", fee="0"))
        self.ledger.process_snapshot(buy_id, snap(1), as_of=T0+timedelta(seconds=2))
        sell_id = self.ledger.create_order(request("sell", side=Side.SELL, qty="3", order_type=OrderType.LIMIT, limit="99", created=T0+timedelta(seconds=3), fee="0"))
        # Bid levels 100, 99, 98; sell limit 99 excludes 98.
        result = self.ledger.process_snapshot(sell_id, snap(4, bids=[(110,2),(109,2)], asks=[(111,1)]), as_of=T0+timedelta(seconds=5))
        self.assertEqual(result.cumulative_filled_quantity, Decimal("3"))
        self.assertEqual(result.vwap, (Decimal("220")+Decimal("109"))/Decimal("3"))
        self.assertTrue(self.ledger.reconcile()["consistent"])

    def test_same_snapshot_cannot_be_reused_for_same_order(self):
        oid = self.ledger.create_order(request(qty="5"))
        book = snap(asks=[(101, 2)])
        a = self.ledger.process_snapshot(oid, book, as_of=T0+timedelta(seconds=2))
        b = self.ledger.process_snapshot(oid, book, as_of=T0+timedelta(seconds=3))
        self.assertEqual(a.cumulative_filled_quantity, b.cumulative_filled_quantity)
        self.assertTrue(b.duplicate)
        self.assertEqual(len(self.ledger.get_fills(oid)), 1)

    def test_snapshot_liquidity_is_shared_across_multiple_orders(self):
        book = snap(asks=[(101, 3)])
        a = self.ledger.create_order(request("a", qty="2", fee="0"))
        b = self.ledger.create_order(request("b", qty="2", fee="0"))
        ra = self.ledger.process_snapshot(a, book, as_of=T0+timedelta(seconds=2))
        rb = self.ledger.process_snapshot(b, book, as_of=T0+timedelta(seconds=2))
        self.assertEqual(ra.cumulative_filled_quantity, Decimal("2"))
        self.assertEqual(rb.cumulative_filled_quantity, Decimal("1"))
        used = self.ledger.snapshot_liquidity(book.snapshot_id)
        askrow = next(x for x in used if x["side"] == "ask" and x["level_index"] == 0)
        self.assertEqual(Decimal(askrow["consumed_qty"]), Decimal("3"))
        self.assertEqual(Decimal(askrow["reserved_qty"]), Decimal("0"))
        self.assertEqual(Decimal(askrow["remaining_qty"]), Decimal("0"))

    def test_identical_side_book_at_new_timestamp_does_not_refresh_consumed_depth(self):
        first_book = snap(1, asks=[(101, 2)])
        first_order = self.ledger.create_order(request("pool-first", qty="2", fee="0"))
        first_result = self.ledger.process_snapshot(first_order, first_book, as_of=T0+timedelta(seconds=2))
        self.assertEqual(first_result.cumulative_filled_quantity, Decimal("2"))

        # A new observation time gives a new snapshot ID, but the ask-side book is unchanged.
        second_book = snap(2, asks=[(101, 2)])
        self.assertNotEqual(first_book.snapshot_id, second_book.snapshot_id)
        self.assertEqual(first_book.liquidity_pool_id("ask"), second_book.liquidity_pool_id("ask"))
        second_order = self.ledger.create_order(request("pool-second", qty="2", created=T0+timedelta(seconds=1), fee="0"))
        second_result = self.ledger.process_snapshot(second_order, second_book, as_of=T0+timedelta(seconds=4))
        self.assertEqual(second_result.cumulative_filled_quantity, Decimal("0"))
        self.assertEqual(second_result.remaining_quantity, Decimal("2"))
        shared = self.ledger.snapshot_liquidity(second_book.snapshot_id)
        ask = next(x for x in shared if x["side"] == "ask" and x["level_index"] == 0)
        self.assertEqual(Decimal(ask["snapshot_consumed_qty"]), Decimal("0"))
        self.assertEqual(Decimal(ask["shared_consumed_qty"]), Decimal("2"))
        self.assertEqual(Decimal(ask["remaining_qty"]), Decimal("0"))
        self.assertTrue(self.ledger.reconcile()["consistent"])

    def test_restart_backfills_shared_liquidity_consumption_from_prior_snapshot_journal(self):
        book = snap(1, asks=[(101, 2)])
        oid = self.ledger.create_order(request("pool-migration-first", qty="2", fee="0"))
        self.ledger.process_snapshot(oid, book, as_of=T0+timedelta(seconds=2))
        # Simulate a database written before shared side-book pool tables existed.
        with closing(sqlite3.connect(self.db_path)) as db:
            db.execute("DROP TABLE snapshot_liquidity_pools")
            db.execute("DROP TABLE liquidity_pool_levels")
            db.execute("DROP TABLE liquidity_pools")
            db.commit()
        reopened = SQLitePaperExecutionLedger(self.db_path, initial_cash_idr=Decimal("10000"))
        next_book = snap(2, asks=[(101, 2)])
        next_oid = reopened.create_order(request("pool-migration-second", qty="1", created=T0+timedelta(seconds=1), fee="0"))
        result = reopened.process_snapshot(next_oid, next_book, as_of=T0+timedelta(seconds=4))
        self.assertEqual(result.cumulative_filled_quantity, Decimal("0"))
        self.assertEqual(Decimal(reopened.snapshot_liquidity(next_book.snapshot_id)[0]["shared_consumed_qty"]), Decimal("2"))
        self.assertTrue(reopened.recover()["consistent"])

    def test_bid_side_change_does_not_refresh_unchanged_ask_side(self):
        first_book = snap(1, bids=[(100, 2)], asks=[(101, 2)])
        first_order = self.ledger.create_order(request("pool-side-first", qty="2", fee="0"))
        self.ledger.process_snapshot(first_order, first_book, as_of=T0+timedelta(seconds=2))
        second_book = snap(2, bids=[(99, 5)], asks=[(101, 2)])
        self.assertNotEqual(first_book.liquidity_pool_id("bid"), second_book.liquidity_pool_id("bid"))
        self.assertEqual(first_book.liquidity_pool_id("ask"), second_book.liquidity_pool_id("ask"))
        second_order = self.ledger.create_order(request("pool-side-second", qty="1", created=T0+timedelta(seconds=1), fee="0"))
        result = self.ledger.process_snapshot(second_order, second_book, as_of=T0+timedelta(seconds=4))
        self.assertEqual(result.cumulative_filled_quantity, Decimal("0"))
        self.assertTrue(self.ledger.reconcile()["consistent"])

    def test_changed_side_book_creates_new_observed_liquidity_pool(self):
        first_book = snap(1, asks=[(101, 2)])
        first_order = self.ledger.create_order(request("pool-change-first", qty="2", fee="0"))
        self.ledger.process_snapshot(first_order, first_book, as_of=T0+timedelta(seconds=2))
        changed_book = snap(2, asks=[(101, 3)])
        self.assertNotEqual(first_book.liquidity_pool_id("ask"), changed_book.liquidity_pool_id("ask"))
        second_order = self.ledger.create_order(request("pool-change-second", qty="2", created=T0+timedelta(seconds=1), fee="0"))
        result = self.ledger.process_snapshot(second_order, changed_book, as_of=T0+timedelta(seconds=4))
        self.assertEqual(result.cumulative_filled_quantity, Decimal("2"))
        self.assertTrue(self.ledger.reconcile()["consistent"])

    def test_repeated_order_snapshot_cycle_after_commit_is_idempotent(self):
        oid = self.ledger.create_order(request(qty="2"))
        book = snap()
        def crash(stage):
            if stage == "after_commit": raise RuntimeError("simulated lost response")
        with self.assertRaisesRegex(RuntimeError, "lost response"):
            self.ledger.process_snapshot(oid, book, as_of=T0+timedelta(seconds=2), failure_hook=crash)
        replay = self.ledger.process_snapshot(oid, book, as_of=T0+timedelta(seconds=3))
        self.assertTrue(replay.duplicate)
        self.assertEqual(replay.cumulative_filled_quantity, Decimal("2"))
        self.assertEqual(len(self.ledger.get_fills(oid)), 2)
        self.assertTrue(self.ledger.recover()["consistent"])

    def test_crash_before_commit_rolls_back_fills_consumption_and_portfolio(self):
        oid = self.ledger.create_order(request(qty="2"))
        book = snap()
        before = self.ledger.portfolio()
        def crash(stage):
            if stage == "before_commit": raise RuntimeError("injected crash")
        with self.assertRaisesRegex(RuntimeError, "injected crash"):
            self.ledger.process_snapshot(oid, book, as_of=T0+timedelta(seconds=2), failure_hook=crash)
        self.assertEqual(self.ledger.get_fills(oid), [])
        self.assertEqual(self.ledger.portfolio(), before)
        liquidity=self.ledger.snapshot_liquidity(book.snapshot_id)
        self.assertTrue(all(Decimal(x["consumed_qty"]) == 0 for x in liquidity))
        retry=self.ledger.process_snapshot(oid, book, as_of=T0+timedelta(seconds=3))
        self.assertEqual(retry.cumulative_filled_quantity, Decimal("2"))
        self.assertTrue(self.ledger.recover()["consistent"])

    def test_crash_after_fill_event_before_portfolio_update_is_atomic(self):
        oid=self.ledger.create_order(request(qty="1"))
        book=snap()
        def crash(stage):
            if stage == "after_fill_event": raise RuntimeError("event committed inside uncommitted transaction")
        with self.assertRaisesRegex(RuntimeError,"uncommitted transaction"):
            self.ledger.process_snapshot(oid,book,as_of=T0+timedelta(seconds=2),failure_hook=crash)
        self.assertEqual(self.ledger.get_fills(oid),[])
        self.assertEqual(self.ledger.get_events(oid)[-1]["event_type"],"OrderAccepted")
        self.assertEqual(self.ledger.portfolio()["cash_idr"],Decimal("10000"))
        self.assertTrue(self.ledger.recover()["consistent"])

    def test_identical_replay_in_two_fresh_ledgers_is_deterministic(self):
        r=request("replay",qty="3",fee="12")
        book=snap()
        results=[]; portfolios=[]
        for name in ("left.sqlite3","right.sqlite3"):
            engine=SQLitePaperExecutionLedger(Path(self.tmp.name)/name,initial_cash_idr=Decimal("10000"))
            oid=engine.create_order(r)
            result=engine.process_snapshot(oid,book,as_of=T0+timedelta(seconds=2))
            results.append(result.as_dict()); portfolios.append({k:str(v) for k,v in engine.portfolio().items()})
        self.assertEqual(results[0],results[1])
        self.assertEqual(portfolios[0],portfolios[1])

    def test_crash_after_fill_insert_is_atomic(self):
        oid=self.ledger.create_order(request(qty="2"))
        book=snap()
        def crash(stage):
            if stage == "after_fill_insert": raise RuntimeError("crash after fill insert")
        with self.assertRaises(RuntimeError): self.ledger.process_snapshot(oid,book,as_of=T0+timedelta(seconds=2),failure_hook=crash)
        self.assertEqual(self.ledger.get_fills(oid),[])
        self.assertEqual(self.ledger.portfolio()["cash_idr"],Decimal("10000"))

    def test_ioc_partial_fill_cancels_remaining_quantity(self):
        oid=self.ledger.create_order(request(qty="5",tif=TimeInForce.IOC,fee="0"))
        result=self.ledger.process_snapshot(oid,snap(asks=[(101,2)]),as_of=T0+timedelta(seconds=2))
        self.assertEqual(result.state,OrderState.CANCELLED_SIMULATED)
        self.assertEqual(result.cumulative_filled_quantity,Decimal("2"))
        self.assertEqual(result.remaining_quantity,Decimal("3"))
        self.assertEqual(self.ledger.get_order(oid)["state"],OrderState.CANCELLED_SIMULATED)

    def test_cancel_preserves_previous_fills_and_remaining_quantity(self):
        oid=self.ledger.create_order(request(qty="5",fee="0"))
        self.ledger.process_snapshot(oid,snap(1,asks=[(101,2)]),as_of=T0+timedelta(seconds=2))
        cancelled=self.ledger.cancel_order(oid,at=T0+timedelta(seconds=3),reason="halt")
        self.assertEqual(cancelled["state"],OrderState.CANCELLED_SIMULATED)
        self.assertEqual(Decimal(cancelled["filled_qty"]),Decimal("2"))
        self.assertEqual(Decimal(cancelled["remaining_qty"]),Decimal("3"))
        self.assertEqual(len(self.ledger.get_fills(oid)),1)
        self.assertTrue(self.ledger.reconcile()["consistent"])

    def test_expiration_is_distinct_from_cancel_and_preserves_fills(self):
        oid=self.ledger.create_order(request(qty="5",expires=T0+timedelta(seconds=10),fee="0"))
        self.ledger.process_snapshot(oid,snap(1,asks=[(101,2)]),as_of=T0+timedelta(seconds=2))
        expired=self.ledger.expire_orders(as_of=T0+timedelta(seconds=11))
        self.assertEqual(expired,[oid])
        self.assertEqual(self.ledger.get_order(oid)["state"],OrderState.EXPIRED_SIMULATED)
        self.assertEqual(len(self.ledger.get_fills(oid)),1)

    def test_expired_before_match_is_not_filled(self):
        oid=self.ledger.create_order(request(qty="2",expires=T0+timedelta(seconds=2)))
        result=self.ledger.process_snapshot(oid,snap(1),as_of=T0+timedelta(seconds=3))
        self.assertEqual(result.state,OrderState.EXPIRED_SIMULATED)
        self.assertEqual(result.cumulative_filled_quantity,Decimal("0"))

    def test_stale_future_and_out_of_order_snapshots_are_not_matched(self):
        oid=self.ledger.create_order(request(qty="5"))
        stale=snap(1)
        r=self.ledger.process_snapshot(oid,stale,as_of=T0+timedelta(seconds=60))
        self.assertIn("snapshot_stale",r.reasons)
        first=self.ledger.process_snapshot(oid,snap(5),as_of=T0+timedelta(seconds=6))
        # Requested size exceeds all visible depth, leaving an open residual.
        large=self.ledger.create_order(request("large",qty="20"))
        self.ledger.process_snapshot(large,snap(5),as_of=T0+timedelta(seconds=6))
        old=self.ledger.process_snapshot(large,snap(4),as_of=T0+timedelta(seconds=7))
        self.assertIn("out_of_order_snapshot",old.reasons)
        future=self.ledger.create_order(request("future",qty="1",created=T0+timedelta(seconds=20)))
        f=self.ledger.process_snapshot(future,snap(6),as_of=T0+timedelta(seconds=10))
        self.assertIn("snapshot_timestamp_ineligible",f.reasons)

    def test_market_identity_mismatch_does_not_fill(self):
        oid=self.ledger.create_order(request(qty="1"))
        wrong=DepthSnapshot("OTHER","SOL/IDR",T0+timedelta(seconds=1),((100,1),),((101,1),))
        result=self.ledger.process_snapshot(oid,wrong,as_of=T0+timedelta(seconds=2))
        self.assertIn("snapshot_market_mismatch",result.reasons)
        self.assertEqual(result.cumulative_filled_quantity,Decimal("0"))

    def test_cash_and_fee_constraints_cap_buy_fill_without_negative_cash(self):
        ledger=SQLitePaperExecutionLedger(Path(self.tmp.name)/"small-cash.sqlite3",initial_cash_idr=Decimal("101"))
        oid=ledger.create_order(request(qty="2",fee="100"))
        result=ledger.process_snapshot(oid,snap(asks=[(100,2)]),as_of=T0+timedelta(seconds=2))
        self.assertEqual(result.cumulative_filled_quantity,Decimal("1"))
        self.assertGreaterEqual(ledger.portfolio()["cash_idr"],Decimal("0"))

    def test_sell_cannot_exceed_reconstructed_inventory(self):
        oid=self.ledger.create_order(request("sell-empty",side=Side.SELL,qty="2",fee="0"))
        result=self.ledger.process_snapshot(oid,snap(),as_of=T0+timedelta(seconds=2))
        self.assertEqual(result.cumulative_filled_quantity,Decimal("0"))
        self.assertIn("portfolio_capacity_exhausted",result.reasons)

    def test_ledger_reopen_and_recovery_reconstructs_same_portfolio(self):
        oid=self.ledger.create_order(request(qty="2"))
        self.ledger.process_snapshot(oid,snap(),as_of=T0+timedelta(seconds=2))
        before=self.ledger.portfolio()
        reopened=SQLitePaperExecutionLedger(self.db_path,initial_cash_idr=Decimal("10000"))
        self.assertEqual(reopened.portfolio(),before)
        report=reopened.recover()
        self.assertTrue(report["consistent"],report["issues"])
        self.assertEqual(report["fill_count"],2)

    def test_portfolio_realized_pnl_reconciles_buy_then_sell(self):
        buy=self.ledger.create_order(request("buy",qty="2",fee="0"))
        self.ledger.process_snapshot(buy,snap(1,asks=[(100,2)],bids=[(99,2)]),as_of=T0+timedelta(seconds=2))
        sell=self.ledger.create_order(request("sell",side=Side.SELL,qty="1",created=T0+timedelta(seconds=3),fee="0"))
        self.ledger.process_snapshot(sell,snap(4,bids=[(110,1)],asks=[(111,1)]),as_of=T0+timedelta(seconds=5))
        p=self.ledger.portfolio()
        self.assertEqual(p["sol_quantity"],Decimal("1"))
        self.assertEqual(p["realized_pnl_idr"],Decimal("10"))
        self.assertTrue(self.ledger.reconcile()["consistent"])

    def test_price_and_quantity_precision_and_limit_contracts(self):
        with self.assertRaisesRegex(ExecutionError,"quantity_step"):
            request(qty="1.1")
        with self.assertRaisesRegex(ExecutionError,"price_tick"):
            request(order_type=OrderType.LIMIT,limit="101.5")
        with self.assertRaisesRegex(ExecutionError,"must not specify"):
            request(order_type=OrderType.MARKET,limit="101")

    def test_nonzero_initial_inventory_is_refused_without_cost_basis(self):
        with self.assertRaisesRegex(ExecutionError,"cost basis"):
            SQLitePaperExecutionLedger(self.db_path,initial_cash_idr=Decimal("10000"),initial_sol=Decimal("1"))

    def test_snapshot_price_and_quantity_precision_must_match_order_contract(self):
        oid=self.ledger.create_order(request(qty="1",fee="0"))
        book=snap(asks=[(101,Decimal("1.5"))])
        result=self.ledger.process_snapshot(oid,book,as_of=T0+timedelta(seconds=2))
        self.assertIn("snapshot_precision_mismatch",result.reasons)
        self.assertEqual(result.cumulative_filled_quantity,Decimal("0"))

    def test_reconciliation_detects_portfolio_tampering(self):
        oid=self.ledger.create_order(request(qty="1",fee="0"))
        self.ledger.process_snapshot(oid,snap(asks=[(101,1)]),as_of=T0+timedelta(seconds=2))
        db=sqlite3.connect(self.db_path)
        try:
            db.execute("UPDATE paper_accounts SET cash_idr='9999' WHERE account_id='paper-default'")
            db.commit()
        finally:
            db.close()
        report=self.ledger.reconcile()
        self.assertFalse(report["consistent"])
        self.assertTrue(any("cash_idr" in issue for issue in report["issues"]))

    def test_reconciliation_detects_overconsumed_snapshot_depth(self):
        oid=self.ledger.create_order(request(qty="1",fee="0"))
        book=snap(asks=[(101,1)])
        self.ledger.process_snapshot(oid,book,as_of=T0+timedelta(seconds=2))
        db=sqlite3.connect(self.db_path)
        try:
            db.execute("UPDATE snapshot_consumption SET consumed_qty='2' WHERE snapshot_id=? AND side='ask' AND level_index=0",(book.snapshot_id,))
            db.commit()
        finally:
            db.close()
        report=self.ledger.reconcile()
        self.assertFalse(report["consistent"])
        self.assertTrue(any("over-allocated" in issue for issue in report["issues"]))

    def test_execution_events_and_fills_are_append_only(self):
        oid=self.ledger.create_order(request(qty="1",fee="0"))
        self.ledger.process_snapshot(oid,snap(asks=[(101,1)]),as_of=T0+timedelta(seconds=2))
        db=sqlite3.connect(self.db_path)
        try:
            with self.assertRaises(sqlite3.IntegrityError):
                db.execute("UPDATE fills SET quantity='0' WHERE order_id=?",(oid,))
            with self.assertRaises(sqlite3.IntegrityError):
                db.execute("DELETE FROM events WHERE order_id=?",(oid,))
            db.rollback()
        finally: db.close()
        self.assertTrue(self.ledger.reconcile()["consistent"])

    def test_sqlite_atomic_transaction_and_unique_fills(self):
        oid=self.ledger.create_order(request(qty="2"))
        self.ledger.process_snapshot(oid,snap(),as_of=T0+timedelta(seconds=2))
        db=sqlite3.connect(self.db_path)
        try:
            self.assertEqual(db.execute("PRAGMA integrity_check").fetchone()[0],"ok")
            self.assertEqual(db.execute("SELECT COUNT(*) FROM fills WHERE order_id=?",(oid,)).fetchone()[0],2)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM execution_cycles WHERE order_id=?",(oid,)).fetchone()[0],1)
        finally:
            db.close()


if __name__ == "__main__": unittest.main()