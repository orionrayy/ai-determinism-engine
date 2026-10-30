from __future__ import annotations

import sqlite3
from contextlib import closing
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlparse

import indodax_microstructure as m

NOW = datetime(2026, 10, 30, 4, 0, tzinfo=timezone.utc)


def payload(bids=None, asks=None):
    return {"buy": bids or [["100", "2"], ["99", "3"], ["98", "4"]],
            "sell": asks or [["102", "1"], ["103", "2"], ["104", "3"]]}


class ParseTests(unittest.TestCase):
    def test_valid_schema_and_spread(self):
        s = m.parse_depth(payload(), NOW)
        self.assertEqual(s.bid, Decimal("100")); self.assertEqual(s.ask, Decimal("102"))
        self.assertEqual(s.mid, Decimal("101")); self.assertEqual(len(s.bids), 3)
        self.assertEqual(s.exchange_timestamp_ms, None)

    def test_timestamp_is_preserved_as_ms(self):
        self.assertEqual(m.parse_depth({**payload(), "timestamp": 1800000000000}, NOW).exchange_timestamp_ms, 1800000000000)

    def test_zero_spread_locked_book_is_valid(self):
        self.assertEqual(m.parse_depth(payload([[100, 1]], [[100, 2]]), NOW).spread_bps, 0)

    def test_crossed_book_rejected(self):
        with self.assertRaisesRegex(m.MicrostructureError, "crossed"):
            m.parse_depth(payload([[103, 1]], [[102, 1]]), NOW)

    def test_empty_sides_and_non_object_rejected(self):
        for raw in ({}, {"buy": [], "sell": [[102,1]]}, {"buy": [[100,1]], "sell": []}, []):
            with self.subTest(raw=raw), self.assertRaises(m.MicrostructureError): m.parse_depth(raw, NOW)

    def test_malformed_and_non_finite_levels_rejected(self):
        for raw in (payload([[100]], [[102,1]]), payload([[0,1]], [[102,1]]), payload([[100,0]], [[102,1]]),
                    payload([[100,"NaN"]], [[102,1]]), payload([[True,1]], [[102,1]])):
            with self.subTest(raw=raw), self.assertRaises(m.MicrostructureError): m.parse_depth(raw, NOW)

    def test_unsorted_and_duplicate_levels_rejected(self):
        for raw in (payload([[99,1],[100,1]], [[102,1]]), payload([[100,1],[100,2]], [[102,1]]),
                    payload([[100,1]], [[103,1],[102,1]]), payload([[100,1]], [[102,1],[102,2]])):
            with self.subTest(raw=raw), self.assertRaises(m.MicrostructureError): m.parse_depth(raw, NOW)

    def test_invalid_timestamp_and_limit_rejected(self):
        for bad in ("NaN", True, 1.5, -1):
            with self.subTest(ts=bad), self.assertRaises(m.MicrostructureError): m.parse_depth({**payload(),"timestamp":bad}, NOW)
        with self.assertRaises(m.MicrostructureError): m.parse_depth(payload(), NOW.replace(tzinfo=None))
        with self.assertRaises(m.MicrostructureError): m.parse_depth(payload(), NOW, max_levels=0)


class ClientTests(unittest.TestCase):
    def test_exact_read_only_public_endpoint(self):
        calls=[]; client=m.DepthClient(transport=lambda url, timeout: (calls.append(url) or payload()), attempts=1)
        self.assertEqual(client.snapshot().ask, Decimal("102"))
        self.assertEqual(calls, [m.ENDPOINT]); self.assertEqual(urlparse(calls[0]).scheme, "https")

    def test_timeout_and_attempt_configuration_is_bounded(self):
        for kw in ({"timeout":0},{"attempts":4},{"interval_seconds":0.5}):
            with self.subTest(kw=kw), self.assertRaises(m.MicrostructureError): m.DepthClient(**kw)

    def test_bad_response_fails_closed(self):
        client=m.DepthClient(transport=lambda *_: {"buy":[],"sell":[]}, attempts=1)
        with self.assertRaises(m.MicrostructureError): client.snapshot()


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.path=Path(self.tmp.name)/"depth.sqlite3"; self.store=m.DepthStore(self.path)

    def tearDown(self): self.tmp.cleanup()

    def test_append_stores_all_levels(self):
        key, inserted=self.store.append(m.parse_depth(payload(), NOW)); self.assertTrue(inserted)
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM observations").fetchone()[0],1)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM levels WHERE observation_id=?",(key,)).fetchone()[0],6)

    def test_identical_snapshot_is_idempotent(self):
        s=m.parse_depth(payload(),NOW); a=self.store.append(s); b=self.store.append(s)
        self.assertTrue(a[1]); self.assertFalse(b[1]); self.assertEqual(a[0],b[0])
        self.assertEqual(self.store.summary()["observation_count"],1)

    def test_same_book_at_different_times_is_kept(self):
        self.store.append(m.parse_depth(payload(),NOW)); self.store.append(m.parse_depth(payload(),NOW+timedelta(seconds=10)))
        self.assertEqual(self.store.summary()["observation_count"],2)

    def test_attempts_and_latency_are_recorded(self):
        self.store.attempt(NOW,12); self.store.attempt(NOW,45,TimeoutError("timeout"))
        s=self.store.summary()["attempts"]
        self.assertEqual((s["total"],s["ok"],s["errors"]),(2,1,1)); self.assertEqual(s["mean_duration_ms"],28.5)

    def test_spread_and_depth_distribution(self):
        self.store.append(m.parse_depth(payload(),NOW))
        self.store.append(m.parse_depth(payload([[99,2]],[[103,1]]),NOW+timedelta(seconds=10)))
        s=self.store.summary(250)
        self.assertEqual(s["observation_count"],2); self.assertIsNotNone(s["spread_bps"]["median"])
        self.assertIsNotNone(s["spread_bps"]["p95_nearest_rank"]); self.assertIsNotNone(s["bid_quote_depth_idr"]["mean"])

    def test_empty_archive_does_not_invent_metrics(self):
        s=self.store.summary(); self.assertEqual(s["observation_count"],0); self.assertIsNone(s["spread_bps"]["mean"])
        self.assertIsNone(s["ask_quote_depth_idr"]["median"])

    def test_reopen_is_durable(self):
        self.store.append(m.parse_depth(payload(),NOW)); reopened=m.DepthStore(self.path)
        self.assertEqual(reopened.summary()["observation_count"],1)

    def test_memory_database_rejected_for_persistence_contract(self):
        with self.assertRaises(m.MicrostructureError): m.DepthStore(":memory:")


class CollectionTests(unittest.TestCase):
    def test_collect_records_observations_and_runs_finite_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=m.DepthStore(Path(tmp)/"d.sqlite3"); calls=[]
            client=m.DepthClient(transport=lambda *_: (calls.append(1) or payload()),attempts=1)
            result=m.collect(client,store,samples=3,interval_seconds=1)
            self.assertEqual((len(calls),result["new_observations_this_run"],result["attempts"]["ok"]),(3,3,3))
            self.assertTrue(result["read_only"]); self.assertFalse(result["orders_enabled"])

    def test_collection_stops_after_consecutive_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=m.DepthStore(Path(tmp)/"d.sqlite3"); client=m.DepthClient(transport=lambda *_: {},attempts=1)
            r=m.collect(client,store,samples=5,interval_seconds=1,max_errors=2)
            self.assertEqual(r["attempts"]["errors"],2); self.assertEqual(r["consecutive_errors_at_end"],2)

    def test_bounds_reject_invalid_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=m.DepthStore(Path(tmp)/"d.sqlite3"); client=m.DepthClient(transport=lambda *_:payload())
            for kw in ({"samples":0,"interval_seconds":10},{"samples":1,"interval_seconds":0.1}):
                with self.subTest(kw=kw), self.assertRaises(m.MicrostructureError): m.collect(client,store,**kw)


if __name__ == "__main__": unittest.main()
