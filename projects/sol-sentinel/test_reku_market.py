import io
import time
import unittest
import urllib.error

import reku_market as market


class RekuMarketParserTests(unittest.TestCase):
    def test_find_coin_from_documented_result_envelope(self):
        payload = {"result": [
            {"id": 7, "accountcode": "BTC", "accountname": "Bitcoin"},
            {"id": 81, "accountcode": "SOL", "accountname": "Solana", "enablebuy": 1, "enablesell": 1},
        ]}
        coin = market.find_coin(payload, "SOL")
        self.assertEqual(coin["id"], 81)
        self.assertEqual(market.row_symbol(coin), "SOL")

    def test_find_coin_fails_closed_on_ambiguous_ids(self):
        payload = [
            {"id": 81, "accountcode": "SOL"},
            {"id": 82, "accountcode": "SOL"},
        ]
        with self.assertRaisesRegex(ValueError, "ambiguous IDs"):
            market.find_coin(payload, "SOL")

    def test_orderbook_sorts_bid_and_ask_levels(self):
        book = market.parse_orderbook({
            "b": [[10.0, 100.0, 0.1], [20.0, 101.0, 0.2]],
            "s": [[30.0, 103.0, 0.3], [40.0, 102.0, 0.4]],
        })
        self.assertEqual(book["bids"][0]["price_idr"], 101.0)
        self.assertEqual(book["asks"][0]["price_idr"], 102.0)

    def test_spread_metrics(self):
        result = market.spread_metrics(99.0, 101.0)
        self.assertEqual(result["mid_idr"], 100.0)
        self.assertEqual(result["spread_idr"], 2.0)
        self.assertEqual(result["spread_bps"], 200.0)

    def test_crossed_quotes_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "crossed"):
            market.spread_metrics(101.0, 100.0)

    def test_parse_chart_payload_sorts_deduplicates_and_parses_ohlcv(self):
        now_ms = 1_800_000_000_000
        payload = [
            [now_ms - 120_000, 100.0, 101.0, 99.0, 102.0, 10.0, 1010.0],
            [now_ms - 60_000, 101.0, 102.0, 100.0, 103.0, 11.0, 1122.0],
            [now_ms - 60_000, 101.0, 102.5, 100.0, 104.0, 12.0, 1230.0],
            [now_ms, 102.5, 103.0, 101.0, 104.0, 13.0, 1339.0],
        ]
        report = market.parse_chart_payload(payload, interval_minutes=1, now_ms=now_ms)
        self.assertEqual(report["data_status"], "OK")
        self.assertEqual(report["points_returned"], 3)
        self.assertEqual(report["last_bar"]["timestamp_ms"], now_ms)
        self.assertEqual(report["last_bar"]["close_idr"], 103.0)
        self.assertEqual(report["last_bar"]["volume_idr"], 1339.0)

    def test_parse_chart_payload_uses_interval_specific_freshness(self):
        now_ms = 1_800_000_000_000
        payload = [[now_ms - 181_000, 100.0, 101.0, 99.0, 102.0, 10.0, 1010.0]]
        one_minute = market.parse_chart_payload(payload, interval_minutes=1, now_ms=now_ms)
        five_minutes = market.parse_chart_payload(payload, interval_minutes=5, now_ms=now_ms)
        self.assertEqual(one_minute["data_status"], "STALE")
        self.assertEqual(one_minute["allowed_age_seconds"], 180)
        self.assertEqual(five_minutes["data_status"], "OK")
        self.assertEqual(five_minutes["allowed_age_seconds"], 900)

    def test_parse_chart_payload_rejects_nonpositive_interval(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            market.parse_chart_payload([], interval_minutes=0, now_ms=1_800_000_000_000)

    def test_fetch_chart_probe_preserves_stale_5m_fallback_reason(self):
        now_ms = int(time.time() * 1000)

        def fake_request(path, params=None, timeout=8.0):
            self.assertEqual(path, "/chart")
            self.assertIn(params["f"], (5, 15))
            return [[now_ms - 1_000_000, 100.0, 101.0, 99.0, 102.0, 10.0, 1010.0]]

        from unittest.mock import patch
        with patch.object(market, "request_json", side_effect=fake_request):
            report, error = market.fetch_chart_probe(51)
        self.assertIsNone(error)
        self.assertEqual(report["interval_minutes"], 15)
        self.assertTrue(report["fallback_used"])
        self.assertEqual(len(report["prior_attempt_errors"]), 1)
        self.assertIn("f=5: chart data stale", report["prior_attempt_errors"][0])

    def test_request_json_includes_http_error_body_for_diagnostics(self):
        err = urllib.error.HTTPError(
            "https://api.reku.id/v2/orderbook?id=51",
            400,
            "Bad Request",
            {},
            io.BytesIO(b'{"message":"query rejected by endpoint"}'),
        )
        from unittest.mock import patch
        with patch.object(market.urllib.request, "urlopen", side_effect=err):
            with self.assertRaisesRegex(RuntimeError, "HTTP 400.*query rejected by endpoint"):
                market.request_json("/orderbook", {"id": 51})

    def test_parse_chart_payload_rejects_payload_without_valid_bars(self):
        with self.assertRaisesRegex(ValueError, "no valid OHLCV bars"):
            market.parse_chart_payload([["bad timestamp", 100, 101, 99, 102, 10, 1010]], interval_minutes=1, now_ms=1_800_000_000_000)

    def test_snapshot_falls_back_to_documented_all_orderbooks_endpoint(self):
        def fake_request(path, params=None, timeout=8.0):
            if path == "/coins":
                return {"result": [{"id": 51, "accountcode": "SOL", "accountname": "Solana", "enablebuy": 1, "enablesell": 1}]}
            if path == "/price":
                return [{"id": 51, "cd": "SOL", "c": 101.0, "o": 100.0, "h": 102.0, "l": 99.0, "v": 1000, "cp": 1.0}]
            if path == "/bidask":
                return [{"code": "SOL", "bid": 100.0, "ask": 102.0}]
            if path == "/orderbook":
                raise RuntimeError("HTTP 400 on scoped order book")
            if path == "/orderbookall":
                return {"b": [[1000.0, 100.0, 10.0]], "s": [[1020.0, 102.0, 10.0]]}
            if path == "/chart":
                stamp = int(time.time()) - 60
                return [[stamp, 100.0, 101.0, 99.0, 102.0, 10.0, 1010.0]]
            raise AssertionError("unexpected Reku endpoint " + path)

        from unittest.mock import patch
        with patch.object(market, "request_json", side_effect=fake_request):
            report = market.fetch_snapshot("SOL")
        self.assertEqual(report["data_status"], "OK")
        self.assertEqual(report["orderbook_status"], "OK")
        self.assertEqual(report["orderbook_endpoint_used"], "orderbookall")
        self.assertEqual(report["orderbook_query_used"], {"symbol": "SOL_IDR"})
        self.assertAlmostEqual(report["orderbook"]["spread_bps"], (2.0 / 101.0) * 10_000.0, places=8)
        self.assertAlmostEqual(report["direct_bidask"]["spread_bps"], (2.0 / 101.0) * 10_000.0, places=8)
        self.assertEqual(report["chart_status"], "OK")
        self.assertEqual(report["chart_probe"]["last_bar"]["close_idr"], 101.0)

    def test_snapshot_marks_partial_if_both_orderbook_endpoints_fail(self):
        def fake_request(path, params=None, timeout=8.0):
            if path == "/coins":
                return {"result": [{"id": 51, "accountcode": "SOL", "accountname": "Solana"}]}
            if path == "/price":
                return [{"id": 51, "cd": "SOL", "c": 101.0}]
            if path == "/bidask":
                return [{"code": "SOL", "bid": 100.0, "ask": 102.0}]
            if path in ("/orderbook", "/orderbookall"):
                raise RuntimeError("book endpoint unavailable")
            if path == "/chart":
                stamp = int(time.time()) - 60
                return [[stamp, 100.0, 101.0, 99.0, 102.0, 10.0, 1010.0]]
            raise AssertionError("unexpected Reku endpoint " + path)

        from unittest.mock import patch
        with patch.object(market, "request_json", side_effect=fake_request):
            report = market.fetch_snapshot("SOL")
        self.assertEqual(report["data_status"], "PARTIAL")
        self.assertEqual(report["orderbook_status"], "ERROR")
        self.assertIsNotNone(report["direct_bidask"])
        self.assertIsNone(report["orderbook"])

    def test_chart_row_parser_uses_documented_ohlcv_field_order(self):
        now_ms = int(time.time() * 1000)
        payload = [[now_ms / 1000.0, 100.0, 101.0, 99.0, 102.0, 10.0, 1010.0]]
        report = market.parse_chart_payload(payload, interval_minutes=1, now_ms=now_ms)
        self.assertEqual(report["data_status"], "OK")
        self.assertEqual(report["last_bar"]["open_idr"], 100.0)
        self.assertEqual(report["last_bar"]["close_idr"], 101.0)
        self.assertEqual(report["last_bar"]["low_idr"], 99.0)
        self.assertEqual(report["last_bar"]["high_idr"], 102.0)
        with self.assertRaisesRegex(ValueError, "no valid OHLCV"):
            market.parse_chart_payload([["bad"]], interval_minutes=1, now_ms=now_ms)

    def test_cross_venue_reference_uses_explicit_fx_proxy(self):
        result = market.compare_reference(1_815_800.0, 100.0, 18_158.0)
        self.assertEqual(result["binance_usdt_usd_proxy_idr"], 1_815_800.0)
        self.assertAlmostEqual(result["reku_vs_binance_proxy_bps"], 0.0)
        self.assertIn("not an executable", result["warning"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
