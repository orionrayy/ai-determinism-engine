import unittest

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
            raise AssertionError("unexpected Reku endpoint " + path)

        from unittest.mock import patch
        with patch.object(market, "request_json", side_effect=fake_request):
            report = market.fetch_snapshot("SOL")
        self.assertEqual(report["data_status"], "OK")
        self.assertEqual(report["orderbook_status"], "OK")
        self.assertEqual(report["orderbook_endpoint_used"], "orderbookall")
        self.assertAlmostEqual(report["orderbook"]["spread_bps"], (2.0 / 101.0) * 10_000.0, places=8)
        self.assertAlmostEqual(report["direct_bidask"]["spread_bps"], (2.0 / 101.0) * 10_000.0, places=8)

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
            raise AssertionError("unexpected Reku endpoint " + path)

        from unittest.mock import patch
        with patch.object(market, "request_json", side_effect=fake_request):
            report = market.fetch_snapshot("SOL")
        self.assertEqual(report["data_status"], "PARTIAL")
        self.assertEqual(report["orderbook_status"], "ERROR")
        self.assertIsNotNone(report["direct_bidask"])
        self.assertIsNone(report["orderbook"])

    def test_cross_venue_reference_uses_explicit_fx_proxy(self):
        result = market.compare_reference(1_815_800.0, 100.0, 18_158.0)
        self.assertEqual(result["binance_usdt_usd_proxy_idr"], 1_815_800.0)
        self.assertAlmostEqual(result["reku_vs_binance_proxy_bps"], 0.0)
        self.assertIn("not an executable", result["warning"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
