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

    def test_cross_venue_reference_uses_explicit_fx_proxy(self):
        result = market.compare_reference(1_815_800.0, 100.0, 18_158.0)
        self.assertEqual(result["binance_usdt_usd_proxy_idr"], 1_815_800.0)
        self.assertAlmostEqual(result["reku_vs_binance_proxy_bps"], 0.0)
        self.assertIn("not an executable", result["warning"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
