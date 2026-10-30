import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from paper_engine import BacktestConfig
from cost_sensitivity import build_scenarios, load_reku_spread_reference, run_sensitivity


class CostSensitivityTests(unittest.TestCase):
    def test_reku_cost_presets_reject_native_indodax_idr_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "native.csv"
            csv_path.write_text("open_time_ms,close_time_ms,open,high,low,close,volume\\n", encoding="utf-8")
            Path(str(csv_path) + ".meta.json").write_text(json.dumps({
                "venue": "INDODAX",
                "pair": "SOL/IDR",
                "input_price_currency": "IDR",
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "native INDODAX SOL/IDR requires a separately declared venue-specific cost model"):
                run_sensitivity(csv_path, quote_to_idr_rate=1.0)

    def test_scenario_costs_match_published_reku_components(self):
        scenarios = {name: (description, config) for name, description, config in build_scenarios(BacktestConfig())}
        pro = scenarios["reku_pro_baseline"][1]
        self.assertEqual(pro.entry_fee_bps, 10.555)
        self.assertEqual(pro.exit_fee_bps, 31.555)
        self.assertEqual(pro.entry_fee_bps + pro.exit_fee_bps, 42.11)
        self.assertEqual(pro.spread_bps_round_trip, 10.0)
        self.assertEqual(pro.slippage_bps_per_side, 10.0)
        self.assertEqual(pro.minimum_order_notional_idr, 30_000.0)
        lightning = scenarios["reku_lightning_fee_only_floor"][1]
        self.assertEqual(lightning.minimum_order_notional_idr, 5_000.0)

    def test_scenarios_are_predeclared_and_not_auto_selected(self):
        scenarios = build_scenarios(BacktestConfig())
        names = [name for name, _, _ in scenarios]
        self.assertEqual(names, [
            "reku_pro_baseline",
            "reku_pro_lower_friction_assumption",
            "reku_pro_fee_only_floor",
            "reku_lightning_fee_only_floor",
        ])
        self.assertEqual(len(set(names)), len(names))
        for _, description, _ in scenarios:
            self.assertTrue(description)

    def test_fresh_partial_reku_snapshot_provides_labeled_spread_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reku.json"
            path.write_text(json.dumps({
                "venue": "Reku",
                "pair": "SOL/IDR",
                "data_status": "PARTIAL",
                "orderbook_status": "ERROR",
                "captured_at_end_utc": datetime.now(timezone.utc).isoformat(),
                "direct_bidask": {
                    "bid_idr": 1_840_000.0,
                    "ask_idr": 1_845_000.0,
                    "spread_idr": 5_000.0,
                    "mid_idr": 1_842_500.0,
                    "spread_bps": (5_000.0 / 1_842_500.0) * 10_000.0,
                },
            }), encoding="utf-8")
            spread, meta = load_reku_spread_reference(path)
        self.assertAlmostEqual(spread, (5_000.0 / 1_842_500.0) * 10_000.0)
        self.assertEqual(meta["orderbook_status"], "ERROR")
        self.assertTrue(meta["applied_as_constant_over_historical_window"])

        scenarios = {name: (description, config) for name, description, config in build_scenarios(BacktestConfig(), spread)}
        measured = scenarios["reku_pro_current_spread_reference"][1]
        self.assertAlmostEqual(measured.spread_bps_round_trip, spread)
        self.assertEqual(measured.slippage_bps_per_side, 10.0)
        self.assertEqual(measured.entry_fee_bps + measured.exit_fee_bps, 42.11)
        total_cost = measured.entry_fee_bps + measured.exit_fee_bps + measured.spread_bps_round_trip + 2 * measured.slippage_bps_per_side
        self.assertAlmostEqual(total_cost, 42.11 + spread + 20.0)

    def test_reku_spread_snapshot_rejects_stale_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reku.json"
            path.write_text(json.dumps({
                "venue": "Reku", "pair": "SOL/IDR", "data_status": "OK",
                "captured_at_end_utc": (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat(),
                "direct_bidask": {"bid_idr": 100.0, "ask_idr": 101.0, "spread_bps": 99.5},
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stale or future-dated"):
                load_reku_spread_reference(path)

    def test_technical_strategy_parameters_are_constant_across_cost_scenarios(self):
        scenarios = build_scenarios(BacktestConfig(minimum_net_edge_bps=25.0))
        for _, _, cfg in scenarios:
            self.assertEqual(cfg.minimum_net_edge_bps, 25.0)
            self.assertEqual(cfg.risk_fraction, 0.005)
            self.assertEqual(cfg.stop_atr_multiple, 1.0)
            self.assertEqual(cfg.target_atr_multiple, 1.5)
            self.assertEqual(cfg.minimum_relative_volume, 1.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
