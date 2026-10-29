import unittest

from paper_engine import BacktestConfig
from cost_sensitivity import build_scenarios


class CostSensitivityTests(unittest.TestCase):
    def test_scenario_costs_match_published_reku_components(self):
        scenarios = {name: (description, config) for name, description, config in build_scenarios(BacktestConfig())}
        pro = scenarios["reku_pro_baseline"][1]
        self.assertEqual(pro.entry_fee_bps, 10.555)
        self.assertEqual(pro.exit_fee_bps, 31.555)
        self.assertEqual(pro.entry_fee_bps + pro.exit_fee_bps, 42.11)
        self.assertEqual(pro.spread_bps_round_trip, 10.0)
        self.assertEqual(pro.slippage_bps_per_side, 10.0)

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
