#!/usr/bin/env python3
"""Compare predeclared execution-cost scenarios on identical SOL history; offline only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

from paper_engine import BacktestConfig, backtest, read_bars_csv, resolve_quote_to_idr


def build_scenarios(base: BacktestConfig) -> list[tuple[str, str, BacktestConfig]]:
    """Fixed sensitivity scenarios; never select one automatically by historical P&L."""
    return [
        (
            "reku_pro_baseline",
            "Published Reku Pro fee components + 10 bps round-trip spread and 10 bps slippage per side (spread/slippage are assumptions).",
            replace(base, entry_fee_bps=10.555, exit_fee_bps=31.555, spread_bps_round_trip=10.0, slippage_bps_per_side=10.0),
        ),
        (
            "reku_pro_lower_friction_assumption",
            "Same published Reku Pro fees + hypothetical 5 bps round-trip spread and 5 bps slippage per side; not measured or guaranteed.",
            replace(base, entry_fee_bps=10.555, exit_fee_bps=31.555, spread_bps_round_trip=5.0, slippage_bps_per_side=5.0),
        ),
        (
            "reku_pro_fee_only_floor",
            "Mathematical lower bound using Reku Pro fee components but zero spread/slippage; not executable or realistic.",
            replace(base, entry_fee_bps=10.555, exit_fee_bps=31.555, spread_bps_round_trip=0.0, slippage_bps_per_side=0.0),
        ),
        (
            "reku_lightning_fee_only_floor",
            "Mathematical fee floor using published Lightning fee components; assumes zero quote spread/slippage, which is not realistic or verified.",
            replace(base, entry_fee_bps=0.555, exit_fee_bps=21.555, spread_bps_round_trip=0.0, slippage_bps_per_side=0.0),
        ),
    ]


def summarize_scenario(name: str, description: str, config: BacktestConfig, report: dict[str, Any]) -> dict[str, Any]:
    total_cost_bps = (
        config.entry_fee_bps + config.exit_fee_bps
        + config.spread_bps_round_trip + 2 * config.slippage_bps_per_side
    )
    return {
        "scenario": name,
        "description": description,
        "fee_source": "Reku published fee schedule" if name.startswith("reku_pro") or name.startswith("reku_lightning") else "configured fee assumption",
        "execution_friction_is_assumption": True,
        "theoretical_floor_only": "fee_only_floor" in name,
        "entry_fee_bps": config.entry_fee_bps,
        "exit_fee_bps": config.exit_fee_bps,
        "spread_bps_round_trip": config.spread_bps_round_trip,
        "slippage_bps_per_side": config.slippage_bps_per_side,
        "modeled_round_trip_cost_bps": total_cost_bps,
        "minimum_gross_move_bps_in_signal_gate": total_cost_bps + config.minimum_net_edge_bps,
        "signal_evaluations": report["signal_evaluations"],
        "signal_warmup_bars": report["signal_warmup_bars"],
        "signal_candidates": report["signal_candidates"],
        "signal_rejection_counts": report["signal_rejection_counts"],
        "closed_trades": report["closed_trades"],
        "initial_equity_idr": report["initial_equity_idr"],
        "final_equity_idr": report["final_equity_idr"],
        "net_pnl_idr": report["net_pnl_idr"],
        "net_return_pct": report["net_return_pct"],
        "win_rate_pct": report["win_rate_pct"],
        "profit_factor": report["profit_factor"],
        "max_drawdown_pct": report["max_drawdown_pct"],
        "total_fees_idr": report["total_fees_idr"],
        "total_execution_friction_idr": report["total_execution_friction_idr"],
    }


def run_sensitivity(
    csv_path: str | Path,
    quote_to_idr_rate: float | None = None,
    quote_to_idr_source: str = "",
    initial_capital_idr: float = 300_000.0,
    minimum_net_edge_bps: float = 25.0,
) -> dict[str, Any]:
    source = Path(csv_path).expanduser()
    rate, rate_source = resolve_quote_to_idr(source, quote_to_idr_rate, quote_to_idr_source)
    bars = read_bars_csv(source)
    dataset_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    base = BacktestConfig(
        initial_capital_idr=initial_capital_idr,
        quote_to_idr_rate=rate,
        quote_to_idr_source=rate_source,
        minimum_net_edge_bps=minimum_net_edge_bps,
    )
    base.validate()
    results = []
    for name, description, config in build_scenarios(base):
        report = backtest(bars, config, dataset_sha256=dataset_sha)
        results.append(summarize_scenario(name, description, config, report))
    return {
        "schema_version": "sol-sentinel-cost-sensitivity-v1",
        "mode": "OFFLINE_COST_SENSITIVITY_ONLY",
        "execution_enabled": False,
        "symbol": "SOLUSDT",
        "venue_reference": "Binance public SOLUSDT candles converted with a dated USD/IDR reference; not Reku executable prices.",
        "dataset_sha256": dataset_sha,
        "bars_1m": len(bars),
        "period_start": report["period_start"],
        "period_end": report["period_end"],
        "quote_to_idr_rate": rate,
        "quote_to_idr_source": rate_source,
        "minimum_net_edge_bps": minimum_net_edge_bps,
        "technical_parameters_frozen_across_scenarios": True,
        "scenarios": results,
        "warnings": [
            "This is a sensitivity analysis, not automatic strategy optimization. Never select a scenario only because its historical P&L is highest.",
            "The published Reku Pro fee components are sourced from https://reku.id/fees/crypto; recheck the current mode and final order preview.",
            "All spread/slippage values are assumptions. Zero-spread and zero-slippage cases are theoretical fee floors, not executable quotes.",
            "Binance SOLUSDT candles plus USD/IDR conversion are not exact Reku SOL/IDR prices.",
            "Zero trades means no evidence of profitability. A positive historical result is still not a forecast.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--quote-to-idr-rate", type=float, default=None)
    parser.add_argument("--quote-to-idr-source", default="")
    parser.add_argument("--initial-capital-idr", type=float, default=300_000.0)
    parser.add_argument("--minimum-net-edge-bps", type=float, default=25.0)
    parser.add_argument("--output", default="", help="optional JSON report path")
    args = parser.parse_args(argv)
    if not math.isfinite(args.initial_capital_idr) or args.initial_capital_idr <= 0:
        parser.error("--initial-capital-idr must be finite and positive")
    if not math.isfinite(args.minimum_net_edge_bps) or args.minimum_net_edge_bps < 0:
        parser.error("--minimum-net-edge-bps must be finite and non-negative")
    try:
        report = run_sensitivity(
            args.csv, args.quote_to_idr_rate, args.quote_to_idr_source,
            args.initial_capital_idr, args.minimum_net_edge_bps,
        )
        if args.output:
            path = Path(args.output).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        print(json.dumps({
            "mode": report["mode"], "execution_enabled": False,
            "period_start": report["period_start"], "period_end": report["period_end"],
            "bars_1m": report["bars_1m"], "dataset_sha256": report["dataset_sha256"],
            "scenarios": [{
                "scenario": s["scenario"],
                "modeled_round_trip_cost_bps": s["modeled_round_trip_cost_bps"],
                "minimum_gross_move_bps_in_signal_gate": s["minimum_gross_move_bps_in_signal_gate"],
                "signal_candidates": s["signal_candidates"],
                "net_pnl_idr": s["net_pnl_idr"],
                "net_return_pct": s["net_return_pct"],
                "closed_trades": s["closed_trades"],
                "max_drawdown_pct": s["max_drawdown_pct"],
                "signal_rejection_counts": s["signal_rejection_counts"],
            } for s in report["scenarios"]],
            "report_path": str(Path(args.output).expanduser()) if args.output else None,
            "warning": "OFFLINE SENSITIVITY ONLY; zero-spread/slippage scenarios are mathematical floors, not trading forecasts.",
        }, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"mode": "OFFLINE_COST_SENSITIVITY_ONLY", "execution_enabled": False, "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
