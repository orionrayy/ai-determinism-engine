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
from datetime import datetime, timezone

from paper_engine import BacktestConfig, backtest, read_bars_csv, resolve_quote_to_idr, resolve_input_market_metadata


def load_reku_spread_reference(path: str | Path, max_age_seconds: float = 300.0) -> tuple[float, dict[str, Any]]:
    """Read a fresh public Reku bid/ask snapshot. A PARTIAL snapshot is acceptable only if bid/ask is valid."""
    snapshot_path = Path(path).expanduser()
    if not snapshot_path.is_file():
        raise ValueError(f"Reku market snapshot not found: {snapshot_path}")
    try:
        report = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read Reku market snapshot: {exc}") from exc
    if not isinstance(report, dict) or report.get("venue") != "Reku" or report.get("pair") != "SOL/IDR":
        raise ValueError("Snapshot must be a Reku SOL/IDR public market report")
    if report.get("data_status") not in ("OK", "PARTIAL"):
        raise ValueError(f"Reku snapshot is not usable: {report.get('data_status') or 'missing data_status'}")
    direct = report.get("direct_bidask")
    if not isinstance(direct, dict):
        raise ValueError("Reku snapshot has no valid direct bid/ask data")
    try:
        spread = float(direct["spread_bps"])
        bid = float(direct["bid_idr"])
        ask = float(direct["ask_idr"])
        timestamp = str(report["captured_at_end_utc"])
        captured = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Reku snapshot has incomplete bid/ask/timestamp fields") from exc
    if captured.tzinfo is None:
        raise ValueError("Reku snapshot timestamp must be timezone-aware")
    age_seconds = (datetime.now(timezone.utc) - captured.astimezone(timezone.utc)).total_seconds()
    if age_seconds < -5.0 or age_seconds > max_age_seconds:
        raise ValueError(f"Reku spread snapshot is stale or future-dated (age_seconds={age_seconds:.1f})")
    if not all(math.isfinite(x) and x > 0 for x in (spread, bid, ask)) or ask < bid:
        raise ValueError("Reku spread snapshot contains invalid/crossed quote data")
    return spread, {
        "source": "Reku public /v2/bidask",
        "captured_at_utc": captured.astimezone(timezone.utc).isoformat(),
        "age_seconds_at_sensitivity_start": age_seconds,
        "data_status": report.get("data_status"),
        "orderbook_status": report.get("orderbook_status"),
        "bid_idr": bid,
        "ask_idr": ask,
        "spread_idr": direct.get("spread_idr"),
        "spread_bps": spread,
        "quote_is_current_at_run_time_only": True,
        "applied_as_constant_over_historical_window": True,
        "warning": "A single current spread is a sensitivity reference, not a historical spread series. Order-book depth may be unavailable.",
    }


def build_scenarios(base: BacktestConfig, observed_spread_bps: float | None = None) -> list[tuple[str, str, BacktestConfig]]:
    """Fixed sensitivity scenarios; never select one automatically by historical P&L."""
    scenarios = [
        (
            "reku_pro_baseline",
            "Published Reku Pro fee components + 10 bps round-trip spread and 10 bps slippage per side (spread/slippage are assumptions).",
            replace(base, entry_fee_bps=10.555, exit_fee_bps=31.555, spread_bps_round_trip=10.0, slippage_bps_per_side=10.0, minimum_order_notional_idr=30_000.0),
        ),
        (
            "reku_pro_lower_friction_assumption",
            "Same published Reku Pro fees + hypothetical 5 bps round-trip spread and 5 bps slippage per side; not measured or guaranteed.",
            replace(base, entry_fee_bps=10.555, exit_fee_bps=31.555, spread_bps_round_trip=5.0, slippage_bps_per_side=5.0, minimum_order_notional_idr=30_000.0),
        ),
        (
            "reku_pro_fee_only_floor",
            "Mathematical lower bound using Reku Pro fee components but zero spread/slippage; not executable or realistic.",
            replace(base, entry_fee_bps=10.555, exit_fee_bps=31.555, spread_bps_round_trip=0.0, slippage_bps_per_side=0.0, minimum_order_notional_idr=30_000.0),
        ),
        (
            "reku_lightning_fee_only_floor",
            "Mathematical fee floor using published Lightning fee components; assumes zero quote spread/slippage, which is not realistic or verified.",
            replace(base, entry_fee_bps=0.555, exit_fee_bps=21.555, spread_bps_round_trip=0.0, slippage_bps_per_side=0.0, minimum_order_notional_idr=5_000.0),
        ),
    ]
    if observed_spread_bps is not None:
        if not math.isfinite(observed_spread_bps) or observed_spread_bps < 0:
            raise ValueError("observed Reku spread must be finite and non-negative")
        scenarios.append((
            "reku_pro_current_spread_reference",
            "Published Reku Pro fees + latest direct Reku SOL/IDR bid/ask spread held constant over this historical window for sensitivity only; 10 bps slippage per side remains an assumption. Not a historical spread series.",
            replace(base, entry_fee_bps=10.555, exit_fee_bps=31.555, spread_bps_round_trip=observed_spread_bps, slippage_bps_per_side=10.0, minimum_order_notional_idr=30_000.0),
        ))
    return scenarios


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
        "minimum_order_notional_idr": config.minimum_order_notional_idr,
        "modeled_round_trip_cost_bps": total_cost_bps,
        "minimum_gross_move_bps_in_signal_gate": total_cost_bps + config.minimum_net_edge_bps,
        "signal_evaluations": report["signal_evaluations"],
        "signal_warmup_bars": report["signal_warmup_bars"],
        "signal_candidates": report["signal_candidates"],
        "post_warmup_signal_evaluations": report["post_warmup_signal_evaluations"],
        "signal_filter_rejection_events": report["signal_filter_rejection_events"],
        "signal_rejection_counts": report["signal_rejection_counts"],
        "signal_rejection_combinations": report["signal_rejection_combinations"],
        "entry_order_attempts": report["entry_order_attempts"],
        "simulated_entry_fills": report["simulated_entry_fills"],
        "entry_size_rejections": report["entry_size_rejections"],
        "entry_size_rejection_counts": report["entry_size_rejection_counts"],
        "daily_loss_lock_blocked_bars": report["daily_loss_lock_blocked_bars"],
        "closed_trades": report["closed_trades"],
        "gross_profit_idr": report["gross_profit_idr"],
        "gross_loss_idr": report["gross_loss_idr"],
        "average_win_idr": report["average_win_idr"],
        "average_loss_idr": report["average_loss_idr"],
        "expectancy_net_per_trade_idr": report["expectancy_net_per_trade_idr"],
        "average_hold_bars": report["average_hold_bars"],
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
    reku_market_json_path: str | Path | None = None,
) -> dict[str, Any]:
    source = Path(csv_path).expanduser()
    source_market = resolve_input_market_metadata(source)
    if source_market != {"venue": "Binance", "symbol": "SOLUSDT", "price_currency": "USDT"}:
        raise ValueError(
            "cost_sensitivity presets currently model the Reku fee reference on Binance SOLUSDT/USDT input only; "
            "native INDODAX SOL/IDR requires a separately declared venue-specific cost model"
        )
    rate, rate_source = resolve_quote_to_idr(source, quote_to_idr_rate, quote_to_idr_source, source_market)
    bars = read_bars_csv(source)
    dataset_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    reku_spread_reference = None
    reku_spread_error = None
    observed_spread_bps = None
    if reku_market_json_path:
        try:
            observed_spread_bps, reku_spread_reference = load_reku_spread_reference(reku_market_json_path)
        except (OSError, ValueError, RuntimeError) as exc:
            reku_spread_error = f"{type(exc).__name__}: {str(exc)[:200]}"
    base = BacktestConfig(
        initial_capital_idr=initial_capital_idr,
        quote_to_idr_rate=rate,
        quote_to_idr_source=rate_source,
        minimum_net_edge_bps=minimum_net_edge_bps,
    )
    base.validate()
    results = []
    for name, description, config in build_scenarios(base, observed_spread_bps):
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
        "reku_spread_reference": reku_spread_reference,
        "reku_spread_reference_error": reku_spread_error,
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
    parser.add_argument("--reku-market-json", default="", help="optional fresh read-only Reku SOL/IDR snapshot; used only for a clearly labeled spread sensitivity")
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
            args.reku_market_json or None,
        )
        if args.output:
            path = Path(args.output).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        print(json.dumps({
            "mode": report["mode"], "execution_enabled": False,
            "period_start": report["period_start"], "period_end": report["period_end"],
            "bars_1m": report["bars_1m"], "dataset_sha256": report["dataset_sha256"],
            "reku_spread_reference": report.get("reku_spread_reference"),
            "reku_spread_reference_error": report.get("reku_spread_reference_error"),
            "scenarios": [{
                "scenario": s["scenario"],
                "modeled_round_trip_cost_bps": s["modeled_round_trip_cost_bps"],
                "minimum_gross_move_bps_in_signal_gate": s["minimum_gross_move_bps_in_signal_gate"],
                "signal_candidates": s["signal_candidates"],
                "net_pnl_idr": s["net_pnl_idr"],
                "net_return_pct": s["net_return_pct"],
                "closed_trades": s["closed_trades"],
                "win_rate_pct": s["win_rate_pct"],
                "profit_factor": s["profit_factor"],
                "max_drawdown_pct": s["max_drawdown_pct"],
                "total_fees_idr": s["total_fees_idr"],
                "total_execution_friction_idr": s["total_execution_friction_idr"],
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
