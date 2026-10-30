#!/usr/bin/env python3
"""Chronological 60/20/20 evaluation of a fixed SOL strategy; no parameter tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from paper_engine import (
    BacktestConfig, Bar, backtest, read_bars_csv, resolve_quote_to_idr,
    resolve_input_market_metadata, resolve_fee_inputs,
)


DEFAULT_RATIOS = (0.60, 0.20, 0.20)


def split_chronologically(
    bars: Sequence[Bar],
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    minimum_segment_bars: int = 600,
) -> dict[str, list[Bar]]:
    if len(ratios) != 3 or any(not math.isfinite(r) or r <= 0 for r in ratios):
        raise ValueError("exactly three positive finite ratios are required")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("split ratios must sum to 1")
    if minimum_segment_bars < 100:
        raise ValueError("minimum segment size must be at least 100 bars")
    n = len(bars)
    first = int(n * ratios[0])
    second = int(n * (ratios[0] + ratios[1]))
    segments = {
        "development": list(bars[:first]),
        "validation": list(bars[first:second]),
        "out_of_sample": list(bars[second:]),
    }
    for name, segment in segments.items():
        if len(segment) < minimum_segment_bars:
            raise ValueError(
                f"{name} segment has {len(segment)} bars; at least {minimum_segment_bars} are required"
            )
        for previous, current in zip(segment, segment[1:]):
            if current.open_time_ms - previous.open_time_ms != 60_000:
                raise ValueError(f"{name} segment contains a timestamp gap")
    # Also ensure boundaries are chronological and disjoint.
    if not (
        segments["development"][-1].open_time_ms < segments["validation"][0].open_time_ms
        < segments["out_of_sample"][0].open_time_ms
    ):
        raise ValueError("segments overlap or are not ordered chronologically")
    return segments


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "period_start", "period_end", "bars_1m", "initial_equity_idr",
        "final_equity_idr", "net_pnl_idr", "net_return_pct",
        "signal_evaluations", "signal_warmup_bars", "signal_candidates", "post_warmup_signal_evaluations",
        "signal_filter_rejection_events", "signal_rejection_counts", "signal_rejection_combinations",
        "entry_order_attempts", "simulated_entry_fills", "entry_size_rejections", "entry_size_rejection_counts",
        "daily_loss_lock_blocked_bars", "positions_opened", "positions_closed", "closed_trades",
        "gross_profit_idr", "gross_loss_idr", "average_win_idr", "average_loss_idr",
        "expectancy_net_per_trade_idr", "gross_pnl_before_exchange_fees_idr", "average_hold_bars",
        "win_rate_pct", "profit_factor", "max_drawdown_pct",
        "total_fees_idr", "total_execution_friction_idr", "daily_loss_lock_events",
    )
    return {key: report[key] for key in keys}


def evaluate(
    bars: Sequence[Bar],
    config: BacktestConfig,
    dataset_sha256: str,
    minimum_segment_bars: int = 600,
) -> dict[str, Any]:
    config.validate()
    segments = split_chronologically(bars, minimum_segment_bars=minimum_segment_bars)
    summaries: dict[str, Any] = {}
    for name, segment in segments.items():
        segment_identity = hashlib.sha256(
            f"{dataset_sha256}:{name}:{segment[0].open_time_ms}:{segment[-1].close_time_ms}".encode()
        ).hexdigest()
        report = backtest(segment, config, dataset_sha256=segment_identity)
        summaries[name] = summarize(report)
        summaries[name]["dataset_segment_sha256"] = segment_identity
        summaries[name]["closed_trade_details"] = report["trades"]
    return {
        "schema_version": "sol-sentinel-chronological-evaluation-v1",
        "mode": "OFFLINE_CHRONOLOGICAL_EVALUATION",
        "execution_enabled": False,
        "symbol": config.input_symbol,
        "input_venue": config.input_venue,
        "venue_reference": (
            "Native INDODAX SOL/IDR public candles; no FX conversion applied; modeled fills and costs are assumptions"
            if config.input_price_currency.upper() == "IDR"
            else "Binance public SOLUSDT candles converted by a dated USD/IDR reference; not Reku or another venue's executable prices"
        ),
        "source_quote_currency": config.input_price_currency.upper(), "simulation_price_currency": "IDR",
        "quote_to_idr_rate": config.quote_to_idr_rate, "quote_to_idr_source": config.quote_to_idr_source,
        "quote_to_idr_assumption": (
            "Native IDR prices are already denominated in the simulator currency; FX conversion is disabled."
            if config.input_price_currency.upper() == "IDR"
            else "A dated USD/IDR reference proxies USDT/IDR; stablecoin premium/discount and execution-venue prices are not modeled."
        ),
        "fee_model_name": config.fee_model_name,
        "fee_model_source": config.fee_model_source,
        "source_dataset_sha256": dataset_sha256,
        "split_ratios": {"development": 0.60, "validation": 0.20, "out_of_sample": 0.20},
        "segment_capital_model": "Each segment starts independently at configured initial equity; returns are not compounded across segments.",
        "config": asdict(config),
        "segments": summaries,
        "warnings": [
            "No parameters are automatically optimized. Keep this configuration frozen when interpreting out-of-sample results.",
            "The development and validation periods are not a promise of future results.",
            "Out-of-sample results are meaningful only with adequate trade count and representative regimes.",
            "Fees, slippage, missed fills, latency and venue mismatch can materially reduce live outcomes.",
            "This evaluation is offline only; it does not connect to an account or place orders.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--quote-to-idr-rate", type=float, default=None, help="manual FX override; otherwise use capture sidecar")
    parser.add_argument("--quote-to-idr-source", default="", help="source label for a manual FX override")
    parser.add_argument("--input-venue", default=None, help="required source venue when capture sidecar is missing")
    parser.add_argument("--input-symbol", default=None, help="required source pair when capture sidecar is missing")
    parser.add_argument("--input-price-currency", choices=("USDT", "IDR"), default=None, help="required source quote currency when capture sidecar is missing")
    parser.add_argument("--fee-model", choices=("reku-pro-reference", "indodax-custom"), default=None)
    parser.add_argument("--fee-source", default="", help="label for explicitly supplied venue fee rates")
    parser.add_argument("--minimum-segment-bars", type=int, default=600)
    parser.add_argument("--initial-capital-idr", type=float, default=300_000.0)
    parser.add_argument("--risk-fraction", type=float, default=0.005)
    parser.add_argument("--max-exposure-fraction", type=float, default=0.50)
    parser.add_argument("--daily-loss-fraction", type=float, default=0.03)
    parser.add_argument("--entry-fee-bps", type=float, default=None)
    parser.add_argument("--exit-fee-bps", type=float, default=None)
    parser.add_argument("--spread-bps-round-trip", type=float, default=None)
    parser.add_argument("--slippage-bps-per-side", type=float, default=None)
    parser.add_argument("--minimum-net-edge-bps", type=float, default=25.0)
    parser.add_argument("--minimum-relative-volume", type=float, default=1.5)
    parser.add_argument("--minimum-order-notional-idr", type=float, default=None)
    parser.add_argument("--stop-atr-multiple", type=float, default=1.0)
    parser.add_argument("--target-atr-multiple", type=float, default=1.5)
    parser.add_argument("--maximum-hold-bars", type=int, default=30)
    parser.add_argument("--output", default="", help="optional JSON report path")
    args = parser.parse_args(argv)
    try:
        source = Path(args.csv).expanduser()
        source_market = resolve_input_market_metadata(
            source, args.input_venue, args.input_symbol, args.input_price_currency
        )
        quote_to_idr_rate, quote_to_idr_source = resolve_quote_to_idr(
            source, args.quote_to_idr_rate, args.quote_to_idr_source, source_market
        )
        fee_inputs = resolve_fee_inputs(
            source_market,
            fee_model=args.fee_model,
            fee_source=args.fee_source,
            entry_fee_bps=args.entry_fee_bps,
            exit_fee_bps=args.exit_fee_bps,
            spread_bps_round_trip=args.spread_bps_round_trip,
            slippage_bps_per_side=args.slippage_bps_per_side,
            minimum_order_notional_idr=args.minimum_order_notional_idr,
        )
        config = BacktestConfig(
            initial_capital_idr=args.initial_capital_idr,
            quote_to_idr_rate=quote_to_idr_rate,
            quote_to_idr_source=quote_to_idr_source,
            input_venue=source_market["venue"],
            input_symbol=source_market["symbol"],
            input_price_currency=source_market["price_currency"],
            fee_model_name=fee_inputs["fee_model_name"],
            fee_model_source=fee_inputs["fee_model_source"],
            risk_fraction=args.risk_fraction,
            max_exposure_fraction=args.max_exposure_fraction, daily_loss_fraction=args.daily_loss_fraction,
            entry_fee_bps=fee_inputs["entry_fee_bps"], exit_fee_bps=fee_inputs["exit_fee_bps"],
            spread_bps_round_trip=fee_inputs["spread_bps_round_trip"],
            slippage_bps_per_side=fee_inputs["slippage_bps_per_side"], minimum_net_edge_bps=args.minimum_net_edge_bps,
            minimum_relative_volume=args.minimum_relative_volume,
            minimum_order_notional_idr=fee_inputs["minimum_order_notional_idr"],
            stop_atr_multiple=args.stop_atr_multiple,
            target_atr_multiple=args.target_atr_multiple, maximum_hold_bars=args.maximum_hold_bars,
        )
        config.validate()
        dataset_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        bars = read_bars_csv(source)
        report = evaluate(bars, config, dataset_sha, args.minimum_segment_bars)
        summary = {
            "mode": report["mode"], "execution_enabled": False,
            "source_dataset_sha256": dataset_sha, "split_ratios": report["split_ratios"],
            "segments": {name: summarize_like for name, summarize_like in (
                (key, {k: val for k, val in value.items() if k != "closed_trade_details"})
                for key, value in report["segments"].items()
            )},
            "warning": "OFFLINE ONLY. No parameter tuning, no live orders, no future return implied.",
        }
        output = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False)
        if args.output:
            out_path = Path(args.output).expanduser()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(output, encoding="utf-8")
            summary["report_path"] = str(out_path)
        print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"mode": "OFFLINE_CHRONOLOGICAL_EVALUATION", "execution_enabled": False, "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
