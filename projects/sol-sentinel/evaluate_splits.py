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

from paper_engine import BacktestConfig, Bar, backtest, read_bars_csv


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
        "final_equity_idr", "net_pnl_idr", "net_return_pct", "closed_trades",
        "win_rate_pct", "profit_factor", "max_drawdown_pct", "total_fees_idr",
        "daily_loss_lock_events",
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
        "symbol": "SOLUSDT",
        "venue_reference": "Binance public spot candles; not Reku SOL/IDR executable prices",
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
    parser.add_argument("--minimum-segment-bars", type=int, default=600)
    parser.add_argument("--initial-capital-idr", type=float, default=300_000.0)
    parser.add_argument("--risk-fraction", type=float, default=0.005)
    parser.add_argument("--max-exposure-fraction", type=float, default=0.50)
    parser.add_argument("--daily-loss-fraction", type=float, default=0.03)
    parser.add_argument("--entry-fee-bps", type=float, default=10.555)
    parser.add_argument("--exit-fee-bps", type=float, default=31.555)
    parser.add_argument("--spread-bps-round-trip", type=float, default=10.0)
    parser.add_argument("--slippage-bps-per-side", type=float, default=10.0)
    parser.add_argument("--minimum-net-edge-bps", type=float, default=25.0)
    parser.add_argument("--minimum-relative-volume", type=float, default=1.5)
    parser.add_argument("--stop-atr-multiple", type=float, default=1.0)
    parser.add_argument("--target-atr-multiple", type=float, default=1.5)
    parser.add_argument("--maximum-hold-bars", type=int, default=30)
    parser.add_argument("--output", default="", help="optional JSON report path")
    args = parser.parse_args(argv)
    config = BacktestConfig(
        initial_capital_idr=args.initial_capital_idr, risk_fraction=args.risk_fraction,
        max_exposure_fraction=args.max_exposure_fraction, daily_loss_fraction=args.daily_loss_fraction,
        entry_fee_bps=args.entry_fee_bps, exit_fee_bps=args.exit_fee_bps,
        spread_bps_round_trip=args.spread_bps_round_trip, slippage_bps_per_side=args.slippage_bps_per_side, minimum_net_edge_bps=args.minimum_net_edge_bps,
        minimum_relative_volume=args.minimum_relative_volume, stop_atr_multiple=args.stop_atr_multiple,
        target_atr_multiple=args.target_atr_multiple, maximum_hold_bars=args.maximum_hold_bars,
    )
    try:
        config.validate()
        source = Path(args.csv).expanduser()
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
