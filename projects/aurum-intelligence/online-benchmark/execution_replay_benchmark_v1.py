#!/usr/bin/env python3
"""Large historical Bid/Ask replay with conservative ambiguity bounds; research only."""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import large_scale_bidask_benchmark_v1_0 as large
from quote_path_replay_v1 import BidAskBar, simulate_bracket_trade

PROTOCOL = "aurum-quote-side-economic-replay-v1"
COST_GRID_R = (0.0, 0.025, 0.05, 0.10)
PARTITIONS = (("development", 0.0, 0.70), ("calibration", 0.70, 0.80), ("final_holdout", 0.80, 1.0))


def _iso_ms(value: int) -> str:
    return datetime.fromtimestamp(int(value) / 1000, timezone.utc).isoformat()


def _bars_for_window(dataset: np.ndarray, start: int, count: int) -> list[BidAskBar]:
    if start < 0 or start + count > len(dataset):
        raise ValueError("path_window_out_of_range")
    rows = []
    for i in range(start, start + count):
        row = dataset[i]
        rows.append(BidAskBar(
            _iso_ms(row["ts"]),
            float(row["bo"]), float(row["bh"]), float(row["bl"]), float(row["bc"]),
            float(row["ao"]), float(row["ah"]), float(row["al"]), float(row["ac"]),
        ))
    return rows


def _bound_metrics(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"n": 0, "mean_r": None, "median_r": None, "q05_r": None, "cvar05_r": None,
                "win_fraction": None, "profit_factor": None}
    losses = float(-array[array < 0].sum())
    wins = float(array[array > 0].sum())
    q05 = float(np.quantile(array, 0.05))
    tail = array[array <= q05]
    return {
        "n": int(len(array)), "mean_r": float(array.mean()), "median_r": float(np.median(array)),
        "q05_r": q05, "cvar05_r": float(tail.mean()) if len(tail) else q05,
        "win_fraction": float(np.mean(array > 0)),
        "profit_factor": wins / losses if losses > 0 else (None if not wins else "no_losses_in_sample"),
    }


def _summarize(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not events:
        return {"events": 0, "ambiguous_events": 0, "pessimistic_gross": _bound_metrics([]),
                "optimistic_gross": _bound_metrics([]), "cost_sensitivity": []}
    lower = np.asarray([float(e["lower_r"]) for e in events], dtype=np.float64)
    upper = np.asarray([float(e["upper_r"]) for e in events], dtype=np.float64)
    ambiguous = sum(e["path_status"] == "ambiguous_same_bar" for e in events)
    return {
        "events": len(events), "ambiguous_events": ambiguous, "ambiguous_fraction": ambiguous / len(events),
        "pessimistic_gross": _bound_metrics(lower), "optimistic_gross": _bound_metrics(upper),
        "cost_sensitivity": [{
            "assumed_additional_round_trip_cost_r": cost,
            "pessimistic_mean_net_r": float(lower.mean() - cost),
            "optimistic_mean_net_r": float(upper.mean() - cost),
            "pessimistic_positive_after_cost_fraction": float(np.mean(lower - cost > 0)),
            "optimistic_positive_after_cost_fraction": float(np.mean(upper - cost > 0)),
            "broker_observed_cost": False,
        } for cost in COST_GRID_R],
        "semantics": "overlapping per-signal events, not a portfolio-equity or fill simulation",
    }


def run(output_dir: Path, *, min_candles: int = large.MIN_CANDLES,
        min_anchors: int = large.MIN_SCENARIO_ANCHORS) -> dict[str, Any]:
    started = time.perf_counter()
    if min_candles < large.MIN_CANDLES or min_anchors < large.MIN_SCENARIO_ANCHORS:
        raise ValueError("minimum_candles_or_anchors_may_not_be_lowered")
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded = large.load_dataset()
    dataset, months, total_bytes, metadata = loaded
    original_loader = large.load_dataset
    large.load_dataset = lambda: loaded
    try:
        large.run(output_dir, min_candles, min_anchors)
    finally:
        large.load_dataset = original_loader

    quality = large.market_quality(dataset)
    features = large.indicators(dataset)
    anchors, eligible_windows = large.candidate_anchor_indices(dataset, min_anchors, features["atr14"])
    signals = large.signals_for_anchors(dataset, anchors, features)
    if len(np.unique(dataset["ts"][anchors])) < min_anchors:
        raise ValueError("distinct_anchor_gate_failed")

    path_started = time.perf_counter()
    outcomes: dict[tuple[int, int], dict[str, Any]] = {}
    status_counts: dict[str, int] = {}
    label_crosschecks = 0
    for ordinal, anchor_value in enumerate(anchors):
        anchor = int(anchor_value)
        atr = float(features["atr14"][anchor])
        local_bars = _bars_for_window(dataset, anchor, large.HORIZON_BARS + 1)
        for direction in (1, -1):
            result = simulate_bracket_trade(
                local_bars, decision_index=0, side="long" if direction == 1 else "short",
                atr=atr, take_profit_atr=1.0, stop_loss_atr=1.0,
                horizon_bars=large.HORIZON_BARS, expected_interval_seconds=60)
            legacy = large.evaluate_path(dataset, features, anchor, direction, 1.0,
                                         horizon=large.HORIZON_BARS)
            if result["outcome_label_tp_before_sl"] != legacy["label"]:
                raise ValueError("path_label_disagreement_anchor_{}_side_{}".format(anchor, direction))
            label_crosschecks += 1
            if result["status"] == "incomplete_path_gap":
                raise ValueError("eligible_anchor_has_path_gap_" + str(anchor))
            status_counts[result["status"]] = status_counts.get(result["status"], 0) + 1
            lower = result["pnl_lower_bound_price_units_per_unit"]
            upper = result["pnl_upper_bound_price_units_per_unit"]
            if lower is None or upper is None:
                raise ValueError("path_did_not_produce_complete_pnl_bounds")
            outcomes[(ordinal, direction)] = {
                "path_status": result["status"], "label": result["outcome_label_tp_before_sl"],
                "lower_r": float(lower) / atr, "upper_r": float(upper) / atr,
                "entry_time_utc": result["entry_time_utc"], "exit_time_utc": result["exit_time_utc"],
                "fill_assumption": result["fill_assumption"],
            }

    split70, split80 = int(len(anchors) * 0.70), int(len(anchors) * 0.80)
    signal_events: list[dict[str, Any]] = []
    for ordinal, anchor_value in enumerate(anchors):
        anchor = int(anchor_value)
        split = "development" if ordinal < split70 else ("calibration" if ordinal < split80 else "final_holdout")
        for rule in large.STRATEGIES:
            direction = int(signals[rule][ordinal])
            if direction == 0:
                continue
            signal_events.append({
                "decision_ordinal": ordinal, "anchor_index": anchor,
                "decision_time_utc": _iso_ms(dataset["ts"][anchor]), "rule": rule,
                "side": "long" if direction == 1 else "short", "split": split,
                **outcomes[(ordinal, direction)],
            })

    by_strategy: dict[str, Any] = {}
    for rule in large.STRATEGIES:
        by_strategy[rule] = {"all": _summarize([e for e in signal_events if e["rule"] == rule])}
        for name, _, _ in PARTITIONS:
            by_strategy[rule][name] = _summarize([e for e in signal_events if e["rule"] == rule and e["split"] == name])
        for side in ("long", "short"):
            by_strategy[rule][side] = _summarize([e for e in signal_events if e["rule"] == rule and e["side"] == side])

    final_events = [e for e in signal_events if e["split"] == "final_holdout"]
    if label_crosschecks != len(anchors) * 2 or len(final_events) < 1000:
        raise ValueError("label_crosscheck_or_holdout_gate_failed")
    manifest_digest = hashlib.sha256(json.dumps(months, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    report = {
        "status": "DIAGNOSTIC_ONLY", "protocol_version": PROTOCOL,
        "dataset": {
            "id": large.DATASET_ID, "repository": large.REPOSITORY_URL, "resolution": "M1",
            "price_sides": ["bid", "ask"], "row_count": int(len(dataset)),
            "month_count": metadata["selected_months"], "period_start_month": metadata["first_month"],
            "period_end_month": metadata["last_month"], "download_bytes": int(total_bytes),
            "monthly_manifest_sha256": manifest_digest,
            "provider_provenance_independently_verified": False, "upstream_data_rights_verified": False,
            "provider_note": large.PROVIDER_NOTE,
        },
        "target_contract": {
            "decision_time": "after completed anchor close", "entry": "next M1 open",
            "long_entry": "Ask", "long_exit": "Bid", "short_entry": "Bid", "short_exit": "Ask",
            "target_atr": 1.0, "stop_atr": 1.0, "horizon_bars": large.HORIZON_BARS,
            "timeout": "executable-side close", "gap": "observed executable-side open",
            "same_bar_collision": "retained with pessimistic/optimistic PnL bounds; no guessed path",
        },
        "coverage": {
            "dataset_rows": int(len(dataset)), "weekday_active_nonflat_rows": int(quality["weekday_active_nonflat_rows"]),
            "eligible_contiguous_windows": int(eligible_windows), "unique_real_anchors": int(len(anchors)),
            "path_outcomes_evaluated": int(len(outcomes)), "legacy_label_crosschecks": int(label_crosschecks),
            "legacy_label_mismatches": 0, "signal_events": int(len(signal_events)),
            "final_holdout_signal_events": int(len(final_events)), "path_status_counts_by_anchor_side": status_counts,
        },
        "metrics_by_strategy": by_strategy,
        "cost_sensitivity": {
            "additional_cost_grid_r": list(COST_GRID_R),
            "interpretation": "assumed aggregate costs per event in R, not broker-observed costs",
            "broker_observed_cost": False, "net_account_currency_pnl_produced": False,
        },
        "limitations": [
            "Event-level metrics overlap; no portfolio position sizing, simultaneous exposure or equity curve.",
            "Bid/Ask OHLC cannot resolve tick order inside a same-bar TP/SL collision; bounds remain explicit.",
            "Touch fills are assumptions; actual queue, partial fill, no-fill, latency, impact and slippage are not modeled.",
            "The additional-cost grid is hypothetical. It is not a measured broker profile.",
            "Source origin, intended-use rights, timezone semantics and broker equivalence remain unverified.",
        ],
        "decision_output": "NO_TRADE", "validated_model": False, "validated_for_profitability": False,
        "deployment_calibrated": False, "live_execution_authorized": False,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "path_replay_runtime_seconds": round(time.perf_counter() - path_started, 3),
    }
    (output_dir / "execution_replay_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        "# Execution-side XAUUSD research replay", "",
        "Status: DIAGNOSTIC_ONLY. Decision: NO_TRADE. Not a broker-executable backtest or profitability claim.", "",
        "- Dataset rows: {:,}; months: {} ({}–{}).".format(len(dataset), metadata["selected_months"], metadata["first_month"], metadata["last_month"]),
        "- Unique real anchors: {:,}; direction paths: {:,}; signal events: {:,}; final holdout events: {:,}.".format(len(anchors), len(outcomes), len(signal_events), len(final_events)),
        "- Existing path-label cross-checks: {:,}; mismatches: 0.".format(label_crosschecks), "",
        "## Per-signal event bounds", "",
        "Ambiguous same-bar collisions remain in the economic sample with explicit bounds.", "",
        "| Strategy | Split | Events | Ambiguous | Mean gross R lower | Mean gross R upper | Mean net R lower @ assumed 0.05R cost | Mean net R upper @ assumed 0.05R cost |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rule in large.STRATEGIES:
        for name, _, _ in PARTITIONS:
            item = by_strategy[rule][name]
            lower, upper = item["pessimistic_gross"]["mean_r"], item["optimistic_gross"]["mean_r"]
            assumed = next((x for x in item["cost_sensitivity"] if x["assumed_additional_round_trip_cost_r"] == 0.05), None)
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                rule, name, item["events"], item["ambiguous_events"],
                "n/a" if lower is None else "{:.6f}".format(lower),
                "n/a" if upper is None else "{:.6f}".format(upper),
                "n/a" if assumed is None else "{:.6f}".format(assumed["pessimistic_mean_net_r"]),
                "n/a" if assumed is None else "{:.6f}".format(assumed["optimistic_mean_net_r"]),
            ))
    lines += ["", "## Interpretation limits", "",
        "- Gross R uses executable quote sides; spread is embedded and not deducted twice.",
        "- Commission, swap, slippage, impact, latency and actual fills are not measured for the intended venue.",
        "- Metrics are per-signal event, not capital-aware portfolio results; nearby signals overlap.",
        "- Source provenance and rights are unverified; preserve NO_TRADE.", ""]
    (output_dir / "execution_replay_report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-candles", type=int, default=large.MIN_CANDLES)
    parser.add_argument("--min-anchors", type=int, default=large.MIN_SCENARIO_ANCHORS)
    args = parser.parse_args()
    try:
        report = run(args.output_dir, min_candles=args.min_candles, min_anchors=args.min_anchors)
        print(json.dumps({
            "status": report["status"], "dataset_rows": report["coverage"]["dataset_rows"],
            "anchors": report["coverage"]["unique_real_anchors"], "path_outcomes": report["coverage"]["path_outcomes_evaluated"],
            "label_crosschecks": report["coverage"]["legacy_label_crosschecks"],
            "signal_events": report["coverage"]["signal_events"], "final_holdout_events": report["coverage"]["final_holdout_signal_events"],
            "decision_output": report["decision_output"], "runtime_seconds": report["runtime_seconds"],
            "path_replay_runtime_seconds": report["path_replay_runtime_seconds"],
        }, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)[:500],
                          "decision_output": "NO_TRADE", "live_execution_authorized": False}, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
