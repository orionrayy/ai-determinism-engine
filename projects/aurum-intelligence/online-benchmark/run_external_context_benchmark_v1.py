#!/usr/bin/env python3
"""Run a bounded, real-data XAUUSD + external-context ablation; never routes trades."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from external_context_schema_v1 import AcquisitionResult, SourceManifest, validate_observations
from cftc_cot_adapter_v1 import acquire_cot_year
from gdelt_gkg_adapter_v1 import acquire_historical_gkg, archive_coverage_intervals, summarize_archive_coverage
from bls_release_adapter_v1 import acquire_bls_series
from fred_alfred_adapter_v1 import acquire_fred
from point_in_time_features_v1 import build_asof_features
from external_context_benchmark_v1 import run_context_ablation_benchmark

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import large_scale_bidask_benchmark_v1_0 as price_benchmark  # noqa: E402

CONTEXT_TARGET = 100_000
PRICE_CANDIDATE_ANCHORS = 20_000
PRICE_FEATURE_KEYS = (
    "return_1", "return_3", "return_6", "return_12", "momentum_6",
    "atr_fraction", "spread_fraction", "spread", "hour_sin", "hour_cos",
    "weekday_sin", "weekday_cos", "volatility_12",
)
PREREGISTERED_GDELT_START = datetime(2023, 1, 1, tzinfo=timezone.utc)
PREREGISTERED_GDELT_END = datetime(2025, 6, 30, 23, 59, tzinfo=timezone.utc)


def scale_gate(real_observations: int, target: int = CONTEXT_TARGET) -> dict[str, Any]:
    """Scale status depends only on counted, real unique source observations."""
    if isinstance(real_observations, bool) or not isinstance(real_observations, int) or real_observations < 0:
        raise ValueError("real_observation_count_must_be_nonnegative_integer")
    scale_met = real_observations >= target
    return {
        "status": "PASS" if scale_met else "NOT_MET",
        "observations": real_observations,
        "target": target,
        "passed": bool(scale_met),
        "synthetic_or_duplicated_observations_added": 0,
    }


def _acquire_sources(
    decision_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[Any], dict[str, Any], dict[str, str], int, tuple[tuple[datetime, datetime], ...]]:
    decision_times = sorted(row["decision_time"] for row in decision_rows)
    if len(decision_times) < 100:
        raise ValueError("insufficient_decision_rows_for_context_sampling")
    dev_stop = int(len(decision_times) * 0.70)
    cal_stop = int(len(decision_times) * 0.80)
    split_boundaries = (decision_times[dev_stop], decision_times[cal_stop])
    block_centers = (
        decision_times[int(len(decision_times) * 0.35)],
        decision_times[int(len(decision_times) * 0.75)],
        decision_times[int(len(decision_times) * 0.90)],
    )
    if not (PREREGISTERED_GDELT_START < split_boundaries[0]
            < split_boundaries[1] < PREREGISTERED_GDELT_END):
        raise ValueError("price_context_split_boundaries_outside_preregistered_gdelt_window")
    if decision_times[0] < PREREGISTERED_GDELT_START or decision_times[-1] > PREREGISTERED_GDELT_END:
        raise ValueError("price_decision_times_outside_preregistered_gdelt_window")
    observations: list[Any] = []
    details: dict[str, Any] = {}
    states: dict[str, str] = {}
    cot_results = [acquire_cot_year(year) for year in (2023, 2024, 2025)]
    cot_obs = [item for result in cot_results for item in result.observations]
    observations.extend(cot_obs)
    cot_statuses = [result.status for result in cot_results]
    if cot_obs and all(status == "PASS" for status in cot_statuses):
        states["cot"] = "PASS"
    elif cot_obs:
        states["cot"] = "PARTIAL"
    else:
        states["cot"] = "FAILED"
    details["cftc_cot"] = {
        "status": states["cot"], "years_requested": [2023, 2024, 2025],
        "observations": len(cot_obs),
        "subresults": [{"status": r.status, "count": len(r.observations), "error_code": r.error_code,
                        "warnings": list(r.warnings)} for r in cot_results],
        "payload_sha256": sorted({m.payload_sha256 for r in cot_results for m in r.manifests}),
        "rights_status": "review-required", "provenance_status": "official-host; upstream independence unverified",
    }

    gdelt_result = acquire_historical_gkg(
        PREREGISTERED_GDELT_START, PREREGISTERED_GDELT_END,
        target_records=CONTEXT_TARGET, max_archives=480, aggregate_cap=2 * 1024 * 1024 * 1024,
        sampling_design="split_stratified_contiguous_blocks",
        split_boundaries=split_boundaries, block_centers=block_centers,
    )
    gdelt_obs = list(gdelt_result.observations)
    observations.extend(gdelt_obs)
    states["news"] = gdelt_result.status
    def warning_metric(name: str) -> int:
        prefix = name + "="
        for warning in gdelt_result.warnings:
            if warning.startswith(prefix):
                try:
                    return int(warning[len(prefix):])
                except ValueError:
                    return 0
        return 0

    details["gdelt_gkg"] = {
        "status": gdelt_result.status, "observations_real_unique": len(gdelt_obs),
        "target_observations": CONTEXT_TARGET, "error_code": gdelt_result.error_code,
        "warnings": list(gdelt_result.warnings),
        "archive_payload_sha256": sorted({m.payload_sha256 for m in gdelt_result.manifests}),
        "archive_count": len(gdelt_result.manifests),
        "discovered_archive_count": warning_metric("discovered_archives"),
        "selected_archive_count": warning_metric("sampled_archives"),
        "processed_archive_count": warning_metric("processed_sampled_archives"),
        "aggregate_bytes_downloaded": warning_metric("aggregate_bytes"),
        "requested_aggregate_byte_cap": 2 * 1024 * 1024 * 1024,
        "rights_status": "review-required", "provenance_status": "official-host; rights/upstream lineage not independently verified",
        "article_body_persisted": False,
        "sampling_design": "split_stratified_contiguous_blocks",
        "max_archives_requested": 480,
        "split_boundaries_utc": [value.isoformat() for value in split_boundaries],
        "block_centers_utc": [value.isoformat() for value in block_centers],
    }
    gdelt_real_count = len(gdelt_obs)
    gdelt_coverage = archive_coverage_intervals(gdelt_result.manifests)
    gdelt_time_coverage = summarize_archive_coverage(
        gdelt_coverage, PREREGISTERED_GDELT_START, PREREGISTERED_GDELT_END
    )
    gdelt_archive_window_complete = any(
        warning == "archive_window_complete=true" for warning in gdelt_result.warnings
    )
    details["gdelt_gkg"]["observed_archive_batch_intervals"] = len(gdelt_coverage)
    details["gdelt_gkg"]["archive_window_complete"] = gdelt_archive_window_complete
    details["gdelt_gkg"]["archive_time_coverage"] = gdelt_time_coverage
    details["gdelt_gkg"]["coverage_semantics"] = (
        "only successfully acquired 15-minute archive batches; gaps remain uncovered; "
        "a batch contributes only after its conservative availability boundary"
    )

    # Public BLS v1 observations are acquired but remain descriptive until matched to
    # an independently sourced official release-time calendar; observation period != publish time.
    bls_result = acquire_bls_series(["CUUR0000SA0"], 2016, 2025)
    bls_obs = list(bls_result.observations)
    observations.extend(bls_obs)
    details["bls"] = {
        "status": bls_result.status, "observations": len(bls_obs), "error_code": bls_result.error_code,
        "release_calendar_status": "NOT_PROVIDED_DESCRIPTIVE_ONLY",
        "causal_feature_status": "UNKNOWN",
        "warnings": ["No external release calendar was supplied; BLS observation periods are not publication times."],
        "payload_sha256": [m.payload_sha256 for m in bls_result.manifests],
    }

    fred_result = acquire_fred({
        "series_id": "GOLDAMGBD228NLBM", "observation_start": "2023-01-01",
        "observation_end": "2025-06-30", "realtime_start": "2023-01-01",
        "realtime_end": "2026-10-17",
    })
    fred_obs = list(fred_result.observations)
    observations.extend(fred_obs)
    details["fred_alfred"] = {
        "status": fred_result.status, "observations": len(fred_obs),
        "error_code": fred_result.error_code, "warnings": list(fred_result.warnings),
        "payload_sha256": sorted({m.payload_sha256 for m in fred_result.manifests}),
        "api_key_value_reported": False,
    }
    states["macro"] = "PASS" if fred_obs else "UNKNOWN"
    states["fred"] = fred_result.status
    return observations, details, states, gdelt_real_count, gdelt_coverage


def _make_price_anchor_rows(
    dataset: dict[str, Any], indicator: dict[str, Any], anchors: Any
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    import numpy as np

    close = indicator["mid_close"]
    atr = indicator["atr14"]
    returns = np.full(len(close), np.nan, dtype=np.float64)
    returns[1:] = close[1:] / close[:-1] - 1.0
    bid_close, ask_close = dataset["bc"], dataset["ac"]
    timestamps = dataset["ts"]
    rows: list[dict[str, Any]] = []
    outcomes: dict[str, int] = {
        "candidate_anchors": int(len(anchors)), "invalid_path": 0,
        "ambiguous_same_bar_excluded": 0, "matured_binary_labels": 0,
    }
    for anchor_value in anchors:
        i = int(anchor_value)
        label = price_benchmark.evaluate_path(
            dataset, indicator, i, direction=1, target_atr_multiple=1.0,
            horizon=price_benchmark.HORIZON_BARS,
        )
        if label["label"] is None:
            outcomes["ambiguous_same_bar_excluded"] += 1
            continue
        if label["status"] == "invalid_atr" or label["bars_to_outcome"] is None:
            outcomes["invalid_path"] += 1
            continue
        if i + int(label["bars_to_outcome"]) > len(close) - 1:
            outcomes["invalid_path"] += 1
            continue
        timepoint = datetime.fromtimestamp(float(timestamps[i]) / 1000.0, tz=timezone.utc)
        mid = float(close[i])
        if not math.isfinite(mid) or mid <= 0:
            outcomes["invalid_path"] += 1
            continue
        past = [float(returns[i - k]) for k in (1, 3, 6, 12)]
        if not all(math.isfinite(v) for v in past):
            outcomes["invalid_path"] += 1
            continue
        volatility_values = [float(v) for v in returns[i-12:i] if math.isfinite(float(v))]
        volatility = statistics.pstdev(volatility_values) if len(volatility_values) >= 10 else 0.0
        spread = max(0.0, float(ask_close[i]) - float(bid_close[i]))
        atr_fraction = float(atr[i]) / mid
        hour = timepoint.hour + timepoint.minute / 60.0
        weekday = timepoint.weekday()
        row = {
            "decision_id": f"xauusd-m1-anchor-{i}",
            "decision_time": timepoint,
            "decision_index": i,
            "target_exit_index": i + int(label["bars_to_outcome"]),
            "label": int(label["label"]),
            "label_mature": True,
            "return_1": past[0], "return_3": past[1], "return_6": past[2], "return_12": past[3],
            "momentum_6": float(close[i] / close[i-6] - 1.0),
            "atr_fraction": atr_fraction,
            "spread": spread,
            "spread_fraction": spread / mid,
            "hour_sin": math.sin(2.0 * math.pi * hour / 24.0),
            "hour_cos": math.cos(2.0 * math.pi * hour / 24.0),
            "weekday_sin": math.sin(2.0 * math.pi * weekday / 7.0),
            "weekday_cos": math.cos(2.0 * math.pi * weekday / 7.0),
            "volatility_12": volatility,
            "market_regime": "high_volatility" if atr_fraction >= 0.001 else "low_volatility",
        }
        rows.append(row)
        outcomes["matured_binary_labels"] += 1
    return rows, outcomes


def run(output_path: Path) -> dict[str, Any]:
    dataset, _months_report, total_bytes, price_meta = price_benchmark.load_dataset()
    market_quality = price_benchmark.market_quality(dataset)
    indicators = price_benchmark.indicators(dataset)
    anchors, eligible_count = price_benchmark.candidate_anchor_indices(
        dataset, PRICE_CANDIDATE_ANCHORS, indicators["atr14"]
    )
    decision_rows, label_audit = _make_price_anchor_rows(dataset, indicators, anchors)
    # Sampling centers use the fixed chronological price-row cohort, not context values.
    observations, source_details, source_states, gdelt_count, gdelt_coverage = _acquire_sources(decision_rows)
    audit = validate_observations(observations)
    source_details["price_dataset"] = {
        "repository": price_benchmark.REPOSITORY,
        "dataset_id": price_benchmark.DATASET_ID,
        "rows": int(len(dataset)), "paired_valid_bid_ask_candles": int(len(dataset)),
        "weekday_active_nonflat_rows": int(market_quality["weekday_active_nonflat_rows"]),
        "months": price_meta["selected_months"],
        "source_audit_sha256": {"bid": price_meta["bid_audit_sha256"], "ask": price_meta["ask_audit_sha256"]},
        "total_download_bytes": total_bytes,
        "eligible_contiguous_atr_windows": eligible_count,
        "provenance_independently_verified": False,
        "rights_verified": False,
    }
    state_for_features = {
        "cot": source_states.get("cot", "FAILED"),
        "macro": source_states.get("macro", "UNKNOWN"),
        "news": source_states.get("news", "FAILED"),
    }
    enriched = build_asof_features(
        decision_rows, observations, source_statuses=state_for_features,
        news_coverage_intervals=gdelt_coverage,
    )
    benchmark = run_context_ablation_benchmark(
        enriched, context_observation_count=gdelt_count,
        target_context_records=CONTEXT_TARGET,
        price_feature_keys=PRICE_FEATURE_KEYS,
        source_statuses=state_for_features,
        bootstrap_replicates=250, block_size=100, seed=20261017,
    )
    scale = scale_gate(gdelt_count)
    feature_coverage = benchmark.get("coverage", {})
    temporal_feature_coverage = {}
    for family in ("cot", "macro", "news", "news_1h", "news_24h"):
        family_coverage = feature_coverage.get(family, {})
        fraction = float(family_coverage.get("coverage_fraction", 0.0))
        temporal_feature_coverage[family] = {
            **family_coverage,
            "status": "PASS" if fraction >= 1.0 else "PARTIAL" if fraction > 0.0 else "NOT_MET",
        }
    source_completeness_status = (
        "PASS" if source_details["gdelt_gkg"]["archive_window_complete"]
        else "PARTIAL" if gdelt_coverage else "NOT_MET"
    )
    report = {
        "benchmark_id": "AURUM-POINT-IN-TIME-EXTERNAL-CONTEXT-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        # PASS is a pipeline/scale-gate state only; source completeness and usable
        # feature coverage are reported independently and can remain PARTIAL/NOT_MET.
        "status": "PASS" if scale["passed"] and source_details["gdelt_gkg"]["status"] == "PASS" else "PARTIAL",
        "status_semantics": (
            "PASS means bounded acquisition/evaluation completed and the unique-observation scale gate passed; "
            "it does not imply full archive-window completeness or eligible news/macro features"
        ),
        "software_execution_status": "PASS",
        "data_acceptance_status": scale["status"],
        "source_acquisition_status": source_details["gdelt_gkg"]["status"],
        "source_completeness_status": source_completeness_status,
        "temporal_feature_coverage": temporal_feature_coverage,
        "research_only": True, "decision_gate": "NO_TRADE",
        "deployment_calibrated": False, "validated": False, "live_signal_enabled": False,
        "scale_gate": scale,
        "source_details": source_details,
        "source_states_for_features": state_for_features,
        "canonical_observation_audit": {
            "row_count": audit.row_count, "valid_count": audit.valid_count,
            "invalid_reasons": dict(audit.invalid_reasons), "duplicate_id_count": audit.duplicate_id_count,
            "conflicting_duplicate_count": audit.conflicting_duplicate_count,
            "unknown_availability_count": audit.unknown_availability_count,
            "conservative_proxy_count": audit.conservative_proxy_count, "status": audit.status,
        },
        "price_label_audit": label_audit,
        "ablation_benchmark": benchmark,
        "artifact_policy": "Metadata, counts, hashes, features and metrics only; no raw price bars, article text, API keys, or raw high-volume datasets.",
        "limitations": [
            "The GDELT event count is a bounded, deterministic archive sample, not proof of complete news coverage.",
            "BLS features are descriptive-only until matched to an authoritative release-time calendar.",
            "CFTC release time uses a conservative following-week Wednesday proxy unless verified evidence is supplied.",
            "Price source provider rights and broker equivalence remain unverified.",
            "No result constitutes an edge, profitability, whale-level performance, or #1-accuracy claim.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = run(args.output)
        print(json.dumps({
            "status": result["status"], "decision_gate": result["decision_gate"],
            "real_gdelt_unique_observations": result["scale_gate"]["observations"],
            "scale_gate_status": result["scale_gate"]["status"],
            "source_acquisition_status": result["source_acquisition_status"],
            "source_completeness_status": result["source_completeness_status"],
            "gdelt_time_coverage": result["source_details"]["gdelt_gkg"]["archive_time_coverage"],
            "news_feature_coverage": result["temporal_feature_coverage"]["news"],
            "news_1h_feature_coverage": result["temporal_feature_coverage"]["news_1h"],
            "news_24h_feature_coverage": result["temporal_feature_coverage"]["news_24h"],
            "gdelt_sampling_design": result["source_details"]["gdelt_gkg"]["sampling_design"],
            "gdelt_archive_count": result["source_details"]["gdelt_gkg"]["archive_count"],
            "gdelt_selected_archive_count": result["source_details"]["gdelt_gkg"]["selected_archive_count"],
            "gdelt_processed_archive_count": result["source_details"]["gdelt_gkg"]["processed_archive_count"],
            "gdelt_aggregate_bytes_downloaded": result["source_details"]["gdelt_gkg"]["aggregate_bytes_downloaded"],
            "price_rows": result["source_details"]["price_dataset"]["rows"],
            "scored_real_labels": result["price_label_audit"]["matured_binary_labels"],
            "ablations": {k: v.get("status") for k, v in result["ablation_benchmark"]["ablations"].items()},
            "report_path": str(args.output),
        }, indent=2))
        # Data scale can be NOT_MET while software/report execution completes.
        return 0
    except Exception as exc:
        # Do not print exception text for provider/network paths that might carry secrets.
        print(json.dumps({"status":"FAILED","error_code":type(exc).__name__}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
