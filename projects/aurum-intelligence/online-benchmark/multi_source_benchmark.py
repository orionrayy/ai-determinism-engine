#!/usr/bin/env python3
"""Read-only multi-source XAUUSD data audit and benchmark. PASS is not trading approval."""
import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from datetime import timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import ci_benchmark as core
import barrier_oos_benchmark as barrier_benchmark

USER_AGENT = "Aurum-Intelligence/multi-source-benchmark"
CASE_COUNT = 50
SCENARIO_ANCHORS = 3
SOURCES = (
    {
        "id": "getdata_xauusd_m5",
        "repository": "getdata-finance/xauusd-5m-ohlcv-metals-historical-data",
        "url": "https://raw.githubusercontent.com/getdata-finance/xauusd-5m-ohlcv-metals-historical-data/main/XAUUSD_5m.csv",
        "license_url": "https://github.com/getdata-finance/xauusd-5m-ohlcv-metals-historical-data/blob/main/LICENSE",
        "terms_url": "https://www.getdata.finance/documentation",
        "license_status": "GitHub repository advertises MIT; GetData's published standard licence for purchased data is purchaser-only and restricts embedding raw files in paid products; how the terms relate to this public sample is unresolved",
        "provenance_claim": "GetData documentation describes broker CFD/cash-style mid-quotes from FXCM and Capital.com; linkage to this exact GitHub sample has not been independently verified",
        "quote_side_claim": "mid_quote_claimed_in_provider_documentation; exact sample quote-side not independently verified",
        "commercial_use_approved": False,
        "underlying_data_rights_verified": False,
        "broker_provenance_verified": False,
        "quote_side_verified": False,
        "timezone_semantics_verified": False,
        "semantics": "community XAUUSD M5 OHLCV; source/quote semantics not independently verified for this exact sample",
        "interval_seconds": 300, "naive_timezone": None, "min_rows": 10000,
        "max_bytes": 16 * 1024 * 1024, "overlap_group": "m5",
    },
    {
        "id": "sai_community_xauusd_m5",
        "repository": "Sai310421/xauusd-data",
        "url": "https://raw.githubusercontent.com/Sai310421/xauusd-data/main/csv/XAUUSD/XAUUSD_M5_2026Q1Q2.csv",
        "license_url": "https://github.com/Sai310421/xauusd-data/blob/main/LICENSE",
        "terms_url": "https://github.com/Sai310421/xauusd-data",
        "license_status": "Repository advertises MIT; underlying broker-feed terms, redistribution scope and commercial rights not independently verified",
        "provenance_claim": "README says 2026 Q1-Q2 bars are aggregated from public broker feeds; specific upstream publishers/feed lineage not independently verified",
        "quote_side_claim": "mid_quote_claimed_in_repository_README; exact source calculations not independently verified",
        "commercial_use_approved": False,
        "underlying_data_rights_verified": False,
        "broker_provenance_verified": False,
        "quote_side_verified": False,
        "timezone_semantics_verified": False,
        "semantics": "community mid-quote OHLCV; timestamp values are naive in CSV and interpreted as UTC based on README claim",
        "interval_seconds": 300, "naive_timezone": "UTC", "min_rows": 10000,
        "max_bytes": 2 * 1024 * 1024, "overlap_group": "m5",
    },
    {
        "id": "sai_community_xauusd_m1",
        "repository": "Sai310421/xauusd-data",
        "url": "https://raw.githubusercontent.com/Sai310421/xauusd-data/main/csv/XAUUSD/XAUUSD_M1_2026Q1Q2.csv",
        "license_url": "https://github.com/Sai310421/xauusd-data/blob/main/LICENSE",
        "terms_url": "https://github.com/Sai310421/xauusd-data",
        "license_status": "Repository advertises MIT; underlying broker-feed terms, redistribution scope and commercial rights not independently verified",
        "provenance_claim": "Same community publisher as the M5 sample; specific upstream publishers/feed lineage not independently verified",
        "quote_side_claim": "mid_quote_claimed_in_repository_README; exact source calculations not independently verified",
        "commercial_use_approved": False,
        "underlying_data_rights_verified": False,
        "broker_provenance_verified": False,
        "quote_side_verified": False,
        "timezone_semantics_verified": False,
        "semantics": "community mid-quote OHLCV; timestamp values are naive in CSV and interpreted as UTC based on README claim",
        "interval_seconds": 60, "naive_timezone": "UTC", "min_rows": 50000,
        "max_bytes": 8 * 1024 * 1024, "overlap_group": None,
    },
)


def source_use_eligibility(spec):
    """Separate successful diagnostics from data-use, model, and deployment permissions."""
    return {
        "technical_benchmark": "DIAGNOSTIC_ONLY",
        "source_rights_review": (
            "VERIFIED" if spec.get("underlying_data_rights_verified", False)
            else "BLOCKED_PENDING_REVIEW"
        ),
        "commercial_use_approved": bool(spec.get("commercial_use_approved", False)),
        "model_training_for_commercial_service": (
            "APPROVED" if spec.get("commercial_use_approved", False)
            and spec.get("underlying_data_rights_verified", False)
            else "BLOCKED_PENDING_WRITTEN_RIGHTS_CLEARANCE"
        ),
        "production_feed_approved": (
            "APPROVED" if spec.get("commercial_use_approved", False)
            and spec.get("underlying_data_rights_verified", False)
            and spec.get("broker_provenance_verified", False)
            and spec.get("quote_side_verified", False)
            and spec.get("timezone_semantics_verified", False)
            else "BLOCKED_PENDING_PROVENANCE_AND_RIGHTS"
        ),
        "provenance_claim": spec.get("provenance_claim", "UNVERIFIED"),
        "quote_side_claim": spec.get("quote_side_claim", "UNVERIFIED"),
        "underlying_data_rights_verified": bool(spec.get("underlying_data_rights_verified", False)),
        "broker_provenance_verified": bool(spec.get("broker_provenance_verified", False)),
        "quote_side_verified": bool(spec.get("quote_side_verified", False)),
        "timezone_semantics_verified": bool(spec.get("timezone_semantics_verified", False)),
        "caveat": "Repository visibility or a repository-level licence claim does not by itself establish rights to the underlying broker data.",
    }


def build_readiness_gates(results, overlap, pipeline_status):
    """Return explicit downstream blockers; never infer trading readiness from CI success."""
    compatibility = overlap.get("close_relative_delta", {}).get("compatibility_status", "UNKNOWN")
    provenance_verified = bool(results) and all(
        item["source"].get("broker_provenance_verified", False)
        and item["source"].get("timezone_semantics_verified", False)
        and item["source"].get("quote_side_verified", False)
        for item in results
    )
    rights_verified = bool(results) and all(
        item["source"].get("underlying_data_rights_verified", False)
        and item["source"].get("commercial_use_approved", False)
        for item in results
    )
    independent = overlap.get("source_independence_verified", False)
    blockers = []
    if compatibility != "COMPARABLE":
        blockers.append("CROSS_SOURCE_PRICE_LEVEL_DIVERGENCE")
    if not independent:
        blockers.append("CROSS_SOURCE_INDEPENDENCE_UNVERIFIED")
    if not provenance_verified:
        blockers.append("SOURCE_PROVENANCE_TIMEZONE_OR_QUOTE_SIDE_UNVERIFIED")
    if not rights_verified:
        blockers.append("UNDERLYING_DATA_RIGHTS_OR_COMMERCIAL_USE_UNVERIFIED")
    return {
        "pipeline_status": pipeline_status,
        "diagnostic_benchmark_status": "PASS" if pipeline_status == "PASS" else "FAIL",
        "source_comparability_gate": "PASS" if compatibility == "COMPARABLE" else "BLOCKED",
        "cross_source_confirmation_gate": "PASS" if compatibility == "COMPARABLE" and independent and provenance_verified else "BLOCKED",
        "model_training_for_commercial_service": "PASS" if rights_verified and provenance_verified else "BLOCKED",
        "commercial_product_use": "PASS" if rights_verified and provenance_verified else "BLOCKED",
        "production_readiness": "PASS" if rights_verified and provenance_verified and compatibility == "COMPARABLE" and independent else "BLOCKED",
        "live_order_execution": "BLOCKED",
        "reason_codes": blockers,
        "interpretation": "CI PASS demonstrates execution of diagnostics only; it does not grant data rights or authorize model training, cross-source confirmation, commercial use, or production trading.",
    }


class SourceError(ValueError):
    pass


def fetch_source(spec, opener=urlopen):
    prefix = "https://raw.githubusercontent.com/" + spec["repository"] + "/"
    if not spec["url"].startswith(prefix):
        raise SourceError("url_not_allowlisted:" + spec["id"])
    request = Request(spec["url"], headers={"User-Agent": USER_AGENT})
    last_error = None
    for attempt in range(3):
        try:
            with opener(request, timeout=60) as response:
                if not response.geturl().startswith(prefix):
                    raise SourceError("redirect_outside_allowlist:" + spec["id"])
                length = response.headers.get("Content-Length")
                if length:
                    try:
                        declared_length = int(length)
                    except (TypeError, ValueError) as exc:
                        raise SourceError("invalid_content_length:" + spec["id"]) from exc
                    if declared_length > spec["max_bytes"]:
                        raise SourceError("content_length_over_limit:" + spec["id"])
                payload = response.read(spec["max_bytes"] + 1)
                if not payload:
                    raise SourceError("empty_payload:" + spec["id"])
                if len(payload) > spec["max_bytes"]:
                    raise SourceError("payload_over_limit:" + spec["id"])
                return payload
        except SourceError:
            raise
        except (OSError, URLError, HTTPError, TimeoutError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.25 * (attempt + 1))
    raise RuntimeError("fetch_failed:" + spec["id"] + ":" + type(last_error).__name__)


def normalize_utc(rows, naive_timezone):
    for index, row in enumerate(rows):
        stamp = row["timestamp"]
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            if naive_timezone != "UTC":
                raise SourceError("naive_timestamp_without_explicit_timezone_policy")
            stamp = stamp.replace(tzinfo=timezone.utc)
        else:
            stamp = stamp.astimezone(timezone.utc)
        row["timestamp"] = stamp
        if index and stamp <= rows[index - 1]["timestamp"]:
            raise SourceError("timestamps_not_strictly_increasing_after_normalization")
    return rows


def percentile(values, q):
    if not values:
        return None
    return sorted(values)[min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))]


def audit_quality(rows, expected_interval_seconds):
    if len(rows) < 2:
        raise SourceError("insufficient_rows_for_interval_audit")
    intervals = [
        (rows[index]["timestamp"] - rows[index - 1]["timestamp"]).total_seconds()
        for index in range(1, len(rows))
    ]
    if any(not math.isfinite(value) or value <= 0 for value in intervals):
        raise SourceError("invalid_time_interval")
    return {
        "rows": len(rows),
        "start_utc": rows[0]["timestamp"].isoformat(),
        "end_utc": rows[-1]["timestamp"].isoformat(),
        "expected_interval_seconds": expected_interval_seconds,
        "median_interval_seconds": statistics.median(intervals),
        "p95_interval_seconds": percentile(intervals, 0.95),
        "max_interval_seconds": max(intervals),
        "intervals_over_1_5x_expected": sum(value > expected_interval_seconds * 1.5 for value in intervals),
        "intervals_over_2x_expected": sum(value > expected_interval_seconds * 2 for value in intervals),
        "strictly_increasing_timestamps": True,
        "gaps_filled": False,
    }


def run_one(spec, payload, cases=CASE_COUNT, bootstrap_replicates=80):
    started = time.perf_counter()
    if len(payload) > spec["max_bytes"]:
        raise SourceError("payload_over_limit:" + spec["id"])
    rows = core.parse_csv(payload, min_rows=spec["min_rows"])
    normalize_utc(rows, spec["naive_timezone"])
    quality = audit_quality(rows, spec["interval_seconds"])
    case_rows = core.case_replay(rows, cases)
    if len(case_rows) != cases * len(core.RULES):
        raise SourceError("real_case_count_mismatch:" + spec["id"])
    full = core.run_full_history_replay(rows)
    probability = core.run_prequential_probability_diagnostics(
        rows, bootstrap_replicates=bootstrap_replicates
    )
    # Evaluate the actual TP-before-SL target on causal chronological splits; do not enable signals.
    barrier_oos = barrier_benchmark.run_barrier_oos(
        rows, source_id=spec["id"], bootstrap_replicates=40
    )
    market = core.market_scenarios(rows, anchors=SCENARIO_ANCHORS)
    software = core.software_scenarios(rows)
    market_passed = sum(item["passed"] for item in market)
    software_passed = sum(item["passed"] for item in software)
    if market_passed != len(market) or software_passed != len(software):
        raise SourceError("scenario_contract_failure:" + spec["id"])
    fixed_rules = []
    for rule in core.RULES:
        selected = [row for row in case_rows if row["rule"] == rule]
        trades = [row for row in selected if row["signal"]]
        correct = sum(row["signal"] == row["next_body_direction"] for row in trades)
        fixed_rules.append({
            "rule": rule,
            "cases": len(selected),
            "trades": len(trades),
            "direction_accuracy": correct / len(trades) if trades else None,
            "net_pnl_units_assumed_costs": round(sum(row["net_pnl_units"] for row in trades), 8),
        })
    return {
        "status": "PASS",
        "source": {
            "id": spec["id"], "repository": spec["repository"], "url": spec["url"],
            "license_url": spec["license_url"], "terms_url": spec["terms_url"],
            "license_status": spec["license_status"],
            "provenance_claim": spec["provenance_claim"],
            "quote_side_claim": spec["quote_side_claim"],
            "semantics": spec["semantics"], "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload), "timeframe_seconds": spec["interval_seconds"],
            "timestamp_interpretation": (
                "explicit offsets normalized to UTC" if spec["naive_timezone"] is None
                else "naive timestamps interpreted as UTC based on source README; not independently verified"
            ),
            "source_timezone_verified": spec["timezone_semantics_verified"],
            "timezone_semantics_verified": spec["timezone_semantics_verified"],
            "broker_provenance_verified": spec["broker_provenance_verified"],
            "quote_side_verified": spec["quote_side_verified"],
            "underlying_data_rights_verified": spec["underlying_data_rights_verified"],
            "commercial_use_approved": spec["commercial_use_approved"],
            "independent_source_verified": False,
            "source_use_eligibility": source_use_eligibility(spec),
        },
        "dataset": quality,
        "real_data_benchmark": {
            "timestamp_spread_cases": cases,
            "rule_comparisons": len(case_rows),
            "fixed_rules": fixed_rules,
        },
        "full_history_replay": full,
        "prequential_probability_diagnostics": probability,
        "target_specific_barrier_oos": barrier_oos,
        "scenarios": {
            "market_paths": len(market), "market_passed": market_passed,
            "software_scenarios": len(software), "software_passed": software_passed,
            "generated_paths_are_synthetic": True,
        },
        "runtime_ms": round((time.perf_counter() - started) * 1000, 3),
        "decision_output": "NO_TRADE", "predictive_model_validated": False,
        "deployment_calibrated": False, "live_signal_enabled": False,
    }, rows


def compare_m5(spec_a, rows_a, spec_b, rows_b, minimum_matches=1000):
    if spec_a["interval_seconds"] != 300 or spec_b["interval_seconds"] != 300:
        raise SourceError("same_timeframe_m5_required_for_overlap")
    by_a = {row["timestamp"]: row for row in rows_a}
    by_b = {row["timestamp"]: row for row in rows_b}
    common = sorted(set(by_a).intersection(by_b))
    if len(common) < minimum_matches:
        raise SourceError("insufficient_exact_timestamp_overlap:" + str(len(common)))

    def measure_offset(source_b_minus_a_seconds):
        offset = timedelta(seconds=source_b_minus_a_seconds)
        pairs = [
            (stamp, by_a[stamp], by_b[stamp + offset])
            for stamp in by_a
            if stamp + offset in by_b
        ]
        if len(pairs) < minimum_matches:
            return {
                "source_b_minus_source_a_timestamp_seconds": source_b_minus_a_seconds,
                "status": "INSUFFICIENT_OVERLAP",
                "matched_timestamps": len(pairs),
                "minimum_required": minimum_matches,
            }
        absolute_deltas = [
            abs(row_a["close"] - row_b["close"]) for _, row_a, row_b in pairs
        ]
        relative_deltas = [
            delta / ((abs(row_a["close"]) + abs(row_b["close"])) / 2.0)
            for delta, (_, row_a, row_b) in zip(absolute_deltas, pairs)
        ]
        return {
            "source_b_minus_source_a_timestamp_seconds": source_b_minus_a_seconds,
            "status": "PASS",
            "matched_timestamps": len(pairs),
            "absolute_close_delta_median_price_units": statistics.median(absolute_deltas),
            "absolute_close_delta_p95_price_units": percentile(absolute_deltas, 0.95),
            "relative_close_delta_median_fraction": statistics.median(relative_deltas),
            "relative_close_delta_p95_fraction": percentile(relative_deltas, 0.95),
        }

    # Probe likely bar-label/timezone offsets for diagnosis only. Never shift or pool source bars.
    alignment_offsets = [-25200, -10800, -7200, -3600, -600, -300, 0, 300, 600, 3600, 7200, 10800, 25200]
    alignment_probe = [measure_offset(offset) for offset in alignment_offsets]
    exact = next(item for item in alignment_probe if item["source_b_minus_source_a_timestamp_seconds"] == 0)
    eligible = [item for item in alignment_probe if item["status"] == "PASS"]
    best = min(eligible, key=lambda item: item["relative_close_delta_median_fraction"])
    exact_median_relative = exact["relative_close_delta_median_fraction"]
    best_median_relative = best["relative_close_delta_median_fraction"]
    relative_median_improvement = (
        (exact_median_relative - best_median_relative) / exact_median_relative
        if exact_median_relative > 0 else 0.0
    )
    material_shift_candidate = (
        best["source_b_minus_source_a_timestamp_seconds"] != 0
        and best_median_relative < exact_median_relative * 0.8
    )
    alignment_diagnosis = (
        "POSSIBLE_TIMESTAMP_LABEL_OR_TIMEZONE_SHIFT"
        if material_shift_candidate
        else "NO_OBVIOUS_BOUNDED_SHIFT_FIX"
    )

    # Source-level diagnostics are aggregate-first; only five paired candle samples are emitted.
    signed_close_deltas = [by_b[stamp]["close"] - by_a[stamp]["close"] for stamp in common]
    differences = [abs(delta) for delta in signed_close_deltas]
    reference_prices = [
        (abs(by_a[stamp]["close"]) + abs(by_b[stamp]["close"])) / 2.0
        for stamp in common
    ]
    relative_differences = [
        delta / reference if reference > 0 else math.inf
        for delta, reference in zip(differences, reference_prices)
    ]
    price_quantiles = {
        "source_a_close": {
            "median": statistics.median(by_a[stamp]["close"] for stamp in common),
            "p05": percentile([by_a[stamp]["close"] for stamp in common], 0.05),
            "p95": percentile([by_a[stamp]["close"] for stamp in common], 0.95),
        },
        "source_b_close": {
            "median": statistics.median(by_b[stamp]["close"] for stamp in common),
            "p05": percentile([by_b[stamp]["close"] for stamp in common], 0.05),
            "p95": percentile([by_b[stamp]["close"] for stamp in common], 0.95),
        },
    }
    ohlc_signed_median = {
        field: statistics.median(by_b[stamp][field] - by_a[stamp][field] for stamp in common)
        for field in ("open", "high", "low", "close")
    }
    monthly_groups = {}
    for stamp in common:
        monthly_groups.setdefault(stamp.strftime("%Y-%m"), []).append(stamp)
    monthly_diagnostics = []
    for month, stamps in sorted(monthly_groups.items()):
        signed = [by_b[stamp]["close"] - by_a[stamp]["close"] for stamp in stamps]
        absolute = [abs(value) for value in signed]
        relative = [
            abs(value) / ((abs(by_a[stamp]["close"]) + abs(by_b[stamp]["close"])) / 2.0)
            for stamp, value in zip(stamps, signed)
        ]
        monthly_diagnostics.append({
            "month_utc": month,
            "matched_timestamps": len(stamps),
            "signed_close_delta_median_b_minus_a": statistics.median(signed),
            "absolute_close_delta_median_price_units": statistics.median(absolute),
            "absolute_close_delta_p95_price_units": percentile(absolute, 0.95),
            "relative_close_delta_median_fraction": statistics.median(relative),
            "relative_close_delta_p95_fraction": percentile(relative, 0.95),
        })
    sample_indexes = sorted(set([
        0,
        len(common) // 4,
        len(common) // 2,
        (3 * len(common)) // 4,
        len(common) - 1,
    ]))
    paired_candle_samples = []
    for index in sample_indexes:
        stamp = common[index]
        a, b = by_a[stamp], by_b[stamp]
        delta = b["close"] - a["close"]
        reference = (abs(a["close"]) + abs(b["close"])) / 2.0
        paired_candle_samples.append({
            "timestamp_utc": stamp.isoformat(),
            "source_a_ohlc": {field: a[field] for field in ("open", "high", "low", "close")},
            "source_b_ohlc": {field: b[field] for field in ("open", "high", "low", "close")},
            "signed_close_delta_b_minus_a": delta,
            "absolute_close_delta_price_units": abs(delta),
            "relative_absolute_close_delta_fraction": abs(delta) / reference if reference > 0 else None,
        })
    median_relative_delta = statistics.median(relative_differences)
    p95_relative_delta = percentile(relative_differences, 0.95)
    # Price-level compatibility gate, not a trading-accuracy target:
    # median delta <= 0.25% and P95 <= 0.50% of each matched price level.
    median_relative_limit = 0.0025
    p95_relative_limit = 0.005
    compatibility_status = (
        "COMPARABLE"
        if median_relative_delta <= median_relative_limit
        and p95_relative_delta <= p95_relative_limit
        else "DIVERGENT"
    )
    signs_agree = []
    for stamp in common:
        next_stamp = stamp + timedelta(seconds=300)
        if next_stamp not in by_a or next_stamp not in by_b:
            continue
        move_a = by_a[next_stamp]["close"] - by_a[stamp]["close"]
        move_b = by_b[next_stamp]["close"] - by_b[stamp]["close"]
        sign_a, sign_b = (move_a > 0) - (move_a < 0), (move_b > 0) - (move_b < 0)
        signs_agree.append(sign_a == sign_b)
    return {
        "status": "PASS",
        "source_a": spec_a["id"],
        "source_b": spec_b["id"],
        "timeframe_seconds": 300,
        "exact_timestamp_matches": len(common),
        "overlap_fraction_a": len(common) / len(by_a),
        "overlap_fraction_b": len(common) / len(by_b),
        "overlap_start_utc": common[0].isoformat(),
        "overlap_end_utc": common[-1].isoformat(),
        "close_abs_delta_price_units": {
            "median": statistics.median(differences),
            "p95": percentile(differences, 0.95),
            "maximum": max(differences),
        },
        "price_level_diagnostics": {
            "source_a_close_quantiles": price_quantiles["source_a_close"],
            "source_b_close_quantiles": price_quantiles["source_b_close"],
            "signed_close_delta_median_b_minus_a": statistics.median(signed_close_deltas),
            "signed_close_delta_mean_b_minus_a": statistics.mean(signed_close_deltas),
            "fraction_source_b_close_above_source_a": sum(delta > 0 for delta in signed_close_deltas) / len(signed_close_deltas),
            "fraction_source_b_close_below_source_a": sum(delta < 0 for delta in signed_close_deltas) / len(signed_close_deltas),
            "fraction_equal_close": sum(delta == 0 for delta in signed_close_deltas) / len(signed_close_deltas),
            "ohlc_signed_delta_median_b_minus_a": ohlc_signed_median,
            "monthly_diagnostics": monthly_diagnostics,
            "representative_paired_candles": paired_candle_samples,
            "sample_policy": "5 deterministic snapshots across exact overlap; not a substitute for full raw data audit",
        },
        "close_relative_delta": {
            "median_fraction": median_relative_delta,
            "p95_fraction": p95_relative_delta,
            "median_limit_fraction": median_relative_limit,
            "p95_limit_fraction": p95_relative_limit,
            "compatibility_status": compatibility_status,
            "meaning": "diagnostic price-level compatibility threshold; not a performance or execution-cost measure",
        },
        "temporal_alignment_probe": {
            "tested_offsets_seconds": alignment_offsets,
            "results": alignment_probe,
            "best_observed_offset_seconds_diagnostic_only": best["source_b_minus_source_a_timestamp_seconds"],
            "best_offset_seconds_diagnostic_only": (
                best["source_b_minus_source_a_timestamp_seconds"] if material_shift_candidate else None
            ),
            "best_median_relative_delta_fraction": best_median_relative,
            "exact_timestamp_median_relative_delta_fraction": exact_median_relative,
            "relative_median_improvement_fraction": relative_median_improvement,
            "material_shift_candidate": material_shift_candidate,
            "diagnosis": alignment_diagnosis,
            "automatic_timestamp_shifting": False,
        },
        "close_to_close_direction_pairs": len(signs_agree),
        "close_to_close_direction_agreement": sum(signs_agree) / len(signs_agree) if signs_agree else None,
        "independence_status": "UNVERIFIED: community sources may share upstream data",
        "datasets_merged": False,
        "cross_source_confirmation_allowed": False,
    }

def build_report(results, overlap, min_total_rows):
    rows_total = sum(item["dataset"]["rows"] for item in results)
    all_cases = sum(item["real_data_benchmark"]["rule_comparisons"] for item in results)
    market_paths = sum(item["scenarios"]["market_paths"] for item in results)
    software_scenarios = sum(item["scenarios"]["software_scenarios"] for item in results)
    price_compatibility = overlap.get("close_relative_delta", {}).get("compatibility_status", "UNKNOWN")
    passed = (
        len(results) == len(SOURCES)
        and all(item["status"] == "PASS" for item in results)
        and all(item["dataset"]["rows"] >= spec["min_rows"] for item, spec in zip(results, SOURCES))
        and rows_total >= min_total_rows
        and overlap["status"] == "PASS"
    )
    pipeline_status = "PASS" if passed else "FAIL"
    readiness_gates = build_readiness_gates(results, overlap, pipeline_status)
    return {
        "schema_version": "1.2.0",
        "benchmark_kind": "multi_source_real_dataset_diagnostics",
        "status": pipeline_status,
        "status_semantics": "pipeline and test contracts only; not trading or data-use approval",
        "source_count": len(results),
        "dataset_rows_sum_not_deduplicated": rows_total,
        "minimum_total_rows": min_total_rows,
        "real_data_rule_comparisons": all_cases,
        "synthetic_market_path_contracts": market_paths,
        "synthetic_software_contracts": software_scenarios,
        "sources": results,
        "overlap_concordance": [overlap],
        "cross_source_compatibility_status": price_compatibility,
        "cross_source_confirmation_allowed": False,
        "readiness_gates": readiness_gates,
        "production_ready": readiness_gates["production_readiness"] == "PASS",
        "decision_output": "NO_TRADE",
        "predictive_model_validated": False,
        "model_calibrated_for_deployment": False,
        "live_signal_enabled": False,
        "live_order_execution_enabled": False,
        "limitations": [
            "Rows from different sources are not deduplicated or concatenated; the total is not an independent sample count.",
            "Community-source broker provenance, timezone semantics, data rights and upstream independence remain unverified.",
            "Repository-level MIT claims do not alone prove rights to upstream broker data or commercial embedding; GetData documentation describes more restrictive purchaser-only terms, so scope must be resolved in writing.",
            "The M5 and M1 results are separate tasks; their metrics are not directly comparable without a normalized protocol.",
            "M5 price-level compatibility is tested separately from direction agreement using explicit relative-delta thresholds.",
            "Cross-source confirmation is disabled while independent source lineage and price-level compatibility are unverified.",
            "Commercial model use, embedding, and production-feed use stay blocked until source rights/provenance are verified.",
            "Costs are assumed, not measured broker spread, commission, slippage, swaps or fills.",
            "Synthetic scenario contracts are code robustness tests, not extra observed market outcomes.",
            "Next-candle body direction is not TP-before-SL calibration or executable strategy performance.",
            "PASS means data and test contracts passed, not that a trading strategy has an edge.",
        ],
    }

def render_markdown(report):
    lines = [
        "# Aurum multi-source benchmark", "",
        "Pipeline result: " + report["status"], "Cross-source price compatibility: " + report.get("cross_source_compatibility_status", "UNKNOWN"), "Decision gate: NO_TRADE", "",
        "PASS means ingestion/audit/replay/scenario contracts ran; it does not approve a trading model.", "",
        "| Source | Rows | Timeframe | UTC start | UTC end | SHA-256 | Real rule comparisons | Market contracts | Software contracts |",
        "|---|---:|---:|---|---|---|---:|---:|---:|",
    ]
    for item in report["sources"]:
        source, data = item["source"], item["dataset"]
        lines.append("| {} | {} | {}s | {} | {} | {} | {} | {} | {} |".format(
            source["id"], data["rows"], source["timeframe_seconds"], data["start_utc"],
            data["end_utc"], source["sha256"], item["real_data_benchmark"]["rule_comparisons"],
            item["scenarios"]["market_passed"], item["scenarios"]["software_passed"]))
    gates = report.get("readiness_gates", {})
    lines.extend([
        "", "## Data-use and production readiness gates", "",
        "Pipeline diagnostics: **{}**; cross-source comparability: **{}**; cross-source confirmation: **{}**; commercial use: **{}**; production: **{}**.".format(
            gates.get("diagnostic_benchmark_status", "UNKNOWN"),
            gates.get("source_comparability_gate", "UNKNOWN"),
            gates.get("cross_source_confirmation_gate", "UNKNOWN"),
            gates.get("commercial_product_use", "UNKNOWN"),
            gates.get("production_readiness", "UNKNOWN")),
        "CI execution success does not grant underlying data rights or authorize commercial model training, source cross-confirmation, or production use.",
        "", "| Source | Rights review | Commercial approval | Commercial model training | Production feed |",
        "|---|---|---|---|---|",
    ])
    for item in report["sources"]:
        source = item["source"]
        use = source.get("source_use_eligibility", {})
        lines.append("| {} | {} | {} | {} | {} |".format(
            source["id"],
            use.get("source_rights_review", "UNKNOWN"),
            use.get("commercial_use_approved", False),
            use.get("model_training_for_commercial_service", "UNKNOWN"),
            use.get("production_feed_approved", "UNKNOWN")))
        lines.append("- {}: {}".format(source["id"], source.get("license_status", "Rights status not recorded.")))
        lines.append("  Terms/source notes: {} — {}".format(
            source.get("terms_url", "n/a"),
            source.get("provenance_claim", "Provenance not recorded.")))
    if gates.get("reason_codes"):
        lines.append("")
        lines.append("Blocked reason codes: " + ", ".join(gates["reason_codes"]))
    overlap = report["overlap_concordance"][0]
    absolute = overlap.get("close_abs_delta_price_units", {
        "median": overlap.get("close_abs_delta_median", 0.0),
        "p95": overlap.get("close_abs_delta_p95", 0.0),
        "maximum": overlap.get("close_abs_delta_max", 0.0),
    })
    relative = overlap.get("close_relative_delta", {})
    agreement = overlap.get("close_to_close_direction_agreement")
    alignment = overlap.get("temporal_alignment_probe", {})
    candidate_offset = alignment.get("best_offset_seconds_diagnostic_only")
    candidate_offset_text = "none" if candidate_offset is None else str(candidate_offset)
    price_diag = overlap.get("price_level_diagnostics", {})
    qa = price_diag.get("source_a_close_quantiles", {})
    qb = price_diag.get("source_b_close_quantiles", {})
    monthly = price_diag.get("monthly_diagnostics", [])
    paired_samples = price_diag.get("representative_paired_candles", [])
    lines.extend([
        "", "## Cross-source M5 overlap and price compatibility", "",
        "Exact shared timestamps: {}.".format(overlap["exact_timestamp_matches"]),
        "Absolute close difference median/P95/max: {:.6f}/{:.6f}/{:.6f} price units.".format(
            absolute["median"], absolute["p95"], absolute["maximum"]),
        "Relative close difference median/P95: {:.4%}/{:.4%}.".format(
            relative.get("median_fraction", 0.0), relative.get("p95_fraction", 0.0)),
        "Price-level compatibility: **{}** (median limit {:.2%}; P95 limit {:.2%}).".format(
            relative.get("compatibility_status", "UNKNOWN"),
            relative.get("median_limit_fraction", 0.0025),
            relative.get("p95_limit_fraction", 0.005)),
        "Consecutive close-to-close direction agreement: {}. Source independence remains UNVERIFIED.".format(
            "n/a" if agreement is None else format(agreement, ".4f")),
        "Bounded timestamp-alignment probe: material candidate offset B−A = {} seconds; best observed offset = {} seconds; relative median {:.4%} vs exact {:.4%} (improvement {:.2%}); diagnosis {}. No automatic shift applied.".format(
            candidate_offset_text,
            alignment.get("best_observed_offset_seconds_diagnostic_only", "unknown"),
            alignment.get("best_median_relative_delta_fraction", 0.0),
            alignment.get("exact_timestamp_median_relative_delta_fraction", 0.0),
            alignment.get("relative_median_improvement_fraction", 0.0),
            alignment.get("diagnosis", "NOT_RECORDED")),
        "Cross-source confirmation enabled: no. High direction agreement does not cure divergent price levels or prove independent provenance.",
        "", "## Signed price-level bias", "",
        "Matched-overlap close medians: source A {:.5f}, source B {:.5f}; median signed B−A delta {:.5f}, mean signed delta {:.5f} price units.".format(
            qa.get("median", 0.0), qb.get("median", 0.0),
            price_diag.get("signed_close_delta_median_b_minus_a", 0.0),
            price_diag.get("signed_close_delta_mean_b_minus_a", 0.0)),
        "Source B closes above/below/equal to source A in {:.2%}/{:.2%}/{:.2%} of matched timestamps.".format(
            price_diag.get("fraction_source_b_close_above_source_a", 0.0),
            price_diag.get("fraction_source_b_close_below_source_a", 0.0),
            price_diag.get("fraction_equal_close", 0.0)),
        "Median signed B−A OHLC differences: " + ", ".join(
            "{}={:.5f}".format(k, v) for k, v in price_diag.get("ohlc_signed_delta_median_b_minus_a", {}).items()
        ) + " price units.",
        "", "### Monthly matched-price discrepancy", "",
        "| UTC month | Matched bars | Median signed B−A | Median absolute delta | P95 absolute delta | Median relative delta | P95 relative delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for item in monthly:
        lines.append("| {} | {} | {:.5f} | {:.5f} | {:.5f} | {:.4%} | {:.4%} |".format(
            item["month_utc"], item["matched_timestamps"],
            item["signed_close_delta_median_b_minus_a"],
            item["absolute_close_delta_median_price_units"],
            item["absolute_close_delta_p95_price_units"],
            item["relative_close_delta_median_fraction"],
            item["relative_close_delta_p95_fraction"]))
    lines.extend([
        "", "### Representative paired candles (diagnostic samples)", "",
        "| UTC timestamp | Source A close | Source B close | Signed B−A delta | Relative absolute delta |",
        "|---|---:|---:|---:|---:|",
    ])
    for sample in paired_samples:
        lines.append("| {} | {:.5f} | {:.5f} | {:.5f} | {:.4%} |".format(
            sample["timestamp_utc"],
            sample["source_a_ohlc"]["close"],
            sample["source_b_ohlc"]["close"],
            sample["signed_close_delta_b_minus_a"],
            sample["relative_absolute_close_delta_fraction"]))
    lines.extend(["", "## Full-history fixed-rule results", ""])
    for item in report["sources"]:
        lines.extend(["### " + item["source"]["id"], "",
                      "| Rule | Full net PnL (assumed costs) | Tail net PnL | Tail trades | Tail PF |",
                      "|---|---:|---:|---:|---:|"])
        for rule, segments in item["full_history_replay"]["rules"].items():
            full, tail = segments["full_history"], segments["chronological_tail_holdout"]
            pf = tail["profit_factor"]
            lines.append("| {} | {:.4f} | {:.4f} | {} | {} |".format(
                rule, full["net_pnl_units"], tail["net_pnl_units"], tail["trades"],
                "n/a" if pf is None else format(pf, ".4f")))
        lines.extend(["", "Prequential direction-probability baselines:", "",
                      "| Baseline | N | Brier | Log loss | Brier skill vs 0.5 |",
                      "|---|---:|---:|---:|---:|"])
        for name, metric in item["prequential_probability_diagnostics"].get("methods", {}).items():
            skill = metric["brier_skill_vs_constant_50pct"]
            lines.append("| {} | {} | {:.6f} | {:.6f} | {} |".format(
                name, metric["n"], metric["brier_score"], metric["log_loss"],
                "n/a" if skill is None else format(skill, ".6f")))
        lines.append("")
    lines.extend([
        "## TP-before-SL chronological out-of-sample diagnostic", "",
        "Target: TP reached before SL within 12 bars using ATR-based barriers; entry is next-bar open, same-bar ambiguity is stop-first, and timeout is reported separately. This is an OHLC research label, not a broker fill.", "",
        "| Source | Side | Train N | Calibration N | Holdout N | TP event rate | Raw Brier | Cal Brier | Brier improvement 95% CI | Raw log loss | Cal log loss | Log-loss improvement 95% CI | Calibrator |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---|",
    ])
    for item in report["sources"]:
        barrier = item.get("target_specific_barrier_oos", {})
        for side in ("long", "short"):
            result = barrier.get("sides", {}).get(side, {})
            raw = result.get("raw_holdout_metrics") or {}
            adjusted = result.get("calibrated_holdout_metrics") or {}
            fit = result.get("calibration", {}).get("calibrator_fit", {})
            holdout = result.get("holdout", {})
            paired = result.get("paired_calibration_improvement_block_bootstrap_ci", {}).get("metrics", {})
            brier_ci = paired.get("brier_score", {})
            log_ci = paired.get("log_loss", {})
            brier_ci_text = (
                "n/a" if not brier_ci else "[{:.6f}, {:.6f}] ({:.0%} positive)".format(
                    brier_ci["lower_95"], brier_ci["upper_95"],
                    brier_ci["fraction_positive_bootstrap_improvements"])
            )
            log_ci_text = (
                "n/a" if not log_ci else "[{:.6f}, {:.6f}] ({:.0%} positive)".format(
                    log_ci["lower_95"], log_ci["upper_95"],
                    log_ci["fraction_positive_bootstrap_improvements"])
            )
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                item["source"]["id"], side,
                result.get("train_rows", 0), result.get("calibration_rows", 0),
                holdout.get("rows", result.get("holdout_rows", 0)),
                "n/a" if holdout.get("positive_rate") is None else format(holdout["positive_rate"], ".4f"),
                "n/a" if raw.get("brier_score") is None else format(raw["brier_score"], ".6f"),
                "n/a" if adjusted.get("brier_score") is None else format(adjusted["brier_score"], ".6f"),
                brier_ci_text,
                "n/a" if raw.get("log_loss") is None else format(raw["log_loss"], ".6f"),
                "n/a" if adjusted.get("log_loss") is None else format(adjusted["log_loss"], ".6f"),
                log_ci_text,
                fit.get("status", "not_fitted")))
    lines.extend([
        "", "All TP-before-SL results are diagnostics only. The model/calibrator are not validated for deployment, the holdout has not authorized model selection, and NO_TRADE/live-execution guards remain unchanged.",
        "", "## Non-negotiable caveats", ""])
    lines.extend("- " + text for text in report["limitations"])
    lines.extend(["", "Do not enable live signals or order execution from this benchmark."])
    return "\n".join(lines) + "\n"


def write_report_files(report, output_dir):
    """Persist machine-readable and human-readable evidence without raw source bars."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "multi_source_dataset_benchmark.json"
    markdown_path = output_dir / "multi_source_dataset_benchmark.md"
    json_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def run_catalog(output_dir, cases=CASE_COUNT, min_total_rows=100000):
    if cases < 30:
        raise ValueError("cases_must_be_at_least_30")
    started = time.perf_counter()
    loaded, results = [], []
    for spec in SOURCES:
        payload = fetch_source(spec)
        report, rows = run_one(spec, payload, cases=cases)
        loaded.append((spec, rows))
        results.append(report)
    by_id = {spec["id"]: rows for spec, rows in loaded}
    by_spec = {spec["id"]: spec for spec, rows in loaded}
    overlap = compare_m5(
        by_spec["getdata_xauusd_m5"], by_id["getdata_xauusd_m5"],
        by_spec["sai_community_xauusd_m5"], by_id["sai_community_xauusd_m5"],
    )
    report = build_report(results, overlap, min_total_rows)
    report["runtime_ms"] = round((time.perf_counter() - started) * 1000, 3)
    write_report_files(report, output_dir)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("multi-source-output"))
    parser.add_argument("--cases", type=int, default=CASE_COUNT)
    parser.add_argument("--min-total-rows", type=int, default=100000)
    args = parser.parse_args(argv)
    try:
        result = run_catalog(args.output_dir, args.cases, args.min_total_rows)
        print(json.dumps({
            "status": result["status"], "source_count": result["source_count"],
            "dataset_rows_sum_not_deduplicated": result["dataset_rows_sum_not_deduplicated"],
            "real_data_rule_comparisons": result["real_data_rule_comparisons"],
            "synthetic_market_path_contracts": result["synthetic_market_path_contracts"],
            "synthetic_software_contracts": result["synthetic_software_contracts"],
            "decision_output": result["decision_output"], "runtime_ms": result["runtime_ms"],
        }, indent=2))
        return 0 if result["status"] == "PASS" else 2
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error_type": type(exc).__name__,
                          "error": str(exc)[:500], "decision_output": "NO_TRADE"},
                         indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
