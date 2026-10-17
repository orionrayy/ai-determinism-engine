#!/usr/bin/env python3
"""Bounded, read-only XAU/USD bid/ask tick quote probe.

Samples three disjoint windows from a much larger public dataset. Windows are
summarized independently and never concatenated into a synthetic time series.
Output is source-quality and spread-distribution evidence, not a strategy test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

API_HOST = "datasets-server.huggingface.co"
API_PATH = "/rows"
DATASET_ID = "CarlosSilva1/xauusd-ticks"
CONFIG = "default"
SPLIT = "train"
USER_AGENT = "Aurum-Intelligence/tick-quote-probe"
PAGE_SIZE = 100
DEFAULT_ROWS_PER_WINDOW = 1000
MAX_ROWS_PER_WINDOW = 2000
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 16 * 1024 * 1024
MAX_RETRIES = 3
REQUIRED_FIELDS = ("timestamp", "bid_price", "ask_price", "bid_volume", "ask_volume")


class ProbeError(ValueError):
    """Expected failure in remote source validation or quote data QA."""


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ProbeError(field + "_must_be_finite_number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProbeError(field + "_must_be_finite_number") from exc
    if not math.isfinite(result):
        raise ProbeError(field + "_must_be_finite_number")
    return result


def parse_timestamp_utc(value: Any) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ProbeError("timestamp_must_be_nonempty_iso8601_string")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProbeError("timestamp_invalid_iso8601") from exc
    # The dataset card claims UTC; viewer timestamps are naive ISO strings.
    # Normalize under that stated assumption but preserve the unverified caveat.
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def build_rows_url(offset: int, length: int = PAGE_SIZE) -> str:
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ProbeError("offset_must_be_nonnegative_integer")
    if isinstance(length, bool) or not isinstance(length, int) or not 1 <= length <= PAGE_SIZE:
        raise ProbeError("page_length_must_be_between_1_and_100")
    query = urlencode({"dataset": DATASET_ID, "config": CONFIG, "split": SPLIT, "offset": offset, "length": length})
    return "https://" + API_HOST + API_PATH + "?" + query


def fetch_json_page(offset: int, length: int = PAGE_SIZE, *, opener=urlopen, timeout_seconds: int = 30,
                    max_response_bytes: int = MAX_RESPONSE_BYTES, sleep=time.sleep,
                    request_counter: list[int] | None = None) -> tuple[dict[str, Any], int]:
    """Fetch one allowlisted 1-100-row page with bounded retries and bytes."""
    request = Request(build_rows_url(offset, length), headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            if request_counter is not None:
                request_counter[0] += 1
            with opener(request, timeout=timeout_seconds) as response:
                final = urlparse(response.geturl())
                if final.scheme != "https" or final.netloc != API_HOST or final.path != API_PATH:
                    raise ProbeError("redirect_outside_allowlist")
                raw = response.read(max_response_bytes + 1)
                if len(raw) > max_response_bytes:
                    raise ProbeError("response_exceeds_byte_limit")
                if not raw:
                    raise ProbeError("empty_response_body")
                try:
                    data = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ProbeError("invalid_json_response") from exc
                if not isinstance(data, dict):
                    raise ProbeError("json_response_must_be_object")
                rows = data.get("rows")
                if not isinstance(rows, list) or len(rows) != length:
                    raise ProbeError("unexpected_page_row_count")
                total = data.get("num_rows_total")
                if isinstance(total, bool) or not isinstance(total, int) or total <= offset or offset + length > total:
                    raise ProbeError("invalid_total_row_count_or_page_bounds")
                features = data.get("features")
                if not isinstance(features, list):
                    raise ProbeError("missing_feature_schema")
                names = {item.get("name") for item in features if isinstance(item, dict)}
                if not set(REQUIRED_FIELDS).issubset(names):
                    raise ProbeError("unexpected_tick_schema")
                return data, len(raw)
        except ProbeError:
            raise
        except (OSError, URLError, HTTPError, TimeoutError) as exc:
            last_error = exc
            if attempt + 1 < MAX_RETRIES:
                sleep(0.5 * (2 ** attempt))
    raise ProbeError("remote_fetch_failed_after_retries:" + type(last_error).__name__)


def plan_window_offsets(total_rows: int, rows_per_window: int = DEFAULT_ROWS_PER_WINDOW) -> list[dict[str, int | str]]:
    if isinstance(total_rows, bool) or not isinstance(total_rows, int) or total_rows < 1:
        raise ProbeError("total_rows_must_be_positive_integer")
    if (isinstance(rows_per_window, bool) or not isinstance(rows_per_window, int)
            or rows_per_window < PAGE_SIZE or rows_per_window > MAX_ROWS_PER_WINDOW
            or rows_per_window % PAGE_SIZE != 0):
        raise ProbeError("rows_per_window_must_be_100_to_2000_in_100_row_steps")
    if total_rows < rows_per_window * 3:
        raise ProbeError("dataset_too_small_for_three_disjoint_windows")
    candidates = [("head", 0), ("middle", (total_rows // 2 // PAGE_SIZE) * PAGE_SIZE),
                  ("tail", ((total_rows - rows_per_window) // PAGE_SIZE) * PAGE_SIZE)]
    result = [{"window": name, "start_offset": offset, "rows_requested": rows_per_window} for name, offset in candidates]
    for window in result:
        if window["start_offset"] < 0 or window["start_offset"] + rows_per_window > total_rows:
            raise ProbeError("window_outside_dataset_bounds")
    for previous, current in zip(result, result[1:]):
        if previous["start_offset"] + rows_per_window > current["start_offset"]:
            raise ProbeError("sample_windows_overlap")
    return result


def _quantile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ProbeError("cannot_compute_quantile_from_empty_values")
    if not 0 <= quantile <= 1:
        raise ProbeError("quantile_out_of_range")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    low, high = int(math.floor(position)), int(math.ceil(position))
    return ordered[low] if low == high else ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize_window(rows: Sequence[Mapping[str, Any]], *, window: str, start_offset: int,
                     payload_sha256: str) -> dict[str, Any]:
    """Validate and summarize one contiguous window; never return raw tick rows."""
    if not rows:
        raise ProbeError("window_must_not_be_empty")
    timestamps, bids, asks, spreads, spreads_bps = [], [], [], [], []
    bid_volumes, ask_volumes = [], []
    crossed, locked, duplicate_timestamps, order_violations = 0, 0, 0, 0
    expected_idx = start_offset
    for item in rows:
        if not isinstance(item, Mapping) or item.get("row_idx") != expected_idx:
            raise ProbeError("source_row_index_not_contiguous")
        row = item.get("row")
        if not isinstance(row, Mapping) or any(field not in row for field in REQUIRED_FIELDS):
            raise ProbeError("tick_row_missing_required_field")
        stamp = parse_timestamp_utc(row["timestamp"])
        bid, ask = _finite(row["bid_price"], "bid_price"), _finite(row["ask_price"], "ask_price")
        bid_volume, ask_volume = _finite(row["bid_volume"], "bid_volume"), _finite(row["ask_volume"], "ask_volume")
        if bid <= 0 or ask <= 0:
            raise ProbeError("bid_ask_price_must_be_positive")
        if bid_volume < 0 or ask_volume < 0:
            raise ProbeError("quote_volume_must_be_nonnegative")
        if timestamps:
            if stamp < timestamps[-1]:
                order_violations += 1
            elif stamp == timestamps[-1]:
                duplicate_timestamps += 1
        spread = ask - bid
        if spread < 0:
            crossed += 1
        elif spread == 0:
            locked += 1
        else:
            spreads.append(spread)
            spreads_bps.append(spread / ((ask + bid) / 2.0) * 10000.0)
        timestamps.append(stamp)
        bids.append(bid)
        asks.append(ask)
        bid_volumes.append(bid_volume)
        ask_volumes.append(ask_volume)
        expected_idx += 1
    ordered = timestamps == sorted(timestamps)
    if not ordered:
        order_violations = max(order_violations, 1)
    flags = []
    if crossed:
        flags.append("CROSSED_BID_ASK_QUOTES_PRESENT")
    if locked:
        flags.append("LOCKED_OR_ZERO_SPREAD_QUOTES_PRESENT")
    if order_violations:
        flags.append("NON_MONOTONIC_TIMESTAMP_WITHIN_WINDOW")
    price_metrics = None if not spreads else {
        "median": statistics.median(spreads), "p95": _quantile(spreads, 0.95),
        "p99": _quantile(spreads, 0.99), "max": max(spreads),
    }
    bps_metrics = None if not spreads_bps else {
        "median": statistics.median(spreads_bps), "p95": _quantile(spreads_bps, 0.95),
        "p99": _quantile(spreads_bps, 0.99), "max": max(spreads_bps),
    }
    return {
        "window": window, "start_offset": start_offset, "end_offset_exclusive": start_offset + len(rows),
        "rows": len(rows), "row_index_contiguous": True,
        "timestamp_start_utc_assumed": timestamps[0].isoformat(),
        "timestamp_end_utc_assumed": timestamps[-1].isoformat(),
        "timestamp_ordered": ordered, "timestamp_order_violations": order_violations,
        "duplicate_adjacent_timestamps": duplicate_timestamps,
        "bid_ask_prices_positive": True, "volume_nonnegative": True,
        "crossed_bid_ask_rows": crossed, "locked_or_zero_spread_rows": locked,
        "positive_spread_rows": len(spreads), "bid_price_median": statistics.median(bids),
        "ask_price_median": statistics.median(asks), "bid_volume_median": statistics.median(bid_volumes),
        "ask_volume_median": statistics.median(ask_volumes),
        "spread_price_units": price_metrics, "spread_basis_points": bps_metrics,
        "quality_flags": flags, "normalized_sample_sha256": payload_sha256, "raw_tick_rows_persisted": False,
    }


def collect_tick_probe(*, rows_per_window: int = DEFAULT_ROWS_PER_WINDOW, opener=urlopen, sleep=time.sleep) -> dict[str, Any]:
    """Fetch three bounded windows from the public bid/ask dataset."""
    counter, total_bytes = [0], [0]
    first_page, first_bytes = fetch_json_page(0, PAGE_SIZE, opener=opener, sleep=sleep, request_counter=counter)
    total = first_page["num_rows_total"]
    windows = plan_window_offsets(total, rows_per_window)
    results = []
    for planned in windows:
        tick_rows, canonical_rows = [], []
        for relative in range(0, int(planned["rows_requested"]), PAGE_SIZE):
            offset = int(planned["start_offset"]) + relative
            if offset == 0:
                page, size = first_page, first_bytes
            else:
                sleep(0.05)
                page, size = fetch_json_page(offset, PAGE_SIZE, opener=opener, sleep=sleep, request_counter=counter)
            if page["num_rows_total"] != total or page.get("partial") is True:
                raise ProbeError("dataset_changed_or_partial_page")
            if total_bytes[0] + size > MAX_TOTAL_BYTES:
                raise ProbeError("total_probe_payload_exceeds_byte_limit")
            total_bytes[0] += size
            for pos, item in enumerate(page["rows"]):
                if item.get("row_idx") != offset + pos:
                    raise ProbeError("row_index_mismatch_at_offset:" + str(offset + pos))
            tick_rows.extend(page["rows"])
            canonical_rows.extend({
                "row_idx": item["row_idx"], "timestamp": item["row"]["timestamp"],
                "bid_price": item["row"]["bid_price"], "ask_price": item["row"]["ask_price"],
                "bid_volume": item["row"]["bid_volume"], "ask_volume": item["row"]["ask_volume"],
            } for item in page["rows"])
        canonical = json.dumps(canonical_rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        results.append(summarize_window(tick_rows, window=str(planned["window"]),
            start_offset=int(planned["start_offset"]), payload_sha256=hashlib.sha256(canonical).hexdigest()))
    total_sampled = sum(item["rows"] for item in results)
    valid = (len(results) == 3 and all(item["rows"] == rows_per_window for item in results)
        and all(item["row_index_contiguous"] and item["timestamp_ordered"] for item in results)
        and all(item["bid_ask_prices_positive"] and item["volume_nonnegative"] for item in results)
        and all(len(item["normalized_sample_sha256"]) == 64 for item in results))
    flags = sorted({flag for item in results for flag in item["quality_flags"]})
    return {
        "schema_version": "1.0.0", "status": "PASS" if valid else "FAIL",
        "benchmark_kind": "bounded_real_bid_ask_tick_quote_microstructure_probe",
        "source": {
            "dataset_id": DATASET_ID, "dataset_url": "https://huggingface.co/datasets/" + DATASET_ID,
            "dataset_card_url": "https://huggingface.co/datasets/" + DATASET_ID + "/blob/main/README.md",
            "api_host": API_HOST, "dataset_card_claimed_license": "CC-BY-4.0",
            "attribution": "Silva, C. (2026). XAU/USD Tick Data (May 2021-May 2026). Hugging Face. https://huggingface.co/datasets/CarlosSilva1/xauusd-ticks",
            "declared_period": "2021-05-24 through 2026-05-24", "declared_granularity": "millisecond bid/ask tick quotes",
            "provenance_claim": "Dataset card describes a broker-reconstructed feed; independent upstream lineage is not verified.",
            "timezone_claim": "Dataset card says UTC; naive API timestamps are normalized as UTC for diagnostics only.",
            "upstream_data_rights_verified": False, "commercial_use_approved": False,
            "model_training_for_commercial_service": "BLOCKED_PENDING_UPSTREAM_RIGHTS_REVIEW",
            "production_feed_approved": False,
        },
        "dataset_rows_reported_by_huggingface": total,
        "sample_design": {
            "window_count": len(results), "rows_per_window": rows_per_window, "sampled_rows_total": total_sampled,
            "window_strategy": "three disjoint contiguous row-index windows at head, middle and tail",
            "windows_concatenated": False, "infer_continuous_time_between_windows": False,
            "request_count_including_retries": counter[0], "total_response_bytes": total_bytes[0],
            "raw_tick_rows_persisted": False,
        },
        "windows": results,
        "structural_quality": {
            "status": "PASS" if valid else "FAIL",
            "timestamp_order_violations": sum(item["timestamp_order_violations"] for item in results),
            "crossed_bid_ask_rows": sum(item["crossed_bid_ask_rows"] for item in results),
            "locked_or_zero_spread_rows": sum(item["locked_or_zero_spread_rows"] for item in results),
            "positive_spread_rows": sum(item["positive_spread_rows"] for item in results), "quality_flags": flags,
        },
        "execution_calibration_eligibility": {
            "spread_profile_is_descriptive_only": True, "actual_fills_or_slippage_observed": False,
            "broker_identity_and_quote_semantics_verified": False, "source_rights_verified": False,
            "eligible_for_production_cost_calibration": False, "eligible_for_commercial_model_training": False,
            "live_signal_enabled": False, "live_order_execution_enabled": False, "decision_gate": "NO_TRADE",
            "reason": "A bounded tick sample describes quoted spreads; it does not establish source rights, broker lineage, executable fill costs or a validated strategy.",
        },
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    src, design = report["source"], report["sample_design"]
    lines = [
        "# Aurum bounded bid/ask tick quote probe", "",
        "Status: {}. This is a small real-data structure/spread probe, not a strategy test.".format(report["status"]), "",
        "Source: [{}]({}) — declared license {}. Upstream broker lineage and commercial data rights are not independently verified.".format(
            src["dataset_id"], src["dataset_url"], src["dataset_card_claimed_license"]),
        "Dataset row count reported by viewer: {:,}. Sampled {:,} rows across {} disjoint windows; windows are not concatenated.".format(
            report["dataset_rows_reported_by_huggingface"], design["sampled_rows_total"], design["window_count"]), "",
        "| Window | Offset | Rows | Start UTC assumed | End UTC assumed | Median spread | P95 | P99 | Median spread bps | P95 bps | Crossed | Locked |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["windows"]:
        spread, bps = item["spread_price_units"], item["spread_basis_points"]
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            item["window"], item["start_offset"], item["rows"], item["timestamp_start_utc_assumed"],
            item["timestamp_end_utc_assumed"], "n/a" if not spread else format(spread["median"], ".6f"),
            "n/a" if not spread else format(spread["p95"], ".6f"), "n/a" if not spread else format(spread["p99"], ".6f"),
            "n/a" if not bps else format(bps["median"], ".4f"), "n/a" if not bps else format(bps["p95"], ".4f"),
            item["crossed_bid_ask_rows"], item["locked_or_zero_spread_rows"]))
    lines.extend([
        "", "## Interpretation boundary", "",
        "- This samples a dataset whose card reports 286M+ tick rows; the probe did not download or benchmark all rows.",
        "- Only the three windows were inspected. Their ranges are distinct; no synthetic continuity is constructed between them.",
        "- Bid/ask spread quantiles are observed quotes, not measured fills, effective execution costs or proof of a tradable edge.",
        "- Dataset-card CC-BY-4.0 declaration is recorded with attribution, but underlying rights and broker provenance remain unverified. Commercial model training and production-feed use remain blocked.",
        "- NO_TRADE remains active. Model validation/deployment calibration are false; live signals and live order execution are disabled.", "",
        "## Reproducibility", "",
        "Requests: {}; response bytes: {}; per-window sample hashes:".format(
            design["request_count_including_retries"], design["total_response_bytes"]),
    ])
    for item in report["windows"]:
        lines.append("- {}: SHA-256 {}".format(item["window"], item["normalized_sample_sha256"]))
    return "\\n".join(lines) + "\\n"


def write_reports(report: Mapping[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path, markdown_path = output_dir / "tick_quote_probe.json", output_dir / "tick_quote_probe.md"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows-per-window", type=int, default=DEFAULT_ROWS_PER_WINDOW)
    args = parser.parse_args(argv)
    try:
        report = collect_tick_probe(rows_per_window=args.rows_per_window)
        files = write_reports(report, args.output_dir)
        print(json.dumps({"status": report["status"], "dataset_rows_reported": report["dataset_rows_reported_by_huggingface"],
            "sampled_rows": report["sample_design"]["sampled_rows_total"], "windows": len(report["windows"]),
            "requests": report["sample_design"]["request_count_including_retries"],
            "report_json": str(files["json"]), "report_markdown": str(files["markdown"]),
            "decision_gate": report["execution_calibration_eligibility"]["decision_gate"]}, indent=2))
        return 0 if report["status"] == "PASS" else 2
    except ProbeError as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc), "live_signal_enabled": False,
            "decision_gate": "NO_TRADE"}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
