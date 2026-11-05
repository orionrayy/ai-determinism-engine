#!/usr/bin/env python3
"""Bounded CC-BY XAUUSD bid/ask tick audit and M5 cross-check; never promotes a source."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

import ci_benchmark as base
from source_qualification_gate_v1 import ALL_EVIDENCE_FIELDS, evaluate_source_qualification

DATASET_ID = "CarlosSilva1/xauusd-ticks"
DATASET_API = (
    "https://huggingface.co/api/datasets/CarlosSilva1/xauusd-ticks/"
    "tree/main/year=2026/month=05"
)
FILE_BASE = "https://huggingface.co/datasets/CarlosSilva1/xauusd-ticks/resolve/main/"
PRIMARY_SOURCE = base.SOURCE
LICENSE_ID = "CC-BY-4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
ATTRIBUTION = (
    "Silva, C. (2026). XAU/USD Tick Data (May 2021–May 2026) [Data set]. "
    "Hugging Face. https://huggingface.co/datasets/CarlosSilva1/xauusd-ticks"
)
MAX_METADATA_BYTES = 2 * 1024 * 1024
MAX_TOTAL_PARQUET_BYTES = 128 * 1024 * 1024
MAX_FILE_BYTES = 32 * 1024 * 1024
MIN_PRIMARY_OVERLAP = 1000
PRICE_FIELDS = ("open", "high", "low", "close")


class TickDatasetError(ValueError):
    """Raised when a bounded tick-data acquisition or quality gate fails."""


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _summary(values):
    if not values:
        return {"n": 0, "median": None, "p05": None, "p95": None, "p99": None, "min": None, "max": None}
    vals = [float(value) for value in values]
    return {
        "n": len(vals),
        "median": statistics.median(vals),
        "p05": _percentile(vals, 0.05),
        "p95": _percentile(vals, 0.95),
        "p99": _percentile(vals, 0.99),
        "min": min(vals),
        "max": max(vals),
    }


def spread_diagnostics(bid_prices, ask_prices):
    """Summarize observed spread without pretending it is the user's broker cost."""
    if len(bid_prices) != len(ask_prices) or len(bid_prices) == 0:
        raise TickDatasetError("bid_ask_arrays_must_be_equal_nonempty")
    spreads, spreads_bps = [], []
    for bid, ask in zip(bid_prices, ask_prices):
        bid, ask = float(bid), float(ask)
        if not math.isfinite(bid) or not math.isfinite(ask) or bid <= 0 or ask < bid:
            raise TickDatasetError("invalid_bid_ask_quote")
        mid = (bid + ask) / 2.0
        spread = ask - bid
        spreads.append(spread)
        spreads_bps.append(spread / mid * 10000.0)
    return {
        "tick_count": len(spreads),
        "spread_price_units": _summary(spreads),
        "spread_basis_points_of_mid": _summary(spreads_bps),
        "execution_cost_is_user_broker_measured": False,
    }


def utc_timestamp(value):
    """Normalize a timestamp for matching only; this does not correct a source."""
    if not isinstance(value, datetime):
        raw = str(value).strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        value = datetime.fromisoformat(raw)
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def compare_price_bars(primary_rows, quote_bars, *, source_label="community OHLCV"):
    """Compare exact bar-open timestamp overlap for bid/mid/ask bars; do not choose a winner."""
    primary_map = {}
    for row in primary_rows:
        stamp = utc_timestamp(row["timestamp"])
        if stamp in primary_map:
            raise TickDatasetError("duplicate_primary_timestamp")
        primary_map[stamp] = row
    comparisons = {}
    sample_rows = []
    for basis in ("bid", "mid", "ask"):
        bars = quote_bars.get(basis, {})
        overlap = sorted(set(primary_map) & set(bars))
        close_bps, all_ohlc_bps, signed_close = [], [], []
        exact_ohlc = 0
        for stamp in overlap:
            primary = primary_map[stamp]
            other = bars[stamp]
            exact_ohlc += int(all(float(primary[f]) == float(other[f]) for f in PRICE_FIELDS))
            for field in PRICE_FIELDS:
                p, q = float(primary[field]), float(other[field])
                all_ohlc_bps.append(abs(p - q) / abs(p) * 10000.0 if p else math.inf)
            pclose, qclose = float(primary["close"]), float(other["close"])
            close_bps.append(abs(pclose - qclose) / abs(pclose) * 10000.0 if pclose else math.inf)
            signed_close.append(pclose - qclose)
        comparisons[basis] = {
            "status": "DIAGNOSTIC_ONLY" if len(overlap) >= MIN_PRIMARY_OVERLAP else "INSUFFICIENT_OVERLAP",
            "exact_timestamp_overlap_bars": len(overlap),
            "exact_ohlc_match_count": exact_ohlc,
            "absolute_close_difference_bps": _summary(close_bps),
            "all_ohlc_field_absolute_difference_bps": _summary(all_ohlc_bps),
            "signed_primary_minus_tick_close_difference_price_units": _summary(signed_close),
            "source_promotion_allowed": False,
        }
        if basis == "mid":
            for stamp in overlap:
                primary = primary_map[stamp]
                sample_rows.append({
                    "timestamp_utc_assumed_bar_open": stamp.isoformat(),
                    **{f"primary_{f}": float(primary[f]) for f in PRICE_FIELDS},
                    **{f"tick_bid_{f}": float(quote_bars["bid"][stamp][f]) for f in PRICE_FIELDS},
                    **{f"tick_mid_{f}": float(quote_bars["mid"][stamp][f]) for f in PRICE_FIELDS},
                    **{f"tick_ask_{f}": float(quote_bars["ask"][stamp][f]) for f in PRICE_FIELDS},
                })
    common = len(set(primary_map) & set(quote_bars.get("mid", {})))
    status = "THIRD_SOURCE_DIAGNOSTIC_ONLY" if common >= MIN_PRIMARY_OVERLAP else (
        "NO_EXACT_OVERLAP" if common == 0 else "INSUFFICIENT_OVERLAP"
    )
    return {
        "status": status,
        "compared_source": source_label,
        "comparison_mode": "exact UTC five-minute bucket starts only; sources remain separate",
        "primary_rows": len(primary_rows),
        "tick_derived_m5_bars": len(quote_bars.get("mid", {})),
        "shared_timestamps": common,
        "basis_comparisons": comparisons,
        "automatic_timestamp_shift": False,
        "automatic_price_rescale": False,
        "automatic_source_promotion": False,
        "limitations": [
            "Primary candle timestamp meaning and provider price basis remain independently unverified.",
            "Tick source is an aggregated reconstructed broker feed, not the user's broker or a universal market truth.",
            "Spread statistics describe this dataset and are not the user's measured execution costs.",
            "OHLC aggregates cannot reconstruct tick sequence for all strategy fills; this is a source diagnostic, not a tick-execution backtest.",
        ],
    }, sample_rows


def classify_lineage_overlap(exact_ohlc_matches, overlap_bars, threshold=0.95, minimum_overlap=1000):
    """Flag possible shared lineage or quote-basis mislabeling; never assert independence."""
    matches = int(exact_ohlc_matches)
    overlap = int(overlap_bars)
    if matches < 0 or overlap < 0 or matches > overlap:
        raise TickDatasetError("invalid_lineage_overlap_counts")
    fraction = matches / overlap if overlap else None
    if overlap < minimum_overlap:
        classification = "INSUFFICIENT_OVERLAP_FOR_LINEAGE_DIAGNOSTIC"
    elif fraction is not None and fraction >= threshold:
        classification = "POTENTIAL_SHARED_LINEAGE_OR_PRICE_BASIS_MISLABEL_REVIEW_REQUIRED"
    else:
        classification = "NO_NEAR_EXACT_OHLC_MATCH_EVIDENCE_NOT_PROOF_OF_INDEPENDENCE"
    return {
        "status": classification,
        "exact_ohlc_match_count": matches,
        "overlap_bars": overlap,
        "exact_ohlc_match_fraction": fraction,
        "review_threshold_fraction": threshold,
        "source_independence_verified": False,
        "independent_source_confirmation": False,
        "automatic_source_promotion": False,
        "interpretation": (
            "near-exact Bid OHLC identity with a repository that describes its bars as mid-quote "
            "can reflect shared data lineage or a price-basis label problem; neither explanation "
            "is assumed without source-owner/provenance evidence"
        ),
    }


def _request_bytes(url, cap, label, timeout=60):
    req = Request(url, headers={"User-Agent": "Aurum-Intelligence/1.4.1 research-benchmark"})
    last = None
    for attempt in range(3):
        try:
            with urlopen(req, timeout=timeout) as response:
                final = urlparse(response.geturl())
                host = (final.hostname or "").lower()
                allowed = (
                    host == "huggingface.co"
                    or host.endswith(".huggingface.co")
                    or host.endswith(".hf.co")
                    or host.endswith(".xethub.hf.co")
                )
                if final.scheme != "https" or not allowed:
                    raise TickDatasetError(label + "_redirect_outside_approved_https_origins")
                size = response.headers.get("Content-Length")
                if size and int(size) > cap:
                    raise TickDatasetError(label + "_content_length_exceeded_cap")
                payload = response.read(cap + 1)
                if len(payload) > cap:
                    raise TickDatasetError(label + "_download_exceeded_cap")
                if size and len(payload) != int(size):
                    raise TickDatasetError(label + "_content_length_mismatch")
                return payload
        except TickDatasetError:
            raise
        except (OSError, URLError, HTTPError, TimeoutError) as exc:
            last = exc
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(label + "_download_failed_after_bounded_retries:" + type(last).__name__)


def discover_monthly_parts():
    url = DATASET_API + "?" + urlencode({"recursive": "false", "expand": "false", "limit": "100"})
    payload = _request_bytes(url, MAX_METADATA_BYTES, "dataset_metadata")
    try:
        entries = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TickDatasetError("invalid_dataset_tree_metadata") from exc
    if isinstance(entries, dict):
        entries = entries.get("items", entries.get("files", []))
    parts = [
        item for item in entries
        if item.get("type") == "file"
        and str(item.get("path", "")).startswith("year=2026/month=05/")
        and str(item.get("path", "")).endswith(".parquet")
    ]
    parts.sort(key=lambda item: str(item["path"]))
    if not parts:
        raise TickDatasetError("no_may_2026_parquet_parts_found")
    for part in parts:
        if not isinstance(part.get("size"), int) or part["size"] < 1:
            raise TickDatasetError("invalid_part_size_metadata")
        if part["size"] > MAX_FILE_BYTES:
            raise TickDatasetError("individual_parquet_part_exceeds_32_mib_limit:" + part["path"])
    if sum(part["size"] for part in parts) > MAX_TOTAL_PARQUET_BYTES:
        raise TickDatasetError("monthly_parquet_total_exceeds_128_mib_limit")
    return parts


def fetch_tick_parts(parts):
    files, payloads = [], []
    for part in parts:
        path = str(part["path"])
        payload = _request_bytes(FILE_BASE + path, MAX_FILE_BYTES, "tick_parquet")
        if len(payload) != part["size"]:
            raise TickDatasetError("parquet_size_differs_from_dataset_tree_metadata:" + path)
        files.append({"path": path, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
        payloads.append(payload)
    return files, payloads


def parquet_parts_to_frame(payloads):
    """Parse only quote columns from bounded monthly Parquet parts."""
    try:
        import numpy as np
        import pandas as pd
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise TickDatasetError("install_pandas_and_pyarrow_for_tick_parquet") from exc
    frames = []
    for payload in payloads:
        table = pq.read_table(io.BytesIO(payload), columns=["timestamp", "bid_price", "ask_price"])
        frames.append(table.to_pandas())
    if not frames:
        raise TickDatasetError("no_tick_parquet_payloads")
    data = pd.concat(frames, ignore_index=True)
    missing = [field for field in ("timestamp", "bid_price", "ask_price") if field not in data.columns]
    if missing:
        raise TickDatasetError("missing_tick_columns:" + ",".join(missing))
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    for field in ("bid_price", "ask_price"):
        data[field] = pd.to_numeric(data[field], errors="coerce")
    if data["timestamp"].isna().any():
        raise TickDatasetError("invalid_or_missing_tick_timestamp")
    quotes = data[["bid_price", "ask_price"]].to_numpy(dtype="float64")
    if not np.isfinite(quotes).all():
        raise TickDatasetError("non_finite_or_missing_bid_ask")
    if (quotes <= 0).any():
        raise TickDatasetError("non_positive_bid_ask")
    if (data["ask_price"] < data["bid_price"]).any():
        raise TickDatasetError("crossed_bid_ask_quote")
    data.sort_values("timestamp", kind="mergesort", inplace=True, ignore_index=True)
    return data


def aggregate_tick_frame(data):
    """Aggregate quote ticks into UTC M5 Bid/Ask/mid OHLC without filling missing bars."""
    import pandas as pd

    if data.empty:
        raise TickDatasetError("empty_tick_frame")
    data = data.copy()
    data["mid_price"] = (data["bid_price"] + data["ask_price"]) / 2.0
    data["bar_timestamp"] = data["timestamp"].dt.floor("5min")
    groups = data.groupby("bar_timestamp", sort=True)
    quote_bars = {}
    for basis, column in (("bid", "bid_price"), ("ask", "ask_price"), ("mid", "mid_price")):
        bars = groups[column].agg(["first", "max", "min", "last"])
        bars.rename(columns={"first": "open", "max": "high", "min": "low", "last": "close"}, inplace=True)
        quote_bars[basis] = {
            stamp.to_pydatetime(): {field: float(row[field]) for field in PRICE_FIELDS}
            for stamp, row in bars.iterrows()
        }
    return quote_bars


def write_sample_csv(path, rows, limit=250):
    if not rows:
        path.write_text("timestamp_utc_assumed_bar_open\n", encoding="utf-8")
        return 0
    if len(rows) > limit:
        indexes = sorted({round(i * (len(rows) - 1) / (limit - 1)) for i in range(limit)})
        rows = [rows[index] for index in indexes]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def run(output_dir):
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    primary_payload = base.fetch()
    primary_rows = base.parse_csv(primary_payload, min_rows=1000)
    primary_hash = hashlib.sha256(primary_payload).hexdigest()
    secondary_payload = base.fetch_secondary()
    secondary_rows = base.parse_csv(secondary_payload, min_rows=1000)
    secondary_hash = hashlib.sha256(secondary_payload).hexdigest()
    parts = discover_monthly_parts()
    downloaded_files, payloads = fetch_tick_parts(parts)
    tick_frame = parquet_parts_to_frame(payloads)
    quote_bars = aggregate_tick_frame(tick_frame)
    comparison, sample_rows = compare_price_bars(
        primary_rows, quote_bars, source_label=base.SOURCE_REPO,
    )
    secondary_comparison, secondary_sample_rows = compare_price_bars(
        secondary_rows, quote_bars, source_label=base.SECONDARY_SOURCE_REPO,
    )
    spread = spread_diagnostics(tick_frame["bid_price"].to_numpy(), tick_frame["ask_price"].to_numpy())
    stamp_min, stamp_max = tick_frame["timestamp"].min(), tick_frame["timestamp"].max()
    secondary_bid = secondary_comparison["basis_comparisons"]["bid"]
    lineage_diagnostic = classify_lineage_overlap(
        secondary_bid["exact_ohlc_match_count"],
        secondary_bid["exact_timestamp_overlap_bars"],
    )
    # These flags intentionally remain false until evidence is reviewed independently.
    # Download success, public URLs, repository license labels and bar concordance are not proof.
    unverified_evidence = {field: False for field in ALL_EVIDENCE_FIELDS}
    source_qualification_gates = {
        "primary_price": evaluate_source_qualification(base.SOURCE_REPO, unverified_evidence, {}),
        "secondary_price": evaluate_source_qualification(base.SECONDARY_SOURCE_REPO, unverified_evidence, {}),
        "tick_bidask": evaluate_source_qualification(DATASET_ID, unverified_evidence, {}),
    }
    report = {
        "status": "PASS" if (
            comparison["status"] == "THIRD_SOURCE_DIAGNOSTIC_ONLY"
            and secondary_comparison["status"] == "THIRD_SOURCE_DIAGNOSTIC_ONLY"
            and len(tick_frame) >= 1000
        ) else "FAIL",
        "benchmark_kind": "real tick bid/ask source integrity and M5 concordance diagnostic; not a strategy-performance benchmark",
        "dataset": {
            "id": DATASET_ID,
            "repository_url": "https://huggingface.co/datasets/CarlosSilva1/xauusd-ticks",
            "license": LICENSE_ID,
            "license_url": LICENSE_URL,
            "attribution": ATTRIBUTION,
            "source_description": "Aggregated tick feed reconstructed from broker data for research; broker provenance is not independently verified.",
            "data_period_documented": "2021-05-24 through 2026-05-24",
            "selected_partition": "year=2026/month=05",
            "files": downloaded_files,
            "bytes_downloaded": sum(item["bytes"] for item in downloaded_files),
            "rows": int(len(tick_frame)),
            "start_utc": stamp_min.isoformat(),
            "end_utc": stamp_max.isoformat(),
            "raw_tick_data_redistributed": False,
        },
        "primary_dataset": {
            "repository": base.SOURCE_REPO,
            "source_url": PRIMARY_SOURCE,
            "rows": len(primary_rows),
            "sha256": primary_hash,
            "timestamp_semantics_independently_verified": False,
        },
        "secondary_dataset": {
            "repository": base.SECONDARY_SOURCE_REPO,
            "source_url": base.SECONDARY_SOURCE,
            "rows": len(secondary_rows),
            "sha256": secondary_hash,
            "license_reference": base.SECONDARY_LICENSE,
            "quote_basis_reported_by_repository": "mid-quote OHLCV",
            "data_rights_independently_verified": False,
            "timestamp_semantics_independently_verified": False,
        },
        "tick_quality": {
            "non_finite_or_missing_rows": 0,
            "non_positive_quote_count": 0,
            "crossed_quote_count": 0,
            "duplicate_millisecond_timestamp_rows": int(tick_frame["timestamp"].duplicated(keep=False).sum()),
            "tick_count": int(len(tick_frame)),
            "distinct_utc_m5_bars": int(len(quote_bars["mid"])),
            "raw_row_count_note": "duplicate millisecond timestamps are reported, not dropped; multiple updates can share timestamp precision",
        },
        "spread": spread,
        "cross_source_comparison": comparison,
        "secondary_community_comparison": secondary_comparison,
        "data_lineage_diagnostic": lineage_diagnostic,
        "source_qualification_gates": source_qualification_gates,
        "runtime_ms": round((time.perf_counter() - started) * 1000, 3),
        "decision_output": "NO_TRADE",
        "reference_only": True,
        "automatic_price_correction": False,
        "automatic_timestamp_shift": False,
        "automatic_source_promotion": False,
        "validated_model": False,
        "calibrated_model": False,
        "live_signal_enabled": False,
    }
    (output_dir / "tick_bidask_benchmark.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    sample_count = write_sample_csv(output_dir / "tick_m5_comparison_sample_primary.csv", sample_rows, limit=250)
    secondary_sample_count = write_sample_csv(
        output_dir / "tick_m5_comparison_sample_secondary.csv", secondary_sample_rows, limit=250,
    )
    md = [
        "# Aurum third-source tick Bid/Ask benchmark",
        "",
        "Status: {}; decision gate: NO_TRADE.".format(report["status"]),
        "",
        "- Dataset: [{}]({})".format(DATASET_ID, report["dataset"]["repository_url"]),
        "- License: [{}]({}); attribution required.".format(LICENSE_ID, LICENSE_URL),
        "- Attribution: " + ATTRIBUTION,
        "- Tick rows acquired: {}; Parquet parts: {}; downloaded bytes: {}.".format(len(tick_frame), len(downloaded_files), report["dataset"]["bytes_downloaded"]),
        "- Tick period observed: {} to {}.".format(stamp_min.isoformat(), stamp_max.isoformat()),
        "- Aggregated M5 quote bars: {}; exact overlap with primary M5: {}; secondary M5: {}.".format(
            len(quote_bars["mid"]), comparison["shared_timestamps"], secondary_comparison["shared_timestamps"],
        ),
        "- Invalid bid/ask rows fail closed instead of being silently discarded; same-millisecond timestamps are counted and retained.",
        "",
        "## Observed spread (this dataset only)",
        "",
        "| Metric | Price units | Basis points of mid |",
        "|---|---:|---:|",
        "| Median | {:.8f} | {:.4f} |".format(spread["spread_price_units"]["median"], spread["spread_basis_points_of_mid"]["median"]),
        "| P95 | {:.8f} | {:.4f} |".format(spread["spread_price_units"]["p95"], spread["spread_basis_points_of_mid"]["p95"]),
        "| P99 | {:.8f} | {:.4f} |".format(spread["spread_price_units"]["p99"], spread["spread_basis_points_of_mid"]["p99"]),
        "",
        "These are historical quotes in the cited reconstructed feed, not measured costs from the user's broker/account.",
        "",
        "## Source qualification",
        "",
        "All three feeds remain reference-only. Provenance, intended-use rights, timestamp/bar semantics and quote basis need referenced independent evidence; execution review additionally needs broker authorization and measured costs.",
        "",
        "## Exact-timestamp M5 comparison",
        "",
        "| Community source | Tick basis | Shared bars | Exact OHLC matches | Close-difference median (bps) | Close-difference P95 (bps) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for label, result in ((base.SOURCE_REPO, comparison), (base.SECONDARY_SOURCE_REPO, secondary_comparison)):
        for basis in ("bid", "mid", "ask"):
            details = result["basis_comparisons"][basis]
            close_stats = details["absolute_close_difference_bps"]
            md.append("| {} | {} | {} | {} | {} | {} |".format(
                label, basis, details["exact_timestamp_overlap_bars"], details["exact_ohlc_match_count"],
                "n/a" if close_stats["median"] is None else "{:.4f}".format(close_stats["median"]),
                "n/a" if close_stats["p95"] is None else "{:.4f}".format(close_stats["p95"]),
            ))
    md += [
        "",
        "## Source-lineage and quote-basis warning",
        "",
        "- Secondary OHLCV versus tick-derived Bid: {} exact OHLC matches out of {} overlapping bars ({:.4%}).".format(
            lineage_diagnostic["exact_ohlc_match_count"], lineage_diagnostic["overlap_bars"],
            lineage_diagnostic["exact_ohlc_match_fraction"] or 0.0,
        ),
        "- Classification: {}.".format(lineage_diagnostic["status"]),
        "- The secondary repository describes its source as mid-quote OHLCV, while its bars are almost exactly identical to the tick-derived Bid bars. This suggests possible shared lineage or a price-basis labeling issue; it does not prove either explanation.",
        "- Treat the tick feed and this secondary OHLCV as non-independent for validation until lineage is verified. No source is automatically promoted.",
        "",
        "Timestamp matching uses exact UTC five-minute bucket starts. No offsets, rescaling, interpolation, synthetic continuity, or source promotion are applied.",
        "Source comparison is diagnostic only. The community feed's candle timestamp convention and executable price basis have not been independently proved; the third feed also discloses reconstructed broker provenance.",
        "",
        "Bounded overlap samples: {} primary and {} secondary timestamps; raw tick files are not included in the artifact.".format(
            sample_count, secondary_sample_count,
        ),
        "",
        "Machine-readable report: tick_bidask_benchmark.json.",
        "",
        "## Limits",
        "",
        "- Historical tick source is aggregated/reconstructed, not broker-authorized data for the user's account.",
        "- CC-BY-4.0 attribution is recorded; data provenance remains separate from license labeling.",
        "- M5 OHLC aggregation cannot establish intraminute TP/SL ordering; this is not a tick-level execution backtest.",
        "- This run validates data acquisition and diagnostic mechanics, not alpha, model calibration, profitability or world-leading accuracy.",
        "",
    ]
    (output_dir / "tick_bidask_benchmark.md").write_text("\n".join(md), encoding="utf-8")
    return {
        "status": report["status"],
        "dataset_rows": int(len(tick_frame)),
        "parquet_parts": len(downloaded_files),
        "distinct_m5_bars": len(quote_bars["mid"]),
        "primary_overlap_bars": comparison["shared_timestamps"],
        "secondary_overlap_bars": secondary_comparison["shared_timestamps"],
        "secondary_mid_close_difference_p95_bps": secondary_comparison["basis_comparisons"]["mid"]["absolute_close_difference_bps"]["p95"],
        "spread_median_price_units": spread["spread_price_units"]["median"],
        "spread_p95_price_units": spread["spread_price_units"]["p95"],
        "cross_source_status": comparison["status"],
        "decision_output": "NO_TRADE",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.output_dir)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "PASS" else 2
    except Exception as exc:
        print(json.dumps({
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc)[:400],
            "decision_output": "NO_TRADE",
        }, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
