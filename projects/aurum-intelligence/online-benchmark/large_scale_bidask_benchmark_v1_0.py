#!/usr/bin/env python3
"""Large-scale, no-trading XAUUSD M1 Bid/Ask benchmark over public historical bars."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np

REPOSITORY = "kevingtlin/Market-Data-Lab"
REPOSITORY_URL = "https://github.com/kevingtlin/Market-Data-Lab"
RAW_BASE = "https://raw.githubusercontent.com/kevingtlin/Market-Data-Lab/main/"
DATASET_ID = "XAUUSD-DUKASCOPY-M1-BID-ASK"
DATASET_METADATA_URL = "https://github.com/kevingtlin/Market-Data-Lab/blob/main/datasets/XAUUSD-DUKASCOPY-M1-BID-ASK/metadata.yaml"
PROVIDER_NOTE = "README/metadata report Dukascopy historical data acquired via dukascopy-node; upstream data rights and broker-feed identity have not been independently verified."
START_MONTH = (2023, 1)
END_MONTH_EXCLUSIVE = (2025, 7)
MIN_CANDLES = 700_000
MIN_SCENARIO_ANCHORS = 20_000
MAX_MONTHLY_FILE_BYTES = 6 * 1024 * 1024
MAX_TOTAL_DOWNLOAD_BYTES = 180 * 1024 * 1024
MAX_AUDIT_BYTES = 2 * 1024 * 1024
FETCH_TIMEOUT_SECONDS = 45
WARMUP_BARS = 30
HORIZON_BARS = 60
TP_MULTIPLIERS = (0.75, 1.0, 1.5)
STOP_ATR_MULTIPLIER = 1.0
STRATEGIES = ("momentum_5", "ema10_vs_ema30", "mean_reversion_z20", "range_breakout_20")
SPLIT_FRACTIONS = (0.70, 0.10, 0.20)
BOOTSTRAP_REPLICATES = 250
BOOTSTRAP_BLOCK_LENGTH = 100
ARTIFACT_NOTE = "Raw bid/ask OHLCV candles are never written to the report artifact."


class BenchmarkError(ValueError):
    """Raised when the source dataset or benchmark contract fails closed."""


def months_between(start=START_MONTH, end_exclusive=END_MONTH_EXCLUSIVE):
    year, month = start
    result = []
    while (year, month) < end_exclusive:
        result.append((year, month))
        month += 1
        if month == 13:
            year += 1
            month = 1
    return result


def month_string(year, month):
    return f"{year:04d}_{month:02d}"


def month_bounds_ms(year, month):
    start = int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp() * 1000)
    if month == 12:
        end = int(datetime(year + 1, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    else:
        end = int(datetime(year, month + 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    return start, end


def _request_bytes(url, cap, label, timeout=FETCH_TIMEOUT_SECONDS):
    req = Request(url, headers={"User-Agent": "Aurum-Intelligence/1.5-large-scale-research"})
    last_error = None
    for attempt in range(3):
        try:
            with urlopen(req, timeout=timeout) as response:
                final = urlparse(response.geturl())
                host = (final.hostname or "").lower()
                if final.scheme != "https" or host != "raw.githubusercontent.com":
                    raise BenchmarkError(label + "_redirect_outside_approved_https_origin")
                if not final.path.startswith("/kevingtlin/Market-Data-Lab/"):
                    raise BenchmarkError(label + "_redirect_outside_approved_repository")
                length = response.headers.get("Content-Length")
                if length and int(length) > cap:
                    raise BenchmarkError(label + "_content_length_exceeds_limit")
                payload = response.read(cap + 1)
                if len(payload) > cap:
                    raise BenchmarkError(label + "_download_exceeds_limit")
                if length and len(payload) != int(length):
                    raise BenchmarkError(label + "_content_length_mismatch")
                return payload
        except BenchmarkError:
            raise
        except (OSError, URLError, HTTPError, TimeoutError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.35 * (attempt + 1))
    raise RuntimeError(label + "_download_failed_after_bounded_retries:" + type(last_error).__name__)


def _parse_number(raw, field, row_number):
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise BenchmarkError(f"nonnumeric_{field}_at_row_{row_number}") from exc
    if not math.isfinite(value):
        raise BenchmarkError(f"nonfinite_{field}_at_row_{row_number}")
    return value


def _parse_timestamp_ms(raw, row_number):
    value = str(raw).strip()
    try:
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("not finite")
        if numeric >= 1_000_000_000_000:
            stamp = int(numeric)
        elif numeric >= 1_000_000_000:
            stamp = int(numeric * 1000)
        else:
            raise ValueError("timestamp numeric magnitude unsupported")
    except ValueError:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise BenchmarkError(f"unparseable_timestamp_at_row_{row_number}") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise BenchmarkError(f"naive_timestamp_at_row_{row_number}")
        stamp = int(parsed.astimezone(timezone.utc).timestamp() * 1000)
    if stamp % 60_000:
        raise BenchmarkError(f"timestamp_not_minute_aligned_at_row_{row_number}")
    return stamp


def parse_month_csv(payload, year, month, side, expected_lines=None):
    """Strictly parse one monthly file; return sorted timestamp and OHLC numpy arrays."""
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise BenchmarkError(f"{side}_{month_string(year, month)}_not_utf8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if not reader.fieldnames:
        raise BenchmarkError("missing_csv_header")
    normalized = {field.strip().lower(): field for field in reader.fieldnames}
    required = ("timestamp", "open", "high", "low", "close")
    if any(field not in normalized for field in required):
        raise BenchmarkError(f"{side}_{month_string(year, month)}_missing_required_columns")
    lower_ms, upper_ms = month_bounds_ms(year, month)
    stamps = []
    prices = []
    last_stamp = None
    for row_number, row in enumerate(reader, 2):
        stamp = _parse_timestamp_ms(row.get(normalized["timestamp"]), row_number)
        if not lower_ms <= stamp < upper_ms:
            raise BenchmarkError(f"{side}_{month_string(year, month)}_timestamp_outside_month")
        if last_stamp is not None and stamp <= last_stamp:
            raise BenchmarkError(f"{side}_{month_string(year, month)}_timestamps_not_strictly_increasing")
        ohlc = tuple(_parse_number(row.get(normalized[f]), f, row_number) for f in ("open", "high", "low", "close"))
        o, h, l, c = ohlc
        if min(ohlc) <= 0 or h < max(o, c) or l > min(o, c) or h < l:
            raise BenchmarkError(f"{side}_{month_string(year, month)}_invalid_ohlc_geometry")
        stamps.append(stamp)
        prices.append(ohlc)
        last_stamp = stamp
    if not stamps:
        raise BenchmarkError(f"{side}_{month_string(year, month)}_empty_month")
    if expected_lines is not None and len(stamps) + 1 != expected_lines:
        raise BenchmarkError(f"{side}_{month_string(year, month)}_audit_line_count_mismatch")
    return np.asarray(stamps, dtype=np.int64), np.asarray(prices, dtype=np.float64)


def parse_audit(payload, side):
    try:
        reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig"), newline=""))
        result = {}
        for row in reader:
            name = row.get("Name", "").strip()
            prefix = f"xauusd_{side}_m1_"
            if not name.startswith(prefix) or not name.endswith(".csv"):
                continue
            ym = name[len(prefix):-4]
            try:
                year_s, month_s = ym.split("_")
                key = (int(year_s), int(month_s))
                result[key] = {
                    "name": name,
                    "bytes": int(row["Bytes"]),
                    "lines": int(row["Lines"]),
                    "expected_lines": int(row["CalendarExpectedLines"]),
                    "status": row.get("Status", "").strip(),
                    "note": row.get("Note", "").strip(),
                    "first_utc": row.get("FirstUtc", "").strip(),
                    "last_utc": row.get("LastUtc", "").strip(),
                    "gap_count": int(row["GapCount"]),
                    "max_gap_minutes": int(row["MaxGapMinutes"]),
                }
            except (ValueError, KeyError) as exc:
                raise BenchmarkError(f"{side}_audit_row_malformed") from exc
        return result
    except UnicodeDecodeError as exc:
        raise BenchmarkError(f"{side}_audit_not_utf8") from exc


def audit_month_entry(audit, year, month, side):
    entry = audit.get((year, month))
    if entry is None:
        raise BenchmarkError(f"{side}_{month_string(year, month)}_missing_from_published_audit")
    if entry["status"] not in ("ok", "ok_market_session"):
        raise BenchmarkError(f"{side}_{month_string(year, month)}_audit_status_not_ok")
    if entry["gap_count"] != 0 and entry["status"] == "ok":
        raise BenchmarkError(f"{side}_{month_string(year, month)}_audit_reports_gaps")
    if entry["lines"] != entry["expected_lines"]:
        raise BenchmarkError(f"{side}_{month_string(year, month)}_audit_calendar_count_mismatch")
    return entry


def _ohlc_arrays_to_dataset(month_items):
    dtype = np.dtype([
        ("ts", "i8"),
        ("bo", "f8"), ("bh", "f8"), ("bl", "f8"), ("bc", "f8"),
        ("ao", "f8"), ("ah", "f8"), ("al", "f8"), ("ac", "f8"),
    ])
    arrays = []
    month_report = []
    total_bytes = 0
    for (year, month), bid_audit, ask_audit in month_items:
        month = int(month)
        stamp = month_string(year, month)
        bid_url = RAW_BASE + f"xauusd/bid/m1/xauusd_bid_m1_{stamp}.csv"
        ask_url = RAW_BASE + f"xauusd/ask/m1/xauusd_ask_m1_{stamp}.csv"
        bid_payload = _request_bytes(bid_url, MAX_MONTHLY_FILE_BYTES, f"bid_{stamp}")
        ask_payload = _request_bytes(ask_url, MAX_MONTHLY_FILE_BYTES, f"ask_{stamp}")
        total_bytes += len(bid_payload) + len(ask_payload)
        if total_bytes > MAX_TOTAL_DOWNLOAD_BYTES:
            raise BenchmarkError("total_download_exceeds_180_mib_limit")
        if len(bid_payload) != bid_audit["bytes"]:
            raise BenchmarkError(f"bid_{stamp}_byte_count_differs_from_audit")
        if len(ask_payload) != ask_audit["bytes"]:
            raise BenchmarkError(f"ask_{stamp}_byte_count_differs_from_audit")
        bid_ts, bid_prices = parse_month_csv(bid_payload, year, month, "bid", bid_audit["lines"])
        ask_ts, ask_prices = parse_month_csv(ask_payload, year, month, "ask", ask_audit["lines"])
        common_ts, bid_idx, ask_idx = np.intersect1d(bid_ts, ask_ts, assume_unique=True, return_indices=True)
        if len(common_ts) < 20_000:
            raise BenchmarkError(f"{stamp}_insufficient_bid_ask_timestamp_join")
        data = np.empty(len(common_ts), dtype=dtype)
        data["ts"] = common_ts
        for i, field in enumerate(("o", "h", "l", "c")):
            data["b" + field] = bid_prices[bid_idx, i]
            data["a" + field] = ask_prices[ask_idx, i]
        for bfield, afield in (("bo", "ao"), ("bh", "ah"), ("bl", "al"), ("bc", "ac")):
            if np.any(data[afield] < data[bfield]):
                raise BenchmarkError(f"{stamp}_ask_{afield}_below_bid_{bfield}")
        arrays.append(data)
        month_report.append({
            "month": stamp,
            "bid_rows": int(len(bid_ts)),
            "ask_rows": int(len(ask_ts)),
            "paired_candles": int(len(common_ts)),
            "bid_ask_unmatched_rows": int(len(bid_ts) + len(ask_ts) - 2 * len(common_ts)),
            "bid_bytes": len(bid_payload),
            "ask_bytes": len(ask_payload),
            "bid_sha256": hashlib.sha256(bid_payload).hexdigest(),
            "ask_sha256": hashlib.sha256(ask_payload).hexdigest(),
            "published_audit_status": bid_audit["status"],
            "published_audit_gaps": bid_audit["gap_count"] + ask_audit["gap_count"],
        })
        del bid_payload, ask_payload, bid_ts, ask_ts, bid_prices, ask_prices, data
    if not arrays:
        raise BenchmarkError("no_month_arrays")
    dataset = np.concatenate(arrays)
    if np.any(dataset["ts"][1:] <= dataset["ts"][:-1]):
        raise BenchmarkError("paired_candle_timestamps_not_globally_strictly_increasing")
    return dataset, month_report, total_bytes


def load_dataset():
    months = months_between()
    if len(months) < 24:
        raise BenchmarkError("selected_historical_window_is_not_long_enough")
    bid_audit_payload = _request_bytes(RAW_BASE + "xauusd/bid/m1/bid_m1_file_audit.csv", MAX_AUDIT_BYTES, "bid_audit")
    ask_audit_payload = _request_bytes(RAW_BASE + "xauusd/ask/m1/ask_m1_file_audit.csv", MAX_AUDIT_BYTES, "ask_audit")
    bid_audit = parse_audit(bid_audit_payload, "bid")
    ask_audit = parse_audit(ask_audit_payload, "ask")
    month_items = []
    for year, month in months:
        month_items.append(( (year, month), audit_month_entry(bid_audit, year, month, "bid"),
                             audit_month_entry(ask_audit, year, month, "ask") ))
    dataset, month_report, total_bytes = _ohlc_arrays_to_dataset(month_items)
    if len(dataset) < MIN_CANDLES:
        raise BenchmarkError(f"paired_candles_below_700000_gate:{len(dataset)}")
    return dataset, month_report, total_bytes, {
        "bid_audit_sha256": hashlib.sha256(bid_audit_payload).hexdigest(),
        "ask_audit_sha256": hashlib.sha256(ask_audit_payload).hexdigest(),
        "selected_months": len(months),
        "first_month": month_string(*months[0]),
        "last_month": month_string(*months[-1]),
    }


def summarize(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"n": 0, "min": None, "p05": None, "median": None, "p95": None, "p99": None, "max": None}
    return {
        "n": int(len(values)),
        "min": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def market_quality(dataset):
    ts = dataset["ts"]
    price_fields = ("bo", "bh", "bl", "bc", "ao", "ah", "al", "ac")
    finite = np.ones(len(dataset), dtype=bool)
    for field in price_fields:
        finite &= np.isfinite(dataset[field])
    valid = finite.copy()
    for field in price_fields:
        valid &= dataset[field] > 0
    valid &= dataset["bh"] >= np.maximum(dataset["bo"], dataset["bc"])
    valid &= dataset["bl"] <= np.minimum(dataset["bo"], dataset["bc"])
    valid &= dataset["bh"] >= dataset["bl"]
    valid &= dataset["ah"] >= np.maximum(dataset["ao"], dataset["ac"])
    valid &= dataset["al"] <= np.minimum(dataset["ao"], dataset["ac"])
    valid &= dataset["ah"] >= dataset["al"]
    valid &= dataset["ao"] >= dataset["bo"]
    valid &= dataset["ah"] >= dataset["bh"]
    valid &= dataset["al"] >= dataset["bl"]
    valid &= dataset["ac"] >= dataset["bc"]
    if not np.all(valid):
        raise BenchmarkError(f"global_paired_candle_quality_failed:{int((~valid).sum())}")
    gaps = np.diff(ts) // 60_000
    weekdays = ((ts // 86_400_000 + 3) % 7) < 5
    mid_o = (dataset["bo"] + dataset["ao"]) / 2.0
    mid_h = (dataset["bh"] + dataset["ah"]) / 2.0
    mid_l = (dataset["bl"] + dataset["al"]) / 2.0
    mid_c = (dataset["bc"] + dataset["ac"]) / 2.0
    zero_range = (mid_h == mid_l)
    return {
        "row_count": int(len(dataset)),
        "rows_mon_fri_utc": int(weekdays.sum()),
        "rows_weekend_utc": int((~weekdays).sum()),
        "duplicate_timestamps": int((gaps == 0).sum()),
        "non_minute_intervals_including_closed_sessions": int((gaps != 1).sum()),
        "intervals_over_5_minutes_including_closed_sessions": int((gaps > 5).sum()),
        "max_chronological_interval_minutes": int(np.max(gaps)) if len(gaps) else 0,
        "zero_range_mid_bars": int(zero_range.sum()),
        "zero_range_fraction": float(zero_range.mean()),
        "weekday_active_nonflat_rows": int((weekdays & ~zero_range).sum()),
        "all_active_nonflat_rows": int((~zero_range).sum()),
        "invalid_or_crossed_quote_rows": 0,
        "spread_close_price_units": summarize(dataset["ac"] - dataset["bc"]),
        "spread_close_bps_mid": summarize((dataset["ac"] - dataset["bc"]) / ((dataset["ac"] + dataset["bc"]) / 2.0) * 10_000.0),
        "time_start_utc": datetime.fromtimestamp(ts[0] / 1000, timezone.utc).isoformat(),
        "time_end_utc": datetime.fromtimestamp(ts[-1] / 1000, timezone.utc).isoformat(),
        "weekdays_masked_for_scenario_windows": True,
    }


def ema(values, period):
    alpha = 2.0 / (period + 1.0)
    result = np.empty(len(values), dtype=np.float64)
    result[0] = values[0]
    for idx in range(1, len(values)):
        result[idx] = alpha * values[idx] + (1.0 - alpha) * result[idx - 1]
    return result


def indicators(dataset):
    mid_o = (dataset["bo"] + dataset["ao"]) / 2.0
    mid_h = (dataset["bh"] + dataset["ah"]) / 2.0
    mid_l = (dataset["bl"] + dataset["al"]) / 2.0
    mid_c = (dataset["bc"] + dataset["ac"]) / 2.0
    previous = np.concatenate(([mid_c[0]], mid_c[:-1]))
    true_range = np.maximum(mid_h - mid_l, np.maximum(np.abs(mid_h - previous), np.abs(mid_l - previous)))
    atr = np.full(len(dataset), np.nan, dtype=np.float64)
    if len(dataset) >= 14:
        atr[13:] = np.convolve(true_range, np.ones(14) / 14.0, mode="valid")
    return {
        "mid_open": mid_o,
        "mid_high": mid_h,
        "mid_low": mid_l,
        "mid_close": mid_c,
        "atr14": atr,
        "ema10": ema(mid_c, 10),
        "ema30": ema(mid_c, 30),
    }


def candidate_anchor_indices(dataset, required_count=MIN_SCENARIO_ANCHORS, atr_values=None):
    ts = dataset["ts"]
    n = len(ts)
    if n <= WARMUP_BARS + HORIZON_BARS:
        raise BenchmarkError("dataset_too_short_for_scenario_windows")
    weekday = ((ts // 86_400_000 + 3) % 7) < 5
    edge_ok = (np.diff(ts) == 60_000) & weekday[:-1] & weekday[1:]
    bad_prefix = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(~edge_ok, dtype=np.int64)))
    anchors = np.arange(WARMUP_BARS, n - HORIZON_BARS, dtype=np.int64)
    continuous = (bad_prefix[anchors + HORIZON_BARS] - bad_prefix[anchors - WARMUP_BARS]) == 0
    if atr_values is not None:
        atr_values = np.asarray(atr_values, dtype=np.float64)
        if len(atr_values) != n:
            raise BenchmarkError("atr_array_length_mismatch")
        continuous &= np.isfinite(atr_values[anchors]) & (atr_values[anchors] > 0)
    eligible = anchors[continuous]
    if len(eligible) < required_count:
        raise BenchmarkError(f"real_contiguous_windows_below_20000_gate:{len(eligible)}")
    if len(eligible) == required_count:
        selected = eligible.copy()
    else:
        offsets = np.rint(np.linspace(0, len(eligible) - 1, required_count)).astype(np.int64)
        selected = eligible[offsets]
    if len(np.unique(selected)) != required_count:
        raise BenchmarkError("scenario_anchor_selection_not_unique")
    return selected, int(len(eligible))


def signals_for_anchors(dataset, anchors, features):
    close = features["mid_close"]
    high = features["mid_high"]
    low = features["mid_low"]
    ema10, ema30 = features["ema10"], features["ema30"]
    signals = {name: np.zeros(len(anchors), dtype=np.int8) for name in STRATEGIES}
    for j, idx in enumerate(anchors):
        idx = int(idx)
        momentum_diff = close[idx] - close[idx - 5]
        signals["momentum_5"][j] = 1 if momentum_diff > 0 else (-1 if momentum_diff < 0 else 0)
        signals["ema10_vs_ema30"][j] = 1 if ema10[idx] > ema30[idx] else (-1 if ema10[idx] < ema30[idx] else 0)
        window = close[idx - 19:idx + 1]
        sd = float(np.std(window, ddof=0))
        zscore = (close[idx] - float(np.mean(window))) / sd if sd > 0 else 0.0
        signals["mean_reversion_z20"][j] = -1 if zscore > 1.0 else (1 if zscore < -1.0 else 0)
        prior_high = float(np.max(high[idx - 20:idx]))
        prior_low = float(np.min(low[idx - 20:idx]))
        signals["range_breakout_20"][j] = 1 if close[idx] > prior_high else (-1 if close[idx] < prior_low else 0)
    return signals


def evaluate_path(dataset, features, anchor_index, direction, target_atr_multiple, horizon=HORIZON_BARS):
    """Bid/Ask execution-side touch labels with explicit same-bar ambiguity handling."""
    idx = int(anchor_index)
    atr = float(features["atr14"][idx])
    if not math.isfinite(atr) or atr <= 0:
        return {"label": None, "status": "invalid_atr", "bars_to_outcome": None}
    if direction == 1:
        entry = float(dataset["ao"][idx + 1])
        target = entry + target_atr_multiple * atr
        stop = entry - STOP_ATR_MULTIPLIER * atr
    elif direction == -1:
        entry = float(dataset["bo"][idx + 1])
        target = entry - target_atr_multiple * atr
        stop = entry + STOP_ATR_MULTIPLIER * atr
    else:
        raise BenchmarkError("invalid_trade_direction")
    for offset in range(1, horizon + 1):
        j = idx + offset
        if direction == 1:
            opened, high, low = float(dataset["bo"][j]), float(dataset["bh"][j]), float(dataset["bl"][j])
            if opened >= target:
                return {"label": 1, "status": "tp_at_open", "bars_to_outcome": offset}
            if opened <= stop:
                return {"label": 0, "status": "sl_at_open", "bars_to_outcome": offset}
            hit_tp, hit_sl = high >= target, low <= stop
        else:
            opened, high, low = float(dataset["ao"][j]), float(dataset["ah"][j]), float(dataset["al"][j])
            if opened <= target:
                return {"label": 1, "status": "tp_at_open", "bars_to_outcome": offset}
            if opened >= stop:
                return {"label": 0, "status": "sl_at_open", "bars_to_outcome": offset}
            hit_tp, hit_sl = low <= target, high >= stop
        if hit_tp and hit_sl:
            return {"label": None, "status": "ambiguous_same_bar", "bars_to_outcome": offset}
        if hit_tp:
            return {"label": 1, "status": "tp_first", "bars_to_outcome": offset}
        if hit_sl:
            return {"label": 0, "status": "sl_first", "bars_to_outcome": offset}
    return {"label": 0, "status": "no_hit_within_horizon", "bars_to_outcome": horizon}


def scenario_matrix(dataset, anchors, features):
    results_by_anchor_side = np.full((len(anchors), 2), -1, dtype=np.int8)
    status_counts = {}
    elapsed = {}
    samples = []
    total = 0
    for k, anchor in enumerate(anchors):
        for side_idx, direction in enumerate((1, -1)):
            for rr in TP_MULTIPLIERS:
                result = evaluate_path(dataset, features, int(anchor), direction, rr)
                key = f"{'long' if direction == 1 else 'short'}_tp{rr:.2f}atr_sl1atr_h60"
                bucket = status_counts.setdefault(key, {
                    "scenarios": 0, "tp_first": 0, "sl_first": 0,
                    "no_hit_within_horizon": 0, "ambiguous_same_bar": 0, "invalid_atr": 0,
                })
                bucket["scenarios"] += 1
                bucket[result["status"]] = bucket.get(result["status"], 0) + 1
                total += 1
                if rr == 1.0:
                    results_by_anchor_side[k, side_idx] = -1 if result["label"] is None else int(result["label"])
                if len(samples) < 25 and k % 777 == 0 and rr == 1.0:
                    samples.append({
                        "scenario_anchor_ordinal": k,
                        "anchor_index": int(anchor),
                        "side": "long" if direction == 1 else "short",
                        "tp_atr_multiple": rr,
                        "sl_atr_multiple": STOP_ATR_MULTIPLIER,
                        "horizon_bars": HORIZON_BARS,
                        "label": result["label"],
                        "status": result["status"],
                    })
    expected = len(anchors) * 2 * len(TP_MULTIPLIERS)
    if total != expected:
        raise BenchmarkError(f"scenario_matrix_count_mismatch:{total}!={expected}")
    return results_by_anchor_side, status_counts, total, samples


def _reliability(probabilities, labels, bins=10):
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    output = []
    ece = 0.0
    mce = 0.0
    for b in range(bins):
        low, high = b / bins, (b + 1) / bins
        mask = (p >= low) & ((p < high) if b < bins - 1 else (p <= high))
        count = int(mask.sum())
        if not count:
            output.append({"bin": b, "lower": low, "upper": high, "n": 0, "mean_probability": None, "observed_rate": None, "absolute_gap": None})
            continue
        mean_p = float(p[mask].mean())
        observed = float(y[mask].mean())
        gap = abs(mean_p - observed)
        ece += count / len(p) * gap
        mce = max(mce, gap)
        output.append({"bin": b, "lower": low, "upper": high, "n": count, "mean_probability": mean_p, "observed_rate": observed, "absolute_gap": gap})
    return {"ece_descriptive": ece, "mce_descriptive": mce, "bins": output}


def _metric_pack(probabilities, labels):
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1 - 1e-6)
    y = np.asarray(labels, dtype=np.float64)
    if len(y) == 0:
        return {"n": 0, "brier": None, "log_loss": None, "base_rate": None, "constant_0_5_brier": None, "brier_skill_vs_0_5": None, "reliability": None}
    brier = float(np.mean((p - y) ** 2))
    baseline = float(np.mean((0.5 - y) ** 2))
    return {
        "n": int(len(y)),
        "brier": brier,
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "base_rate": float(np.mean(y)),
        "constant_0_5_brier": baseline,
        "brier_skill_vs_0_5": float(1 - brier / baseline) if baseline > 0 else None,
        "reliability": _reliability(p, y),
    }


def _moving_block_ci(improvements, replicates=BOOTSTRAP_REPLICATES, block_length=BOOTSTRAP_BLOCK_LENGTH, seed=20261017):
    values = np.asarray(improvements, dtype=np.float64)
    if len(values) < 50:
        return {"status": "INSUFFICIENT_SCORED_SIGNALS", "replicates": 0, "block_length": block_length, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    count = len(values)
    blocks = math.ceil(count / block_length)
    positions = np.arange(block_length, dtype=np.int64)
    means = np.empty(replicates, dtype=np.float64)
    for r in range(replicates):
        starts = rng.integers(0, count, size=blocks)
        indices = ((starts[:, None] + positions[None, :]) % count).reshape(-1)[:count]
        means[r] = float(values[indices].mean())
    return {
        "status": "DIAGNOSTIC_ONLY",
        "replicates": replicates,
        "block_length_scenario_events": block_length,
        "seed": seed,
        "point_estimate_mean_brier_improvement_vs_constant_0_5": float(values.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "interpretation": "positive values favor the matured-label probability baseline; interval is descriptive, not proof of alpha",
    }


def prequential_evaluation(dataset, anchors, signals, labels_by_anchor_side, minimum_scored_events=MIN_SCENARIO_ANCHORS, minimum_final_test_events=1000):
    counts = {(rule, side): [0, 0] for rule in STRATEGIES for side in (-1, 1)}
    global_counts = [0, 0]  # matured event count, matured TP-before-SL wins
    pending = []
    pending_cursor = 0
    prediction_rows = []
    by_rule_records = {rule: [] for rule in STRATEGIES}
    all_event_count = 0
    neutral_count = 0
    ambiguous_count = 0
    final_test_start = int(len(anchors) * 0.80)
    frozen_final_test_counts = None
    frozen_final_test_global_counts = None
    for k, anchor in enumerate(anchors):
        anchor = int(anchor)
        # Freeze the probability baseline at the first final-test timestamp.
        # Only events matured strictly before that timestamp may seed final-test predictions.
        if k <= final_test_start:
            while pending_cursor < len(pending) and pending[pending_cursor][0] < anchor:
                _, rule, side, label = pending[pending_cursor]
                counts[(rule, side)][0] += 1
                counts[(rule, side)][1] += int(label)
                global_counts[0] += 1
                global_counts[1] += int(label)
                pending_cursor += 1
            if k == final_test_start:
                frozen_final_test_counts = {key: list(value) for key, value in counts.items()}
                frozen_final_test_global_counts = list(global_counts)
        elif frozen_final_test_counts is not None:
            counts = {key: list(value) for key, value in frozen_final_test_counts.items()}
            global_counts = list(frozen_final_test_global_counts)
        for rule in STRATEGIES:
            direction = int(signals[rule][k])
            if direction == 0:
                neutral_count += 1
                continue
            all_event_count += 1
            side_idx = 0 if direction == 1 else 1
            raw_label = int(labels_by_anchor_side[k, side_idx])
            if raw_label < 0:
                ambiguous_count += 1
                continue
            n, wins = counts[(rule, direction)]
            probability = (wins + 1.0) / (n + 2.0)
            global_prior_probability = (global_counts[1] + 1.0) / (global_counts[0] + 2.0)
            row = {
                "anchor_index": anchor,
                "maturity_index": anchor + HORIZON_BARS,
                "rule": rule,
                "side": "long" if direction == 1 else "short",
                "side_index": side_idx,
                "probability": probability,
                "global_prior_probability": global_prior_probability,
                "label": raw_label,
                "prior_matured_labels": n,
                "split": "development" if k < int(len(anchors) * 0.70) else (
                    "calibration" if k < int(len(anchors) * 0.80) else "final_holdout"
                ),
            }
            prediction_rows.append(row)
            by_rule_records[rule].append(row)
            pending.append((anchor + HORIZON_BARS, rule, direction, raw_label))
    # Score all prequential predictions by rule and by chronological partition.
    probs = np.asarray([row["probability"] for row in prediction_rows], dtype=np.float64)
    labels = np.asarray([row["label"] for row in prediction_rows], dtype=np.int8)
    global_probs = np.asarray([row["global_prior_probability"] for row in prediction_rows], dtype=np.float64)
    improvements = (0.5 - labels) ** 2 - (probs - labels) ** 2
    global_improvements = (global_probs - labels) ** 2 - (probs - labels) ** 2
    aggregate_metrics = _metric_pack(probs, labels)
    global_baseline_metrics = _metric_pack(global_probs, labels)
    global_baseline_brier = global_baseline_metrics["brier"]
    model_brier = aggregate_metrics["brier"]
    output = {
        "status": "DIAGNOSTIC_ONLY",
        "matured_label_rule": "Only outcomes whose end index is strictly before the next decision index update development/calibration priors; the final 20% test freezes those priors at its first timestamp and never consumes any final-test labels.",
        "probability_method": "rule-and-side-specific Laplace-smoothed historical prevalence from previously matured labels; development/calibration are prequential, final 20% uses frozen matured priors from before its first test timestamp; no optimized model parameters",
        "strategy_event_candidates": all_event_count,
        "neutral_abstentions": neutral_count,
        "ambiguous_same_bar_events_excluded": ambiguous_count,
        "binary_scored_strategy_events": len(prediction_rows),
        "prediction_event_rows_are_raw_price_free": True,
        "aggregate": {
            **aggregate_metrics,
            "brier_improvement_vs_global_matured_label_baseline": (
                float(global_baseline_brier - model_brier)
                if global_baseline_brier is not None and model_brier is not None else None
            ),
            "brier_skill_vs_global_matured_label_baseline": (
                float(1.0 - model_brier / global_baseline_brier)
                if global_baseline_brier is not None and global_baseline_brier > 0 and model_brier is not None else None
            ),
        },
        "global_prequential_base_rate_baseline": global_baseline_metrics,
        "moving_block_bootstrap_brier_improvement_vs_constant_0_5": _moving_block_ci(improvements),
        "moving_block_bootstrap_brier_improvement_vs_global_prequential_base_rate": _moving_block_ci(global_improvements, seed=20261018),
        "by_rule": {},
        "chronological_partitions": {},
    }
    for rule, rows in by_rule_records.items():
        output["by_rule"][rule] = _metric_pack([row["probability"] for row in rows], [row["label"] for row in rows])
    for partition in ("development", "calibration", "final_holdout"):
        rows = [row for row in prediction_rows if row["split"] == partition]
        partition_metrics = _metric_pack([row["probability"] for row in rows], [row["label"] for row in rows])
        global_partition_metrics = _metric_pack(
            [row["global_prior_probability"] for row in rows], [row["label"] for row in rows]
        )
        p_brier = partition_metrics["brier"]
        g_brier = global_partition_metrics["brier"]
        output["chronological_partitions"][partition] = {
            **partition_metrics,
            "global_matured_label_baseline": global_partition_metrics,
            "brier_improvement_vs_global_matured_label_baseline": (
                float(g_brier - p_brier) if g_brier is not None and p_brier is not None else None
            ),
            "brier_skill_vs_global_matured_label_baseline": (
                float(1.0 - p_brier / g_brier)
                if g_brier is not None and g_brier > 0 and p_brier is not None else None
            ),
            "endpoints_are_chronological": True,
            "parameter_fitting_on_partition": False,
            "final_test_priors_frozen": partition == "final_holdout",
        }
    if output["binary_scored_strategy_events"] < minimum_scored_events:
        raise BenchmarkError(
            f"binary_scored_strategy_events_below_gate:{output['binary_scored_strategy_events']}<{minimum_scored_events}"
        )
    if output["chronological_partitions"]["final_holdout"]["n"] < minimum_final_test_events:
        raise BenchmarkError(
            "final_holdout_scored_events_below_gate:"
            + str(output["chronological_partitions"]["final_holdout"]["n"])
            + "<" + str(minimum_final_test_events)
        )
    return output, prediction_rows


def _report_timestamp(dataset, index):
    return datetime.fromtimestamp(int(dataset["ts"][int(index)]) / 1000, timezone.utc).isoformat()


def _write_signal_sample(path, dataset, rows, maximum=250):
    if not rows:
        path.write_text("scenario_id,timestamp_utc,rule,side,probability,label,prior_matured_labels,split\n", encoding="utf-8")
        return 0
    if len(rows) > maximum:
        indices = np.linspace(0, len(rows) - 1, maximum).round().astype(int)
        rows = [rows[int(idx)] for idx in indices]
    fields = ("scenario_id", "timestamp_utc", "rule", "side", "probability", "global_prior_probability", "label", "prior_matured_labels", "split")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows, 1):
            writer.writerow({
                "scenario_id": index,
                "timestamp_utc": _report_timestamp(dataset, row["anchor_index"]),
                "rule": row["rule"],
                "side": row["side"],
                "probability": f'{row["probability"]:.8f}',
                "global_prior_probability": f'{row["global_prior_probability"]:.8f}',
                "label": row["label"],
                "prior_matured_labels": row["prior_matured_labels"],
                "split": row["split"],
            })
    return len(rows)


def run(output_dir, required_candles=MIN_CANDLES, required_scenarios=MIN_SCENARIO_ANCHORS):
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset, months_report, total_bytes, audit_meta = load_dataset()
    quality = market_quality(dataset)
    if len(dataset) < required_candles:
        raise BenchmarkError(f"paired_candle_contract_failed:{len(dataset)}<{required_candles}")
    if quality["rows_mon_fri_utc"] < required_candles:
        raise BenchmarkError(
            f"weekday_candle_contract_failed:{quality['rows_mon_fri_utc']}<{required_candles}"
        )
    if quality["weekday_active_nonflat_rows"] < required_candles:
        raise BenchmarkError(
            f"weekday_active_candle_contract_failed:{quality['weekday_active_nonflat_rows']}<{required_candles}"
        )
    features = indicators(dataset)
    anchors, eligible_windows = candidate_anchor_indices(dataset, required_scenarios, features["atr14"])
    signals = signals_for_anchors(dataset, anchors, features)
    labels_by_anchor_side, scenario_outcomes, scenario_count, scenario_samples = scenario_matrix(dataset, anchors, features)
    evaluation, prediction_rows = prequential_evaluation(dataset, anchors, signals, labels_by_anchor_side)
    anchor_count = len(anchors)
    unique_anchor_timestamps = len(np.unique(dataset["ts"][anchors]))
    if unique_anchor_timestamps < required_scenarios:
        raise BenchmarkError(f"unique_real_scenario_timestamps_below_gate:{unique_anchor_timestamps}")
    if scenario_count < required_scenarios:
        raise BenchmarkError(f"scenario_matrix_below_gate:{scenario_count}")
    report = {
        "status": "PASS",
        "benchmark_id": "AURUM-LARGE-SCALE-BIDASK-M1-v1.0",
        "benchmark_kind": "real historical M1 Bid/Ask candles with causal TP-before-SL scenario scoring; not a live execution or profitability claim",
        "decision_output": "NO_TRADE",
        "dataset": {
            "id": DATASET_ID,
            "repository": REPOSITORY,
            "repository_url": REPOSITORY_URL,
            "metadata_url": DATASET_METADATA_URL,
            "provider_reported_by_repository": "Dukascopy via dukascopy-node",
            "provider_provenance_independently_verified": False,
            "upstream_data_rights_verified": False,
            "intended_use": "research/benchmark only until provenance and usage rights are verified",
            "price_sides": ["bid", "ask"],
            "resolution": "M1",
            "timezone": "UTC",
            "timestamp_unit": "Unix epoch milliseconds per repository metadata",
            "selected_period": {"start": audit_meta["first_month"], "end_inclusive": audit_meta["last_month"]},
            "months_requested": audit_meta["selected_months"],
            "total_monthly_file_bytes": total_bytes,
            "raw_candles_redistributed": False,
            "audit_log_sha256": {
                "bid": audit_meta["bid_audit_sha256"],
                "ask": audit_meta["ask_audit_sha256"],
            },
            "monthly_manifest": months_report,
            "dataset_rows": int(len(dataset)),
            "paired_valid_bid_ask_candles": int(len(dataset)),
            "minimum_candles_required": required_candles,
            "minimum_dataset_rows_required": 20_000,
            "all_prices_validated": True,
            "data_quality": quality,
        },
        "scenario_contract": {
            "minimum_unique_real_timestamp_scenarios_required": required_scenarios,
            "minimum_binary_scored_strategy_events_required": required_scenarios,
            "selected_unique_anchor_timestamps": unique_anchor_timestamps,
            "selected_anchor_timestamps_with_valid_atr": int(np.sum(np.isfinite(features["atr14"][anchors]) & (features["atr14"][anchors] > 0))),
            "eligible_real_contiguous_windows_with_valid_atr": eligible_windows,
            "anchor_contexts": anchor_count,
            "lookback_warmup_bars": WARMUP_BARS,
            "entry_convention": "signals evaluated at completed anchor close; entries use next M1 open",
            "outcome_horizon_bars": HORIZON_BARS,
            "same_bar_tp_sl_collision": "excluded from binary scoring, counted explicitly",
            "unresolved_horizon": "negative for the finite-horizon event TP-before-SL within 60 contiguous minutes",
            "atr_source": "14-period mean true range of bid/ask mid OHLC using only bars through decision close",
            "take_profit_atr_multipliers": list(TP_MULTIPLIERS),
            "stop_loss_atr_multiplier": STOP_ATR_MULTIPLIER,
            "directions_per_anchor": 2,
            "tp_sl_path_scenarios": scenario_count,
            "scenario_outcome_counts": scenario_outcomes,
            "strategy_rules": list(STRATEGIES),
            "strategy_decision_slots": int(anchor_count * len(STRATEGIES)),
            "unique_scenario_timestamps_satisfy_gate": unique_anchor_timestamps >= required_scenarios,
        },
        "prequential_probability_evaluation": evaluation,
        "scenario_sample_summary": {
            "sample_count": len(scenario_samples),
            "raw_prices_included": False,
            "sample_events": scenario_samples,
        },
        "methodological_limits": [
            PROVIDER_NOTE,
            "Public repository access is not proof of redistribution or commercial-use permission; raw monthly data are not included in artifacts.",
            "This benchmark samples 20,000 evenly spaced, contiguous real timestamp windows across the selected period; windows are timestamp-distinct but not statistically independent by construction.",
            "The strategy rules are simple fixed baselines, not a tuned Aurum production model.",
            "The probability baseline is a Laplace-smoothed rule-side historical prevalence using only labels matured before each decision; it does not prove predictive edge.",
            "Bid/Ask OHLC touch labels cannot reconstruct tick order inside the same minute. Same-bar TP/SL collisions are excluded rather than resolving the order optimistically.",
            "The public source's quote provenance, tradeability at the user's intended venue, historical spread representativeness, and exact data-use rights remain unverified.",
            "No commission, swap, slippage, market impact, latency or user-broker fill model is available from this source alone.",
        ],
        "artifact_policy": ARTIFACT_NOTE,
        "live_signal_enabled": False,
        "deployment_calibrated": False,
        "validated_model": False,
        "validated_for_profitability": False,
        "raw_data_included_in_artifact": False,
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    report_path = output_dir / "large_scale_bidask_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    sample_count = _write_signal_sample(output_dir / "large_scale_scenario_sample.csv", dataset, prediction_rows)
    md = [
        "# Aurum large-scale XAUUSD Bid/Ask benchmark v1.0",
        "",
        "Result: PASS for the acquisition, data-contract and benchmark-run gates only. Trading decision: NO_TRADE.",
        "",
        "## Real historical dataset",
        "",
        f"- Repository: {REPOSITORY_URL}",
        f"- Dataset: {DATASET_ID}; price sides: Bid + Ask; resolution: M1; timezone: UTC.",
        f"- Window: {audit_meta['first_month']} through {audit_meta['last_month']}; {audit_meta['selected_months']} calendar months.",
        f"- Paired valid Bid/Ask candles: {len(dataset):,}; minimum: {required_candles:,}.",
        f"- Weekday active non-flat Bid/Ask candles: {quality['weekday_active_nonflat_rows']:,}; minimum: {required_candles:,}.",
        f"- Unique real timestamp windows with valid ATR: {unique_anchor_timestamps:,}; minimum: {required_scenarios:,}.",
        f"- TP/SL market-path scenarios: {scenario_count:,}; minimum: {required_scenarios:,}.",
        f"- Binary-scored directional strategy events: {evaluation['binary_scored_strategy_events']:,}.",
        f"- Total bounded monthly downloads: {total_bytes:,} bytes; raw prices are not redistributed.",
        "",
        "## Data quality and spread",
        "",
        f"- Invalid or crossed Bid/Ask rows: {quality['invalid_or_crossed_quote_rows']}.",
        f"- Weekday UTC rows: {quality['rows_mon_fri_utc']:,}; weekend UTC rows: {quality['rows_weekend_utc']:,}.",
        f"- Non-minute intervals, including scheduled market closures: {quality['non_minute_intervals_including_closed_sessions']:,}; max chronological interval: {quality['max_chronological_interval_minutes']} minutes.",
        f"- Flat Mid OHLC bars: {quality['zero_range_mid_bars']:,} ({quality['zero_range_fraction']:.4%}).",
        f"- Close spread median/P95: {quality['spread_close_price_units']['median']:.6f} / {quality['spread_close_price_units']['p95']:.6f} price units.",
        f"- Close spread median/P95: {quality['spread_close_bps_mid']['median']:.4f} / {quality['spread_close_bps_mid']['p95']:.4f} bps of mid.",
        "",
        "## Causal TP-before-SL calibration diagnostic",
        "",
        f"- Strategy events eligible: {evaluation['strategy_event_candidates']:,}; neutral abstentions: {evaluation['neutral_abstentions']:,}.",
        f"- Binary-scored strategy events: {evaluation['binary_scored_strategy_events']:,}; ambiguous same-bar events excluded: {evaluation['ambiguous_same_bar_events_excluded']:,}.",
        f"- Prequential Brier: {evaluation['aggregate']['brier']:.6f}; constant-0.5 Brier: {evaluation['aggregate']['constant_0_5_brier']:.6f}; skill vs constant-0.5: {evaluation['aggregate']['brier_skill_vs_0_5']:.6f}.",
        f"- Global matured-label baseline Brier: {evaluation['global_prequential_base_rate_baseline']['brier']:.6f}; model skill vs global baseline: {evaluation['aggregate']['brier_skill_vs_global_matured_label_baseline']:.6f}.",
        f"- ECE/MCE (descriptive): {evaluation['aggregate']['reliability']['ece_descriptive']:.6f} / {evaluation['aggregate']['reliability']['mce_descriptive']:.6f}.",
        f"- Moving-block 95% interval for mean Brier improvement over constant 0.5: {evaluation['moving_block_bootstrap_brier_improvement_vs_constant_0_5']['ci95']}.",
        f"- Moving-block 95% interval for mean Brier improvement over global matured-label baseline: {evaluation['moving_block_bootstrap_brier_improvement_vs_global_prequential_base_rate']['ci95']}.",
        "",
        "### Chronological partitions (not used for tuning)",
        "",
        "| Partition | Scored events | Brier | Log loss | Base rate | ECE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for partition in ("development", "calibration", "final_holdout"):
        details = evaluation["chronological_partitions"][partition]
        ece = details["reliability"]["ece_descriptive"] if details["reliability"] else None
        md.append("| {} | {} | {} | {} | {} | {} |".format(
            partition, details["n"],
            "n/a" if details["brier"] is None else f'{details["brier"]:.6f}',
            "n/a" if details["log_loss"] is None else f'{details["log_loss"]:.6f}',
            "n/a" if details["base_rate"] is None else f'{details["base_rate"]:.6f}',
            "n/a" if ece is None else f"{ece:.6f}",
        ))
    md += [
        "",
        "## Interpretation and restrictions",
        "",
        "- PASS certifies only that the declared minimum rows/scenarios, parser, contracts, metrics and artifact generation succeeded.",
        "- Source metadata says Dukascopy via dukascopy-node; upstream provenance and data-use rights are not independently verified. Use for research only pending review.",
        "- The development/calibration predictor uses only earlier, fully matured TP/SL outcomes. At the final-test boundary, prior counts are frozen and no final-test label is used to update predictions; this is a fixed prequential prevalence baseline, not a trained or deployment-calibrated Aurum model.",
        "- Any model-vs-0.5 Brier skill and bootstrap interval are descriptive; they do not establish alpha, profitability or world-leading accuracy.",
        "- Same-bar TP/SL collisions are excluded; missing minute bars are not interpolated for scenario windows.",
        "- Execution cost stats reflect this public Bid/Ask feed, not the user's intended broker's measured spread, commissions, swaps, slippage, or fills.",
        f"- A bounded sample of {sample_count} signal outcomes is written to large_scale_scenario_sample.csv without raw prices.",
        "",
    ]
    (output_dir / "large_scale_bidask_benchmark.md").write_text("\n".join(md), encoding="utf-8")
    return {
        "status": report["status"],
        "dataset_rows": int(len(dataset)),
        "minimum_candles_required": required_candles,
        "selected_months": audit_meta["selected_months"],
        "download_bytes": total_bytes,
        "unique_real_scenario_timestamps": unique_anchor_timestamps,
        "path_scenarios": scenario_count,
        "binary_scored_strategy_events": evaluation["binary_scored_strategy_events"],
        "final_holdout_events": evaluation["chronological_partitions"]["final_holdout"]["n"],
        "prequential_brier": evaluation["aggregate"]["brier"],
        "constant_0_5_brier": evaluation["aggregate"]["constant_0_5_brier"],
        "decision_output": "NO_TRADE",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-candles", type=int, default=MIN_CANDLES)
    parser.add_argument("--min-scenarios", type=int, default=MIN_SCENARIO_ANCHORS)
    args = parser.parse_args()
    try:
        if args.min_candles < MIN_CANDLES:
            raise BenchmarkError("minimum_candles_may_not_be_lowered_below_700000")
        if args.min_scenarios < MIN_SCENARIO_ANCHORS:
            raise BenchmarkError("minimum_scenarios_may_not_be_lowered_below_20000")
        print(json.dumps(run(args.output_dir, args.min_candles, args.min_scenarios), indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
            "decision_output": "NO_TRADE",
            "live_signal_enabled": False,
        }, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
