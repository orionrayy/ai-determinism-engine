"""Deterministic as-of feature joins; unknown/future information is excluded."""
from __future__ import annotations

import math
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Mapping, Sequence

from external_context_schema_v1 import ExternalObservation

VALID_SOURCE_STATES = frozenset({
    "PASS", "PARTIAL", "SKIPPED_CREDENTIAL_REQUIRED", "SKIPPED_POLICY", "NOT_MET", "FAILED"
})
NEWS_BATCH_INTERVAL = timedelta(minutes=15)
THEME_WHITELIST = (
    "ECON_GOLD", "ECON_MARKET", "ECON_MONEY", "ECON_INFLATION",
    "ECON_CENTRAL_BANK", "ECON_STOCKMARKET", "ECON_COMMODITY", "ECON_EXCHANGE_RATE",
)


def _utc_aware(value: datetime | None, name: str, *, optional: bool = False) -> datetime | None:
    if value is None and optional:
        return None
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name}_must_be_timezone_aware")
    return value.astimezone(timezone.utc)


def visible_as_of(
    observations: Sequence[ExternalObservation], decision_time: datetime
) -> list[ExternalObservation]:
    cutoff = _utc_aware(decision_time, "decision_time")
    visible: list[ExternalObservation] = []
    for item in observations:
        available = _utc_aware(item.available_at, "available_at", optional=True)
        # Unknown timing is descriptive-only; never substitute event/published/retrieved time.
        if available is None or item.availability_quality == "unknown" or available > cutoff:
            continue
        if item.value is not None and not math.isfinite(float(item.value)):
            continue
        visible.append(item)
    return sorted(visible, key=lambda x: (
        x.available_at.astimezone(timezone.utc), x.source_id, x.source_record_id
    ))


def _family(source_id: str) -> str | None:
    if source_id.startswith("cftc-"):
        return "cot"
    if source_id in {"bls-public-api-v1", "fred-alfred"}:
        return "macro"
    if source_id.startswith("gdelt-"):
        return "news"
    return None


def _source_state(source_statuses: Mapping[str, str], family: str) -> str:
    value = source_statuses.get(family, "UNKNOWN")
    upper = str(value).upper()
    return upper if upper in VALID_SOURCE_STATES else "UNKNOWN"


def _merge_coverage_intervals(
    intervals: Sequence[tuple[datetime, datetime]],
) -> list[tuple[datetime, datetime]]:
    """Merge touching/overlapping observed archive windows, preserving real gaps."""
    normalized = []
    for index, pair in enumerate(intervals):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(f"invalid_news_coverage_interval_at_{index}")
        start = _utc_aware(pair[0], "news_coverage_start")
        end = _utc_aware(pair[1], "news_coverage_end")
        if start >= end:
            raise ValueError(f"invalid_news_coverage_interval_order_at_{index}")
        normalized.append((start, end))
    normalized.sort(key=lambda item: (item[0], item[1]))
    merged: list[tuple[datetime, datetime]] = []
    for start, end in normalized:
        if merged and start <= merged[-1][1]:
            prior_start, prior_end = merged[-1]
            merged[-1] = (prior_start, max(prior_end, end))
        else:
            merged.append((start, end))
    return merged


def _window_fully_covered(
    start: datetime,
    end: datetime,
    intervals: Sequence[tuple[datetime, datetime]],
) -> bool:
    """A window is complete only when one merged observed interval spans it."""
    return any(left <= start and right >= end for left, right in intervals)


def _archive_batch_window_fully_covered(
    decision_time: datetime,
    duration: timedelta,
    intervals: Sequence[tuple[datetime, datetime]],
) -> bool:
    """Require each complete 15-minute GDELT batch available in a rolling feature window."""
    decision = _utc_aware(decision_time, "decision_time")
    batch_seconds = int(NEWS_BATCH_INTERVAL.total_seconds())
    latest_complete_end = int(decision.timestamp() // batch_seconds) * batch_seconds
    lower_epoch = (decision - duration).timestamp()
    first_required_end = (int(lower_epoch // batch_seconds) + 1) * batch_seconds
    if first_required_end > latest_complete_end:
        return False
    batch_end_epoch = first_required_end
    checked = 0
    while batch_end_epoch <= latest_complete_end:
        batch_end = datetime.fromtimestamp(batch_end_epoch, tz=timezone.utc)
        if not _window_fully_covered(batch_end - NEWS_BATCH_INTERVAL, batch_end, intervals):
            return False
        checked += 1
        batch_end_epoch += batch_seconds
    return checked > 0


def _news_deduplicate(rows: Sequence[ExternalObservation]) -> tuple[list[ExternalObservation], int]:
    unique: list[ExternalObservation] = []
    seen_documents: set[str] = set()
    seen_records: set[tuple[str, str]] = set()
    removed = 0
    for item in sorted(rows, key=lambda x: (x.available_at, x.source_record_id)):
        record_key = (item.source_id, item.source_record_id)
        document_hash = item.attributes.get("document_sha256")
        if record_key in seen_records:
            removed += 1
            continue
        seen_records.add(record_key)
        if isinstance(document_hash, str) and document_hash:
            if document_hash in seen_documents:
                removed += 1
                continue
            seen_documents.add(document_hash)
        unique.append(item)
    return unique, removed


def _cot_position_stats(item: ExternalObservation | None) -> dict[str, float | None]:
    """Return stable, size-normalized COT positions for one already-available report."""
    result: dict[str, float | None] = {
        "net_oi_fraction": None,
        "managed_money_imbalance": None,
        "open_interest": None,
    }
    if item is None or item.value is None:
        return result
    try:
        net = float(item.value)
    except (TypeError, ValueError, OverflowError):
        return result
    if not math.isfinite(net):
        return result

    def finite_attribute(name: str) -> float | None:
        raw = item.attributes.get(name)
        if isinstance(raw, bool) or raw is None:
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError):
            return None
        return value if math.isfinite(value) else None

    open_interest = finite_attribute("open_interest")
    long_position = finite_attribute("long_managed_money")
    short_position = finite_attribute("short_managed_money")
    if open_interest is not None and open_interest > 0:
        result["open_interest"] = open_interest
        result["net_oi_fraction"] = net / open_interest
    if (long_position is not None and short_position is not None
            and long_position >= 0 and short_position >= 0
            and long_position + short_position > 0):
        result["managed_money_imbalance"] = net / (long_position + short_position)
    return result


def build_asof_features(
    decision_rows: Sequence[Mapping[str, object]],
    observations: Sequence[ExternalObservation],
    *,
    source_statuses: Mapping[str, str] | None = None,
    news_coverage_intervals: Sequence[tuple[datetime, datetime]] = (),
) -> list[dict[str, object]]:
    """Create stable as-of features in O((anchors + events) log events) time.

    Anchors and external observations are sorted once; each observation enters the
    rolling state only after its available_at time. This keeps large real-data joins
    practical and prevents future records from influencing earlier rows.
    """
    statuses = source_statuses or {}
    anchors: list[tuple[datetime, str, dict[str, object]]] = []
    seen_ids: set[str] = set()
    for source_row in decision_rows:
        if "decision_time" not in source_row:
            raise ValueError("decision_time_required")
        dt = _utc_aware(source_row["decision_time"], "decision_time")  # type: ignore[arg-type]
        decision_id = str(source_row.get("decision_id", ""))
        if not decision_id:
            raise ValueError("decision_id_required")
        if decision_id in seen_ids:
            raise ValueError("duplicate_decision_id")
        seen_ids.add(decision_id)
        row = dict(source_row)
        row["decision_time"] = dt
        anchors.append((dt, decision_id, row))
    anchors.sort(key=lambda item: (item[0], item[1]))

    # Archive windows become observable only at their end time (the conservative
    # archive-availability boundary). Keep raw intervals separate here: merging an
    # interval with a future contiguous batch would delay visibility of the earlier
    # batch and distort as-of coverage at that boundary.
    raw_news_coverage: list[tuple[datetime, datetime]] = []
    for index, pair in enumerate(news_coverage_intervals):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(f"invalid_news_coverage_interval_at_{index}")
        coverage_start = _utc_aware(pair[0], "news_coverage_start")
        coverage_end = _utc_aware(pair[1], "news_coverage_end")
        if coverage_start >= coverage_end:
            raise ValueError(f"invalid_news_coverage_interval_order_at_{index}")
        raw_news_coverage.append((coverage_start, coverage_end))
    raw_news_coverage.sort(key=lambda item: (item[1], item[0]))
    visible_news_coverage: list[tuple[datetime, datetime]] = []
    merged_visible_news_coverage: list[tuple[datetime, datetime]] = []
    coverage_cursor = 0

    # Keep only explicitly timed rows. The unknown-timing count remains visible in every output.
    unknown_count = 0
    timed: list[ExternalObservation] = []
    for item in observations:
        available = _utc_aware(item.available_at, "available_at", optional=True)
        if item.event_time is not None:
            _utc_aware(item.event_time, "event_time")
        if available is None or item.availability_quality == "unknown":
            unknown_count += 1
            continue
        if item.value is not None and not math.isfinite(float(item.value)):
            continue
        timed.append(item)
    timed.sort(key=lambda x: (
        x.available_at.astimezone(timezone.utc), x.source_id, x.source_record_id
    ))

    future_index = 0
    cot_history: list[ExternalObservation] = []
    macro_history: dict[str, list[ExternalObservation]] = {}
    news_window = deque()
    seen_records: dict[tuple[str, str], ExternalObservation] = {}
    seen_documents: set[str] = set()
    lineage: dict[str, dict[str, object]] = {}
    family_latest: dict[str, datetime] = {}
    proxy_count = 0
    duplicate_count = 0
    visible_record_count = 0
    output: list[dict[str, object]] = []

    for decision_time, _decision_id, base in anchors:
        prior_coverage_cursor = coverage_cursor
        while (coverage_cursor < len(raw_news_coverage)
               and raw_news_coverage[coverage_cursor][1] <= decision_time):
            visible_news_coverage.append(raw_news_coverage[coverage_cursor])
            coverage_cursor += 1
        if coverage_cursor > prior_coverage_cursor:
            merged_visible_news_coverage = _merge_coverage_intervals(visible_news_coverage)
        news_window_1h_fully_covered = _archive_batch_window_fully_covered(
            decision_time, timedelta(hours=1), merged_visible_news_coverage
        )
        news_window_24h_fully_covered = _archive_batch_window_fully_covered(
            decision_time, timedelta(hours=24), merged_visible_news_coverage
        )

        while future_index < len(timed):
            item = timed[future_index]
            available = item.available_at.astimezone(timezone.utc)
            if available > decision_time:
                break
            future_index += 1
            family = _family(item.source_id)
            if family is None:
                continue
            record_key = (item.source_id, item.source_record_id)
            previous_record = seen_records.get(record_key)
            if previous_record is not None:
                # Same source record id should be immutable across archive snapshots.
                duplicate_count += 1
                continue
            seen_records[record_key] = item
            if family == "news":
                doc_hash = item.attributes.get("document_sha256")
                if isinstance(doc_hash, str) and doc_hash:
                    if doc_hash in seen_documents:
                        duplicate_count += 1
                        continue
                    seen_documents.add(doc_hash)
                news_window.append(item)
            elif family == "cot":
                if item.value is not None:
                    cot_history.append(item)
            elif family == "macro":
                if item.value is not None:
                    series = item.dataset_or_series_id
                    bucket = macro_history.setdefault(series, [])
                    bucket.append(item)
                    if len(bucket) > 2:
                        del bucket[:-2]
            visible_record_count += 1
            proxy_count += int(item.availability_quality == "conservative_proxy")
            family_latest[family] = max(family_latest.get(family, available), available)
            node = lineage.setdefault(item.source_id, {"payload_hashes": set(), "record_count": 0})
            node["payload_hashes"].add(item.payload_sha256)  # type: ignore[union-attr]
            node["record_count"] = int(node["record_count"]) + 1

        day_floor = decision_time - timedelta(hours=24)
        while news_window and news_window[0].available_at.astimezone(timezone.utc) <= day_floor:
            news_window.popleft()
        cot = cot_history
        latest_cot = cot[-1] if cot else None
        prior_cot = cot[-2] if len(cot) >= 2 else None
        latest_macro_values: list[float] = []
        macro_changes: list[float] = []
        for series_id in sorted(macro_history):
            values = macro_history[series_id]
            if values:
                latest_macro_values.append(float(values[-1].value))  # type: ignore[arg-type]
            if len(values) >= 2:
                macro_changes.append(float(values[-1].value) - float(values[-2].value))  # type: ignore[arg-type]

        news_24h = list(news_window)
        one_hour_floor = decision_time - timedelta(hours=1)
        news_1h = [x for x in news_24h if one_hour_floor < x.available_at.astimezone(timezone.utc) <= decision_time]
        tone_values = [
            float(x.value) for x in news_24h
            if x.value is not None and math.isfinite(float(x.value))
        ]
        out = dict(base)
        out["feature_cot_net_position"] = float(latest_cot.value) if latest_cot else None
        out["feature_cot_net_position_change"] = (
            float(latest_cot.value) - float(prior_cot.value)
            if latest_cot is not None and prior_cot is not None else None
        )
        latest_cot_stats = _cot_position_stats(latest_cot)
        prior_cot_stats = _cot_position_stats(prior_cot)
        out["feature_cot_net_position_oi_fraction"] = latest_cot_stats["net_oi_fraction"]
        out["feature_cot_managed_money_imbalance"] = latest_cot_stats["managed_money_imbalance"]
        out["feature_cot_net_position_oi_change"] = (
            float(latest_cot_stats["net_oi_fraction"]) - float(prior_cot_stats["net_oi_fraction"])
            if latest_cot_stats["net_oi_fraction"] is not None
            and prior_cot_stats["net_oi_fraction"] is not None else None
        )
        current_oi = latest_cot_stats["open_interest"]
        prior_oi = prior_cot_stats["open_interest"]
        out["feature_cot_open_interest_change_fraction"] = (
            (float(current_oi) - float(prior_oi)) / float(prior_oi)
            if current_oi is not None and prior_oi is not None and float(prior_oi) > 0
            else None
        )
        out["feature_macro_level"] = sum(latest_macro_values) / len(latest_macro_values) if latest_macro_values else None
        out["feature_macro_change"] = sum(macro_changes) / len(macro_changes) if macro_changes else None
        # A sparse archive sample cannot turn unsampled periods into genuine zero-news
        # observations. Keep numeric news features only when the respective historical
        # window is fully covered by acquired, already-available archive batches.
        out["news_window_1h_fully_covered"] = news_window_1h_fully_covered
        out["news_window_24h_fully_covered"] = news_window_24h_fully_covered
        out["news_coverage_intervals_visible"] = len(visible_news_coverage)
        out["feature_news_count_1h"] = len(news_1h) if news_window_1h_fully_covered else None
        out["feature_news_count_24h"] = len(news_24h) if news_window_24h_fully_covered else None
        out["feature_news_tone_mean_24h"] = (
            sum(tone_values) / len(tone_values)
            if tone_values and news_window_24h_fully_covered else None
        )
        for theme in THEME_WHITELIST:
            out[f"feature_news_{theme.lower()}_count_24h"] = (
                sum(1 for item in news_24h if theme in item.attributes.get("themes", ()))
                if news_window_24h_fully_covered else None
            )
        out["news_duplicate_records_removed"] = duplicate_count
        out["availability_proxy_count"] = proxy_count
        out["unknown_availability_excluded_total"] = unknown_count
        out["visible_external_record_count"] = visible_record_count
        cot_state = _source_state(statuses, "cot")
        macro_state = _source_state(statuses, "macro")
        news_state = _source_state(statuses, "news")
        out["coverage_cot"] = "available" if latest_cot else (
            "not_yet_available" if cot_state in {"PASS", "PARTIAL"} else
            "source_unavailable" if cot_state in {
                "NOT_MET", "FAILED", "SKIPPED_CREDENTIAL_REQUIRED", "SKIPPED_POLICY"
            } else "unknown"
        )
        out["coverage_macro"] = "available" if latest_macro_values else (
            "not_yet_available" if macro_state in {"PASS", "PARTIAL"} else
            "source_unavailable" if macro_state in {"NOT_MET", "FAILED", "SKIPPED_CREDENTIAL_REQUIRED", "SKIPPED_POLICY"} else "unknown"
        )
        if news_state in {"NOT_MET", "FAILED", "SKIPPED_CREDENTIAL_REQUIRED", "SKIPPED_POLICY"}:
            news_coverage_1h_status = "source_unavailable"
            news_coverage_24h_status = "source_unavailable"
        elif news_state in {"PASS", "PARTIAL"}:
            news_coverage_1h_status = (
                "available" if news_1h else "available_zero_events"
            ) if news_window_1h_fully_covered else "incomplete_source_window"
            news_coverage_24h_status = (
                "available" if news_24h else "available_zero_events"
            ) if news_window_24h_fully_covered else "incomplete_source_window"
        else:
            news_coverage_1h_status = "unknown"
            news_coverage_24h_status = "unknown"
        out["coverage_news_1h"] = news_coverage_1h_status
        out["coverage_news_24h"] = news_coverage_24h_status
        # Compatibility field keeps its 24-hour interpretation.
        out["coverage_news"] = news_coverage_24h_status
        out["feature_family_latest_available_at"] = {
            family: family_latest[family] for family in sorted(family_latest)
        }
        out["feature_lineage"] = {
            source: {"payload_hashes": tuple(sorted(node["payload_hashes"])[-5:]),
                     "payload_hash_count_total": len(node["payload_hashes"]),
                     "record_count": node["record_count"]}
            for source, node in sorted(lineage.items())
        }
        output.append(out)
    return output
