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


def build_asof_features(
    decision_rows: Sequence[Mapping[str, object]],
    observations: Sequence[ExternalObservation],
    *,
    source_statuses: Mapping[str, str] | None = None,
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
        out["feature_macro_level"] = sum(latest_macro_values) / len(latest_macro_values) if latest_macro_values else None
        out["feature_macro_change"] = sum(macro_changes) / len(macro_changes) if macro_changes else None
        out["feature_news_count_1h"] = len(news_1h)
        out["feature_news_count_24h"] = len(news_24h)
        out["feature_news_tone_mean_24h"] = sum(tone_values) / len(tone_values) if tone_values else None
        for theme in THEME_WHITELIST:
            out[f"feature_news_{theme.lower()}_count_24h"] = sum(
                1 for item in news_24h if theme in item.attributes.get("themes", ())
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
        out["coverage_news"] = (
            "partial_coverage" if news_state == "PARTIAL" else
            "available" if news_24h and news_state == "PASS" else
            "available_zero_events" if news_state == "PASS" else
            "source_unavailable" if news_state in {"NOT_MET", "FAILED", "SKIPPED_CREDENTIAL_REQUIRED", "SKIPPED_POLICY"} else "unknown"
        )
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
