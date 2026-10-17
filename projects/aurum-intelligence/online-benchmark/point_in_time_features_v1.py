"""Deterministic as-of feature joins; unknown/future information is excluded."""
from __future__ import annotations

import math
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
    """Create stable same-row features; retain anchors even when a source is missing."""
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

    # Validate all timestamp fields once. All downstream joins compare normalized UTC instants.
    normalized: list[ExternalObservation] = []
    for item in observations:
        _utc_aware(item.available_at, "available_at", optional=True)
        if item.event_time is not None:
            _utc_aware(item.event_time, "event_time")
        if item.available_at is None or item.availability_quality == "unknown":
            # Keep it out of the causal join; the source audit separately records unknown timing.
            continue
        normalized.append(item)

    output: list[dict[str, object]] = []
    for decision_time, _decision_id, base in anchors:
        visible = visible_as_of(normalized, decision_time)
        cot = [x for x in visible if _family(x.source_id) == "cot" and x.value is not None]
        macro = [x for x in visible if _family(x.source_id) == "macro" and x.value is not None]
        news_all = [x for x in visible if _family(x.source_id) == "news"]
        news_all, duplicate_count = _news_deduplicate(news_all)
        one_hour_start = decision_time - timedelta(hours=1)
        one_day_start = decision_time - timedelta(hours=24)
        news_1h = [x for x in news_all if one_hour_start < x.available_at.astimezone(timezone.utc) <= decision_time]
        news_24h = [x for x in news_all if one_day_start < x.available_at.astimezone(timezone.utc) <= decision_time]
        cot.sort(key=lambda x: (x.available_at, x.source_record_id))
        latest_cot = cot[-1] if cot else None
        prior_cot = cot[-2] if len(cot) >= 2 else None

        latest_macro_by_series: dict[str, list[ExternalObservation]] = {}
        for item in macro:
            latest_macro_by_series.setdefault(item.dataset_or_series_id, []).append(item)
        latest_macro_values: list[float] = []
        macro_changes: list[float] = []
        for series_id in sorted(latest_macro_by_series):
            values = sorted(latest_macro_by_series[series_id], key=lambda x: (x.available_at, x.source_record_id))
            latest = float(values[-1].value)  # type: ignore[arg-type]
            latest_macro_values.append(latest)
            if len(values) >= 2:
                macro_changes.append(latest - float(values[-2].value))  # type: ignore[arg-type]

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
        tone_values = [float(x.value) for x in news_24h if x.value is not None and math.isfinite(float(x.value))]
        out["feature_news_tone_mean_24h"] = sum(tone_values) / len(tone_values) if tone_values else None
        for theme in THEME_WHITELIST:
            out[f"feature_news_{theme.lower()}_count_24h"] = sum(
                1 for item in news_24h
                if theme in item.attributes.get("themes", ())
            )
        out["news_duplicate_records_removed"] = duplicate_count
        out["availability_proxy_count"] = sum(
            1 for item in visible if item.availability_quality == "conservative_proxy"
        )
        out["unknown_availability_excluded_total"] = sum(
            1 for item in observations if item.available_at is None or item.availability_quality == "unknown"
        )
        cot_state = _source_state(statuses, "cot")
        macro_state = _source_state(statuses, "macro")
        news_state = _source_state(statuses, "news")
        out["coverage_cot"] = "available" if latest_cot else (
            "not_yet_available" if cot_state in {"PASS", "PARTIAL"} else
            "partial_coverage" if cot_state == "NOT_MET" else "source_unavailable" if cot_state in {
                "FAILED", "SKIPPED_CREDENTIAL_REQUIRED", "SKIPPED_POLICY"
            } else "unknown"
        )
        out["coverage_macro"] = "available" if latest_macro_values else (
            "not_yet_available" if macro_state in {"PASS", "PARTIAL"} else
            "source_unavailable" if macro_state in {"FAILED", "SKIPPED_CREDENTIAL_REQUIRED", "SKIPPED_POLICY"} else "unknown"
        )
        out["coverage_news"] = (
            "available" if news_24h else
            "available_zero_events" if news_state == "PASS" else
            "partial_coverage" if news_state in {"PARTIAL", "NOT_MET"} else
            "source_unavailable" if news_state in {"FAILED", "SKIPPED_CREDENTIAL_REQUIRED", "SKIPPED_POLICY"} else "unknown"
        )
        lineage: dict[str, dict[str, object]] = {}
        for item in visible:
            if _family(item.source_id) is None:
                continue
            node = lineage.setdefault(item.source_id, {"payload_hashes": set(), "record_count": 0})
            node["payload_hashes"].add(item.payload_sha256)  # type: ignore[union-attr]
            node["record_count"] = int(node["record_count"]) + 1
        out["feature_lineage"] = {
            source: {"payload_hashes": tuple(sorted(node["payload_hashes"])),
                     "record_count": node["record_count"]}
            for source, node in sorted(lineage.items())
        }
        output.append(out)
    return output
