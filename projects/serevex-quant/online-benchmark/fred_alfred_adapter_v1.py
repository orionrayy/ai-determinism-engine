"""Optional FRED/ALFRED observations with vintage-safe as-of selection."""
from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, time as dt_time, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Mapping, Sequence

from external_context_schema_v1 import AcquisitionResult, ExternalObservation, SourceManifest, manifest_sha256

FRED_HOST = "api.stlouisfed.org"
FRED_ENDPOINT = "https://api.stlouisfed.org/fred/series/observations"
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_PAGE_SIZE = 1000
MAX_RESULT_COUNT = 50_000
MAX_PAGES = 50
TIMEOUT_SECONDS = 20
CENTRAL = ZoneInfo("America/Chicago")


def validate_fred_pagination(*, offset: int, limit: int, records_received: int, total_count: int) -> int:
    if offset < 0 or records_received < 0 or total_count < 0:
        raise ValueError("fred_pagination_values_must_be_nonnegative")
    if limit < 1 or limit > MAX_PAGE_SIZE:
        raise ValueError("fred_page_limit_exceeded")
    if offset + records_received > MAX_RESULT_COUNT or total_count > MAX_RESULT_COUNT:
        raise ValueError("fred_result_cap_exceeded")
    if records_received > limit:
        raise ValueError("fred_received_more_than_limit")
    return offset + records_received


def _safe_date(value: object) -> date | None:
    try:
        return date.fromisoformat(str(value))
    except (ValueError, TypeError):
        return None


def _available_from_vintage(vintage_date: date) -> datetime:
    # Date-only vintage has no precise public hour; using next local midnight is conservative.
    local_next_day = datetime.combine(vintage_date + timedelta(days=1), dt_time.min, tzinfo=CENTRAL)
    return local_next_day.astimezone(timezone.utc)


def parse_fred_observations(payload: bytes, manifest: SourceManifest) -> list[ExternalObservation]:
    if not isinstance(payload, bytes) or not payload or len(payload) > MAX_RESPONSE_BYTES:
        raise ValueError("fred_payload_empty_or_too_large")
    parsed = urllib.parse.urlparse(manifest.source_url)
    if parsed.scheme != "https" or parsed.hostname != FRED_HOST or parsed.path != "/fred/series/observations":
        raise ValueError("fred_manifest_url_not_allowlisted")
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("fred_response_invalid_json") from exc
    raw_rows = decoded.get("observations")
    if not isinstance(raw_rows, list):
        raise ValueError("fred_observations_missing")
    count = decoded.get("count", len(raw_rows))
    if not isinstance(count, int) or count > MAX_RESULT_COUNT or len(raw_rows) > MAX_PAGE_SIZE:
        raise ValueError("fred_result_or_page_cap_exceeded")
    rows: list[ExternalObservation] = []
    for row in raw_rows:
        if not isinstance(row, dict):
            continue
        obs_date = _safe_date(row.get("date"))
        vintage_start = _safe_date(row.get("realtime_start"))
        vintage_end = _safe_date(row.get("realtime_end"))
        raw_value = row.get("value")
        if obs_date is None or vintage_start is None or vintage_end is None or raw_value in (None, "", "."):
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(value):
            continue
        available_at = _available_from_vintage(vintage_start)
        record_id = f"{manifest.dataset_or_series_id}:{obs_date.isoformat()}:{vintage_start.isoformat()}"
        rows.append(ExternalObservation(
            source_id="fred-alfred", source_record_id=record_id,
            dataset_or_series_id=manifest.dataset_or_series_id, schema_version="1.0",
            event_time=datetime.combine(obs_date, dt_time.min, tzinfo=timezone.utc),
            published_at=None, available_at=available_at, availability_quality="conservative_proxy",
            retrieved_at=manifest.retrieved_at, source_url=FRED_ENDPOINT,
            payload_sha256=manifest.payload_sha256, source_revision=vintage_start.isoformat(),
            value=value, unit="FRED reported value",
            attributes={"observation_date": obs_date.isoformat(),
                        "realtime_start": vintage_start.isoformat(),
                        "realtime_end": vintage_end.isoformat(),
                        "vintage_date": vintage_start.isoformat()},
            rights_status=manifest.rights_status, provenance_status=manifest.provenance_status,
            quality_flags=("vintage_start_plus_one_local_day_proxy",),
        ))
    return rows


def observation_as_of_vintage(
    observations: Sequence[ExternalObservation], as_of_date: date
) -> list[ExternalObservation]:
    """Select the latest vintage that was available by the requested local calendar date."""
    if not isinstance(as_of_date, date) or isinstance(as_of_date, datetime):
        raise ValueError("as_of_date_must_be_date")
    cutoff = datetime.combine(as_of_date + timedelta(days=1), dt_time.min, tzinfo=CENTRAL).astimezone(timezone.utc)
    selected: dict[tuple[str, str], ExternalObservation] = {}
    for item in observations:
        if item.source_id != "fred-alfred" or item.available_at is None or item.available_at > cutoff:
            continue
        observation_date = str(item.attributes.get("observation_date", ""))
        key = (item.dataset_or_series_id, observation_date)
        current = selected.get(key)
        if current is None or (item.source_revision or "") > (current.source_revision or ""):
            selected[key] = item
    return sorted(selected.values(), key=lambda x: (x.dataset_or_series_id, x.event_time, x.source_revision or ""))


def acquire_fred(config: Mapping[str, object]) -> AcquisitionResult:
    """Acquire FRED vintages using FRED_API_KEY from process environment only."""
    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        return AcquisitionResult("SKIPPED_CREDENTIAL_REQUIRED", error_code="FRED_API_KEY_NOT_CONFIGURED")
    series_id = config.get("series_id")
    start = config.get("observation_start")
    end = config.get("observation_end")
    if not isinstance(series_id, str) or not series_id.strip() or len(series_id) > 40:
        return AcquisitionResult("FAILED", error_code="invalid_series_id")
    if _safe_date(start) is None or _safe_date(end) is None or str(start) > str(end):
        return AcquisitionResult("FAILED", error_code="invalid_observation_date_range")
    rows: list[ExternalObservation] = []
    manifests: list[SourceManifest] = []
    offset = 0
    limit = MAX_PAGE_SIZE
    for _ in range(MAX_PAGES):
        query = urllib.parse.urlencode({
            "series_id": series_id, "api_key": api_key, "file_type": "json",
            "observation_start": str(start), "observation_end": str(end),
            "realtime_start": str(config.get("realtime_start", "1776-07-04")),
            "realtime_end": str(config.get("realtime_end", "9999-12-31")),
            "limit": str(limit), "offset": str(offset),
        })
        request = urllib.request.Request(f"{FRED_ENDPOINT}?{query}",
            headers={"User-Agent":"SerevexResearch/1.0 bounded FRED adapter"})
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                final = urllib.parse.urlparse(response.geturl())
                if final.scheme != "https" or final.hostname != FRED_HOST or final.path != "/fred/series/observations":
                    return AcquisitionResult("FAILED", error_code="fred_redirect_outside_allowlist")
                payload = response.read(MAX_RESPONSE_BYTES + 1)
            if len(payload) > MAX_RESPONSE_BYTES:
                return AcquisitionResult("FAILED", error_code="fred_response_too_large")
            manifest = SourceManifest("fred-alfred", series_id, FRED_ENDPOINT,
                datetime.now(timezone.utc), manifest_sha256(payload), len(payload), 0, 0,
                "FRED/ALFRED observations", "license-review-required", "official-host;lineage-review-required")
            try:
                page = parse_fred_observations(payload, manifest)
                document = json.loads(payload.decode("utf-8"))
                total_count = int(document.get("count", len(page)))
            except (ValueError, json.JSONDecodeError, TypeError):
                return AcquisitionResult("FAILED", manifests=tuple(manifests), error_code="fred_response_invalid")
            manifests.append(SourceManifest(manifest.source_id, manifest.dataset_or_series_id,
                manifest.source_url, manifest.retrieved_at, manifest.payload_sha256, manifest.payload_bytes,
                len(page), len(page), manifest.provider_version, manifest.rights_status, manifest.provenance_status))
            rows.extend(page)
            next_offset = validate_fred_pagination(offset=offset, limit=limit,
                records_received=len(page), total_count=total_count)
            if not page or next_offset >= total_count:
                break
            offset = next_offset
            if len(rows) >= MAX_RESULT_COUNT:
                return AcquisitionResult("PARTIAL", tuple(rows[:MAX_RESULT_COUNT]), tuple(manifests),
                    warnings=("fred_result_cap_reached",))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            # Deliberately omit exception text because URLs contain the API key.
            return AcquisitionResult("FAILED", tuple(rows), tuple(manifests),
                error_code=f"fred_request_failed:{type(exc).__name__}")
    return AcquisitionResult("PASS" if rows else "PARTIAL", tuple(rows), tuple(manifests),
        warnings=(f"records={len(rows)}", "vintage timestamps are conservative date proxies"))
