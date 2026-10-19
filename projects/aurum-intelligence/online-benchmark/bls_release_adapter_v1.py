"""BLS public API v1 adapter preserving observation and public-release times."""
from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Mapping, Sequence

from external_context_schema_v1 import AcquisitionResult, ExternalObservation, SourceManifest, manifest_sha256

BLS_API_URL = "https://api.bls.gov/publicAPI/v1/timeseries/data/"
BLS_HOST = "api.bls.gov"
MAX_REQUEST_BYTES = 1_000_000
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_SERIES_PER_REQUEST = 25
MAX_YEARS_PER_REQUEST = 10
MAX_REQUESTS_PER_DAY = 25
TIMEOUT_SECONDS = 20


class BLSAPIError(ValueError):
    def __init__(self, status: str, message: str = "") -> None:
        self.status = status
        self.code = _safe_error_code(message or status)
        super().__init__(f"bls_api_{status.lower()}:{self.code}")


def _safe_error_code(value: str) -> str:
    normalized = "".join(c.lower() if c.isalnum() else "_" for c in value)[:80].strip("_")
    return normalized or "unspecified"


def validate_bls_request(
    series_ids: Sequence[str], start_year: int, end_year: int, *, requests_today: int = 0
) -> None:
    if not series_ids or len(series_ids) > MAX_SERIES_PER_REQUEST:
        raise ValueError("bls_series_count_out_of_range")
    if any(not isinstance(s, str) or not s.strip() or len(s) > 40 for s in series_ids):
        raise ValueError("bls_series_id_invalid")
    if len(set(series_ids)) != len(series_ids):
        raise ValueError("bls_duplicate_series_id")
    if not isinstance(start_year, int) or not isinstance(end_year, int) or start_year > end_year:
        raise ValueError("bls_year_range_invalid")
    if end_year - start_year >= MAX_YEARS_PER_REQUEST:
        raise ValueError("bls_year_range_exceeds_10_years")
    if requests_today < 0 or requests_today >= MAX_REQUESTS_PER_DAY:
        raise ValueError("bls_daily_request_limit_reached")


def map_release_availability(
    series_id: str, observation_period: str, release_calendar: Mapping[str, datetime]
) -> datetime | None:
    value = release_calendar.get(f"{series_id}|{observation_period}")
    if value is None:
        return None
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("bls_release_time_must_be_timezone_aware")
    return value.astimezone(timezone.utc)


def classify_bls_error(code: str) -> str:
    normalized = (code or "").lower()
    if any(term in normalized for term in ("limit", "rate", "daily", "request_not_processed", "failed")):
        return "FAILED"
    if any(term in normalized for term in ("partial", "series", "warning")):
        return "PARTIAL"
    return "FAILED"


def _period_time(year: int, period: str) -> datetime | None:
    # Observation timestamp is only the period's start, never a publication timestamp.
    if len(period) == 3 and period.startswith("M") and period[1:].isdigit():
        month = int(period[1:])
        if 1 <= month <= 12:
            return datetime(year, month, 1, tzinfo=timezone.utc)
    if len(period) == 3 and period.startswith("Q") and period[1:].isdigit():
        quarter = int(period[1:])
        if 1 <= quarter <= 4:
            return datetime(year, 1 + (quarter - 1) * 3, 1, tzinfo=timezone.utc)
    return None


def parse_bls_api_response(
    payload: bytes, manifest: SourceManifest,
    release_calendar: Mapping[str, datetime] | None = None,
) -> list[ExternalObservation]:
    if not isinstance(payload, bytes) or not payload or len(payload) > MAX_RESPONSE_BYTES:
        raise ValueError("bls_payload_empty_or_too_large")
    host = manifest.source_url.split("/")[2] if "://" in manifest.source_url else ""
    if host != BLS_HOST or not manifest.source_url.startswith("https://"):
        raise ValueError("bls_manifest_url_not_allowlisted")
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("bls_response_invalid_json") from exc
    status = decoded.get("status")
    if status != "REQUEST_SUCCEEDED":
        messages = decoded.get("message") or []
        code = str(messages[0]) if messages else str(status or "missing_status")
        raise BLSAPIError(str(status or "MISSING_STATUS"), code)
    result = decoded.get("Results", {}).get("series")
    if not isinstance(result, list):
        raise BLSAPIError("MALFORMED_RESPONSE", "series_missing")
    if len(result) > MAX_SERIES_PER_REQUEST:
        raise ValueError("bls_response_series_limit_exceeded")
    calendar = release_calendar or {}
    output: list[ExternalObservation] = []
    for series in result:
        if not isinstance(series, dict) or not isinstance(series.get("data"), list):
            continue
        series_id = str(series.get("seriesID", "")).strip()
        if not series_id or len(series["data"]) > 100_000:
            continue
        for entry in series["data"]:
            if not isinstance(entry, dict):
                continue
            year_raw, period = str(entry.get("year", "")), str(entry.get("period", ""))
            if not year_raw.isdigit():
                continue
            year = int(year_raw)
            period_time = _period_time(year, period)
            raw_value = str(entry.get("value", "")).strip()
            try:
                value = float(raw_value)
            except (ValueError, TypeError):
                continue
            if not math.isfinite(value):
                continue
            period_label = f"{year}-{period}"
            release_at = map_release_availability(series_id, period_label, calendar)
            flags = ("publication_time_calendar_matched",) if release_at else ("unknown_publication_time_descriptive_only",)
            output.append(ExternalObservation(
                source_id="bls-public-api-v1", source_record_id=f"{series_id}:{period_label}",
                dataset_or_series_id=series_id, schema_version="1.0",
                event_time=period_time, published_at=release_at, available_at=release_at,
                availability_quality="verified" if release_at else "unknown",
                retrieved_at=manifest.retrieved_at, source_url=manifest.source_url,
                payload_sha256=manifest.payload_sha256, source_revision=None,
                value=value, unit="BLS_reported_value",
                attributes={"series_id": series_id, "observation_period": period_label,
                            "period_name": str(entry.get("periodName", ""))[:40],
                            "year": year, "period": period,
                            "footnotes_present": bool(entry.get("footnotes"))},
                rights_status=manifest.rights_status, provenance_status=manifest.provenance_status,
                quality_flags=flags,
            ))
    return output


def acquire_bls_series(
    series_ids: Sequence[str], start_year: int, end_year: int, *,
    requests_today: int = 0, release_calendar: Mapping[str, datetime] | None = None,
) -> AcquisitionResult:
    try:
        validate_bls_request(series_ids, start_year, end_year, requests_today=requests_today)
    except ValueError as exc:
        return AcquisitionResult("FAILED", error_code=str(exc))
    body = json.dumps({"seriesid": list(series_ids), "startyear": str(start_year),
                       "endyear": str(end_year)}, separators=(",", ":")).encode()
    if len(body) > MAX_REQUEST_BYTES:
        return AcquisitionResult("FAILED", error_code="bls_request_body_too_large")
    request = urllib.request.Request(BLS_API_URL, data=body, method="POST",
        headers={"Content-Type":"application/json", "User-Agent":"AurumResearch/1.0 bounded BLS adapter"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            if response.geturl() != BLS_API_URL:
                return AcquisitionResult("FAILED", error_code="bls_redirect_outside_allowlist")
            payload = response.read(MAX_RESPONSE_BYTES + 1)
        if len(payload) > MAX_RESPONSE_BYTES:
            return AcquisitionResult("FAILED", error_code="bls_response_too_large")
        manifest = SourceManifest("bls-public-api-v1", ",".join(series_ids), BLS_API_URL,
            datetime.now(timezone.utc), manifest_sha256(payload), len(payload), 0, 0,
            "BLS public API v1", "public-data-rights-review-required", "official-host")
        observations = parse_bls_api_response(payload, manifest, release_calendar)
        final_manifest = SourceManifest(manifest.source_id,manifest.dataset_or_series_id,
            manifest.source_url,manifest.retrieved_at,manifest.payload_sha256,manifest.payload_bytes,
            len(observations),len(observations),manifest.provider_version,manifest.rights_status,
            manifest.provenance_status)
        return AcquisitionResult("PASS" if observations else "PARTIAL",tuple(observations),(final_manifest,))
    except BLSAPIError as exc:
        return AcquisitionResult(classify_bls_error(exc.code), error_code=exc.code)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return AcquisitionResult("FAILED", error_code=f"bls_request_failed:{type(exc).__name__}")
