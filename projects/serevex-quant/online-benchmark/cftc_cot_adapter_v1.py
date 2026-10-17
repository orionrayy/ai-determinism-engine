"""Causal CFTC Gold COT adapter. Network acquisition is bounded and read-only."""
from __future__ import annotations

import csv
import io
import math
import time
import urllib.error
import urllib.request
import zipfile
from datetime import date, datetime, time as dt_time, timedelta
from io import BytesIO
from typing import Iterable
from zoneinfo import ZoneInfo

from external_context_schema_v1 import (
    AcquisitionResult, ExternalObservation, SourceManifest, manifest_sha256,
    validate_observations,
)

OFFICIAL_HOST = "www.cftc.gov"
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 100 * 1024 * 1024
MAX_ROWS = 2_000_000
TIMEOUT_SECONDS = 20
MAX_RETRIES = 2
ET = ZoneInfo("America/New_York")
BASE_URL = "https://www.cftc.gov/files/dea/history/"


def conservative_cot_available_at(report_date: date) -> datetime:
    """00:00 ET Wednesday of the calendar week after the usual Tuesday report.

    The report-date Tuesday is not the public-release timestamp. This deliberately
    conservative fallback is report_date + 8 days for ordinary Tuesday reports.
    """
    if not isinstance(report_date, date) or isinstance(report_date, datetime):
        raise ValueError("report_date_must_be_date")
    return datetime.combine(report_date + timedelta(days=8), dt_time.min, tzinfo=ET)


def resolve_cot_available_at(
    report_date: date,
    verified_release_at: datetime | None = None,
    ambiguous_calendar: bool = False,
) -> tuple[datetime | None, str]:
    if ambiguous_calendar and verified_release_at is None:
        return None, "unknown"
    if verified_release_at is not None:
        if verified_release_at.tzinfo is None or verified_release_at.utcoffset() is None:
            raise ValueError("verified_release_must_be_timezone_aware")
        release = verified_release_at.astimezone(ZoneInfo("UTC"))
        if release.date() < report_date:
            raise ValueError("release_precedes_report_date")
        return release, "verified"
    return conservative_cot_available_at(report_date), "conservative_proxy"


def _decode_payload(payload: bytes) -> bytes:
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("empty_cot_payload")
    if len(payload) > MAX_DOWNLOAD_BYTES:
        raise ValueError("cot_payload_too_large")
    if payload.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(BytesIO(payload)) as archive:
                candidates = [i for i in archive.infolist()
                              if not i.is_dir() and i.filename.lower().endswith((".txt", ".csv"))]
                if len(candidates) != 1:
                    raise ValueError("cot_zip_expected_one_data_file")
                info = candidates[0]
                if info.file_size > MAX_UNCOMPRESSED_BYTES:
                    raise ValueError("cot_uncompressed_payload_too_large")
                with archive.open(info) as stream:
                    data = stream.read(MAX_UNCOMPRESSED_BYTES + 1)
                if len(data) > MAX_UNCOMPRESSED_BYTES:
                    raise ValueError("cot_uncompressed_payload_too_large")
                return data
        except (zipfile.BadZipFile, OSError, RuntimeError) as exc:
            raise ValueError("invalid_or_truncated_cot_zip") from exc
    if payload[:2] == b"PK":
        raise ValueError("invalid_or_truncated_cot_zip")
    return payload


def _parse_rows(data: bytes) -> list[dict[str, str]]:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    sample = text[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",|\t")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    if not reader.fieldnames:
        raise ValueError("cot_header_missing")
    normalized = {name.strip().strip('"').lower() for name in reader.fieldnames if name}
    # The official Disaggregated COT file uses M_Money_* and
    # As_of_Date_Form_YYYY-MM-DD. Accept older API-style aliases as well.
    required_groups = (
        {"market_and_exchange_names"},
        {"futonly_or_combined"},
        {"as_of_date_form_yyyy-mm-dd", "report_date_as_yyyy-mm-dd", "report_date_as_yyyy_mm_dd"},
        {"m_money_positions_long_all", "lev_money_positions_long_all"},
        {"m_money_positions_short_all", "lev_money_positions_short_all"},
    )
    if any(not (group & normalized) for group in required_groups):
        raise ValueError("cot_required_columns_missing")
    records: list[dict[str, str]] = []
    for row in reader:
        if len(records) >= MAX_ROWS:
            raise ValueError("cot_row_limit_exceeded")
        records.append({(k or "").strip(): (v or "").strip() for k, v in row.items()})
    return records


def parse_cot_csv(payload: bytes, manifest: SourceManifest) -> list[ExternalObservation]:
    """Parse official disaggregated futures-only data; quarantine malformed rows."""
    if "cftc.gov" not in manifest.source_url.lower():
        raise ValueError("cot_manifest_not_official_host")
    if not manifest.source_url.startswith("https://www.cftc.gov/files/dea/history/"):
        raise ValueError("cot_manifest_path_not_allowlisted")
    data = _decode_payload(payload)
    # Keep the row parser independent from network and return only valid Gold futures records.
    output: list[ExternalObservation] = []
    for row in _parse_rows(data):
        lower = {k.strip().strip('"').lower(): v for k, v in row.items()}
        market = lower.get("market_and_exchange_names", "").strip()
        future_mode = lower.get("futonly_or_combined", "").strip().lower()
        if "gold" not in market.lower() or future_mode not in {"futonly", "futures only", "futures-only"}:
            continue
        try:
            date_raw = (lower.get("as_of_date_form_yyyy-mm-dd")
                        or lower.get("report_date_as_yyyy-mm-dd")
                        or lower.get("report_date_as_yyyy_mm_dd") or "")
            long_raw = lower.get("m_money_positions_long_all") or lower.get("lev_money_positions_long_all")
            short_raw = lower.get("m_money_positions_short_all") or lower.get("lev_money_positions_short_all")
            if not date_raw or long_raw is None or short_raw is None:
                continue
            report_date = date.fromisoformat(date_raw[:10])
            long_pos = float(long_raw.replace(",", ""))
            short_pos = float(short_raw.replace(",", ""))
            open_interest_raw = lower.get("open_interest_all", "")
            open_interest = float(open_interest_raw.replace(",", "")) if open_interest_raw else None
            net_position = long_pos - short_pos
            if not (math.isfinite(long_pos) and math.isfinite(short_pos) and math.isfinite(net_position)):
                continue
            if long_pos < 0 or short_pos < 0 or (open_interest is not None and
                                                 (not math.isfinite(open_interest) or open_interest < 0)):
                continue
        except (KeyError, ValueError, OverflowError):
            continue
        available_at = conservative_cot_available_at(report_date)
        output.append(ExternalObservation(
            source_id="cftc-cot", source_record_id=f"{report_date.isoformat()}:{market}:FutOnly",
            dataset_or_series_id="cftc-disaggregated-futures-only-gold",
            schema_version="1.0", event_time=datetime.combine(report_date, dt_time.min, tzinfo=ZoneInfo("UTC")),
            published_at=None, available_at=available_at, availability_quality="conservative_proxy",
            retrieved_at=manifest.retrieved_at, source_url=manifest.source_url,
            payload_sha256=manifest.payload_sha256, source_revision=None,
            value=net_position, unit="contracts_net_managed_money",
            attributes={"market": market, "report_date": report_date.isoformat(),
                        "long_managed_money": long_pos, "short_managed_money": short_pos,
                        "open_interest": open_interest, "fut_only": True},
            rights_status=manifest.rights_status, provenance_status=manifest.provenance_status,
            quality_flags=("release_time_conservative_proxy",),
        ))
    validate_observations(output)
    return output


def select_gold_futures(observations: Iterable[ExternalObservation]) -> list[ExternalObservation]:
    return [item for item in observations
            if item.attributes.get("fut_only") is True
            and "gold" in str(item.attributes.get("market", "")).lower()]


def acquire_cot_year(year: int, *, max_bytes: int = MAX_DOWNLOAD_BYTES) -> AcquisitionResult:
    """Bounded fetch of a single official CFTC annual futures-only archive."""
    if not isinstance(year, int) or year < 2000 or year > datetime.now().year:
        return AcquisitionResult("FAILED", error_code="year_out_of_supported_range")
    if not 1 <= max_bytes <= MAX_DOWNLOAD_BYTES:
        return AcquisitionResult("FAILED", error_code="invalid_byte_limit")
    url = f"{BASE_URL}fut_disagg_txt_{year}.zip"
    request = urllib.request.Request(url, headers={"User-Agent": "SerevexResearch/1.0 (bounded public-data audit)"})
    payload = None
    last_error = "download_failed"
    for attempt in range(MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                final = response.geturl()
                if not final.startswith("https://www.cftc.gov/files/dea/history/"):
                    return AcquisitionResult("FAILED", error_code="redirect_outside_allowlist")
                declared = response.headers.get("Content-Length")
                if declared and int(declared) > max_bytes:
                    return AcquisitionResult("FAILED", error_code="content_length_exceeds_limit")
                data = response.read(max_bytes + 1)
                if len(data) > max_bytes:
                    return AcquisitionResult("FAILED", error_code="download_exceeds_limit")
                if not data:
                    return AcquisitionResult("FAILED", error_code="empty_download")
                payload = data
                break
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            last_error = type(exc).__name__
            if attempt < MAX_RETRIES:
                time.sleep(0.25 * (attempt + 1))
    if payload is None:
        return AcquisitionResult("FAILED", error_code=f"download:{last_error}")
    now = datetime.now(tz=ZoneInfo("UTC"))
    manifest = SourceManifest("cftc-cot", "disaggregated-futures-only",
        url, now, manifest_sha256(payload), len(payload), 0, 0,
        f"CFTC archive {year}", "review-required", "official-source;independent-upstream-unverified")
    try:
        observations = parse_cot_csv(payload, manifest)
    except ValueError as exc:
        return AcquisitionResult("FAILED", manifests=(manifest,), error_code=str(exc))
    final_manifest = SourceManifest(manifest.source_id, manifest.dataset_or_series_id,
        manifest.source_url, manifest.retrieved_at, manifest.payload_sha256,
        manifest.payload_bytes, len(observations), len(observations), manifest.provider_version,
        manifest.rights_status, manifest.provenance_status)
    status = "PASS" if observations else "PARTIAL"
    return AcquisitionResult(status, tuple(observations), (final_manifest,),
        warnings=("single_year_smoke_does_not_equal_full_historical_coverage",))
