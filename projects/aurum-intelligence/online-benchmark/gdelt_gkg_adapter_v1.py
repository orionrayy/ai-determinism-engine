"""Bounded point-in-time GDELT GKG metadata adapter (never persists article bodies)."""
from __future__ import annotations

import csv
import hashlib
import io
import math
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from io import BytesIO
from urllib.parse import urlparse

from external_context_schema_v1 import AcquisitionResult, ExternalObservation, SourceManifest, manifest_sha256

OFFICIAL_HOST = "data.gdeltproject.org"
MASTER_URL = "https://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
MAX_PAYLOAD_BYTES = 32 * 1024 * 1024
# The official GDELT 2.0 master list is currently >120 MB. Keep it bounded,
# but large enough to select historical 15-minute GKG archives.
MAX_MASTER_BYTES = 160 * 1024 * 1024
MAX_ARCHIVE_BYTES = 24 * 1024 * 1024
MAX_AGGREGATE_BYTES = 320 * 1024 * 1024
MAX_RECORDS_PER_ARCHIVE = 500_000
MAX_ARCHIVES_PER_RUN = 100
TIMEOUT_SECONDS = 20
MAX_RETRIES = 2
BATCH_INTERVAL = timedelta(minutes=15)


def normalize_archive_url(url: str) -> str:
    """Upgrade only public GDELT master-list links on the exact official host to HTTPS."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname != OFFICIAL_HOST:
        raise ValueError("gdelt_url_outside_official_allowlist")
    if parsed.username is not None or parsed.password is not None or parsed.port not in (None, 80, 443):
        raise ValueError("gdelt_url_has_unsafe_authority")
    if not parsed.path.startswith("/gdeltv2/") or parsed.query or parsed.fragment:
        raise ValueError("gdelt_path_not_allowlisted")
    return f"https://{OFFICIAL_HOST}{parsed.path}"


def _validate_official_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != OFFICIAL_HOST:
        raise ValueError("gdelt_url_outside_https_allowlist")
    if not parsed.path.startswith("/gdeltv2/"):
        raise ValueError("gdelt_path_not_allowlisted")


def archive_available_at(record_id: str) -> datetime:
    """Conservative first availability proxy: record batch time + one GKG interval."""
    if not isinstance(record_id, str) or len(record_id) < 14 or not record_id[:14].isdigit():
        raise ValueError("invalid_gdelt_record_id")
    try:
        generated_at = datetime.strptime(record_id[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError("invalid_gdelt_record_timestamp") from exc
    return generated_at + BATCH_INTERVAL


def classify_doc_api_history(*, is_doc_api: bool, claimed_multiyear: bool) -> str:
    if is_doc_api and claimed_multiyear:
        return "NOT_MET"
    return "SMOKE_ONLY" if is_doc_api else "ARCHIVE_CANDIDATE"


def parse_gkg_tsv(payload: bytes, manifest: SourceManifest) -> list[ExternalObservation]:
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("empty_gdelt_payload")
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("gdelt_payload_too_large")
    parsed_url = urlparse(manifest.source_url)
    if parsed_url.scheme != "https" or parsed_url.hostname != OFFICIAL_HOST:
        raise ValueError("gdelt_manifest_source_not_allowlisted")
    try:
        text = payload.decode("utf-8", errors="replace")
    except Exception as exc:
        raise ValueError("gdelt_decode_failed") from exc
    reader = csv.reader(io.StringIO(text), delimiter="\t", quotechar='"')
    observations: list[ExternalObservation] = []
    for row_no, fields in enumerate(reader, start=1):
        if row_no > MAX_RECORDS_PER_ARCHIVE:
            raise ValueError("gdelt_record_limit_exceeded")
        if not fields or not fields[0].strip():
            continue
        if len(fields) < 11:
            continue
        record_id = fields[0].strip()
        try:
            # GKG record ID is the generation timestamp; only an availability proxy.
            event_time = datetime.strptime(record_id[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
            available_at = archive_available_at(record_id)
        except ValueError:
            continue
        doc_url = fields[4].strip()
        if doc_url and not doc_url.startswith(("https://", "http://")):
            doc_url = ""
        domain = urlparse(doc_url).hostname.lower() if doc_url else ""
        theme_field = fields[6].strip()
        themes = tuple(sorted({part.strip().split(",", 1)[0][:100]
                               for part in theme_field.split(";") if part.strip()}))[:40]
        tone = None
        tone_raw = fields[10].split(",", 1)[0].strip() if fields[10].strip() else ""
        if tone_raw:
            try:
                candidate = float(tone_raw)
                if math.isfinite(candidate):
                    tone = max(-100.0, min(100.0, candidate))
            except ValueError:
                tone = None
        url_digest = hashlib.sha256(doc_url.encode("utf-8")).hexdigest() if doc_url else None
        domain_digest = hashlib.sha256(domain.encode("utf-8")).hexdigest() if domain else None
        observations.append(ExternalObservation(
            source_id="gdelt-gkg", source_record_id=record_id,
            dataset_or_series_id="gdelt-gkg-v2-metadata", schema_version="1.0",
            event_time=event_time, published_at=event_time, available_at=available_at,
            availability_quality="conservative_proxy", retrieved_at=manifest.retrieved_at,
            source_url=manifest.source_url, payload_sha256=manifest.payload_sha256,
            source_revision=None, value=tone, unit="gkg_tone_first_component" if tone is not None else None,
            attributes={"document_sha256": url_digest, "domain_sha256": domain_digest,
                        "themes": themes, "tone": tone, "source_name": fields[3][:160],
                        "archive_record_timestamp": event_time.isoformat()},
            rights_status=manifest.rights_status, provenance_status=manifest.provenance_status,
            quality_flags=("archive_generation_time_plus_15m_proxy", "article_body_not_stored"),
        ))
    return observations


def deduplicate_snapshots(observations: list[ExternalObservation]) -> list[ExternalObservation]:
    """Keep earliest-available observation for each GKG record and document hash."""
    ordered = sorted(observations, key=lambda x: (
        x.available_at is None, x.available_at or datetime.max.replace(tzinfo=timezone.utc),
        x.source_record_id))
    by_record: dict[tuple[str, str], ExternalObservation] = {}
    by_doc: dict[str, ExternalObservation] = {}
    output: list[ExternalObservation] = []
    for item in ordered:
        record_key = (item.source_id, item.source_record_id)
        previous = by_record.get(record_key)
        if previous is not None:
            if previous.payload_sha256 != item.payload_sha256 or previous.attributes != item.attributes:
                flags = tuple(sorted(set(previous.quality_flags) | {"conflicting_snapshot_record_id"}))
                revised = replace(previous, quality_flags=flags)
                by_record[record_key] = revised
                output = [revised if x is previous else x for x in output]
            continue
        doc_hash = item.attributes.get("document_sha256")
        if isinstance(doc_hash, str) and doc_hash and doc_hash in by_doc:
            first = by_doc[doc_hash]
            flags = tuple(sorted(set(first.quality_flags) | {"duplicate_document_first_seen_retained"}))
            revised = replace(first, quality_flags=flags)
            by_doc[doc_hash] = revised
            output = [revised if x is first else x for x in output]
            by_record[record_key] = revised
            continue
        by_record[record_key] = item
        if isinstance(doc_hash, str) and doc_hash:
            by_doc[doc_hash] = item
        output.append(item)
    return sorted(output, key=lambda x: (x.available_at, x.source_record_id))


def _bounded_get(url: str, cap: int) -> bytes:
    _validate_official_url(url)
    request = urllib.request.Request(url, headers={"User-Agent": "AurumResearch/1.0 bounded metadata audit"})
    last_error = "download_failed"
    for attempt in range(MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                _validate_official_url(response.geturl())
                length = response.headers.get("Content-Length")
                if length and int(length) > cap:
                    raise ValueError("content_length_exceeds_limit")
                payload = response.read(cap + 1)
                if len(payload) > cap:
                    raise ValueError("download_exceeds_limit")
                return payload
        except ValueError:
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = type(exc).__name__
            if attempt < MAX_RETRIES:
                time.sleep(0.25 * (attempt + 1))
    raise RuntimeError(f"gdelt_download_failed:{last_error}")


def acquire_historical_gkg(
    start: datetime, end: datetime, *, target_records: int = 100_000,
    max_archives: int = MAX_ARCHIVES_PER_RUN, aggregate_cap: int = MAX_AGGREGATE_BYTES,
) -> AcquisitionResult:
    """Acquire metadata archives in a bounded time window; honest NOT_MET if under target."""
    if start.tzinfo is None or end.tzinfo is None or start >= end:
        return AcquisitionResult("FAILED", error_code="invalid_time_window")
    if target_records < 1 or max_archives < 1 or max_archives > MAX_ARCHIVES_PER_RUN:
        return AcquisitionResult("FAILED", error_code="invalid_acquisition_limits")
    if aggregate_cap < 1 or aggregate_cap > MAX_AGGREGATE_BYTES:
        return AcquisitionResult("FAILED", error_code="invalid_aggregate_byte_cap")
    try:
        listing = _bounded_get(MASTER_URL, MAX_MASTER_BYTES).decode("utf-8", errors="replace")
        candidates = []
        for line in listing.splitlines():
            bits = line.split()
            if len(bits) < 3:
                continue
            url = bits[-1]
            try:
                _validate_official_url(url)
                archive_name = url.rsplit("/", 1)[-1]
                if not archive_name.endswith(".gkg.csv.zip"):
                    continue
                digits = "".join(c for c in archive_name[:14] if c.isdigit())
                if len(digits) < 8:
                    continue
                when = datetime.strptime(digits[:8], "%Y%m%d").replace(tzinfo=timezone.utc)
                if start.astimezone(timezone.utc).date() <= when.date() <= end.astimezone(timezone.utc).date():
                    candidates.append((when, url))
            except (ValueError, OverflowError):
                continue
        candidates = sorted(candidates)
        discovered_archive_count = len(candidates)
        if len(candidates) > max_archives:
            # Deterministic evenly spaced selection over the preregistered period.
            indices = sorted({round(i * (len(candidates)-1) / (max_archives-1))
                              for i in range(max_archives)}) if max_archives > 1 else [0]
            candidates = [candidates[i] for i in indices]
        observations = []
        manifests = []
        total_bytes = len(listing)
        for _, url in candidates:
            if total_bytes >= aggregate_cap:
                break
            archive = _bounded_get(url, min(MAX_ARCHIVE_BYTES, aggregate_cap-total_bytes))
            total_bytes += len(archive)
            digest = manifest_sha256(archive)
            try:
                with zipfile.ZipFile(BytesIO(archive)) as zf:
                    files = [i for i in zf.infolist() if not i.is_dir() and i.filename.endswith(".csv")]
                    if len(files) != 1 or files[0].file_size > MAX_PAYLOAD_BYTES:
                        continue
                    payload = zf.read(files[0])
            except (zipfile.BadZipFile, OSError, RuntimeError):
                continue
            retrieved = datetime.now(timezone.utc)
            manifest = SourceManifest("gdelt-gkg", "historical-gkg", url, retrieved, digest,
                len(archive), 0, 0, "GKG v2 archive", "review-required",
                "official-host;independent-rights-review-required")
            parsed = parse_gkg_tsv(payload, manifest)
            manifests.append(SourceManifest(manifest.source_id, manifest.dataset_or_series_id,
                manifest.source_url, manifest.retrieved_at, manifest.payload_sha256,
                manifest.payload_bytes, len(parsed), len(parsed), manifest.provider_version,
                manifest.rights_status, manifest.provenance_status))
            observations.extend(parsed)
            if len(observations) >= target_records:
                break
        deduped = deduplicate_snapshots(observations)
        archive_window_complete = len(candidates) == discovered_archive_count
        if len(deduped) >= target_records and archive_window_complete:
            status = "PASS"
        elif len(deduped) > 0:
            status = "PARTIAL"
        else:
            status = "NOT_MET"
        return AcquisitionResult(status, tuple(deduped), tuple(manifests),
            warnings=(f"real_unique_observations={len(deduped)}", f"aggregate_bytes={total_bytes}",
                      f"target_records={target_records}",
                      f"discovered_archives={discovered_archive_count}",
                      f"sampled_archives={len(candidates)}",
                      f"archive_window_complete={str(archive_window_complete).lower()}",
                      "sampling_design=deterministic_evenly_spaced_archive_sample",
                      "no_article_body_retained"))
    except (ValueError, RuntimeError, UnicodeError) as exc:
        return AcquisitionResult("FAILED", error_code=str(exc))
