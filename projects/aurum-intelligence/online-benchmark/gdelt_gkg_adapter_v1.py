"""Bounded point-in-time GDELT GKG metadata adapter (never persists article bodies)."""
from __future__ import annotations

import csv
from bisect import bisect_left
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
from typing import Sequence

from external_context_schema_v1 import AcquisitionResult, ExternalObservation, SourceManifest, manifest_sha256

OFFICIAL_HOST = "data.gdeltproject.org"
MASTER_URL = "https://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
# The official GDELT 2.0 master list is currently >120 MB. Keep it bounded,
# but large enough to select historical 15-minute GKG archives.
MAX_MASTER_BYTES = 160 * 1024 * 1024
MAX_ARCHIVE_BYTES = 48 * 1024 * 1024
MAX_AGGREGATE_BYTES = 800 * 1024 * 1024
MAX_RECORDS_PER_ARCHIVE = 500_000
MAX_ARCHIVES_PER_RUN = 480
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
            row_content_conflicts = (
                previous.event_time != item.event_time
                or previous.published_at != item.published_at
                or previous.available_at != item.available_at
                or previous.value != item.value
                or previous.unit != item.unit
                or previous.attributes != item.attributes
            )
            # payload_sha256 is the enclosing archive hash, not a per-record hash.
            if row_content_conflicts:
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


def _download_archive_with_budget(
    url: str,
    *,
    remaining_bytes: int,
    prior_observation_count: int,
) -> tuple[bytes | None, str | None]:
    """Download one archive without throwing away earlier records on budget exhaustion."""
    if isinstance(remaining_bytes, bool) or not isinstance(remaining_bytes, int) or remaining_bytes < 1:
        if prior_observation_count > 0:
            return None, "aggregate_byte_cap_reached_before_next_archive"
        raise ValueError("aggregate_byte_cap_exceeded_before_first_archive")
    cap = min(MAX_ARCHIVE_BYTES, remaining_bytes)
    try:
        return _bounded_get(url, cap), None
    except ValueError as exc:
        code = str(exc)
        if code in {"content_length_exceeds_limit", "download_exceeds_limit"}:
            # Distinguish a single archive that exceeds its own bound from exhaustion
            # of the whole remaining budget. The old branch mislabeled both cases as
            # aggregate exhaustion and stopped the entire multi-year sample early.
            if remaining_bytes > MAX_ARCHIVE_BYTES:
                return None, "archive_skipped_per_archive_byte_cap"
            if prior_observation_count > 0:
                return None, "aggregate_byte_cap_reached_before_next_archive"
        raise


def select_historical_gkg_archives(
    master_listing: str,
    start: datetime,
    end: datetime,
    *,
    max_archives: int,
    sampling_design: str = "evenly_spaced",
    split_boundaries: Sequence[datetime] | None = None,
    block_centers: Sequence[datetime] | None = None,
) -> tuple[list[tuple[datetime, str]], int]:
    """Extract historical GKG URLs from the official size/hash/URL master-list rows.

    The manifest format is three whitespace-delimited columns: byte size, checksum,
    URL. Archive URLs may use legacy HTTP, which is upgraded only for the exact
    allowlisted official host/path. Return (bounded deterministic sample, total
    matching archive count) so sampling is never misreported as complete coverage.
    """
    if start.tzinfo is None or end.tzinfo is None or start >= end:
        raise ValueError("invalid_time_window")
    if isinstance(max_archives, bool) or not isinstance(max_archives, int) or max_archives < 1:
        raise ValueError("invalid_max_archives")
    if sampling_design not in {"evenly_spaced", "split_stratified_contiguous_blocks"}:
        raise ValueError("invalid_sampling_design")
    start_utc = start.astimezone(timezone.utc)
    end_utc = end.astimezone(timezone.utc)
    normalized_boundaries: tuple[datetime, datetime] | None = None
    normalized_centers: tuple[datetime, datetime, datetime] | None = None
    if sampling_design == "split_stratified_contiguous_blocks":
        if split_boundaries is None or len(split_boundaries) != 2:
            raise ValueError("split_sampling_requires_two_chronological_boundaries")
        bounds = []
        for index, value in enumerate(split_boundaries):
            if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"split_boundary_must_be_timezone_aware_at_{index}")
            bounds.append(value.astimezone(timezone.utc))
        if not start_utc < bounds[0] < bounds[1] < end_utc:
            raise ValueError("split_boundaries_must_be_inside_window_and_increasing")
        normalized_boundaries = (bounds[0], bounds[1])
        if block_centers is not None:
            if len(block_centers) != 3:
                raise ValueError("split_sampling_requires_three_block_centers")
            centers = []
            for index, value in enumerate(block_centers):
                if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
                    raise ValueError(f"block_center_must_be_timezone_aware_at_{index}")
                centers.append(value.astimezone(timezone.utc))
            if not (start_utc <= centers[0] < bounds[0]
                    and bounds[0] <= centers[1] < bounds[1]
                    and bounds[1] <= centers[2] <= end_utc):
                raise ValueError("block_centers_must_fall_inside_their_chronological_splits")
            normalized_centers = (centers[0], centers[1], centers[2])
    candidates: list[tuple[datetime, str]] = []
    for line in master_listing.splitlines():
        bits = line.split()
        if len(bits) < 3:
            continue
        try:
            url = normalize_archive_url(bits[-1])
            _validate_official_url(url)
            archive_name = url.rsplit("/", 1)[-1]
            if not archive_name.endswith(".gkg.csv.zip"):
                continue
            timestamp = archive_name[:14]
            if len(timestamp) != 14 or not timestamp.isdigit():
                continue
            archive_time = datetime.strptime(timestamp, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
            if start_utc <= archive_time <= end_utc:
                candidates.append((archive_time, url))
        except (ValueError, OverflowError):
            continue
    candidates.sort()
    discovered_count = len(candidates)
    if discovered_count > max_archives:
        if sampling_design == "evenly_spaced":
            # Compatibility mode: evenly spaced timestamps do not imply continuous coverage.
            indices = sorted({
                round(i * (discovered_count - 1) / (max_archives - 1))
                for i in range(max_archives)
            }) if max_archives > 1 else [0]
            candidates = [candidates[i] for i in indices]
        else:
            assert normalized_boundaries is not None
            edges = (start_utc, normalized_boundaries[0], normalized_boundaries[1], end_utc)
            centers = normalized_centers or tuple(
                left + (right - left) / 2 for left, right in zip(edges[:-1], edges[1:])
            )
            base_budget, remainder = divmod(max_archives, 3)
            budgets = [base_budget + int(i < remainder) for i in range(3)]
            blocks: list[list[tuple[datetime, str]]] = []
            for split_index, (left, right, center, budget) in enumerate(
                zip(edges[:-1], edges[1:], centers, budgets)
            ):
                segment = [item for item in candidates
                           if left <= item[0] < right or (split_index == 2 and item[0] == right)]
                if budget <= 0 or not segment:
                    blocks.append([])
                elif len(segment) <= budget:
                    blocks.append(segment)
                else:
                    times = [item[0] for item in segment]
                    center_index = min(bisect_left(times, center), len(segment) - 1)
                    block_start = max(0, min(center_index - budget // 2, len(segment) - budget))
                    blocks.append(segment[block_start:block_start + budget])
            # Interleave the segments so a strict aggregate-byte cap cannot spend
            # all available bytes on the earliest chronological region.
            selected: list[tuple[datetime, str]] = []
            for rank in range(max((len(block) for block in blocks), default=0)):
                for block in blocks:
                    if rank < len(block):
                        selected.append(block[rank])
            candidates = selected
    return candidates, discovered_count


def archive_coverage_intervals(
    manifests: Sequence[SourceManifest],
) -> tuple[tuple[datetime, datetime], ...]:
    """Return observed 15-minute archive batches; gaps are intentionally preserved.

    These intervals represent only archives actually acquired and parsed. A large
    record count from sparse archives is not evidence of continuous news coverage.
    """
    intervals: set[tuple[datetime, datetime]] = set()
    for manifest in manifests:
        if manifest.source_id != "gdelt-gkg":
            continue
        parsed = urlparse(manifest.source_url)
        if parsed.scheme != "https" or parsed.hostname != OFFICIAL_HOST:
            raise ValueError("gdelt_manifest_source_not_allowlisted")
        name = parsed.path.rsplit("/", 1)[-1]
        if not name.endswith(".gkg.csv.zip") or len(name) < 14 or not name[:14].isdigit():
            raise ValueError("invalid_gdelt_archive_manifest_timestamp")
        start = datetime.strptime(name[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        intervals.add((start, start + BATCH_INTERVAL))
    return tuple(sorted(intervals))


def summarize_archive_coverage(
    intervals: Sequence[tuple[datetime, datetime]],
    window_start: datetime,
    window_end: datetime,
) -> dict[str, object]:
    """Measure actual observed time coverage; count density never fills archive gaps."""
    if (window_start.tzinfo is None or window_start.utcoffset() is None
            or window_end.tzinfo is None or window_end.utcoffset() is None):
        raise ValueError("coverage_window_must_be_timezone_aware")
    start = window_start.astimezone(timezone.utc)
    end = window_end.astimezone(timezone.utc)
    if start >= end:
        raise ValueError("invalid_coverage_window_order")
    unique_clipped: set[tuple[datetime, datetime]] = set()
    for index, pair in enumerate(intervals):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(f"invalid_news_coverage_interval_at_{index}")
        left, right = pair
        if (left.tzinfo is None or left.utcoffset() is None
                or right.tzinfo is None or right.utcoffset() is None):
            raise ValueError(f"coverage_interval_must_be_timezone_aware_at_{index}")
        left = max(left.astimezone(timezone.utc), start)
        right = min(right.astimezone(timezone.utc), end)
        if left >= right:
            continue
        unique_clipped.add((left, right))
    ordered = sorted(unique_clipped)
    merged: list[tuple[datetime, datetime]] = []
    for left, right in ordered:
        if merged and left <= merged[-1][1]:
            prior_left, prior_right = merged[-1]
            merged[-1] = (prior_left, max(prior_right, right))
        else:
            merged.append((left, right))
    covered_seconds = sum((right-left).total_seconds() for left, right in merged)
    requested_seconds = (end-start).total_seconds()
    fraction = min(1.0, covered_seconds/requested_seconds)
    if fraction >= 1.0 - 1e-12:
        status = "PASS"
    elif covered_seconds > 0:
        status = "PARTIAL"
    else:
        status = "NOT_MET"
    return {
        "status": status,
        "observed_interval_count": len(unique_clipped),
        "merged_coverage_interval_count": len(merged),
        "covered_minutes": round(covered_seconds/60.0, 3),
        "requested_window_minutes": round(requested_seconds/60.0, 3),
        "coverage_fraction": round(fraction, 9),
        "coverage_percent": round(100.0*fraction, 6),
    }


def acquire_historical_gkg(
    start: datetime, end: datetime, *, target_records: int = 100_000,
    max_archives: int = MAX_ARCHIVES_PER_RUN, aggregate_cap: int = MAX_AGGREGATE_BYTES,
    sampling_design: str = "evenly_spaced",
    split_boundaries: Sequence[datetime] | None = None,
    block_centers: Sequence[datetime] | None = None,
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
        candidates, discovered_archive_count = select_historical_gkg_archives(
            listing, start, end, max_archives=max_archives, sampling_design=sampling_design,
            split_boundaries=split_boundaries, block_centers=block_centers,
        )
        observations = []
        manifests = []
        acquisition_warnings: list[str] = []
        total_bytes = len(listing.encode("utf-8"))
        for _, url in candidates:
            remaining = aggregate_cap - total_bytes
            archive, stop_warning = _download_archive_with_budget(
                url, remaining_bytes=remaining, prior_observation_count=len(observations)
            )
            if stop_warning:
                acquisition_warnings.append(stop_warning)
                if stop_warning == "aggregate_byte_cap_reached_before_next_archive":
                    break
                # One oversized/corrupt archive must not truncate all later dates.
                continue
            if archive is None:
                acquisition_warnings.append("archive_download_returned_no_payload")
                break
            total_bytes += len(archive)
            digest = manifest_sha256(archive)
            try:
                with zipfile.ZipFile(BytesIO(archive)) as zf:
                    files = [i for i in zf.infolist() if not i.is_dir() and i.filename.endswith(".csv")]
                    if len(files) != 1 or files[0].file_size > MAX_PAYLOAD_BYTES:
                        acquisition_warnings.append("archive_skipped_invalid_csv_file_count_or_size")
                        continue
                    payload = zf.read(files[0])
            except (zipfile.BadZipFile, OSError, RuntimeError):
                acquisition_warnings.append("archive_skipped_invalid_zip")
                continue
            retrieved = datetime.now(timezone.utc)
            source_manifest = SourceManifest("gdelt-gkg", "historical-gkg", url, retrieved, digest,
                len(archive), 0, 0, "GKG v2 archive", "review-required",
                "official-host;independent-rights-review-required")
            try:
                parsed = parse_gkg_tsv(payload, source_manifest)
            except (ValueError, UnicodeError):
                acquisition_warnings.append("archive_skipped_parser_rejection")
                continue
            manifests.append(SourceManifest(source_manifest.source_id, source_manifest.dataset_or_series_id,
                source_manifest.source_url, source_manifest.retrieved_at, source_manifest.payload_sha256,
                source_manifest.payload_bytes, len(parsed), len(parsed), source_manifest.provider_version,
                source_manifest.rights_status, source_manifest.provenance_status))
            observations.extend(parsed)
            # Do not stop on raw rows: repeated document snapshots are deduplicated
            # after acquisition, and an early stop would bias the sample toward the
            # oldest archives instead of preserving temporal spread.
        deduped = deduplicate_snapshots(observations)
        archive_window_complete = len(candidates) == discovered_archive_count and not acquisition_warnings
        sampled_archive_count = len(candidates)
        processed_archive_count = len(manifests)
        sampled_window_coverage = (
            processed_archive_count / sampled_archive_count if sampled_archive_count else 0.0
        )
        # The candidate set is intentionally a bounded deterministic sample of the
        # 85k+ archive window. Full-window coverage is not a realistic CI gate; PASS
        # means the selected sample was acquired without skips and met the real-data
        # target. Coverage fields remain explicit so this is never called a complete
        # archive census.
        sample_completed = (
            processed_archive_count == sampled_archive_count and not acquisition_warnings
        )
        if len(deduped) >= target_records and sample_completed:
            status = "PASS"
        elif len(deduped) >= target_records or len(deduped) > 0:
            status = "PARTIAL"
        else:
            status = "NOT_MET"
        return AcquisitionResult(status, tuple(deduped), tuple(manifests),
            warnings=(f"real_unique_observations={len(deduped)}", f"aggregate_bytes={total_bytes}",
                      f"target_records={target_records}",
                      f"discovered_archives={discovered_archive_count}",
                      f"sampled_archives={sampled_archive_count}",
                      f"processed_sampled_archives={processed_archive_count}",
                      f"sampled_window_coverage={sampled_window_coverage:.6f}",
                      f"sample_completed={str(sample_completed).lower()}",
                      f"archive_window_complete={str(archive_window_complete).lower()}",
                      f"sampling_design={sampling_design}",
                      "no_article_body_retained", *acquisition_warnings))
    except (ValueError, RuntimeError, UnicodeError) as exc:
        return AcquisitionResult("FAILED", error_code=str(exc))
