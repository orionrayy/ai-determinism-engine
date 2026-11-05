"""Canonical, auditable point-in-time contract for external context observations."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Mapping, Sequence


AVAILABILITY_QUALITIES = frozenset({"verified", "conservative_proxy", "unknown", "not_applicable"})
ACQUISITION_STATUSES = frozenset({
    "PASS", "PARTIAL", "SKIPPED_CREDENTIAL_REQUIRED", "SKIPPED_POLICY", "NOT_MET", "FAILED"
})


def parse_utc_timestamp(value: str) -> datetime:
    """Parse a timezone-explicit ISO-8601 timestamp and normalize it to UTC."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("timestamp_required")
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        result = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError("invalid_iso8601_timestamp") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("timezone_required")
    return result.astimezone(timezone.utc)


def _validate_datetime(value: datetime | None, name: str) -> None:
    if value is not None and (not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None):
        raise ValueError(f"{name}_must_be_timezone_aware")


@dataclass(frozen=True)
class ExternalObservation:
    source_id: str
    source_record_id: str
    dataset_or_series_id: str
    schema_version: str
    event_time: datetime | None
    published_at: datetime | None
    available_at: datetime | None
    availability_quality: str
    retrieved_at: datetime | None
    source_url: str
    payload_sha256: str
    source_revision: str | None = None
    value: float | None = None
    unit: str | None = None
    attributes: Mapping[str, object] = field(default_factory=dict)
    rights_status: str = "unknown"
    provenance_status: str = "unverified"
    quality_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceManifest:
    source_id: str
    dataset_or_series_id: str
    source_url: str
    retrieved_at: datetime
    payload_sha256: str
    payload_bytes: int
    row_count_raw: int
    row_count_valid: int
    provider_version: str = "unknown"
    rights_status: str = "unknown"
    provenance_status: str = "unverified"


@dataclass(frozen=True)
class AcquisitionResult:
    status: str
    observations: tuple[ExternalObservation, ...] = ()
    manifests: tuple[SourceManifest, ...] = ()
    warnings: tuple[str, ...] = ()
    error_code: str | None = None

    def __post_init__(self) -> None:
        if self.status not in ACQUISITION_STATUSES:
            raise ValueError("invalid_acquisition_status")


@dataclass(frozen=True)
class SourceAudit:
    row_count: int
    valid_count: int
    invalid_reasons: Mapping[str, int]
    duplicate_id_count: int
    conflicting_duplicate_count: int
    unknown_availability_count: int
    conservative_proxy_count: int
    status: str


def manifest_sha256(payload: bytes) -> str:
    if not isinstance(payload, bytes):
        raise TypeError("payload_must_be_bytes")
    return hashlib.sha256(payload).hexdigest()


def _signature(observation: ExternalObservation) -> tuple[object, ...]:
    return (
        observation.event_time, observation.published_at, observation.available_at,
        observation.value, observation.unit, observation.source_revision,
        tuple(sorted((str(k), repr(v)) for k, v in observation.attributes.items())),
    )


def validate_observations(observations: Sequence[ExternalObservation]) -> SourceAudit:
    """Validate records without mutating or silently overwriting duplicates."""
    invalid: dict[str, int] = {}
    duplicate_count = 0
    conflicting_count = 0
    unknown_count = 0
    proxy_count = 0
    seen: dict[tuple[str, str], tuple[object, ...]] = {}
    for index, item in enumerate(observations):
        if not isinstance(item, ExternalObservation):
            raise ValueError(f"invalid_observation_type_at_{index}")
        for name in ("event_time", "published_at", "available_at", "retrieved_at"):
            _validate_datetime(getattr(item, name), name)
        if not item.source_id.strip() or not item.source_record_id.strip() or not item.dataset_or_series_id.strip():
            raise ValueError(f"missing_identity_at_{index}")
        if item.availability_quality not in AVAILABILITY_QUALITIES:
            raise ValueError(f"invalid_availability_quality_at_{index}")
        if item.value is not None and not math.isfinite(float(item.value)):
            raise ValueError(f"nonfinite_value_at_{index}")
        if item.available_at is not None and item.retrieved_at is not None and item.available_at > item.retrieved_at:
            raise ValueError(f"available_at_after_retrieved_at_at_{index}")
        if item.available_at is None and item.availability_quality != "unknown":
            raise ValueError(f"availability_timestamp_missing_without_unknown_quality_at_{index}")
        if item.available_at is None or item.availability_quality == "unknown":
            unknown_count += 1
        if item.availability_quality == "conservative_proxy":
            proxy_count += 1
        key = (item.source_id, item.source_record_id)
        signature = _signature(item)
        if key in seen:
            duplicate_count += 1
            if seen[key] != signature:
                conflicting_count += 1
                invalid["conflicting_duplicate_id"] = invalid.get("conflicting_duplicate_id", 0) + 1
        else:
            seen[key] = signature
    valid = len(observations) - invalid.get("conflicting_duplicate_id", 0)
    return SourceAudit(
        row_count=len(observations), valid_count=max(0, valid),
        invalid_reasons=dict(sorted(invalid.items())),
        duplicate_id_count=duplicate_count,
        conflicting_duplicate_count=conflicting_count,
        unknown_availability_count=unknown_count,
        conservative_proxy_count=proxy_count,
        status="PASS" if not invalid else "PARTIAL",
    )
