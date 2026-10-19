"""Strict temporal-semantic data contracts for Serevex research.

This module validates whether a feature fact could have been known at a declared
decision time, and whether an OHLC candle is internally coherent. It is deliberately
fail-closed and is not connected to live signal or order-routing code.

The contract distinguishes event/reference time from information-availability time.
A future scheduled event may be a legitimate known-ahead feature, but only when its
publication/availability metadata proves it was available before the decision. A
future market-bar close is never eligible for a decision made before that close.
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

CONTRACT_VERSION = "serevex-temporal-semantic-contract-v1.0.0"
LIVE_SIGNAL_ENABLED = False
LIVE_ORDER_EXECUTION_ENABLED = False
DECISION_OUTPUT = "NO_TRADE"

SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
SEMANTIC_TYPES = {
    "market_bar",
    "derived_market_feature",
    "macro_observation",
    "positioning_report",
    "news_event",
    "scheduled_event",
    "other",
}
EVENT_TIME_ROLES = {
    "bar_close",
    "past_market_event",
    "observation_period",
    "reporting_period",
    "scheduled_future_event",
    "effective_from",
    "not_applicable",
}
PAST_ONLY_EVENT_ROLES = {
    "bar_close",
    "past_market_event",
    "observation_period",
    "reporting_period",
}
VALUE_TYPES = {"number", "string", "boolean", "category"}
QUOTE_BASES = {"bid", "ask", "mid", "last", "none", "not_applicable", "unknown"}
MANDATORY_FIELDS = (
    "schema_version", "feature_id", "source_id", "instrument", "semantic_type",
    "event_time_role", "value_type", "value", "unit", "quote_basis", "event_time",
    "published_at", "available_at", "observed_at", "source_record_id",
    "source_revision", "source_sha256", "transform_id", "transform_version",
    "transform_sha256", "timezone_semantics_verified",
    "availability_semantics_verified", "revision_semantics_verified",
    "provenance_status", "rights_status", "quality_status",
)


def _aware_utc(value: object, field_name: str) -> datetime:
    """Parse an aware datetime/ISO timestamp; never infer a missing timezone."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            raise ValueError(f"{field_name}_empty_timestamp")
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise ValueError(f"{field_name}_invalid_iso8601") from exc
    else:
        raise ValueError(f"{field_name}_timestamp_type_invalid")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name}_timezone_unverified")
    return parsed.astimezone(timezone.utc)


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _sha256_ok(value: object) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value.strip()) is not None


def _number_ok(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _value_is_well_typed(value: object, value_type: str) -> bool:
    if value_type == "number":
        return _number_ok(value)
    if value_type == "boolean":
        return isinstance(value, bool)
    if value_type in {"string", "category"}:
        return isinstance(value, str) and bool(value.strip())
    return False


def validate_feature_fact(
    fact: Mapping[str, Any],
    *,
    decision_time: datetime | str,
    max_age_seconds: float | None = None,
) -> dict[str, Any]:
    """Audit one immutable feature fact against the information set at decision_time.

    A fact is eligible only if required semantics, timestamps, provenance, licensing,
    content/transform hashes, and quality controls are all explicitly verified.
    Evidence references and flags are assertions requiring separate external audit;
    this function does not authenticate them.
    """
    if not isinstance(fact, Mapping):
        raise ValueError("fact_must_be_mapping")
    decision = _aware_utc(decision_time, "decision_time")
    if max_age_seconds is not None:
        if (
            isinstance(max_age_seconds, bool)
            or not isinstance(max_age_seconds, (int, float))
            or not math.isfinite(float(max_age_seconds))
            or max_age_seconds < 0
        ):
            raise ValueError("max_age_seconds_must_be_finite_nonnegative_number")

    blockers: list[str] = []
    missing = [key for key in MANDATORY_FIELDS if key not in fact]
    if missing:
        blockers.extend(f"missing_field:{key}" for key in missing)

    for key in (
        "schema_version", "feature_id", "source_id", "instrument", "unit",
        "source_record_id", "source_revision", "transform_id", "transform_version",
    ):
        if key in fact and not _nonempty_string(fact[key]):
            blockers.append(f"invalid_nonempty_string:{key}")

    if fact.get("schema_version") != CONTRACT_VERSION:
        blockers.append("schema_version_mismatch")
    semantic_type = fact.get("semantic_type")
    if not isinstance(semantic_type, str) or semantic_type not in SEMANTIC_TYPES:
        blockers.append("semantic_type_unknown")
    event_role = fact.get("event_time_role")
    if not isinstance(event_role, str) or event_role not in EVENT_TIME_ROLES:
        blockers.append("event_time_role_unknown")
    if isinstance(semantic_type, str) and isinstance(event_role, str):
        expected_event_roles = {
            "market_bar": {"bar_close", "past_market_event"},
            "derived_market_feature": {"bar_close", "past_market_event"},
            "macro_observation": {"observation_period", "reporting_period"},
            "positioning_report": {"reporting_period", "past_market_event"},
            "news_event": {"past_market_event", "scheduled_future_event"},
            "scheduled_event": {"scheduled_future_event"},
        }
        allowed_roles = expected_event_roles.get(semantic_type)
        if allowed_roles is not None and event_role not in allowed_roles:
            blockers.append("semantic_type_event_time_role_mismatch")
    value_type = fact.get("value_type")
    if not isinstance(value_type, str) or value_type not in VALUE_TYPES:
        blockers.append("value_type_unknown")
    elif "value" in fact and not _value_is_well_typed(fact.get("value"), value_type):
        blockers.append("value_does_not_match_declared_type")

    quote_basis = fact.get("quote_basis")
    if not isinstance(quote_basis, str) or quote_basis not in QUOTE_BASES:
        blockers.append("quote_basis_unknown")
    elif isinstance(semantic_type, str) and semantic_type in {"market_bar", "derived_market_feature"} and quote_basis not in {"bid", "ask", "mid", "last"}:
        blockers.append("market_feature_requires_explicit_quote_basis")

    for key in ("source_sha256", "transform_sha256"):
        if key in fact and not _sha256_ok(fact.get(key)):
            blockers.append(f"invalid_sha256:{key}")

    for key in (
        "timezone_semantics_verified",
        "availability_semantics_verified",
        "revision_semantics_verified",
    ):
        if fact.get(key) is not True:
            blockers.append(f"unverified_semantics:{key}")

    if fact.get("provenance_status") != "VERIFIED":
        blockers.append("source_provenance_not_verified")
    if fact.get("rights_status") != "VERIFIED":
        blockers.append("source_usage_rights_not_verified")
    if fact.get("quality_status") != "PASS":
        blockers.append("data_quality_not_passed")

    parsed: dict[str, datetime] = {}
    for key in ("event_time", "published_at", "available_at", "observed_at"):
        try:
            parsed[key] = _aware_utc(fact.get(key), key)
        except ValueError as exc:
            blockers.append(str(exc))

    published = parsed.get("published_at")
    available = parsed.get("available_at")
    observed = parsed.get("observed_at")
    event = parsed.get("event_time")

    if published is not None and published > decision:
        blockers.append("published_after_decision")
    if available is not None and available > decision:
        blockers.append("unavailable_at_decision_time")
    if published is not None and available is not None and available < published:
        blockers.append("available_before_published_timestamp")
    if available is not None and observed is not None and observed < available:
        blockers.append("observed_before_available_timestamp")
    if event is not None and isinstance(event_role, str) and event_role in PAST_ONLY_EVENT_ROLES and event > decision:
        blockers.append("event_time_after_decision_for_past_only_role")
    if max_age_seconds is not None and available is not None:
        age = (decision - available).total_seconds()
        if age < 0:
            # Keep a distinct reason even though unavailable_at_decision_time is also set.
            blockers.append("negative_information_age")
        elif age > float(max_age_seconds):
            blockers.append("feature_exceeds_maximum_age")

    # Stable order helps deterministic reports and tests while preserving first occurrence.
    blockers = list(dict.fromkeys(blockers))
    available_age = (
        (decision - available).total_seconds()
        if available is not None
        else None
    )
    eligible = not blockers
    return {
        "contract_version": CONTRACT_VERSION,
        "feature_id": fact.get("feature_id"),
        "source_id": fact.get("source_id"),
        "instrument": fact.get("instrument"),
        "decision_time_utc": decision.isoformat(),
        "eligible_at_decision_time": eligible,
        "status": "ELIGIBLE_FOR_RESEARCH_FEATURE_SET" if eligible else "BLOCKED",
        "blockers": blockers,
        "information_age_seconds": available_age,
        "source_sha256_valid": _sha256_ok(fact.get("source_sha256")),
        "transform_sha256_valid": _sha256_ok(fact.get("transform_sha256")),
        "promotion_status": "DIAGNOSTIC_ONLY",
        "decision_output": DECISION_OUTPUT,
        "live_signal_enabled": LIVE_SIGNAL_ENABLED,
        "live_order_execution_enabled": LIVE_ORDER_EXECUTION_ENABLED,
        "interpretation": (
            "A pass means metadata passes this contract only; it does not prove that "
            "the content is authentic, complete, economically useful, or independently audited."
        ),
    }


def validate_ohlc_candle(
    candle: Mapping[str, Any],
    *,
    decision_time: datetime | str,
    expected_interval_seconds: int,
    timestamp_semantics_verified: bool,
    quote_basis: str,
) -> dict[str, Any]:
    """Validate one bar's temporal and OHLC geometry without repairing its prices."""
    if not isinstance(candle, Mapping):
        raise ValueError("candle_must_be_mapping")
    if (
        isinstance(expected_interval_seconds, bool)
        or not isinstance(expected_interval_seconds, int)
        or expected_interval_seconds <= 0
    ):
        raise ValueError("expected_interval_seconds_must_be_positive_integer")
    decision = _aware_utc(decision_time, "decision_time")
    blockers: list[str] = []

    if timestamp_semantics_verified is not True:
        blockers.append("candle_timestamp_semantics_unverified")
    if not isinstance(quote_basis, str) or quote_basis not in {"bid", "ask", "mid", "last"}:
        blockers.append("candle_quote_basis_not_explicit")

    for key in ("instrument", "source_id", "session_id"):
        if not _nonempty_string(candle.get(key)):
            blockers.append(f"invalid_nonempty_string:{key}")

    try:
        bar_start = _aware_utc(candle.get("bar_start"), "bar_start")
    except ValueError as exc:
        bar_start = None
        blockers.append(str(exc))
    try:
        bar_end = _aware_utc(candle.get("bar_end"), "bar_end")
    except ValueError as exc:
        bar_end = None
        blockers.append(str(exc))

    if bar_start is not None and bar_end is not None:
        if bar_end <= bar_start:
            blockers.append("bar_end_must_follow_bar_start")
        elif (bar_end - bar_start).total_seconds() != expected_interval_seconds:
            blockers.append("bar_duration_mismatch")
        if bar_end > decision:
            blockers.append("bar_not_closed_by_decision_time")

    prices: dict[str, float] = {}
    for key in ("open", "high", "low", "close"):
        value = candle.get(key)
        if not _number_ok(value):
            blockers.append(f"invalid_finite_numeric_price:{key}")
        else:
            prices[key] = float(value)
            if float(value) <= 0:
                blockers.append(f"nonpositive_price:{key}")

    if len(prices) == 4:
        if prices["high"] < max(prices["open"], prices["close"], prices["low"]):
            blockers.append("high_violates_ohlc_geometry")
        if prices["low"] > min(prices["open"], prices["close"], prices["high"]):
            blockers.append("low_violates_ohlc_geometry")

    if "volume" in candle and candle["volume"] is not None:
        if not _number_ok(candle["volume"]) or float(candle["volume"]) < 0:
            blockers.append("volume_must_be_finite_nonnegative")

    blockers = list(dict.fromkeys(blockers))
    return {
        "contract_version": CONTRACT_VERSION,
        "instrument": candle.get("instrument"),
        "source_id": candle.get("source_id"),
        "bar_start_utc": bar_start.isoformat() if bar_start is not None else None,
        "bar_end_utc": bar_end.isoformat() if bar_end is not None else None,
        "quote_basis": quote_basis,
        "eligible_for_closed_bar_research": not blockers,
        "status": "PASS" if not blockers else "BLOCKED",
        "blockers": blockers,
        "price_repair_applied": False,
        "source_promoted": False,
        "decision_output": DECISION_OUTPUT,
        "live_signal_enabled": LIVE_SIGNAL_ENABLED,
        "live_order_execution_enabled": LIVE_ORDER_EXECUTION_ENABLED,
    }


def validate_candle_series(
    candles: Sequence[Mapping[str, Any]],
    *,
    decision_time: datetime | str,
    expected_interval_seconds: int,
    timestamp_semantics_verified: bool,
    quote_basis: str,
) -> dict[str, Any]:
    """Audit candle order, identity, duplicate bars, overlap and session-aware gaps.

    A gap is permitted only across an explicit session boundary. Within one session,
    each bar must begin exactly when the previous bar ends. Session IDs may not recur
    after switching away from them. This rule prevents accidental gap imputation.
    """
    if isinstance(candles, (str, bytes)) or not isinstance(candles, Sequence) or not candles:
        raise ValueError("candles_must_be_nonempty_sequence")
    decision = _aware_utc(decision_time, "decision_time")
    reports = [
        validate_ohlc_candle(
            row,
            decision_time=decision,
            expected_interval_seconds=expected_interval_seconds,
            timestamp_semantics_verified=timestamp_semantics_verified,
            quote_basis=quote_basis,
        )
        for row in candles
    ]
    blockers: list[str] = []
    seen_keys: set[tuple[object, ...]] = set()
    seen_sessions: set[str] = set()
    previous_session: str | None = None
    previous_start: datetime | None = None
    previous_end: datetime | None = None

    for index, (candle, report) in enumerate(zip(candles, reports)):
        for reason in report["blockers"]:
            blockers.append(f"row_{index}:{reason}")
        session = candle.get("session_id")
        same_session_as_previous = session == previous_session
        start = _safe_time(candle.get("bar_start"))
        end = _safe_time(candle.get("bar_end"))
        source_key = candle.get("source_id")
        instrument_key = candle.get("instrument")
        if isinstance(source_key, str) and isinstance(instrument_key, str) and start is not None:
            key = (source_key, instrument_key, start, quote_basis)
            if key in seen_keys:
                blockers.append(f"row_{index}:duplicate_bar_identity")
            seen_keys.add(key)
        else:
            blockers.append(f"row_{index}:duplicate_identity_not_checkable")

        if isinstance(session, str) and session:
            if previous_session is None:
                previous_session = session
                same_session_as_previous = True
            elif session != previous_session:
                seen_sessions.add(previous_session)
                if session in seen_sessions:
                    blockers.append(f"row_{index}:session_id_recurred_after_boundary")
                previous_session = session
                same_session_as_previous = False

        if start is not None and previous_start is not None and start <= previous_start:
            blockers.append(f"row_{index}:bar_starts_not_strictly_increasing")
        if start is not None and previous_end is not None:
            if start < previous_end:
                blockers.append(f"row_{index}:bars_overlap")
            elif same_session_as_previous and start != previous_end:
                blockers.append(f"row_{index}:implicit_gap_within_session")
        previous_start = start if start is not None else previous_start
        previous_end = end if end is not None else previous_end

    blockers = list(dict.fromkeys(blockers))
    return {
        "contract_version": CONTRACT_VERSION,
        "candle_count": len(candles),
        "valid_candle_count": sum(bool(item["eligible_for_closed_bar_research"]) for item in reports),
        "invalid_candle_count": sum(not bool(item["eligible_for_closed_bar_research"]) for item in reports),
        "status": "PASS" if not blockers else "BLOCKED",
        "blockers": blockers,
        "row_reports": reports,
        "data_repaired": False,
        "missing_bars_synthesized": False,
        "promotion_status": "DIAGNOSTIC_ONLY",
        "decision_output": DECISION_OUTPUT,
        "live_signal_enabled": LIVE_SIGNAL_ENABLED,
        "live_order_execution_enabled": LIVE_ORDER_EXECUTION_ENABLED,
    }


def _safe_time(value: object) -> datetime | None:
    try:
        return _aware_utc(value, "bar_timestamp")
    except ValueError:
        return None
