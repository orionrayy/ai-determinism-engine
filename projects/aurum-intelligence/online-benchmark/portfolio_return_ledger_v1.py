"""Fail-closed portfolio-level return ledger for research use only.

Gross PnL must be marked at executable quote side (long at Bid, short at Ask);
spread is therefore already embedded in gross PnL. spread_cost is audit metadata
and is not deducted a second time. No broker or order-routing capability exists.
"""
from __future__ import annotations
import hashlib
import json
import re
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

SCHEMA_VERSION = "portfolio-return-ledger-v1.0.0"
DECISION_OUTPUT = "NO_TRADE"
LIVE_SIGNAL_ENABLED = False
LIVE_ORDER_EXECUTION_ENABLED = False
COST_FIELDS = ("spread_cost", "commission_cost", "slippage_cost", "other_fees_cost", "market_impact_cost", "financing_pnl")
NONNEGATIVE_COST_FIELDS = frozenset(COST_FIELDS[:-1])
EVIDENCE_STATES = frozenset({"observed", "assumed", "unknown", "not_applicable"})
GATE_FLAGS = ("source_data_qualified", "execution_costs_verified", "portfolio_replay_verified", "locked_holdout_unseen", "trial_registry_complete")


def _dec(value: object, name: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{name}_must_be_finite_decimal")
    try:
        x = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name}_must_be_finite_decimal") from exc
    if not x.is_finite():
        raise ValueError(f"{name}_must_be_finite_decimal")
    return x


def _time(value: object, name: str) -> datetime:
    if isinstance(value, str):
        raw = value.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            value = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise ValueError(f"{name}_must_be_iso8601") from exc
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name}_must_be_timezone_aware")
    return value.astimezone(timezone.utc)


def _sha(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None


def _ref(value: object) -> bool:
    if not isinstance(value, str) or not value.strip() or len(value) > 2048:
        return False
    raw = value.strip()
    if raw.startswith("https://"):
        p = urlparse(raw)
        return bool(p.hostname) and p.username is None and p.password is None
    if raw.startswith(("repo://", "evidence://")):
        scheme, ident = raw.split("://", 1)
        return scheme in {"repo", "evidence"} and bool(ident.strip("/")) and not any(c.isspace() for c in ident)
    return False


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 512:
        raise ValueError(f"{name}_must_be_nonempty_string")
    return value.strip()


def _cost_evidence(raw: object, *, field_name: str, amount: Decimal, trade_count: int) -> tuple[dict[str, Any], str]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{field_name}_evidence_must_be_mapping")
    state = raw.get("state")
    if state not in EVIDENCE_STATES:
        raise ValueError(f"{field_name}_evidence_state_invalid")
    ref, reason = raw.get("reference"), raw.get("reason")
    out = {"state": state, "reference": None, "reason": None}
    if state == "observed":
        if not _ref(ref):
            raise ValueError(f"{field_name}_observed_evidence_requires_stable_reference")
        out["reference"] = ref.strip()
        return out, state
    if state in {"assumed", "unknown"}:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{field_name}_{state}_evidence_requires_reason")
        out["reason"] = reason.strip()[:512]
        if ref is not None:
            if not _ref(ref):
                raise ValueError(f"{field_name}_evidence_reference_invalid")
            out["reference"] = ref.strip()
        return out, state
    if amount != 0:
        raise ValueError(f"{field_name}_not_applicable_but_amount_nonzero")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(f"{field_name}_not_applicable_requires_reason")
    if trade_count > 0 and field_name != "financing_pnl":
        raise ValueError(f"{field_name}_not_applicable_during_fill_interval")
    out["reason"] = reason.strip()[:512]
    return out, state


def validate_portfolio_ledger(
    rows: Sequence[Mapping[str, Any]], *,
    expected_interval_seconds: int,
    starting_equity: object,
    max_gross_leverage: object,
    source_data_qualified: bool = False,
    execution_costs_verified: bool = False,
    portfolio_replay_verified: bool = False,
    locked_holdout_unseen: bool = False,
    trial_registry_complete: bool = False,
    gate_evidence_refs: Mapping[str, str] | None = None,
    equity_tolerance: object = "0.01",
) -> dict[str, Any]:
    """Validate a full, regular-clock ledger; never drops intervals or authorizes trades."""
    if isinstance(expected_interval_seconds, bool) or not isinstance(expected_interval_seconds, int) or expected_interval_seconds <= 0:
        raise ValueError("expected_interval_seconds_must_be_positive_integer")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise ValueError("ledger_rows_must_be_nonempty_sequence")
    flags = {
        "source_data_qualified": source_data_qualified,
        "execution_costs_verified": execution_costs_verified,
        "portfolio_replay_verified": portfolio_replay_verified,
        "locked_holdout_unseen": locked_holdout_unseen,
        "trial_registry_complete": trial_registry_complete,
    }
    if any(not isinstance(value, bool) for value in flags.values()):
        raise ValueError("gate_flags_must_be_boolean")
    start_eq = _dec(starting_equity, "starting_equity")
    leverage_cap = _dec(max_gross_leverage, "max_gross_leverage")
    tolerance = _dec(equity_tolerance, "equity_tolerance")
    if start_eq <= 0:
        raise ValueError("starting_equity_must_be_positive")
    if leverage_cap <= 0:
        raise ValueError("max_gross_leverage_must_be_positive")
    if tolerance < 0:
        raise ValueError("equity_tolerance_must_be_nonnegative")

    out_rows, seen_ids, seen_sessions = [], set(), set()
    last_time = last_session = None
    last_equity = None
    identity = None
    unknown, assumed, observed_only = False, False, True
    observed_fills = True
    no_exposure_count = trade_interval_count = trade_count_total = 0
    gross_total = net_total = spread_audit_total = Decimal("0")
    peak, max_dd = start_eq, Decimal("0")
    warnings = []

    for i, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"row_{i}_must_be_mapping")
        stamp = _time(row.get("timestamp"), f"row_{i}_timestamp")
        session = _text(row.get("session_id"), f"row_{i}_session_id")
        if last_time is not None:
            if stamp <= last_time:
                raise ValueError("timestamps_must_be_strictly_increasing")
            if session == last_session:
                if (stamp - last_time).total_seconds() != expected_interval_seconds:
                    raise ValueError("within_session_timestamps_must_match_expected_cadence")
            else:
                if session in seen_sessions:
                    raise ValueError("session_ids_must_be_contiguous")
                seen_sessions.add(session)
        else:
            seen_sessions.add(session)
        last_time, last_session = stamp, session

        ident = {k: _text(row.get(k), f"row_{i}_{k}") for k in ("portfolio_id", "strategy_id", "trial_id")}
        if identity is None:
            identity = ident
        elif ident != identity:
            raise ValueError("portfolio_strategy_trial_identity_must_be_stable")
        decision_id = _text(row.get("decision_id"), f"row_{i}_decision_id")
        if decision_id in seen_ids:
            raise ValueError("decision_ids_must_be_unique")
        seen_ids.add(decision_id)
        source_id = _text(row.get("source_id"), f"row_{i}_source_id")
        market_hash, trace_hash = row.get("market_snapshot_sha256"), row.get("decision_trace_sha256")
        if not _sha(market_hash):
            raise ValueError(f"row_{i}_market_snapshot_sha256_invalid")
        if not _sha(trace_hash):
            raise ValueError(f"row_{i}_decision_trace_sha256_invalid")
        if row.get("pnl_basis") != "quote_side_after_spread":
            raise ValueError("pnl_basis_must_be_quote_side_after_spread")

        eq_before = _dec(row.get("equity_before"), f"row_{i}_equity_before")
        eq_after = _dec(row.get("equity_after"), f"row_{i}_equity_after")
        gross = _dec(row.get("gross_pnl_quote_side"), f"row_{i}_gross_pnl_quote_side")
        if eq_before <= 0 or eq_after <= 0:
            raise ValueError(f"row_{i}_equity_must_remain_positive")
        if i == 0:
            if abs(eq_before - start_eq) > tolerance:
                raise ValueError("first_equity_before_must_match_starting_equity")
        elif last_equity is not None and abs(eq_before - last_equity) > tolerance:
            raise ValueError("equity_chain_discontinuity")

        costs = {name: _dec(row.get(name), f"row_{i}_{name}") for name in COST_FIELDS}
        for name in NONNEGATIVE_COST_FIELDS:
            if costs[name] < 0:
                raise ValueError(f"row_{i}_{name}_must_be_nonnegative")
        exposed = row.get("exposure_during_interval")
        if not isinstance(exposed, bool):
            raise ValueError(f"row_{i}_exposure_during_interval_must_be_boolean")
        long_exp = _dec(row.get("long_exposure_cash"), f"row_{i}_long_exposure_cash")
        short_exp = _dec(row.get("short_exposure_cash"), f"row_{i}_short_exposure_cash")
        turnover = _dec(row.get("turnover_cash"), f"row_{i}_turnover_cash")
        trades = row.get("trade_count")
        if isinstance(trades, bool) or not isinstance(trades, int) or trades < 0:
            raise ValueError(f"row_{i}_trade_count_must_be_nonnegative_integer")
        if min(long_exp, short_exp, turnover) < 0:
            raise ValueError(f"row_{i}_exposure_and_turnover_must_be_nonnegative")
        leverage = (long_exp + short_exp) / eq_before
        if leverage > leverage_cap:
            raise ValueError(f"row_{i}_gross_leverage_limit_exceeded")

        fill_mode, fill_ref, fill_sha = row.get("fill_mode"), row.get("fill_reference"), row.get("fill_sha256")
        if trades == 0:
            if fill_mode != "none" or fill_ref is not None or fill_sha is not None:
                raise ValueError(f"row_{i}_no_trade_fill_fields_must_be_empty")
            if turnover != 0:
                raise ValueError(f"row_{i}_turnover_requires_trade_count")
            no_exposure_count += int(not exposed)
        else:
            trade_interval_count += 1
            trade_count_total += trades
            if not exposed or turnover <= 0:
                raise ValueError(f"row_{i}_trades_require_interval_exposure_and_positive_turnover")
            if fill_mode not in {"observed", "simulated"}:
                raise ValueError(f"row_{i}_trade_fill_mode_invalid")
            if not _ref(fill_ref) or not _sha(fill_sha):
                raise ValueError(f"row_{i}_trade_fill_requires_reference_and_sha256")
            if fill_mode != "observed":
                observed_fills = False
                warnings.append("simulated_fill_present")

        if not exposed:
            if trades != 0 or long_exp != 0 or short_exp != 0 or turnover != 0:
                raise ValueError(f"row_{i}_flat_interval_has_exposure_or_turnover")
            if gross != 0 or any(v != 0 for v in costs.values()):
                raise ValueError(f"row_{i}_flat_interval_must_have_zero_pnl_and_costs")
        elif not (long_exp > 0 or short_exp > 0 or trades > 0):
            raise ValueError(f"row_{i}_exposure_flag_without_position_or_fill")

        evidence = row.get("cost_evidence")
        if not isinstance(evidence, Mapping) or set(evidence) != set(COST_FIELDS):
            raise ValueError(f"row_{i}_cost_evidence_must_cover_exact_cost_fields")
        normalized_evidence, states = {}, {}
        for name in COST_FIELDS:
            normalized_evidence[name], state = _cost_evidence(
                evidence[name], field_name=name, amount=costs[name], trade_count=trades
            )
            states[name] = state
            if state == "unknown":
                unknown, observed_only = True, False
            elif state == "assumed":
                assumed, observed_only = True, False
        if any(state == "not_applicable" and costs[name] != 0 for name, state in states.items()):
            raise ValueError(f"row_{i}_not_applicable_cost_has_nonzero_amount")

        # Gross PnL is quote-side marked, so spread is already reflected in gross.
        net = gross - costs["commission_cost"] - costs["slippage_cost"] - costs["other_fees_cost"] - costs["market_impact_cost"] + costs["financing_pnl"]
        expected_after = eq_before + net
        if abs(eq_after - expected_after) > tolerance:
            raise ValueError(f"row_{i}_equity_pnl_reconciliation_failed")
        if eq_after != expected_after:
            warnings.append("equity_reconciled_within_explicit_tolerance")
        gross_total += gross
        net_total += net
        spread_audit_total += costs["spread_cost"]
        last_equity = eq_after
        peak = max(peak, eq_after)
        max_dd = max(max_dd, (peak - eq_after) / peak)
        fill_ref_norm = fill_ref.strip() if isinstance(fill_ref, str) else None

        out_rows.append({
            "timestamp": stamp.isoformat().replace("+00:00", "Z"),
            "session_id": session,
            "decision_id": decision_id,
            **ident,
            "source_id": source_id,
            "market_snapshot_sha256": market_hash.lower(),
            "decision_trace_sha256": trace_hash.lower(),
            "pnl_basis": "quote_side_after_spread",
            "equity_before": str(eq_before),
            "equity_after": str(eq_after),
            "gross_pnl_quote_side": str(gross),
            "spread_cost_audit_only_embedded_in_gross": str(costs["spread_cost"]),
            "commission_cost": str(costs["commission_cost"]),
            "slippage_cost": str(costs["slippage_cost"]),
            "other_fees_cost": str(costs["other_fees_cost"]),
            "market_impact_cost": str(costs["market_impact_cost"]),
            "financing_pnl": str(costs["financing_pnl"]),
            "net_pnl_quote_side_after_costs": str(net),
            "gross_return": float(gross / eq_before),
            "net_return": float(net / eq_before),
            "exposure_during_interval": exposed,
            "long_exposure_cash": str(long_exp),
            "short_exposure_cash": str(short_exp),
            "gross_leverage": float(leverage),
            "trade_count": trades,
            "turnover_cash": str(turnover),
            "fill_mode": fill_mode,
            "fill_reference": fill_ref_norm,
            "fill_sha256": fill_sha.lower() if isinstance(fill_sha, str) else None,
            "cost_evidence": normalized_evidence,
        })

    refs = gate_evidence_refs if isinstance(gate_evidence_refs, Mapping) else {}
    valid_refs = {k: refs[k].strip() for k in GATE_FLAGS if k in refs and _ref(refs[k])}
    gates = {k: flags[k] is True and k in valid_refs for k in GATE_FLAGS}
    evidence_status = "INCOMPLETE" if unknown else "ASSUMPTION_ONLY" if assumed else "OBSERVED_OR_NOT_APPLICABLE_ONLY"
    eligible = not unknown and not assumed and observed_only and observed_fills and all(gates.values())
    canonical = json.dumps(out_rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "contract_status": "PASS",
        "economic_evidence_status": evidence_status,
        "row_count": len(out_rows),
        "portfolio_id": identity["portfolio_id"],
        "strategy_id": identity["strategy_id"],
        "trial_id": identity["trial_id"],
        "starting_equity": str(start_eq),
        "ending_equity": str(last_equity),
        "gross_pnl_total": str(gross_total),
        "net_pnl_total": str(net_total),
        "spread_cost_audit_total_embedded_in_gross": str(spread_audit_total),
        "mean_period_net_return": sum(x["net_return"] for x in out_rows) / len(out_rows),
        "maximum_equity_drawdown_fraction_descriptive": float(max_dd),
        "no_exposure_interval_count": sum(not x["exposure_during_interval"] for x in out_rows),
        "intervals_with_fills": trade_interval_count,
        "total_fill_count": trade_count_total,
        "expected_interval_seconds": expected_interval_seconds,
        "timestamps": [x["timestamp"] for x in out_rows],
        "session_ids": [x["session_id"] for x in out_rows],
        "periods": out_rows,
        "net_returns": [x["net_return"] for x in out_rows],
        "gross_returns": [x["gross_return"] for x in out_rows],
        "ledger_sha256": digest,
        "gate_status": {k: "PASS" if v else "BLOCKED" for k, v in gates.items()},
        "gate_evidence_refs": valid_refs,
        "selection_guard_input_eligible": eligible,
        "selection_guard_blockers": [
            *([] if not unknown else ["material_cost_evidence_unknown"]),
            *([] if not assumed else ["assumed_cost_evidence_present"]),
            *([] if observed_fills else ["simulated_trade_fill_present"]),
            *[k for k, ok in gates.items() if not ok],
        ],
        "decision_output": DECISION_OUTPUT,
        "live_signal_enabled": LIVE_SIGNAL_ENABLED,
        "live_order_execution_enabled": LIVE_ORDER_EXECUTION_ENABLED,
        "warnings": sorted(set(warnings)),
        "limitations": [
            "Gross PnL must be reconciled independently to executable Bid/Ask marks; hashes do not authenticate the source.",
            "Spread is embedded in gross PnL and must not be deducted twice.",
            "Evidence references and declared gate flags are not self-authenticating.",
            "A validated ledger does not prove alpha, profitability, broker equivalence, or deployment readiness.",
        ],
    }


def build_synchronized_return_panel(
    ledger_reports: Mapping[str, Mapping[str, Any]], *,
    expected_interval_seconds: int,
    benchmark_returns: Sequence[object],
) -> dict[str, Any]:
    """Create a synchronized strategy-return panel without dropping unmatched periods."""
    if not isinstance(ledger_reports, Mapping) or not ledger_reports:
        raise ValueError("ledger_reports_mapping_required")
    if isinstance(expected_interval_seconds, bool) or not isinstance(expected_interval_seconds, int) or expected_interval_seconds <= 0:
        raise ValueError("expected_interval_seconds_must_be_positive_integer")
    base_times = base_sessions = None
    strategy_returns, digests = {}, {}
    eligible = True
    for key in sorted(ledger_reports):
        report = ledger_reports[key]
        if not isinstance(key, str) or not key.strip() or not isinstance(report, Mapping):
            raise ValueError("ledger_report_entries_invalid")
        if report.get("contract_status") != "PASS" or report.get("expected_interval_seconds") != expected_interval_seconds:
            raise ValueError(f"ledger_{key}_not_valid_or_interval_mismatch")
        if report.get("strategy_id") != key:
            raise ValueError(f"ledger_{key}_strategy_identity_mismatch")
        periods = report.get("periods")
        if not isinstance(periods, list) or not periods or any(not isinstance(row, Mapping) for row in periods):
            raise ValueError(f"ledger_{key}_period_rows_missing")
        claimed_ledger_sha = report.get("ledger_sha256")
        if not _sha(claimed_ledger_sha):
            raise ValueError(f"ledger_{key}_digest_invalid")
        canonical_ledger = json.dumps(periods, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        computed_ledger_sha = hashlib.sha256(canonical_ledger.encode("utf-8")).hexdigest()
        if computed_ledger_sha != claimed_ledger_sha.lower():
            raise ValueError(f"ledger_{key}_content_sha256_mismatch")
        times_from_periods = [row.get("timestamp") for row in periods]
        sessions_from_periods = [row.get("session_id") for row in periods]
        returns_from_periods = [row.get("net_return") for row in periods]
        times, sessions, values = report.get("timestamps"), report.get("session_ids"), report.get("net_returns")
        if not isinstance(times, list) or not isinstance(sessions, list) or not isinstance(values, list) or not times:
            raise ValueError(f"ledger_{key}_series_missing")
        if report.get("row_count") != len(periods):
            raise ValueError(f"ledger_{key}_row_count_mismatch")
        if times != times_from_periods or sessions != sessions_from_periods or values != returns_from_periods:
            raise ValueError(f"ledger_{key}_series_projection_mismatch")
        if not (len(times) == len(sessions) == len(values)):
            raise ValueError(f"ledger_{key}_series_length_mismatch")
        for row_index, period in enumerate(periods):
            equity_before = _dec(period.get("equity_before"), f"ledger_{key}_period_{row_index}_equity_before")
            net_pnl = _dec(period.get("net_pnl_quote_side_after_costs"), f"ledger_{key}_period_{row_index}_net_pnl")
            net_return = _dec(period.get("net_return"), f"ledger_{key}_period_{row_index}_net_return")
            expected_return = net_pnl / equity_before
            if abs(expected_return - net_return) > Decimal("1e-12"):
                raise ValueError(f"ledger_{key}_period_return_arithmetic_mismatch")
        if base_times is None:
            base_times, base_sessions = list(times), list(sessions)
        elif times != base_times or sessions != base_sessions:
            raise ValueError("strategy_ledgers_not_synchronized_no_rows_may_be_dropped")
        if key in strategy_returns:
            raise ValueError("duplicate_strategy_label")
        strategy_returns[key] = [float(_dec(v, f"ledger_{key}_return")) for v in values]
        eligible = eligible and report.get("selection_guard_input_eligible") is True
        digests[key] = claimed_ledger_sha.lower()
    benchmark = [float(_dec(x, "benchmark_return")) for x in benchmark_returns]
    if base_times is None or base_sessions is None:
        raise ValueError("no_ledger_series")
    if len(benchmark) != len(base_times):
        raise ValueError("benchmark_length_mismatch_no_rows_may_be_dropped")
    manifest = {
        "schema_version": "synchronized-portfolio-return-panel-v1.0.0",
        "expected_interval_seconds": expected_interval_seconds,
        "timestamps": base_times,
        "session_ids": base_sessions,
        "strategy_ledger_sha256": digests,
        "strategy_returns": strategy_returns,
        "benchmark_returns": benchmark,
    }
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
    return {
        **manifest,
        "panel_sha256": digest,
        "row_count": len(base_times),
        "strategy_count": len(strategy_returns),
        "time_alignment_status": "PASS",
        "selection_guard_input_eligible": eligible,
        "panel_status": "EVIDENCE_ELIGIBLE" if eligible else "DIAGNOSTIC_ONLY",
        "decision_output": DECISION_OUTPUT,
        "live_signal_enabled": LIVE_SIGNAL_ENABLED,
        "live_order_execution_enabled": LIVE_ORDER_EXECUTION_ENABLED,
        "limitations": [
            "All strategies must use the exact same clock; unmatched periods are never dropped.",
            "Panel structure does not qualify the source, costs, holdout, portfolio replay or full trial registry.",
            "Inference must use the complete registered strategy family and a locked chronological holdout.",
        ],
    }
