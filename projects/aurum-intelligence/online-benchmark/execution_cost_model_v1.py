"""Auditable per-trade quote-side PnL and cost accounting for research replay.

Pure and offline: this is not an order router, broker adapter, market-impact model, or
certification of inputs. Unknown material costs fail closed for net-PnL output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping
from urllib.parse import urlparse

EVIDENCE_STATES = frozenset({"observed", "assumed", "unknown"})


def _decimal(value: object, name: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name}_must_be_finite_number")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"{name}_must_be_finite_number") from exc
    if not number.is_finite():
        raise ValueError(f"{name}_must_be_finite_number")
    return number


def _timestamp(value: object, name: str) -> datetime:
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


def _valid_evidence_ref(value: object) -> bool:
    if not isinstance(value, str) or not value.strip() or len(value) > 2048:
        return False
    raw = value.strip()
    if raw.startswith("https://"):
        parsed = urlparse(raw)
        return bool(parsed.hostname) and parsed.username is None and parsed.password is None
    if raw.startswith(("evidence://", "repo://")):
        _, identifier = raw.split("://", 1)
        return bool(identifier.strip("/")) and not any(ch.isspace() for ch in identifier)
    return False


@dataclass(frozen=True)
class Evidence:
    """Evidence declaration; a reference string is not independently verified."""
    state: str = "unknown"
    reference: str | None = None

    def __post_init__(self) -> None:
        if self.state not in EVIDENCE_STATES:
            raise ValueError("evidence_state_must_be_observed_assumed_or_unknown")
        if self.reference is not None and not _valid_evidence_ref(self.reference):
            raise ValueError("evidence_reference_must_be_https_or_stable_evidence_uri")

    @property
    def has_reference(self) -> bool:
        return _valid_evidence_ref(self.reference)


@dataclass(frozen=True)
class QuoteSnapshot:
    timestamp: datetime | str
    bid: Decimal | int | float | str
    ask: Decimal | int | float | str
    source_id: str
    evidence: Evidence = field(default_factory=Evidence)

    def normalized(self, name: str) -> dict[str, Any]:
        stamp = _timestamp(self.timestamp, f"{name}_timestamp")
        bid = _decimal(self.bid, f"{name}_bid")
        ask = _decimal(self.ask, f"{name}_ask")
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError(f"{name}_source_id_required")
        if bid <= 0 or ask <= 0:
            raise ValueError(f"{name}_prices_must_be_positive")
        if ask < bid:
            raise ValueError(f"{name}_crossed_quote")
        if not isinstance(self.evidence, Evidence):
            raise ValueError(f"{name}_evidence_must_use_Evidence_contract")
        return {"timestamp": stamp, "bid": bid, "ask": ask, "mid": (bid + ask) / 2,
                "spread": ask - bid, "source_id": self.source_id, "evidence": self.evidence}


@dataclass(frozen=True)
class ExecutionProfile:
    """Versioned cost assumptions in instrument price/currency units.

    Contract units multiply lots. Commission is per lot per side. Slippage is adverse
    price units per fill. Swap is signed cash PnL per lot per rollover. Currency
    conversion fails closed until independently implemented and validated.
    """
    profile_id: str
    instrument_id: str
    quote_currency: str
    account_currency: str
    contract_units_per_lot: Decimal | int | float | str
    commission_per_lot_per_side: Decimal | int | float | str
    entry_slippage_price_units: Decimal | int | float | str
    exit_slippage_price_units: Decimal | int | float | str
    commission_evidence: Evidence = field(default_factory=Evidence)
    slippage_evidence: Evidence = field(default_factory=Evidence)
    swap_evidence: Evidence = field(default_factory=Evidence)
    swap_long_per_lot_per_rollover: Decimal | int | float | str = "0"
    swap_short_per_lot_per_rollover: Decimal | int | float | str = "0"
    broker_authorized: bool = False
    broker_authorization_evidence_ref: str | None = None

    def normalized(self) -> dict[str, Any]:
        if not isinstance(self.profile_id, str) or not self.profile_id.strip():
            raise ValueError("profile_id_required")
        if not isinstance(self.instrument_id, str) or not self.instrument_id.strip():
            raise ValueError("instrument_id_required")
        quote_currency = self.quote_currency.strip().upper()
        account_currency = self.account_currency.strip().upper()
        if not quote_currency or not account_currency:
            raise ValueError("quote_and_account_currency_required")
        if quote_currency != account_currency:
            raise ValueError("currency_conversion_not_implemented_fail_closed")
        units = _decimal(self.contract_units_per_lot, "contract_units_per_lot")
        commission = _decimal(self.commission_per_lot_per_side, "commission_per_lot_per_side")
        entry_slip = _decimal(self.entry_slippage_price_units, "entry_slippage_price_units")
        exit_slip = _decimal(self.exit_slippage_price_units, "exit_slippage_price_units")
        swap_long = _decimal(self.swap_long_per_lot_per_rollover, "swap_long_per_lot_per_rollover")
        swap_short = _decimal(self.swap_short_per_lot_per_rollover, "swap_short_per_lot_per_rollover")
        if units <= 0:
            raise ValueError("contract_units_per_lot_must_be_positive")
        if commission < 0 or entry_slip < 0 or exit_slip < 0:
            raise ValueError("commission_and_slippage_assumptions_must_be_nonnegative")
        if not isinstance(self.broker_authorized, bool):
            raise ValueError("broker_authorized_must_be_boolean")
        if self.broker_authorization_evidence_ref is not None and not _valid_evidence_ref(self.broker_authorization_evidence_ref):
            raise ValueError("broker_authorization_reference_invalid")
        for field_name, evidence in (
            ("commission_evidence", self.commission_evidence),
            ("slippage_evidence", self.slippage_evidence),
            ("swap_evidence", self.swap_evidence),
        ):
            if not isinstance(evidence, Evidence):
                raise ValueError(f"{field_name}_must_use_Evidence_contract")
        return {
            "profile_id": self.profile_id.strip(), "instrument_id": self.instrument_id.strip(),
            "quote_currency": quote_currency, "account_currency": account_currency,
            "contract_units_per_lot": units, "commission_per_lot_per_side": commission,
            "entry_slippage_price_units": entry_slip, "exit_slippage_price_units": exit_slip,
            "swap_long_per_lot_per_rollover": swap_long, "swap_short_per_lot_per_rollover": swap_short,
        }


def calculate_closed_trade(
    *,
    side: str,
    lots: Decimal | int | float | str,
    entry_quote: QuoteSnapshot,
    exit_quote: QuoteSnapshot,
    profile: ExecutionProfile,
    rollovers: int = 0,
    initial_risk_cash: Decimal | int | float | str | None = None,
) -> dict[str, Any]:
    """Calculate a closed trade without double-counting Bid/Ask spread.

    Long fills use Ask then Bid; short fills use Bid then Ask. Spread is itemized for
    audit but not deducted twice. An unknown material cost prevents net-PnL output.
    Assumption-based PnL is explicitly an estimate, never broker-validation.
    """
    risk_cash = None if initial_risk_cash is None else _decimal(initial_risk_cash, "initial_risk_cash")
    if risk_cash is not None and risk_cash <= 0:
        raise ValueError("initial_risk_cash_must_be_positive")
    normalized_profile = profile.normalized()
    normalized_entry = entry_quote.normalized("entry")
    normalized_exit = exit_quote.normalized("exit")
    direction = {"long": 1, "short": -1}.get(side)
    if direction is None:
        raise ValueError("side_must_be_long_or_short")
    quantity_lots = _decimal(lots, "lots")
    if quantity_lots <= 0:
        raise ValueError("lots_must_be_positive")
    if isinstance(rollovers, bool) or not isinstance(rollovers, int) or rollovers < 0:
        raise ValueError("rollovers_must_be_nonnegative_integer")
    if normalized_exit["timestamp"] <= normalized_entry["timestamp"]:
        raise ValueError("exit_timestamp_must_follow_entry_timestamp")

    components = {
        "commission": profile.commission_evidence,
        "slippage": profile.slippage_evidence,
    }
    if rollovers > 0:
        components["swap"] = profile.swap_evidence
    unknown_costs = sorted(name for name, item in components.items() if item.state == "unknown")
    assumption_costs = sorted(name for name, item in components.items() if item.state == "assumed")
    unreferenced_observed_costs = sorted(name for name, item in components.items()
                                         if item.state == "observed" and not item.has_reference)
    quote_evidence_complete = all(
        item["evidence"].state == "observed" and item["evidence"].has_reference
        for item in (normalized_entry, normalized_exit)
    )
    broker_auth_complete = profile.broker_authorized and _valid_evidence_ref(profile.broker_authorization_evidence_ref)

    units = normalized_profile["contract_units_per_lot"] * quantity_lots
    entry = normalized_entry["ask"] if direction == 1 else normalized_entry["bid"]
    exit_ = normalized_exit["bid"] if direction == 1 else normalized_exit["ask"]
    entry_mid, exit_mid = normalized_entry["mid"], normalized_exit["mid"]
    entry_slip = normalized_profile["entry_slippage_price_units"]
    exit_slip = normalized_profile["exit_slippage_price_units"]
    mid_move_pnl = Decimal(direction) * (exit_mid - entry_mid) * units
    spread_drag = (normalized_entry["spread"] + normalized_exit["spread"]) * units / 2
    quote_side_gross = Decimal(direction) * (exit_ - entry) * units
    slippage_drag = (entry_slip + exit_slip) * units
    commission = normalized_profile["commission_per_lot_per_side"] * quantity_lots * 2
    swap_per_lot = (normalized_profile["swap_long_per_lot_per_rollover"] if direction == 1
                    else normalized_profile["swap_short_per_lot_per_rollover"])
    swap_pnl = swap_per_lot * quantity_lots * rollovers

    if mid_move_pnl - spread_drag != quote_side_gross:
        raise ArithmeticError("quote_side_pnl_decomposition_invariant_failed")

    cost_components = {
        "commission": {"amount": str(commission), "unit": normalized_profile["account_currency"],
                       "evidence_state": profile.commission_evidence.state,
                       "evidence_ref_present": profile.commission_evidence.has_reference},
        "spread_drag": {"amount": str(spread_drag), "unit": normalized_profile["quote_currency"],
                        "measurement": "derived_from_entry_exit_bid_ask_quotes"},
        "slippage": {"amount": str(slippage_drag), "unit": normalized_profile["quote_currency"],
                     "entry_price_units": str(entry_slip), "exit_price_units": str(exit_slip),
                     "evidence_state": profile.slippage_evidence.state,
                     "evidence_ref_present": profile.slippage_evidence.has_reference},
        "swap_pnl": {"amount": str(swap_pnl), "unit": normalized_profile["account_currency"],
                     "rollovers": rollovers,
                     "evidence_state": profile.swap_evidence.state if rollovers else "not_applicable",
                     "evidence_ref_present": profile.swap_evidence.has_reference if rollovers else None},
    }
    base = {
        "schema_version": "1.0.0",
        "profile_id": normalized_profile["profile_id"],
        "instrument_id": normalized_profile["instrument_id"],
        "side": side,
        "lots": str(quantity_lots),
        "quantity_units": str(units),
        "quote_currency": normalized_profile["quote_currency"],
        "account_currency": normalized_profile["account_currency"],
        "entry": {"timestamp_utc": normalized_entry["timestamp"].isoformat(),
                  "source_id": normalized_entry["source_id"], "bid": str(normalized_entry["bid"]),
                  "ask": str(normalized_entry["ask"]), "mid": str(entry_mid),
                  "fill_quote_side_before_slippage": str(entry),
                  "evidence_state": normalized_entry["evidence"].state,
                  "evidence_ref_present": normalized_entry["evidence"].has_reference},
        "exit": {"timestamp_utc": normalized_exit["timestamp"].isoformat(),
                 "source_id": normalized_exit["source_id"], "bid": str(normalized_exit["bid"]),
                 "ask": str(normalized_exit["ask"]), "mid": str(exit_mid),
                 "fill_quote_side_before_slippage": str(exit_),
                 "evidence_state": normalized_exit["evidence"].state,
                 "evidence_ref_present": normalized_exit["evidence"].has_reference},
        "pnl_decomposition": {
            "gross_mid_move_pnl": str(mid_move_pnl),
            "spread_drag_included_in_quote_side_pnl": str(spread_drag),
            "quote_side_gross_pnl_before_slippage_and_fees": str(quote_side_gross),
            "slippage_drag": str(slippage_drag),
            "commission_cost": str(commission),
            "swap_pnl": str(swap_pnl),
        },
        "cost_components": cost_components,
        "unknown_material_costs": unknown_costs,
        "assumption_costs": assumption_costs,
        "observed_costs_missing_evidence_reference": unreferenced_observed_costs,
        "quote_evidence_complete": quote_evidence_complete,
        "broker_authorization_evidence_complete": broker_auth_complete,
        "live_execution_authorized": False,
        "execution_evidence_status": "BLOCKED_UNKNOWN_COSTS" if unknown_costs else (
            "ASSUMPTION_ONLY" if assumption_costs else (
                "ELIGIBLE_FOR_MANUAL_REVIEW" if quote_evidence_complete and broker_auth_complete
                and not unreferenced_observed_costs else "EVIDENCE_INCOMPLETE"
            )
        ),
        "interpretation": "Accounting estimate only; evidence references require independent review.",
    }
    if unknown_costs:
        base["gross_quote_side_pnl"] = str(quote_side_gross)
        base["net_pnl"] = None
        base["net_r"] = None
        base["pnl_status"] = "NET_PNL_NOT_COMPUTABLE_UNKNOWN_COSTS"
        return base

    net_pnl = quote_side_gross - slippage_drag - commission + swap_pnl
    base["gross_quote_side_pnl"] = str(quote_side_gross)
    base["net_pnl"] = str(net_pnl)
    base["pnl_status"] = "ESTIMATE_ONLY" if assumption_costs or not quote_evidence_complete else "ACCOUNTING_RESULT_REQUIRES_EXTERNAL_VALIDATION"
    if risk_cash is None:
        base["initial_risk_cash"] = None
        base["net_r"] = None
    else:
        base["initial_risk_cash"] = str(risk_cash)
        base["net_r"] = str(net_pnl / risk_cash)
    return base
