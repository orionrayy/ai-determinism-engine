#!/usr/bin/env python3
"""Venue-neutral SOL Sentinel adapter contracts and deterministic paper execution.

This module deliberately contains no exchange order client, wallet signer, transfer
sender, credential loader, or live execution implementation. All amounts are Decimal.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class AdapterContractError(ValueError):
    """A platform contract or safety precondition failed."""


def _decimal(value: Any, field: str, *, allow_zero: bool = False) -> Decimal:
    if isinstance(value, bool):
        raise AdapterContractError(f"{field} must be a finite decimal")
    try:
        out = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        raise AdapterContractError(f"{field} must be a finite decimal") from None
    if not out.is_finite() or (out < 0 if allow_zero else out <= 0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise AdapterContractError(f"{field} must be finite and {qualifier}")
    return out


def _aware_utc(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise AdapterContractError(f"{field} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _is_multiple(value: Decimal, step: Decimal) -> bool:
    return value % step == 0


@dataclass(frozen=True)
class MarketSpec:
    """Canonical market contract; venue-specific ticker symbols stay separate."""

    venue: str
    symbol: str
    venue_symbol: str
    base_asset: str
    quote_asset: str
    price_tick: Decimal
    quantity_step: Decimal
    min_notional_quote: Decimal
    base_decimals: int
    quote_decimals: int
    min_quantity_base: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        for field in ("venue", "symbol", "venue_symbol", "base_asset", "quote_asset"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise AdapterContractError(f"{field} must be a non-empty string")
            object.__setattr__(self, field, value.strip())
        if self.base_asset.upper() == self.quote_asset.upper():
            raise AdapterContractError("base_asset and quote_asset must differ")
        object.__setattr__(self, "base_asset", self.base_asset.upper())
        object.__setattr__(self, "quote_asset", self.quote_asset.upper())
        object.__setattr__(self, "price_tick", _decimal(self.price_tick, "price_tick"))
        object.__setattr__(self, "quantity_step", _decimal(self.quantity_step, "quantity_step"))
        object.__setattr__(self, "min_notional_quote", _decimal(self.min_notional_quote, "min_notional_quote", allow_zero=True))
        object.__setattr__(self, "min_quantity_base", _decimal(self.min_quantity_base, "min_quantity_base", allow_zero=True))
        for field in ("base_decimals", "quote_decimals"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 18:
                raise AdapterContractError(f"{field} must be an integer in 0..18")


@dataclass(frozen=True)
class MarketQuote:
    venue: str
    symbol: str
    bid: Decimal
    ask: Decimal
    observed_at: datetime
    source: str = "mock"

    def __post_init__(self) -> None:
        object.__setattr__(self, "bid", _decimal(self.bid, "bid"))
        object.__setattr__(self, "ask", _decimal(self.ask, "ask"))
        object.__setattr__(self, "observed_at", _aware_utc(self.observed_at, "observed_at"))
        if self.ask < self.bid:
            raise AdapterContractError("crossed quote (ask < bid)")
        for field in ("venue", "symbol", "source"):
            if not isinstance(getattr(self, field), str) or not getattr(self, field).strip():
                raise AdapterContractError(f"{field} must be a non-empty string")

    @property
    def midpoint(self) -> Decimal:
        return (self.bid + self.ask) / Decimal("2")

    @property
    def spread_bps(self) -> Decimal:
        return (self.ask - self.bid) / self.midpoint * Decimal("10000")


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class OrderIntent:
    idempotency_key: str
    venue: str
    symbol: str
    side: Side
    quantity_base: Decimal
    limit_price_quote: Decimal
    created_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.idempotency_key, str) or not self.idempotency_key.strip():
            raise AdapterContractError("idempotency_key must be a non-empty string")
        if len(self.idempotency_key) > 200:
            raise AdapterContractError("idempotency_key is too long")
        if not isinstance(self.side, Side):
            try:
                object.__setattr__(self, "side", Side(str(self.side).upper()))
            except ValueError:
                raise AdapterContractError("side must be BUY or SELL") from None
        object.__setattr__(self, "quantity_base", _decimal(self.quantity_base, "quantity_base"))
        object.__setattr__(self, "limit_price_quote", _decimal(self.limit_price_quote, "limit_price_quote"))
        object.__setattr__(self, "created_at", _aware_utc(self.created_at, "created_at"))
        for field in ("venue", "symbol"):
            if not isinstance(getattr(self, field), str) or not getattr(self, field).strip():
                raise AdapterContractError(f"{field} must be a non-empty string")

    @property
    def notional_quote(self) -> Decimal:
        return self.quantity_base * self.limit_price_quote

    def fingerprint(self) -> str:
        canonical = {
            "venue": self.venue,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity_base": str(self.quantity_base),
            "limit_price_quote": str(self.limit_price_quote),
            "created_at": self.created_at.isoformat(),
        }
        return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class OrderCheck:
    accepted: bool
    reasons: tuple[str, ...]
    reference_price: Decimal | None
    quote_age_seconds: Decimal | None
    spread_bps: Decimal | None


def validate_order(
    intent: OrderIntent,
    market: MarketSpec,
    quote: MarketQuote,
    *,
    now: datetime | None = None,
    max_quote_age_seconds: Decimal = Decimal("30"),
    max_spread_bps: Decimal = Decimal("100"),
    future_clock_skew_seconds: Decimal = Decimal("5"),
) -> OrderCheck:
    """Validate venue identity, lot/tick/minimum, and quote integrity before paper fill."""
    current = _aware_utc(now or datetime.now(timezone.utc), "now")
    reasons: list[str] = []
    if intent.venue != market.venue or intent.symbol != market.symbol:
        reasons.append("market_identity_mismatch")
    if quote.venue != market.venue or quote.symbol != market.symbol:
        reasons.append("quote_identity_mismatch")
    if not _is_multiple(intent.quantity_base, market.quantity_step):
        reasons.append("quantity_step_mismatch")
    if not _is_multiple(intent.limit_price_quote, market.price_tick):
        reasons.append("price_tick_mismatch")
    if intent.notional_quote < market.min_notional_quote:
        reasons.append("minimum_notional_not_met")
    if intent.quantity_base < market.min_quantity_base:
        reasons.append("minimum_base_quantity_not_met")

    age = Decimal(str((current - quote.observed_at).total_seconds()))
    spread = quote.spread_bps
    if age < -_decimal(future_clock_skew_seconds, "future_clock_skew_seconds", allow_zero=True):
        reasons.append("quote_from_future")
    elif age > _decimal(max_quote_age_seconds, "max_quote_age_seconds", allow_zero=True):
        reasons.append("quote_stale")
    if spread > _decimal(max_spread_bps, "max_spread_bps", allow_zero=True):
        reasons.append("spread_above_limit")

    reference = quote.ask if intent.side is Side.BUY else quote.bid
    return OrderCheck(not reasons, tuple(reasons), reference, age, spread)


class ReceiptStatus(str, Enum):
    FILLED_SIMULATED = "FILLED_SIMULATED"
    NO_FILL = "NO_FILL"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class ExecutionReceipt:
    order_id: str
    idempotency_key: str
    status: ReceiptStatus
    venue: str
    symbol: str
    side: Side
    requested_quantity_base: Decimal
    filled_quantity_base: Decimal
    fill_price_quote: Decimal | None
    fee_quote: Decimal
    quote_observed_at: datetime
    reasons: tuple[str, ...]
    execution_enabled: bool = False
    live_order_submitted: bool = False


@runtime_checkable
class MarketDataAdapter(Protocol):
    def get_quote(self, market: MarketSpec) -> MarketQuote: ...


@runtime_checkable
class CapitalSourceAdapter(Protocol):
    def source_id(self) -> str: ...


@runtime_checkable
class CustodyAdapter(Protocol):
    def get_balance(self, asset: str, network: str) -> Decimal: ...


@runtime_checkable
class TransferPlanner(Protocol):
    def plan(self, route: "TransferRoute", amount: Decimal) -> "TransferDecision": ...


@runtime_checkable
class TransactionSigner(Protocol):
    def sign(self, unsigned_transaction: bytes) -> bytes: ...


@runtime_checkable
class ChainTransactionSubmitter(Protocol):
    def submit(self, signed_transaction: bytes) -> str: ...


@runtime_checkable
class ExecutionAdapter(Protocol):
    def execute(self, intent: OrderIntent, market: MarketSpec, quote: MarketQuote, **kwargs: Any) -> ExecutionReceipt: ...


@runtime_checkable
class LedgerAdapter(Protocol):
    def record(self, receipt: ExecutionReceipt) -> None: ...


@dataclass(frozen=True)
class TransferRoute:
    source: str
    destination: str
    asset: str
    source_network: str
    destination_network: str
    minimum_withdrawal: Decimal | None
    withdrawal_fee: Decimal | None
    minimum_deposit: Decimal | None
    source_network_verified: bool = False
    destination_network_verified: bool = False
    cost_verified_at: datetime | None = None

    def __post_init__(self) -> None:
        for field in ("source", "destination", "asset", "source_network", "destination_network"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise AdapterContractError(f"{field} must be a non-empty string")
            object.__setattr__(self, field, value.strip())
        for field in ("minimum_withdrawal", "withdrawal_fee", "minimum_deposit"):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, _decimal(value, field, allow_zero=True))
        if self.cost_verified_at is not None:
            object.__setattr__(self, "cost_verified_at", _aware_utc(self.cost_verified_at, "cost_verified_at"))


@dataclass(frozen=True)
class TransferDecision:
    approved_for_planning: bool
    reasons: tuple[str, ...]
    amount_requested: Decimal
    estimated_fee: Decimal | None
    estimated_amount_received: Decimal | None
    transfer_sent: bool = False


def assess_transfer_route(
    route: TransferRoute,
    amount: Decimal,
    *,
    now: datetime | None = None,
    max_cost_age_seconds: Decimal = Decimal("86400"),
) -> TransferDecision:
    """Validate metadata only. This function can never send or sign a transfer."""
    requested = _decimal(amount, "amount")
    current = _aware_utc(now or datetime.now(timezone.utc), "now")
    reasons: list[str] = []
    if route.source.lower() == route.destination.lower():
        reasons.append("source_and_destination_match")
    if route.source_network.upper() != route.destination_network.upper():
        reasons.append("network_mismatch")
    if not route.source_network_verified:
        reasons.append("source_network_not_verified")
    if not route.destination_network_verified:
        reasons.append("destination_network_not_verified")
    if route.minimum_withdrawal is None:
        reasons.append("minimum_withdrawal_unknown")
    elif requested < route.minimum_withdrawal:
        reasons.append("below_minimum_withdrawal")
    if route.withdrawal_fee is None:
        reasons.append("withdrawal_fee_unknown")
    if route.minimum_deposit is None:
        reasons.append("minimum_deposit_unknown")
    if route.cost_verified_at is None:
        reasons.append("route_cost_verification_missing")
    else:
        cost_age = Decimal(str((current - route.cost_verified_at).total_seconds()))
        max_age = _decimal(max_cost_age_seconds, "max_cost_age_seconds", allow_zero=True)
        if cost_age < Decimal("-5") or cost_age > max_age:
            reasons.append("route_cost_verification_stale")
    received: Decimal | None = None
    if route.withdrawal_fee is not None:
        if route.withdrawal_fee > requested:
            reasons.append("withdrawal_fee_exceeds_amount")
        else:
            received = requested - route.withdrawal_fee
            if route.minimum_deposit is not None and received < route.minimum_deposit:
                reasons.append("net_amount_below_minimum_deposit")
    return TransferDecision(
        approved_for_planning=not reasons,
        reasons=tuple(reasons),
        amount_requested=requested,
        estimated_fee=route.withdrawal_fee,
        estimated_amount_received=received,
        transfer_sent=False,
    )


class ExecutionMode(str, Enum):
    READ_ONLY = "READ_ONLY"
    PAPER = "PAPER"
    LIVE = "LIVE"


@dataclass(frozen=True)
class ExecutionPolicy:
    mode: ExecutionMode = ExecutionMode.PAPER
    explicit_live_approval: bool = False
    readiness_gates_passed: bool = False
    approval_reference: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.mode, ExecutionMode):
            try:
                object.__setattr__(self, "mode", ExecutionMode(str(self.mode).upper()))
            except ValueError:
                raise AdapterContractError("mode must be READ_ONLY, PAPER, or LIVE") from None

    def validate(self) -> None:
        if self.mode is not ExecutionMode.LIVE:
            return
        if not self.explicit_live_approval:
            raise AdapterContractError("live mode requires separate explicit user approval")
        if not self.readiness_gates_passed:
            raise AdapterContractError("live mode is blocked until every readiness gate passes")
        if not self.approval_reference.strip():
            raise AdapterContractError("live mode requires a traceable approval reference")
        raise AdapterContractError("no live execution adapter is implemented in SOL Sentinel")



class VenueDataError(RuntimeError):
    """Public venue market data could not be safely retrieved or parsed."""


class IndodaxPublicMarketDataAdapter:
    """Read-only INDODAX public depth adapter; never accesses balances or places orders.

    Official public REST docs list https://indodax.com/api/depth/$pair_id and
    specify that public endpoints need no API key. The caller must provide the
    pair id from the current /api/pairs catalogue, e.g. "solidr".
    """

    BASE_URL = "https://indodax.com/api/depth/"
    PAIRS_URL = "https://indodax.com/api/pairs"
    MAX_RESPONSE_BYTES = 1_000_000
    RETRYABLE_HTTP_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})

    def __init__(self, *, timeout: float = 5.0, attempts: int = 2, pause_seconds: float = 0.25, transport: Any = None) -> None:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0.1 <= timeout <= 20:
            raise AdapterContractError("timeout must be between 0.1 and 20 seconds")
        if isinstance(attempts, bool) or not isinstance(attempts, int) or not 1 <= attempts <= 3:
            raise AdapterContractError("attempts must be an integer in 1..3")
        if isinstance(pause_seconds, bool) or not isinstance(pause_seconds, (int, float)) or not 0 <= pause_seconds <= 2:
            raise AdapterContractError("pause_seconds must be between 0 and 2 seconds")
        self._timeout = float(timeout)
        self._attempts = attempts
        self._pause_seconds = float(pause_seconds)
        self._transport = transport or self._http_get_json

    @classmethod
    def _http_get_json(cls, url: str, timeout: float) -> Any:
        request = urllib.request.Request(
            url,
            headers={"Accept": "application/json", "User-Agent": "SOL-Sentinel-ReadOnly/1.0"},
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise VenueDataError(f"INDODAX public API HTTP {response.status}")
            raw = response.read(cls.MAX_RESPONSE_BYTES + 1)
        if len(raw) > cls.MAX_RESPONSE_BYTES:
            raise VenueDataError("INDODAX public API response exceeded 1 MB safety limit")
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise VenueDataError("INDODAX public API returned invalid JSON") from None

    def _get_public_json(self, url: str) -> Any:
        last_error: Exception | None = None
        for attempt in range(self._attempts):
            try:
                return self._transport(url, self._timeout)
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in self.RETRYABLE_HTTP_CODES:
                    raise VenueDataError(f"INDODAX public API HTTP {exc.code}") from None
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
            except VenueDataError:
                raise
            if attempt + 1 < self._attempts:
                time.sleep(min(2.0, self._pause_seconds * (2 ** attempt)))
        raise VenueDataError(
            f"INDODAX public API request failed after {self._attempts} attempt(s): "
            f"{type(last_error).__name__ if last_error else 'unknown error'}"
        ) from None

    def get_market_spec(self, symbol: str = "SOL/IDR") -> MarketSpec:
        """Load the exact current pair increments from the public /api/pairs catalogue."""
        if not isinstance(symbol, str) or not symbol.strip():
            raise AdapterContractError("symbol must be a non-empty canonical pair such as SOL/IDR")
        canonical = symbol.replace("/", "").replace("-", "").strip().upper()
        if not canonical.isalnum() or len(canonical) > 30:
            raise AdapterContractError("symbol must contain only asset codes and an optional separator")
        payload = self._get_public_json(self.PAIRS_URL)
        if not isinstance(payload, list):
            raise VenueDataError("INDODAX public /api/pairs response must be an array")
        matches = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            pair_symbol = str(item.get("symbol", "")).replace("/", "").upper()
            pair_id = str(item.get("id", "")).lower()
            description = str(item.get("description", "")).replace("/", "").upper()
            if canonical in (pair_symbol, description) or canonical.lower() == pair_id:
                matches.append(item)
        if len(matches) != 1:
            raise VenueDataError(f"INDODAX pair catalogue matched {len(matches)} rows for {canonical}; refusing ambiguity")
        pair = matches[0]
        base_asset = str(pair.get("traded_currency_unit") or pair.get("traded_currency") or "").upper()
        quote_asset = str(pair.get("base_currency") or "").upper()
        pair_id = str(pair.get("id") or "").lower()
        price_tick = pair.get("price_precision")
        if price_tick is None or str(price_tick).strip() in ("", "0", "0.0"):
            price_tick = pair.get("pricescale")
        quantity_step = pair.get("quantity_increment")
        min_notional = pair.get("trade_min_base_currency")
        min_quantity = pair.get("trade_min_traded_currency")
        if not all((base_asset, quote_asset, pair_id)) or any(
            value is None or str(value).strip() == "" for value in
            (price_tick, quantity_step, min_notional, min_quantity)
        ):
            raise VenueDataError("INDODAX pair metadata omitted a required tick, step, or order minimum")
        try:
            step_decimal = _decimal(quantity_step, "quantity_increment")
            base_decimals = max(0, -step_decimal.normalize().as_tuple().exponent)
            return MarketSpec(
                venue="INDODAX",
                symbol=f"{base_asset}/{quote_asset}",
                venue_symbol=pair_id,
                base_asset=base_asset,
                quote_asset=quote_asset,
                price_tick=_decimal(price_tick, "price_tick"),
                quantity_step=step_decimal,
                min_notional_quote=_decimal(min_notional, "trade_min_base_currency", allow_zero=True),
                base_decimals=base_decimals,
                quote_decimals=0 if quote_asset == "IDR" else 6,
                min_quantity_base=_decimal(min_quantity, "trade_min_traded_currency", allow_zero=True),
            )
        except AdapterContractError as exc:
            raise VenueDataError(f"INDODAX pair metadata invalid: {exc}") from None

    @staticmethod
    def _best_level(levels: Any, *, side: str) -> Decimal:
        if not isinstance(levels, list):
            raise VenueDataError(f"INDODAX depth {side} must be an array")
        prices: list[Decimal] = []
        for level in levels:
            if not isinstance(level, (list, tuple)) or len(level) < 2:
                continue
            try:
                price = _decimal(level[0], f"{side} price")
                quantity = _decimal(level[1], f"{side} quantity")
            except AdapterContractError:
                continue
            if price > 0 and quantity > 0:
                prices.append(price)
        if not prices:
            raise VenueDataError(f"INDODAX depth contains no valid {side} levels")
        return max(prices) if side == "buy" else min(prices)

    def get_quote(self, market: MarketSpec) -> MarketQuote:
        if market.venue.upper() != "INDODAX":
            raise AdapterContractError("Indodax adapter requires venue=INDODAX")
        if market.quote_asset.upper() != "IDR":
            raise AdapterContractError("This public depth adapter currently supports IDR-quoted pairs only")
        pair_id = market.venue_symbol.lower()
        if not pair_id or len(pair_id) > 40 or not pair_id.isalnum():
            raise AdapterContractError("venue_symbol must be the alphanumeric pair id returned by /api/pairs")
        payload = self._get_public_json(self.BASE_URL + pair_id)
        if not isinstance(payload, dict) or ("error" in payload and payload["error"]):
            raise VenueDataError("INDODAX depth response is not a valid order-book object")
        bid = self._best_level(payload.get("buy"), side="buy")
        ask = self._best_level(payload.get("sell"), side="sell")
        return MarketQuote(
            venue="INDODAX",
            symbol=market.symbol,
            bid=bid,
            ask=ask,
            observed_at=datetime.now(timezone.utc),
            source="INDODAX public /api/depth (client receive timestamp)",
        )


class StaticMockMarketDataAdapter:
    """Fixture-only read adapter. It never makes network requests."""

    def __init__(self, quotes: list[MarketQuote]) -> None:
        self._quotes = {(item.venue, item.symbol): item for item in quotes}

    def get_quote(self, market: MarketSpec) -> MarketQuote:
        try:
            return self._quotes[(market.venue, market.symbol)]
        except KeyError:
            raise AdapterContractError(f"no mock quote for {market.venue}:{market.symbol}") from None


class MockCapitalSourceAdapter:
    """Metadata-only source label; does not connect to an exchange account."""

    def __init__(self, identifier: str = "reku-onramp-reference") -> None:
        if not isinstance(identifier, str) or not identifier.strip():
            raise AdapterContractError("capital source identifier must be non-empty")
        self._identifier = identifier.strip()

    def source_id(self) -> str:
        return self._identifier


class MockCustodyAdapter:
    """Fixture-only balance provider; balances are not account or chain observations."""

    def __init__(self, balances: dict[tuple[str, str], Decimal]) -> None:
        self._balances = {
            (asset.upper(), network.upper()): _decimal(amount, "mock balance", allow_zero=True)
            for (asset, network), amount in balances.items()
        }

    def get_balance(self, asset: str, network: str) -> Decimal:
        key = (asset.upper(), network.upper())
        if key not in self._balances:
            raise AdapterContractError(f"mock balance not configured for {key[0]} on {key[1]}")
        return self._balances[key]


class PlanningOnlyTransferAdapter:
    """Transfer planner only. No withdrawal or blockchain-send method exists."""

    def plan(self, route: TransferRoute, amount: Decimal) -> TransferDecision:
        return assess_transfer_route(route, amount)


class InMemoryIdempotentLedger:
    """Test-only paper journal; duplicate identical receipts are idempotent."""

    def __init__(self) -> None:
        self._receipts: dict[str, str] = {}

    def record(self, receipt: ExecutionReceipt) -> None:
        canonical = json.dumps({
            "order_id": receipt.order_id,
            "idempotency_key": receipt.idempotency_key,
            "status": receipt.status.value,
            "venue": receipt.venue,
            "symbol": receipt.symbol,
            "side": receipt.side.value,
            "requested_quantity_base": str(receipt.requested_quantity_base),
            "filled_quantity_base": str(receipt.filled_quantity_base),
            "fill_price_quote": str(receipt.fill_price_quote) if receipt.fill_price_quote is not None else None,
            "fee_quote": str(receipt.fee_quote),
            "quote_observed_at": receipt.quote_observed_at.isoformat(),
            "reasons": list(receipt.reasons),
        }, sort_keys=True, separators=(",", ":"))
        fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
        previous = self._receipts.get(receipt.order_id)
        if previous is not None and previous != fingerprint:
            raise AdapterContractError("ledger idempotency conflict for existing order_id")
        self._receipts[receipt.order_id] = fingerprint

    @property
    def count(self) -> int:
        return len(self._receipts)


class PaperExecutionAdapter:
    """Deterministic quote-crossing simulator. It has no network or signing code."""

    def __init__(self, ledger: LedgerAdapter, *, fee_bps: Decimal = Decimal("0")) -> None:
        self._ledger = ledger
        self._fee_bps = _decimal(fee_bps, "fee_bps", allow_zero=True)
        self._seen: dict[str, tuple[str, ExecutionReceipt]] = {}

    def execute(
        self,
        intent: OrderIntent,
        market: MarketSpec,
        quote: MarketQuote,
        *,
        now: datetime | None = None,
        max_quote_age_seconds: Decimal = Decimal("30"),
        max_spread_bps: Decimal = Decimal("100"),
    ) -> ExecutionReceipt:
        fingerprint = intent.fingerprint()
        seen = self._seen.get(intent.idempotency_key)
        if seen is not None:
            if seen[0] != fingerprint:
                raise AdapterContractError("idempotency key reused for a different order intent")
            return seen[1]

        check = validate_order(
            intent, market, quote, now=now,
            max_quote_age_seconds=max_quote_age_seconds,
            max_spread_bps=max_spread_bps,
        )
        order_id = hashlib.sha256((intent.idempotency_key + ":" + fingerprint).encode()).hexdigest()[:24]
        status = ReceiptStatus.REJECTED
        reasons = list(check.reasons)
        filled = Decimal("0")
        fill_price: Decimal | None = None
        fee = Decimal("0")
        if check.accepted:
            marketable = (
                intent.limit_price_quote >= quote.ask
                if intent.side is Side.BUY
                else intent.limit_price_quote <= quote.bid
            )
            if marketable:
                status = ReceiptStatus.FILLED_SIMULATED
                filled = intent.quantity_base
                fill_price = check.reference_price
                fee = filled * fill_price * self._fee_bps / Decimal("10000")
            else:
                status = ReceiptStatus.NO_FILL
                reasons.append("limit_not_marketable_at_observed_quote")
        receipt = ExecutionReceipt(
            order_id=order_id,
            idempotency_key=intent.idempotency_key,
            status=status,
            venue=market.venue,
            symbol=market.symbol,
            side=intent.side,
            requested_quantity_base=intent.quantity_base,
            filled_quantity_base=filled,
            fill_price_quote=fill_price,
            fee_quote=fee,
            quote_observed_at=quote.observed_at,
            reasons=tuple(reasons),
        )
        self._ledger.record(receipt)
        self._seen[intent.idempotency_key] = (fingerprint, receipt)
        return receipt
