#!/usr/bin/env python3
"""Deterministic, SQLite-backed, paper-only multi-level depth execution.

This module deliberately has no exchange order, account, wallet, signing, or
transaction-submission interface. It consumes immutable *observed* book depth
inside a local simulation journal. A snapshot's displayed liquidity is shared
across all simulated orders and can be consumed only once.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_DOWN, ROUND_HALF_UP
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

SCHEMA_VERSION = 1
ENGINE_VERSION = "sol-sentinel-depth-paper-v1"
ACCOUNT_ID = "paper-default"


class ExecutionError(ValueError):
    """Invalid execution input or operation."""


class LedgerIntegrityError(RuntimeError):
    """Persisted order/fill/portfolio state failed reconciliation."""


def D(value: Any, field: str, *, allow_zero: bool = False) -> Decimal:
    if isinstance(value, bool):
        raise ExecutionError(f"{field} must be a finite decimal")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        raise ExecutionError(f"{field} must be a finite decimal") from None
    if not result.is_finite() or (result < 0 if allow_zero else result <= 0):
        raise ExecutionError(f"{field} must be finite and {'non-negative' if allow_zero else 'positive'}")
    return result


def aware(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ExecutionError(f"{field} must be timezone-aware")
    return value.astimezone(timezone.utc)


def iso(value: datetime) -> str:
    return aware(value, "timestamp").isoformat(timespec="microseconds")


def floor_step(value: Decimal, step: Decimal) -> Decimal:
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def canonical_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class TimeInForce(str, Enum):
    GTC = "GTC"  # Residual stays open until continued, cancelled or expired.
    IOC = "IOC"  # Unfilled residual is cancelled after this snapshot attempt.


class OrderState(str, Enum):
    CREATED = "CREATED"
    OPEN_SIMULATED = "OPEN_SIMULATED"
    PARTIALLY_FILLED_SIMULATED = "PARTIALLY_FILLED_SIMULATED"
    FILLED_SIMULATED = "FILLED_SIMULATED"
    CANCELLED_SIMULATED = "CANCELLED_SIMULATED"
    EXPIRED_SIMULATED = "EXPIRED_SIMULATED"
    NO_FILL = "NO_FILL"
    REJECTED = "REJECTED"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"


TERMINAL_STATES = {
    OrderState.FILLED_SIMULATED,
    OrderState.CANCELLED_SIMULATED,
    OrderState.EXPIRED_SIMULATED,
    OrderState.REJECTED,
}
TRANSITIONS = {
    OrderState.CREATED: {OrderState.OPEN_SIMULATED, OrderState.REJECTED},
    OrderState.OPEN_SIMULATED: {
        OrderState.OPEN_SIMULATED, OrderState.PARTIALLY_FILLED_SIMULATED,
        OrderState.FILLED_SIMULATED, OrderState.CANCELLED_SIMULATED,
        OrderState.EXPIRED_SIMULATED, OrderState.REJECTED, OrderState.RECOVERY_REQUIRED,
    },
    OrderState.PARTIALLY_FILLED_SIMULATED: {
        OrderState.PARTIALLY_FILLED_SIMULATED, OrderState.FILLED_SIMULATED,
        OrderState.CANCELLED_SIMULATED, OrderState.EXPIRED_SIMULATED,
        OrderState.RECOVERY_REQUIRED,
    },
    OrderState.RECOVERY_REQUIRED: {
        OrderState.OPEN_SIMULATED, OrderState.PARTIALLY_FILLED_SIMULATED,
        OrderState.FILLED_SIMULATED, OrderState.CANCELLED_SIMULATED,
        OrderState.EXPIRED_SIMULATED, OrderState.REJECTED,
    },
    OrderState.NO_FILL: {OrderState.OPEN_SIMULATED, OrderState.CANCELLED_SIMULATED, OrderState.EXPIRED_SIMULATED},
}


@dataclass(frozen=True)
class BookLevel:
    price: Decimal
    quantity: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(self, "price", D(self.price, "level.price"))
        object.__setattr__(self, "quantity", D(self.quantity, "level.quantity"))


@dataclass(frozen=True)
class DepthSnapshot:
    venue: str
    symbol: str
    observed_at: datetime
    bids: tuple[BookLevel, ...]
    asks: tuple[BookLevel, ...]
    exchange_timestamp_ms: int | None = None
    snapshot_id: str = ""

    def __post_init__(self) -> None:
        for name in ("venue", "symbol"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ExecutionError(f"{name} must be a non-empty string")
            object.__setattr__(self, name, value.strip().upper())
        object.__setattr__(self, "observed_at", aware(self.observed_at, "observed_at"))
        bids = tuple(x if isinstance(x, BookLevel) else BookLevel(*x) for x in self.bids)
        asks = tuple(x if isinstance(x, BookLevel) else BookLevel(*x) for x in self.asks)
        if not bids or not asks:
            raise ExecutionError("both bid and ask depth sides must be non-empty")
        if any(bids[i].price <= bids[i + 1].price for i in range(len(bids) - 1)):
            raise ExecutionError("bid levels must be strictly descending without duplicates")
        if any(asks[i].price >= asks[i + 1].price for i in range(len(asks) - 1)):
            raise ExecutionError("ask levels must be strictly ascending without duplicates")
        if bids[0].price > asks[0].price:
            raise ExecutionError("crossed order book")
        if self.exchange_timestamp_ms is not None:
            ts = self.exchange_timestamp_ms
            if isinstance(ts, bool) or not isinstance(ts, int) or ts <= 0:
                raise ExecutionError("exchange_timestamp_ms must be a positive integer")
        object.__setattr__(self, "bids", bids)
        object.__setattr__(self, "asks", asks)
        content = {
            "venue": self.venue,
            "symbol": self.symbol,
            "observed_at": iso(self.observed_at),
            "exchange_timestamp_ms": self.exchange_timestamp_ms,
            "bids": [[str(x.price), str(x.quantity)] for x in bids],
            "asks": [[str(x.price), str(x.quantity)] for x in asks],
        }
        digest = canonical_hash(content)
        if self.snapshot_id and self.snapshot_id != digest:
            raise ExecutionError("supplied snapshot_id does not match immutable snapshot content")
        object.__setattr__(self, "snapshot_id", digest)

    @property
    def midpoint(self) -> Decimal:
        """Top-of-book midpoint in the snapshot's quote currency."""
        return (self.bids[0].price + self.asks[0].price) / Decimal("2")

    @property
    def spread_bps(self) -> Decimal:
        """Quoted top-of-book spread in basis points."""
        midpoint = self.midpoint
        if midpoint <= 0:
            raise ExecutionError("snapshot midpoint must be positive")
        return (self.asks[0].price - self.bids[0].price) / midpoint * Decimal("10000")

    def liquidity_pool_id(self, side: str) -> str:
        """Identify an unchanged side book independently of observation timestamp and the opposite side."""
        normalized = str(side).strip().lower()
        if normalized not in ("bid", "ask"):
            raise ExecutionError("liquidity side must be bid or ask")
        levels = self.bids if normalized == "bid" else self.asks
        return canonical_hash({
            "venue": self.venue,
            "symbol": self.symbol,
            "side": normalized,
            "levels": [[str(x.price), str(x.quantity)] for x in levels],
        })

    @classmethod
    def from_indodax(cls, parsed_snapshot: Any) -> "DepthSnapshot":
        """Convert indodax_microstructure.Snapshot without importing its module."""
        try:
            return cls(
                venue="INDODAX", symbol="SOL/IDR", observed_at=parsed_snapshot.observed_at,
                bids=tuple(BookLevel(x.price, x.quantity) for x in parsed_snapshot.bids),
                asks=tuple(BookLevel(x.price, x.quantity) for x in parsed_snapshot.asks),
                exchange_timestamp_ms=parsed_snapshot.exchange_timestamp_ms,
            )
        except AttributeError as exc:
            raise ExecutionError("source object is not a compatible INDODAX depth snapshot") from exc


@dataclass(frozen=True)
class OrderRequest:
    client_order_id: str
    venue: str
    symbol: str
    side: Side
    quantity: Decimal
    order_type: OrderType = OrderType.MARKET
    limit_price: Decimal | None = None
    time_in_force: TimeInForce = TimeInForce.GTC
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None
    quantity_step: Decimal = Decimal("0.00000001")
    price_tick: Decimal = Decimal("1")
    fee_bps: Decimal = Decimal("0")
    fee_quantum: Decimal = Decimal("0.01")
    max_quote_age_seconds: Decimal = Decimal("30")

    def __post_init__(self) -> None:
        if not isinstance(self.client_order_id, str) or not self.client_order_id.strip() or len(self.client_order_id) > 200:
            raise ExecutionError("client_order_id must be a non-empty string of at most 200 characters")
        if not isinstance(self.side, Side): object.__setattr__(self, "side", Side(str(self.side).upper()))
        if not isinstance(self.order_type, OrderType): object.__setattr__(self, "order_type", OrderType(str(self.order_type).upper()))
        if not isinstance(self.time_in_force, TimeInForce): object.__setattr__(self, "time_in_force", TimeInForce(str(self.time_in_force).upper()))
        for name in ("venue", "symbol"):
            v = getattr(self, name)
            if not isinstance(v, str) or not v.strip(): raise ExecutionError(f"{name} must be non-empty")
            object.__setattr__(self, name, v.strip().upper())
        object.__setattr__(self, "quantity", D(self.quantity, "quantity"))
        object.__setattr__(self, "quantity_step", D(self.quantity_step, "quantity_step"))
        object.__setattr__(self, "price_tick", D(self.price_tick, "price_tick"))
        object.__setattr__(self, "fee_bps", D(self.fee_bps, "fee_bps", allow_zero=True))
        object.__setattr__(self, "fee_quantum", D(self.fee_quantum, "fee_quantum"))
        object.__setattr__(self, "max_quote_age_seconds", D(self.max_quote_age_seconds, "max_quote_age_seconds"))
        object.__setattr__(self, "created_at", aware(self.created_at, "created_at"))
        if self.expires_at is not None:
            object.__setattr__(self, "expires_at", aware(self.expires_at, "expires_at"))
            if self.expires_at <= self.created_at: raise ExecutionError("expires_at must be later than created_at")
        if self.quantity % self.quantity_step != 0: raise ExecutionError("quantity violates quantity_step")
        if self.order_type is OrderType.MARKET and self.limit_price is not None:
            raise ExecutionError("market order must not specify limit_price")
        if self.order_type is OrderType.LIMIT:
            if self.limit_price is None: raise ExecutionError("limit order requires limit_price")
            limit = D(self.limit_price, "limit_price")
            if limit % self.price_tick != 0: raise ExecutionError("limit_price violates price_tick")
            object.__setattr__(self, "limit_price", limit)

    def fingerprint(self) -> str:
        return canonical_hash({
            "client_order_id": self.client_order_id, "venue": self.venue, "symbol": self.symbol,
            "side": self.side.value, "quantity": str(self.quantity), "order_type": self.order_type.value,
            "limit_price": str(self.limit_price) if self.limit_price is not None else None,
            "time_in_force": self.time_in_force.value,
            "expires_at": iso(self.expires_at) if self.expires_at else None,
            "quantity_step": str(self.quantity_step), "price_tick": str(self.price_tick),
            "fee_bps": str(self.fee_bps), "fee_quantum": str(self.fee_quantum),
            "max_quote_age_seconds": str(self.max_quote_age_seconds),
        })


@dataclass(frozen=True)
class Fill:
    fill_id: str
    snapshot_id: str
    level_index: int
    price: Decimal
    quantity: Decimal
    gross_idr: Decimal
    fee_idr: Decimal


@dataclass(frozen=True)
class ExecutionResult:
    order_id: str
    snapshot_id: str
    state: OrderState
    requested_quantity: Decimal
    cumulative_filled_quantity: Decimal
    remaining_quantity: Decimal
    fills: tuple[Fill, ...]
    vwap: Decimal | None
    gross_idr: Decimal
    fee_idr: Decimal
    reasons: tuple[str, ...]
    duplicate: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id, "snapshot_id": self.snapshot_id, "state": self.state.value,
            "requested_quantity": str(self.requested_quantity),
            "cumulative_filled_quantity": str(self.cumulative_filled_quantity),
            "remaining_quantity": str(self.remaining_quantity),
            "fills": [{"fill_id":f.fill_id,"snapshot_id":f.snapshot_id,"level_index":f.level_index,
                       "price":str(f.price),"quantity":str(f.quantity),"gross_idr":str(f.gross_idr),"fee_idr":str(f.fee_idr)} for f in self.fills],
            "vwap": str(self.vwap) if self.vwap is not None else None,
            "gross_idr":str(self.gross_idr),"fee_idr":str(self.fee_idr),"reasons":list(self.reasons),
        }


class SQLitePaperExecutionLedger:
    """Atomic order/fill/snapshot-consumption journal for a single simulated portfolio."""

    def __init__(self, path: str | Path, *, initial_cash_idr: Decimal = Decimal("300000"), initial_sol: Decimal = Decimal("0")):
        self.path = str(Path(path).expanduser())
        if self.path == ":memory:": raise ExecutionError("use file-backed SQLite to exercise restart recovery")
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.initial_cash = D(initial_cash_idr, "initial_cash_idr", allow_zero=True)
        self.initial_sol = D(initial_sol, "initial_sol", allow_zero=True)
        if self.initial_sol != 0:
            raise ExecutionError("non-zero initial SOL is disabled until an explicit acquisition cost basis is supported")
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=10000")
        return db

    def _initialize(self) -> None:
        with closing(self._connect()) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS paper_accounts(
              account_id TEXT PRIMARY KEY, initial_cash_idr TEXT NOT NULL, initial_sol TEXT NOT NULL,
              cash_idr TEXT NOT NULL, sol_quantity TEXT NOT NULL, avg_entry_cost_idr TEXT NOT NULL,
              realized_pnl_idr TEXT NOT NULL, fees_paid_idr TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS orders(
              order_id TEXT PRIMARY KEY, client_order_id TEXT NOT NULL UNIQUE, fingerprint TEXT NOT NULL,
              venue TEXT NOT NULL, symbol TEXT NOT NULL, side TEXT NOT NULL, order_type TEXT NOT NULL,
              requested_qty TEXT NOT NULL, filled_qty TEXT NOT NULL, remaining_qty TEXT NOT NULL,
              limit_price TEXT, tif TEXT NOT NULL, created_at TEXT NOT NULL, expires_at TEXT,
              quantity_step TEXT NOT NULL, price_tick TEXT NOT NULL, fee_bps TEXT NOT NULL, fee_quantum TEXT NOT NULL,
              max_quote_age_seconds TEXT NOT NULL, state TEXT NOT NULL, gross_idr TEXT NOT NULL,
              fees_idr TEXT NOT NULL, last_snapshot_at TEXT, updated_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS snapshots(
              snapshot_id TEXT PRIMARY KEY, venue TEXT NOT NULL, symbol TEXT NOT NULL,
              observed_at TEXT NOT NULL, exchange_timestamp_ms INTEGER, book_hash TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS snapshot_levels(
              snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id), side TEXT NOT NULL,
              level_index INTEGER NOT NULL, price TEXT NOT NULL, quantity TEXT NOT NULL,
              PRIMARY KEY(snapshot_id,side,level_index));
            CREATE TABLE IF NOT EXISTS snapshot_consumption(
              snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id), side TEXT NOT NULL,
              level_index INTEGER NOT NULL, reserved_qty TEXT NOT NULL, consumed_qty TEXT NOT NULL,
              PRIMARY KEY(snapshot_id,side,level_index));
            CREATE TABLE IF NOT EXISTS liquidity_pools(
              pool_id TEXT PRIMARY KEY, venue TEXT NOT NULL, symbol TEXT NOT NULL, side TEXT NOT NULL,
              book_hash TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS liquidity_pool_levels(
              pool_id TEXT NOT NULL REFERENCES liquidity_pools(pool_id), level_index INTEGER NOT NULL,
              price TEXT NOT NULL, quantity TEXT NOT NULL, reserved_qty TEXT NOT NULL, consumed_qty TEXT NOT NULL,
              PRIMARY KEY(pool_id,level_index));
            CREATE TABLE IF NOT EXISTS snapshot_liquidity_pools(
              snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id), side TEXT NOT NULL,
              pool_id TEXT NOT NULL REFERENCES liquidity_pools(pool_id), PRIMARY KEY(snapshot_id,side));
            CREATE TABLE IF NOT EXISTS fills(
              seq INTEGER PRIMARY KEY AUTOINCREMENT, fill_id TEXT NOT NULL UNIQUE,
              order_id TEXT NOT NULL REFERENCES orders(order_id), snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id),
              level_index INTEGER NOT NULL, side TEXT NOT NULL, price TEXT NOT NULL, quantity TEXT NOT NULL,
              gross_idr TEXT NOT NULL, fee_idr TEXT NOT NULL, filled_at TEXT NOT NULL,
              UNIQUE(order_id,snapshot_id,level_index));
            CREATE TABLE IF NOT EXISTS events(
              seq INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL UNIQUE, order_id TEXT NOT NULL REFERENCES orders(order_id),
              snapshot_id TEXT, event_type TEXT NOT NULL, occurred_at TEXT NOT NULL, payload_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS execution_cycles(
              cycle_id TEXT PRIMARY KEY, order_id TEXT NOT NULL REFERENCES orders(order_id),
              snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id), result_json TEXT NOT NULL,
              UNIQUE(order_id,snapshot_id));
            CREATE INDEX IF NOT EXISTS orders_state_idx ON orders(state,expires_at);
            CREATE INDEX IF NOT EXISTS fills_order_idx ON fills(order_id,seq);
            CREATE INDEX IF NOT EXISTS events_order_idx ON events(order_id,seq);
            CREATE INDEX IF NOT EXISTS snapshots_time_idx ON snapshots(observed_at);
            CREATE TRIGGER IF NOT EXISTS fills_no_update BEFORE UPDATE ON fills BEGIN SELECT RAISE(ABORT,'fills are append-only'); END;
            CREATE TRIGGER IF NOT EXISTS fills_no_delete BEFORE DELETE ON fills BEGIN SELECT RAISE(ABORT,'fills are append-only'); END;
            CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events BEGIN SELECT RAISE(ABORT,'events are append-only'); END;
            CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT,'events are append-only'); END;
            CREATE TRIGGER IF NOT EXISTS snapshot_levels_no_update BEFORE UPDATE ON snapshot_levels BEGIN SELECT RAISE(ABORT,'snapshot levels are immutable'); END;
            CREATE TRIGGER IF NOT EXISTS snapshot_levels_no_delete BEFORE DELETE ON snapshot_levels BEGIN SELECT RAISE(ABORT,'snapshot levels are immutable'); END;
            CREATE TRIGGER IF NOT EXISTS snapshots_no_update BEFORE UPDATE ON snapshots BEGIN SELECT RAISE(ABORT,'snapshots are immutable'); END;
            CREATE TRIGGER IF NOT EXISTS snapshots_no_delete BEFORE DELETE ON snapshots BEGIN SELECT RAISE(ABORT,'snapshots are immutable'); END;
            """)
            row = db.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
            if row and int(row["value"]) != SCHEMA_VERSION:
                raise LedgerIntegrityError("unsupported execution ledger schema version")
            db.execute("INSERT OR IGNORE INTO metadata VALUES('schema_version',?)", (str(SCHEMA_VERSION),))
            db.execute("BEGIN IMMEDIATE")
            try:
                existing = db.execute("SELECT * FROM paper_accounts WHERE account_id=?", (ACCOUNT_ID,)).fetchone()
                if existing is None:
                    now = iso(datetime.now(timezone.utc))
                    avg = "0" if self.initial_sol == 0 else "0"
                    db.execute("INSERT INTO paper_accounts VALUES(?,?,?,?,?,?,?,?,?)", (
                        ACCOUNT_ID,str(self.initial_cash),str(self.initial_sol),str(self.initial_cash),str(self.initial_sol),avg,"0","0",now))
                else:
                    if D(existing["initial_cash_idr"],"stored initial cash",allow_zero=True) != self.initial_cash or D(existing["initial_sol"],"stored initial SOL",allow_zero=True) != self.initial_sol:
                        raise LedgerIntegrityError("initial portfolio differs from persisted account; refusing implicit reset")
                db.execute("COMMIT")
            except Exception:
                db.execute("ROLLBACK"); raise
            self._backfill_liquidity_pools(db)

    def _register_side_pool(self, db: sqlite3.Connection, snapshot_id: str, venue: str, symbol: str,
                            side: str, levels: Sequence[BookLevel], *, migrated_consumed: Decimal = Decimal("0")) -> str:
        pool_id = canonical_hash({"venue":venue,"symbol":symbol,"side":side,
                                  "levels":[[str(x.price),str(x.quantity)] for x in levels]})
        book_hash = canonical_hash([[str(x.price),str(x.quantity)] for x in levels])
        pool = db.execute("SELECT venue,symbol,side,book_hash FROM liquidity_pools WHERE pool_id=?",(pool_id,)).fetchone()
        if pool is None:
            db.execute("INSERT INTO liquidity_pools VALUES(?,?,?,?,?)",(pool_id,venue,symbol,side,book_hash))
            db.executemany("INSERT INTO liquidity_pool_levels VALUES(?,?,?,?,?,?)",[
                (pool_id,i,str(level.price),str(level.quantity),"0","0") for i,level in enumerate(levels)])
        elif (pool["venue"],pool["symbol"],pool["side"],pool["book_hash"]) != (venue,symbol,side,book_hash):
            raise LedgerIntegrityError("liquidity pool identity conflicts with captured levels")
        else:
            persisted=db.execute("SELECT price,quantity FROM liquidity_pool_levels WHERE pool_id=? ORDER BY level_index",(pool_id,)).fetchall()
            expected=[(str(x.price),str(x.quantity)) for x in levels]
            if [(x["price"],x["quantity"]) for x in persisted] != expected:
                raise LedgerIntegrityError("liquidity pool levels differ from immutable side-book identity")
        mapping=db.execute("SELECT pool_id FROM snapshot_liquidity_pools WHERE snapshot_id=? AND side=?",(snapshot_id,side)).fetchone()
        if mapping is None:
            db.execute("INSERT INTO snapshot_liquidity_pools VALUES(?,?,?)",(snapshot_id,side,pool_id))
            if migrated_consumed:
                for i,level in enumerate(levels):
                    old=db.execute("SELECT consumed_qty FROM snapshot_consumption WHERE snapshot_id=? AND side=? AND level_index=?",
                                   (snapshot_id,side,i)).fetchone()
                    old_qty=D(old["consumed_qty"],"legacy snapshot consumption",allow_zero=True) if old else Decimal(0)
                    if old_qty:
                        row=db.execute("SELECT quantity,consumed_qty,reserved_qty FROM liquidity_pool_levels WHERE pool_id=? AND level_index=?",
                                       (pool_id,i)).fetchone()
                        combined=D(row["consumed_qty"],"pool consumed",allow_zero=True)+old_qty
                        if combined+D(row["reserved_qty"],"pool reserved",allow_zero=True)>D(row["quantity"],"pool original"):
                            raise LedgerIntegrityError("legacy snapshot fills over-consume identical side-book liquidity; manual audit required")
                        db.execute("UPDATE liquidity_pool_levels SET consumed_qty=? WHERE pool_id=? AND level_index=?",
                                   (str(combined),pool_id,i))
        elif mapping["pool_id"] != pool_id:
            raise LedgerIntegrityError("snapshot side maps to a different immutable liquidity pool")
        return pool_id

    def _backfill_liquidity_pools(self, db: sqlite3.Connection) -> None:
        """Attach previously persisted snapshots to shared side-book pools exactly once."""
        db.execute("BEGIN IMMEDIATE")
        try:
            rows=db.execute("SELECT snapshot_id,venue,symbol FROM snapshots ORDER BY observed_at,snapshot_id").fetchall()
            for row in rows:
                for side in ("bid","ask"):
                    exists=db.execute("SELECT 1 FROM snapshot_liquidity_pools WHERE snapshot_id=? AND side=?",
                                      (row["snapshot_id"],side)).fetchone()
                    if exists:
                        continue
                    level_rows=db.execute("SELECT price,quantity FROM snapshot_levels WHERE snapshot_id=? AND side=? ORDER BY level_index",
                                          (row["snapshot_id"],side)).fetchall()
                    levels=tuple(BookLevel(D(x["price"],"stored pool price"),D(x["quantity"],"stored pool qty")) for x in level_rows)
                    if not levels:
                        raise LedgerIntegrityError("legacy snapshot lacks depth levels during liquidity-pool migration")
                    self._register_side_pool(db,row["snapshot_id"],row["venue"],row["symbol"],side,levels,migrated_consumed=Decimal("1"))
            db.execute("COMMIT")
        except Exception:
            db.execute("ROLLBACK"); raise

    @staticmethod
    def _event(db: sqlite3.Connection, order_id: str, event_type: str, occurred_at: str,
               payload: dict[str, Any], snapshot_id: str | None = None, discriminator: str = "") -> None:
        event_id = canonical_hash({"order_id":order_id,"snapshot_id":snapshot_id,"event_type":event_type,
                                   "occurred_at":occurred_at,"payload":payload,"discriminator":discriminator})
        db.execute("INSERT OR IGNORE INTO events(event_id,order_id,snapshot_id,event_type,occurred_at,payload_json) VALUES(?,?,?,?,?,?)",
                   (event_id,order_id,snapshot_id,event_type,occurred_at,json.dumps(payload,sort_keys=True,separators=(",",":"))))

    @staticmethod
    def _transition(current: OrderState, new: OrderState) -> None:
        if new not in TRANSITIONS.get(current,set()):
            raise ExecutionError(f"invalid order state transition: {current.value} -> {new.value}")

    def create_order(self, request: OrderRequest) -> str:
        fingerprint = request.fingerprint()
        order_id = canonical_hash({"client_order_id":request.client_order_id,"fingerprint":fingerprint})[:32]
        now = iso(request.created_at)
        with closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                old = db.execute("SELECT order_id,fingerprint FROM orders WHERE client_order_id=?",(request.client_order_id,)).fetchone()
                if old:
                    if old["fingerprint"] != fingerprint: raise LedgerIntegrityError("client_order_id reused with different order content")
                    db.execute("COMMIT"); return old["order_id"]
                db.execute("INSERT INTO orders VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                    order_id,request.client_order_id,fingerprint,request.venue,request.symbol,request.side.value,request.order_type.value,
                    str(request.quantity),"0",str(request.quantity),str(request.limit_price) if request.limit_price is not None else None,
                    request.time_in_force.value,now,iso(request.expires_at) if request.expires_at else None,
                    str(request.quantity_step),str(request.price_tick),str(request.fee_bps),str(request.fee_quantum),str(request.max_quote_age_seconds),
                    OrderState.CREATED.value,"0","0",None,now))
                self._event(db,order_id,"OrderCreated",now,{"request_fingerprint":fingerprint})
                self._transition(OrderState.CREATED,OrderState.OPEN_SIMULATED)
                db.execute("UPDATE orders SET state=?,updated_at=? WHERE order_id=?",(OrderState.OPEN_SIMULATED.value,now,order_id))
                self._event(db,order_id,"OrderAccepted",now,{"state":OrderState.OPEN_SIMULATED.value})
                db.execute("COMMIT")
                return order_id
            except Exception:
                db.execute("ROLLBACK"); raise

    def _persist_snapshot(self, db: sqlite3.Connection, snapshot: DepthSnapshot) -> None:
        observed = iso(snapshot.observed_at)
        book_hash = canonical_hash({"bids":[[str(x.price),str(x.quantity)] for x in snapshot.bids],
                                    "asks":[[str(x.price),str(x.quantity)] for x in snapshot.asks]})
        existing = db.execute("SELECT venue,symbol,observed_at,exchange_timestamp_ms,book_hash FROM snapshots WHERE snapshot_id=?",
                              (snapshot.snapshot_id,)).fetchone()
        if existing:
            if (existing["venue"],existing["symbol"],existing["observed_at"],existing["exchange_timestamp_ms"],existing["book_hash"]) != (
                snapshot.venue,snapshot.symbol,observed,snapshot.exchange_timestamp_ms,book_hash):
                raise LedgerIntegrityError("snapshot identity conflicts with persisted snapshot")
        else:
            db.execute("INSERT INTO snapshots VALUES(?,?,?,?,?,?)",(snapshot.snapshot_id,snapshot.venue,snapshot.symbol,observed,snapshot.exchange_timestamp_ms,book_hash))
            db.executemany("INSERT INTO snapshot_levels VALUES(?,?,?,?,?)",[
                (snapshot.snapshot_id,side,i,str(level.price),str(level.quantity))
                for side,levels in (("bid",snapshot.bids),("ask",snapshot.asks)) for i,level in enumerate(levels)])
            db.executemany("INSERT INTO snapshot_consumption VALUES(?,?,?,?,?)",[
                (snapshot.snapshot_id,side,i,"0","0")
                for side,levels in (("bid",snapshot.bids),("ask",snapshot.asks)) for i,_ in enumerate(levels)])
        self._register_side_pool(db,snapshot.snapshot_id,snapshot.venue,snapshot.symbol,"bid",snapshot.bids)
        self._register_side_pool(db,snapshot.snapshot_id,snapshot.venue,snapshot.symbol,"ask",snapshot.asks)

    def _account(self, db: sqlite3.Connection) -> sqlite3.Row:
        row = db.execute("SELECT * FROM paper_accounts WHERE account_id=?",(ACCOUNT_ID,)).fetchone()
        if row is None: raise LedgerIntegrityError("paper account is missing")
        return row

    def _account_apply_fill(self, db: sqlite3.Connection, side: Side, qty: Decimal, price: Decimal, gross: Decimal,
                            fee: Decimal, filled_at: str) -> None:
        account = self._account(db)
        cash = D(account["cash_idr"],"cash",allow_zero=True)
        sol = D(account["sol_quantity"],"SOL inventory",allow_zero=True)
        avg = D(account["avg_entry_cost_idr"],"average cost",allow_zero=True)
        realized = Decimal(account["realized_pnl_idr"])
        fees = D(account["fees_paid_idr"],"fees",allow_zero=True)
        if side is Side.BUY:
            total = gross + fee
            if total > cash: raise ExecutionError("insufficient paper cash for depth fill")
            new_sol = sol + qty
            new_avg = (sol * avg + total) / new_sol
            new_cash, new_realized = cash-total, realized
        else:
            if qty > sol: raise ExecutionError("insufficient paper SOL inventory for sell fill")
            removed_cost = qty * avg
            new_cash = cash + gross - fee
            new_sol = sol - qty
            new_avg = Decimal("0") if new_sol == 0 else avg
            new_realized = realized + gross - fee - removed_cost
        db.execute("UPDATE paper_accounts SET cash_idr=?,sol_quantity=?,avg_entry_cost_idr=?,realized_pnl_idr=?,fees_paid_idr=?,updated_at=? WHERE account_id=?",
                   (str(new_cash),str(new_sol),str(new_avg),str(new_realized),str(fees+fee),filled_at,ACCOUNT_ID))

    def process_snapshot(self, order_id: str, snapshot: DepthSnapshot, *, as_of: datetime | None = None,
                         failure_hook: Callable[[str],None] | None = None) -> ExecutionResult:
        current_time = aware(as_of or datetime.now(timezone.utc), "as_of")
        current_iso = iso(current_time)
        with closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                order = db.execute("SELECT * FROM orders WHERE order_id=?",(order_id,)).fetchone()
                if order is None: raise ExecutionError("unknown order_id")
                self._persist_snapshot(db,snapshot)
                prior = db.execute("SELECT result_json FROM execution_cycles WHERE order_id=? AND snapshot_id=?",(order_id,snapshot.snapshot_id)).fetchone()
                if prior:
                    db.execute("COMMIT")
                    payload=json.loads(prior["result_json"])
                    return self._result_from_dict(payload,duplicate=True)
                state = OrderState(order["state"])
                if state in TERMINAL_STATES:
                    result=self._result_from_order(order,snapshot.snapshot_id,(),("order_terminal",))
                    self._store_cycle(db,result); db.execute("COMMIT"); return result
                if (order["venue"],order["symbol"]) != (snapshot.venue,snapshot.symbol):
                    result=self._result_from_order(order,snapshot.snapshot_id,(),("snapshot_market_mismatch",))
                    self._store_cycle(db,result); db.execute("COMMIT"); return result
                observed = aware(snapshot.observed_at,"snapshot.observed_at")
                created = datetime.fromisoformat(order["created_at"])
                if observed <= created or observed > current_time:
                    result=self._result_from_order(order,snapshot.snapshot_id,(),("snapshot_timestamp_ineligible",))
                    self._store_cycle(db,result); db.execute("COMMIT"); return result
                if order["last_snapshot_at"] and observed < datetime.fromisoformat(order["last_snapshot_at"]):
                    result=self._result_from_order(order,snapshot.snapshot_id,(),("out_of_order_snapshot",))
                    self._store_cycle(db,result); db.execute("COMMIT"); return result
                expires = datetime.fromisoformat(order["expires_at"]) if order["expires_at"] else None
                if expires and current_time >= expires:
                    self._transition(state,OrderState.EXPIRED_SIMULATED)
                    db.execute("UPDATE orders SET state=?,updated_at=? WHERE order_id=?",(OrderState.EXPIRED_SIMULATED.value,current_iso,order_id))
                    self._event(db,order_id,"OrderExpired",current_iso,{"remaining_qty":order["remaining_qty"]},snapshot.snapshot_id)
                    refreshed=db.execute("SELECT * FROM orders WHERE order_id=?",(order_id,)).fetchone()
                    result=self._result_from_order(refreshed,snapshot.snapshot_id,(),("order_expired",))
                    self._store_cycle(db,result); db.execute("COMMIT"); return result
                age = Decimal(str((current_time-observed).total_seconds()))
                max_age = D(order["max_quote_age_seconds"],"max_quote_age_seconds")
                if age < 0 or age > max_age:
                    result=self._result_from_order(order,snapshot.snapshot_id,(),("snapshot_stale",))
                    self._store_cycle(db,result); db.execute("COMMIT"); return result

                side = Side(order["side"])
                # The current order's market filters are authoritative for its execution.
                tick = D(order["price_tick"], "price_tick") if "price_tick" in order.keys() else Decimal("1")
                qty_step = D(order["quantity_step"], "quantity_step")
                all_levels = tuple(snapshot.bids) + tuple(snapshot.asks)
                if any(level.price % tick != 0 or level.quantity % qty_step != 0 for level in all_levels):
                    result=self._result_from_order(order,snapshot.snapshot_id,(),("snapshot_precision_mismatch",))
                    self._store_cycle(db,result); db.execute("COMMIT"); return result
                book_side = "ask" if side is Side.BUY else "bid"
                pool_id = snapshot.liquidity_pool_id(book_side)
                levels = snapshot.asks if side is Side.BUY else snapshot.bids
                remaining = D(order["remaining_qty"],"remaining_qty",allow_zero=True)
                previous_filled = D(order["filled_qty"],"filled_qty",allow_zero=True)
                new_fills: list[Fill] = []
                reasons: list[str] = []
                account = self._account(db)
                account_cash = D(account["cash_idr"],"cash",allow_zero=True)
                account_sol = D(account["sol_quantity"],"SOL inventory",allow_zero=True)
                fee_bps = D(order["fee_bps"],"fee_bps",allow_zero=True)
                fee_quantum = D(order["fee_quantum"],"fee_quantum")
                rate = fee_bps / Decimal("10000")
                limit = D(order["limit_price"],"limit_price") if order["limit_price"] is not None else None
                for index, level in enumerate(levels):
                    if remaining <= 0: break
                    if limit is not None and ((side is Side.BUY and level.price > limit) or (side is Side.SELL and level.price < limit)):
                        reasons.append("limit_price_boundary_reached"); break
                    consumed_row = db.execute("SELECT consumed_qty,reserved_qty FROM liquidity_pool_levels WHERE pool_id=? AND level_index=?",
                                               (pool_id,index)).fetchone()
                    if consumed_row is None: raise LedgerIntegrityError("snapshot level is missing from its shared liquidity pool")
                    consumed = D(consumed_row["consumed_qty"],"shared liquidity consumed",allow_zero=True)
                    reserved=D(consumed_row["reserved_qty"],"shared liquidity reserved",allow_zero=True)
                    available = level.quantity-consumed-reserved
                    if available <= 0: continue
                    qty = min(remaining,available)
                    step = D(order["quantity_step"],"quantity_step")
                    qty = floor_step(qty,step)
                    if side is Side.BUY:
                        cash_cap = floor_step(account_cash / (level.price * (Decimal("1")+rate)),step)
                        qty = min(qty,cash_cap)
                    else:
                        qty = min(qty,floor_step(account_sol,step))
                    if qty <= 0:
                        reasons.append("portfolio_capacity_exhausted")
                        break
                    gross = qty*level.price
                    fee = (gross*rate).quantize(fee_quantum,rounding=ROUND_HALF_UP)
                    if side is Side.BUY and gross+fee > account_cash:
                        qty = floor_step(qty-step,step)
                        if qty <= 0: reasons.append("insufficient_cash_after_fee_rounding"); break
                        gross=qty*level.price; fee=(gross*rate).quantize(fee_quantum,rounding=ROUND_HALF_UP)
                    if side is Side.SELL and qty > account_sol:
                        qty=floor_step(account_sol,step); gross=qty*level.price; fee=(gross*rate).quantize(fee_quantum,rounding=ROUND_HALF_UP)
                    fill_id=canonical_hash({"order_id":order_id,"snapshot_id":snapshot.snapshot_id,"level_index":index})
                    fill=Fill(fill_id,snapshot.snapshot_id,index,level.price,qty,gross,fee)
                    new_fills.append(fill)
                    remaining -= qty
                    account_cash = account_cash-gross-fee if side is Side.BUY else account_cash+gross-fee
                    account_sol = account_sol+qty if side is Side.BUY else account_sol-qty
                    db.execute("UPDATE snapshot_consumption SET consumed_qty=? WHERE snapshot_id=? AND side=? AND level_index=?",
                               (str(D(db.execute("SELECT consumed_qty FROM snapshot_consumption WHERE snapshot_id=? AND side=? AND level_index=?",
                                                  (snapshot.snapshot_id,book_side,index)).fetchone()[0],"snapshot consumed",allow_zero=True)+qty),
                                snapshot.snapshot_id,book_side,index))
                    db.execute("UPDATE liquidity_pool_levels SET consumed_qty=? WHERE pool_id=? AND level_index=?",
                               (str(consumed+qty),pool_id,index))
                    db.execute("INSERT INTO fills(fill_id,order_id,snapshot_id,level_index,side,price,quantity,gross_idr,fee_idr,filled_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                               (fill_id,order_id,snapshot.snapshot_id,index,side.value,str(level.price),str(qty),str(gross),str(fee),iso(snapshot.observed_at)))
                    if failure_hook: failure_hook("after_fill_insert")
                    self._event(db,order_id,"FillRecorded",current_iso,{
                        "fill_id":fill.fill_id,"snapshot_id":fill.snapshot_id,"level_index":fill.level_index,
                        "side":side.value,"price":str(fill.price),"quantity":str(fill.quantity),
                        "gross_idr":str(fill.gross_idr),"fee_idr":str(fill.fee_idr)},snapshot.snapshot_id,str(index))
                    if failure_hook: failure_hook("after_fill_event")
                    self._account_apply_fill(db,side,qty,level.price,gross,fee,iso(snapshot.observed_at))
                    if failure_hook: failure_hook("after_portfolio_fill")

                filled_total = previous_filled + sum((f.quantity for f in new_fills),Decimal("0"))
                gross_total = D(order["gross_idr"],"gross",allow_zero=True) + sum((f.gross_idr for f in new_fills),Decimal("0"))
                fees_total = D(order["fees_idr"],"fees",allow_zero=True) + sum((f.fee_idr for f in new_fills),Decimal("0"))
                if remaining == 0:
                    new_state=OrderState.FILLED_SIMULATED
                elif new_fills:
                    new_state=OrderState.PARTIALLY_FILLED_SIMULATED
                else:
                    new_state=OrderState.OPEN_SIMULATED
                if order["tif"] == TimeInForce.IOC.value and remaining > 0:
                    new_state=OrderState.CANCELLED_SIMULATED
                    reasons.append("ioc_residual_cancelled")
                    self._event(db,order_id,"OrderCancelled",current_iso,{"reason":"IOC residual","remaining_qty":str(remaining)},snapshot.snapshot_id)
                self._transition(state,new_state)
                db.execute("UPDATE orders SET filled_qty=?,remaining_qty=?,state=?,gross_idr=?,fees_idr=?,last_snapshot_at=?,updated_at=? WHERE order_id=?",
                           (str(filled_total),str(remaining),new_state.value,str(gross_total),str(fees_total),iso(snapshot.observed_at),current_iso,order_id))
                if failure_hook: failure_hook("after_portfolio_update")
                if new_fills:
                    self._event(db,order_id,"ResidualUpdated",current_iso,{"filled_qty":str(filled_total),"remaining_qty":str(remaining)},snapshot.snapshot_id)
                result=self._result_from_values(order_id,snapshot.snapshot_id,new_state,D(order["requested_qty"],"requested_qty"),
                    filled_total,remaining,tuple(new_fills),gross_total,fees_total,tuple(reasons))
                self._store_cycle(db,result)
                if failure_hook: failure_hook("before_commit")
                db.execute("COMMIT")
            except Exception:
                db.execute("ROLLBACK")
                raise
        if failure_hook: failure_hook("after_commit")
        return result

    def _store_cycle(self, db: sqlite3.Connection, result: ExecutionResult) -> None:
        cycle_id=canonical_hash({"order_id":result.order_id,"snapshot_id":result.snapshot_id})
        db.execute("INSERT INTO execution_cycles VALUES(?,?,?,?)",(cycle_id,result.order_id,result.snapshot_id,json.dumps(result.as_dict(),sort_keys=True,separators=(",",":"))))

    @staticmethod
    def _result_from_values(order_id,snapshot_id,state,requested,filled,remaining,fills,gross,fees,reasons):
        # gross and filled are cumulative for the order, so VWAP must be cumulative too.
        vwap=(gross/filled) if filled else None
        return ExecutionResult(order_id,snapshot_id,state,requested,filled,remaining,tuple(fills),vwap,gross,fees,tuple(reasons))

    def _result_from_order(self, order, snapshot_id, fills, reasons):
        return self._result_from_values(order["order_id"],snapshot_id,OrderState(order["state"]),D(order["requested_qty"],"requested"),
            D(order["filled_qty"],"filled",allow_zero=True),D(order["remaining_qty"],"remaining",allow_zero=True),tuple(fills),
            D(order["gross_idr"],"gross",allow_zero=True),D(order["fees_idr"],"fees",allow_zero=True),tuple(reasons))

    def _result_from_dict(self,payload,duplicate=False):
        fills=tuple(Fill(x["fill_id"],x["snapshot_id"],x["level_index"],D(x["price"],"price"),D(x["quantity"],"quantity"),
                         D(x["gross_idr"],"gross",allow_zero=True),D(x["fee_idr"],"fee",allow_zero=True)) for x in payload["fills"])
        return ExecutionResult(payload["order_id"],payload["snapshot_id"],OrderState(payload["state"]),D(payload["requested_quantity"],"requested"),
            D(payload["cumulative_filled_quantity"],"filled",allow_zero=True),D(payload["remaining_quantity"],"remaining",allow_zero=True),fills,
            D(payload["vwap"],"vwap") if payload["vwap"] is not None else None,D(payload["gross_idr"],"gross",allow_zero=True),
            D(payload["fee_idr"],"fee",allow_zero=True),tuple(payload["reasons"]),duplicate)

    def cancel_order(self, order_id: str, *, at: datetime | None = None, reason: str = "operator_cancel") -> dict[str, Any]:
        when=iso(at or datetime.now(timezone.utc))
        with closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                order=db.execute("SELECT * FROM orders WHERE order_id=?",(order_id,)).fetchone()
                if order is None: raise ExecutionError("unknown order_id")
                state=OrderState(order["state"])
                if state in TERMINAL_STATES:
                    db.execute("COMMIT"); return self.get_order(order_id)
                self._transition(state,OrderState.CANCELLED_SIMULATED)
                db.execute("UPDATE orders SET state=?,updated_at=? WHERE order_id=?",(OrderState.CANCELLED_SIMULATED.value,when,order_id))
                self._event(db,order_id,"OrderCancelled",when,{"reason":reason,"remaining_qty":order["remaining_qty"]})
                db.execute("COMMIT")
            except Exception:
                db.execute("ROLLBACK"); raise
        return self.get_order(order_id)

    def expire_orders(self, *, as_of: datetime) -> list[str]:
        now=aware(as_of,"as_of"); now_s=iso(now); expired=[]
        with closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                rows=db.execute("SELECT * FROM orders WHERE state IN (?,?) AND expires_at IS NOT NULL AND expires_at<=?",
                    (OrderState.OPEN_SIMULATED.value,OrderState.PARTIALLY_FILLED_SIMULATED.value,now_s)).fetchall()
                for row in rows:
                    state=OrderState(row["state"]); self._transition(state,OrderState.EXPIRED_SIMULATED)
                    db.execute("UPDATE orders SET state=?,updated_at=? WHERE order_id=?",(OrderState.EXPIRED_SIMULATED.value,now_s,row["order_id"]))
                    self._event(db,row["order_id"],"OrderExpired",now_s,{"remaining_qty":row["remaining_qty"]})
                    expired.append(row["order_id"])
                db.execute("COMMIT")
            except Exception:
                db.execute("ROLLBACK"); raise
        return expired

    def get_order(self, order_id: str) -> dict[str, Any]:
        with closing(self._connect()) as db:
            row=db.execute("SELECT * FROM orders WHERE order_id=?",(order_id,)).fetchone()
            if row is None: raise ExecutionError("unknown order_id")
            result=dict(row)
            result["state"]=OrderState(result["state"])
            return result

    def find_order_by_client_order_id(self, client_order_id: str) -> dict[str, Any] | None:
        """Return a durable order by its caller-supplied idempotency key."""
        if not isinstance(client_order_id, str) or not client_order_id.strip():
            raise ExecutionError("client_order_id must be a non-empty string")
        with closing(self._connect()) as db:
            row = db.execute("SELECT * FROM orders WHERE client_order_id=?", (client_order_id,)).fetchone()
            if row is None:
                return None
            result = dict(row)
            result["state"] = OrderState(result["state"])
            return result

    def list_open_orders(self) -> list[dict[str, Any]]:
        """Return every recoverable simulated residual in deterministic FIFO order."""
        open_states = (
            OrderState.OPEN_SIMULATED.value,
            OrderState.PARTIALLY_FILLED_SIMULATED.value,
            OrderState.RECOVERY_REQUIRED.value,
        )
        with closing(self._connect()) as db:
            rows = db.execute(
                "SELECT * FROM orders WHERE state IN (?,?,?) ORDER BY created_at,order_id",
                open_states,
            ).fetchall()
            results = []
            for row in rows:
                item = dict(row)
                item["state"] = OrderState(item["state"])
                results.append(item)
            return results

    def get_fills(self, order_id: str | None = None) -> list[dict[str, Any]]:
        with closing(self._connect()) as db:
            if order_id:
                rows=db.execute("SELECT * FROM fills WHERE order_id=? ORDER BY seq",(order_id,)).fetchall()
            else:
                rows=db.execute("SELECT * FROM fills ORDER BY seq").fetchall()
            return [dict(row) for row in rows]

    def get_events(self, order_id: str | None = None) -> list[dict[str, Any]]:
        with closing(self._connect()) as db:
            rows=db.execute("SELECT * FROM events WHERE order_id=? ORDER BY seq",(order_id,)).fetchall() if order_id else db.execute("SELECT * FROM events ORDER BY seq").fetchall()
            return [dict(row) for row in rows]

    def portfolio(self) -> dict[str, Decimal]:
        with closing(self._connect()) as db:
            row=self._account(db)
            return {k:D(row[k],k,allow_zero=True) if k != "realized_pnl_idr" else Decimal(row[k])
                    for k in ("cash_idr","sol_quantity","avg_entry_cost_idr","realized_pnl_idr","fees_paid_idr")}

    def reconcile(self) -> dict[str, Any]:
        """Rebuild order totals and portfolio from immutable fill rows; never silently repair."""
        with closing(self._connect()) as db:
            orders=db.execute("SELECT * FROM orders ORDER BY created_at,order_id").fetchall()
            fills=db.execute("SELECT * FROM fills ORDER BY seq").fetchall()
            account=self._account(db)
            per_order: dict[str,dict[str,Decimal]]={}
            cash=D(account["initial_cash_idr"],"initial cash",allow_zero=True)
            sol=D(account["initial_sol"],"initial SOL",allow_zero=True)
            avg=Decimal("0"); realized=Decimal("0"); fees=Decimal("0")
            for f in fills:
                qty=D(f["quantity"],"fill quantity"); price=D(f["price"],"fill price"); gross=D(f["gross_idr"],"gross"); fee=D(f["fee_idr"],"fee",allow_zero=True)
                order=db.execute("SELECT side FROM orders WHERE order_id=?",(f["order_id"],)).fetchone()
                if order is None: raise LedgerIntegrityError("orphan fill detected")
                side=Side(order["side"])
                state=per_order.setdefault(f["order_id"],{"qty":Decimal(0),"gross":Decimal(0),"fees":Decimal(0)})
                state["qty"]+=qty; state["gross"]+=gross; state["fees"]+=fee
                if side is Side.BUY:
                    total=gross+fee; cash-=total; new=sol+qty; avg=(sol*avg+total)/new if new else Decimal(0); sol=new
                else:
                    if qty>sol: raise LedgerIntegrityError("sell fills exceed reconstructed inventory")
                    cash+=gross-fee; realized+=gross-fee-qty*avg; sol-=qty
                    if sol==0: avg=Decimal(0)
                fees+=fee
            issues=[]
            for order in orders:
                req=D(order["requested_qty"],"requested"); fqty=D(order["filled_qty"],"order filled",allow_zero=True); rem=D(order["remaining_qty"],"order remaining",allow_zero=True)
                aggregate=per_order.get(order["order_id"],{"qty":Decimal(0),"gross":Decimal(0),"fees":Decimal(0)})
                if req != fqty+rem: issues.append(f"{order['order_id']}: requested != filled + remaining")
                if aggregate["qty"] != fqty: issues.append(f"{order['order_id']}: fill rows != order filled quantity")
                if aggregate["gross"] != D(order["gross_idr"],"order gross",allow_zero=True): issues.append(f"{order['order_id']}: fill gross != order gross")
                if aggregate["fees"] != D(order["fees_idr"],"order fees",allow_zero=True): issues.append(f"{order['order_id']}: fill fees != order fees")
            # Snapshot-consumption ledger must never exceed captured liquidity and must equal fills.
            consumed_rows=db.execute("SELECT snapshot_id,side,level_index,reserved_qty,consumed_qty FROM snapshot_consumption").fetchall()
            for c in consumed_rows:
                level=db.execute("SELECT quantity FROM snapshot_levels WHERE snapshot_id=? AND side=? AND level_index=?",
                                 (c["snapshot_id"],c["side"],c["level_index"])).fetchone()
                if level is None:
                    issues.append(f"orphan snapshot consumption {c['snapshot_id']}:{c['side']}:{c['level_index']}")
                    continue
                original=D(level["quantity"],"original snapshot liquidity")
                consumed=D(c["consumed_qty"],"consumed snapshot liquidity",allow_zero=True)
                if c["side"] == "ask":
                    fill_rows=db.execute("SELECT quantity FROM fills WHERE snapshot_id=? AND level_index=? AND side='BUY'",
                                         (c["snapshot_id"],c["level_index"])).fetchall()
                else:
                    fill_rows=db.execute("SELECT quantity FROM fills WHERE snapshot_id=? AND level_index=? AND side='SELL'",
                                         (c["snapshot_id"],c["level_index"])).fetchall()
                fills_used=sum((D(row["quantity"],"fill quantity") for row in fill_rows),Decimal("0"))
                reserved=D(c["reserved_qty"],"reserved snapshot liquidity",allow_zero=True)
                if consumed + reserved > original: issues.append(f"snapshot liquidity over-allocated {c['snapshot_id']}:{c['side']}:{c['level_index']}")
                if fills_used != consumed: issues.append(f"snapshot consumption differs from fill journal {c['snapshot_id']}:{c['side']}:{c['level_index']}")
            pools=db.execute("SELECT pool_id,venue,symbol,side FROM liquidity_pools ORDER BY pool_id").fetchall()
            for pool in pools:
                pool_levels=db.execute("SELECT level_index,price,quantity,reserved_qty,consumed_qty FROM liquidity_pool_levels WHERE pool_id=? ORDER BY level_index",
                                       (pool["pool_id"],)).fetchall()
                for level in pool_levels:
                    original=D(level["quantity"],"pool original quantity")
                    consumed=D(level["consumed_qty"],"pool consumed",allow_zero=True)
                    reserved=D(level["reserved_qty"],"pool reserved",allow_zero=True)
                    if consumed+reserved>original:
                        issues.append(f"shared depth pool over-allocated {pool['pool_id']}:{level['level_index']}")
                    fill_side="BUY" if pool["side"]=="ask" else "SELL"
                    rows=db.execute("SELECT f.quantity FROM fills f JOIN snapshot_liquidity_pools m ON m.snapshot_id=f.snapshot_id AND m.side=? WHERE m.pool_id=? AND f.level_index=? AND f.side=?",
                                    (pool["side"],pool["pool_id"],level["level_index"],fill_side)).fetchall()
                    used=sum((D(x["quantity"],"pool fill quantity") for x in rows),Decimal("0"))
                    if used!=consumed:
                        issues.append(f"shared depth pool fill rows differ from consumed qty {pool['pool_id']}:{level['level_index']}")
            for order in orders:
                st=OrderState(order["state"])
                requested=D(order["requested_qty"],"requested")
                filled_qty=D(order["filled_qty"],"filled",allow_zero=True)
                remaining_qty=D(order["remaining_qty"],"remaining",allow_zero=True)
                if st is OrderState.FILLED_SIMULATED and remaining_qty != 0: issues.append(f"{order['order_id']}: filled state with residual")
                if st is OrderState.PARTIALLY_FILLED_SIMULATED and not (0 < filled_qty < requested and remaining_qty > 0): issues.append(f"{order['order_id']}: invalid partial state quantities")
                if st is OrderState.OPEN_SIMULATED and remaining_qty <= 0: issues.append(f"{order['order_id']}: open state without residual")
            stored={k:Decimal(account[k]) for k in ("cash_idr","sol_quantity","avg_entry_cost_idr","realized_pnl_idr","fees_paid_idr")}
            reconstructed={"cash_idr":cash,"sol_quantity":sol,"avg_entry_cost_idr":avg,"realized_pnl_idr":realized,"fees_paid_idr":fees}
            for key,value in reconstructed.items():
                if stored[key] != value: issues.append(f"portfolio {key} differs from fill replay")
            return {"consistent":not issues,"issues":issues,"order_count":len(orders),"fill_count":len(fills),
                    "stored_portfolio":{k:str(v) for k,v in stored.items()},
                    "replayed_portfolio":{k:str(v) for k,v in reconstructed.items()},
                    "event_count":db.execute("SELECT COUNT(*) FROM events").fetchone()[0],
                    "snapshot_count":db.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]}

    def recover(self) -> dict[str, Any]:
        """Validate durable state after restart. Atomic transactions leave no half-cycle."""
        report=self.reconcile()
        if not report["consistent"]:
            raise LedgerIntegrityError("recovery reconciliation failed: " + "; ".join(report["issues"]))
        return report

    def snapshot_liquidity(self, snapshot_id: str) -> list[dict[str, str]]:
        with closing(self._connect()) as db:
            rows=db.execute("""SELECT l.side,l.level_index,l.price,l.quantity,
                c.consumed_qty snapshot_consumed_qty,p.reserved_qty,p.consumed_qty shared_consumed_qty,
                p.quantity shared_original_qty
                FROM snapshot_levels l
                JOIN snapshot_consumption c USING(snapshot_id,side,level_index)
                JOIN snapshot_liquidity_pools m ON m.snapshot_id=l.snapshot_id AND m.side=l.side
                JOIN liquidity_pool_levels p ON p.pool_id=m.pool_id AND p.level_index=l.level_index
                WHERE l.snapshot_id=? ORDER BY l.side,l.level_index""",(snapshot_id,)).fetchall()
            result=[]
            for row in rows:
                item=dict(row)
                item["consumed_qty"]=row["shared_consumed_qty"]
                item["remaining_qty"]=str(D(row["shared_original_qty"],"pool original")-
                    D(row["reserved_qty"],"pool reserved",allow_zero=True)-D(row["shared_consumed_qty"],"pool consumed",allow_zero=True))
                result.append(item)
            return result



# Operator-facing alias.
DepthPaperExecutionEngine = SQLitePaperExecutionLedger