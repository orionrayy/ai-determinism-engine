#!/usr/bin/env python3
"""Read-only INDODAX SOL/IDR depth sampler with an append-only SQLite archive."""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
import math
import sqlite3
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

ENDPOINT = "https://indodax.com/api/depth/solidr"
VERSION = "indodax-depth-collector-v1"
MAX_RESPONSE_BYTES = 1_000_000
MAX_LEVELS = 500
MAX_SAMPLES = 10_000


class MicrostructureError(ValueError):
    """Invalid public market data or collector configuration."""


def dec(value: Any, field: str, *, zero: bool = False) -> Decimal:
    if isinstance(value, bool):
        raise MicrostructureError(f"{field} must be finite decimal")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        raise MicrostructureError(f"{field} must be finite decimal") from None
    if not result.is_finite() or (result < 0 if zero else result <= 0):
        raise MicrostructureError(f"{field} must be {'non-negative' if zero else 'positive'}")
    return result


def utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise MicrostructureError("observation timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class Level:
    price: Decimal
    quantity: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(self, "price", dec(self.price, "price"))
        object.__setattr__(self, "quantity", dec(self.quantity, "quantity"))


@dataclass(frozen=True)
class Snapshot:
    observed_at: datetime
    bids: tuple[Level, ...]
    asks: tuple[Level, ...]
    exchange_timestamp_ms: int | None
    book_hash: str

    @property
    def bid(self) -> Decimal:
        return self.bids[0].price

    @property
    def ask(self) -> Decimal:
        return self.asks[0].price

    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / Decimal(2)

    @property
    def spread_bps(self) -> Decimal:
        return (self.ask - self.bid) / self.mid * Decimal(10_000)


def parse_depth(payload: Any, observed_at: datetime | None = None, *, max_levels: int = MAX_LEVELS) -> Snapshot:
    """Parse public API 'buy'/'sell' arrays of [price, amount] levels."""
    if isinstance(max_levels, bool) or not isinstance(max_levels, int) or not 1 <= max_levels <= MAX_LEVELS:
        raise MicrostructureError(f"max_levels must be 1..{MAX_LEVELS}")
    if not isinstance(payload, dict):
        raise MicrostructureError("depth response must be a JSON object")
    sides: list[tuple[Level, ...]] = []
    for api_side, label in (("buy", "bid"), ("sell", "ask")):
        rows = payload.get(api_side)
        if not isinstance(rows, list) or not rows:
            raise MicrostructureError(f"{api_side} side must be a non-empty array")
        levels = []
        for i, row in enumerate(rows[:max_levels]):
            if not isinstance(row, (list, tuple)) or len(row) != 2:
                raise MicrostructureError(f"malformed {label} level {i}")
            levels.append(Level(dec(row[0], f"{label} price"), dec(row[1], f"{label} quantity")))
        if any((levels[i].price <= levels[i+1].price if label == "bid" else levels[i].price >= levels[i+1].price) for i in range(len(levels)-1)):
            raise MicrostructureError(f"{label} levels must be strictly {'descending' if label == 'bid' else 'ascending'}")
        sides.append(tuple(levels))
    bids, asks = sides
    if bids[0].price > asks[0].price:
        raise MicrostructureError("crossed order book")
    ts = payload.get("timestamp")
    if ts is not None:
        if isinstance(ts, bool):
            raise MicrostructureError("exchange timestamp must be integer milliseconds")
        try:
            dts = Decimal(str(ts))
            if not dts.is_finite() or dts != dts.to_integral_value() or dts <= 0:
                raise ValueError
            ts = int(dts)
        except (InvalidOperation, ValueError):
            raise MicrostructureError("exchange timestamp must be integer milliseconds") from None
    canonical = {"buy": [[str(x.price), str(x.quantity)] for x in bids], "sell": [[str(x.price), str(x.quantity)] for x in asks]}
    digest = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return Snapshot(utc(observed_at or datetime.now(timezone.utc)), bids, asks, ts, digest)


class DepthClient:
    """Bounded unauthenticated GET client. At least 1 second between samples."""
    def __init__(self, *, timeout: float = 5.0, attempts: int = 2, interval_seconds: float = 1.0, transport=None):
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0.1 <= timeout <= 20:
            raise MicrostructureError("timeout must be between 0.1 and 20 seconds")
        if isinstance(attempts, bool) or not isinstance(attempts, int) or not 1 <= attempts <= 3:
            raise MicrostructureError("attempts must be 1..3")
        if isinstance(interval_seconds, bool) or not isinstance(interval_seconds, (int, float)) or interval_seconds < 1:
            raise MicrostructureError("interval_seconds must be >= 1 second")
        self.timeout, self.attempts, self.interval = float(timeout), attempts, float(interval_seconds)
        self.transport = transport or self._get_json
        self.last_request: float | None = None

    @staticmethod
    def _get_json(url: str, timeout: float) -> Any:
        req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "SOL-Sentinel-ReadOnly/1.0"}, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if response.status != 200 or len(raw) > MAX_RESPONSE_BYTES:
                raise MicrostructureError("INDODAX depth HTTP error or response exceeds 1 MB")
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise MicrostructureError("INDODAX depth response is invalid JSON") from None

    def snapshot(self) -> Snapshot:
        if self.last_request is not None:
            time.sleep(max(0.0, self.interval - (time.monotonic() - self.last_request)))
        self.last_request = time.monotonic()
        last = None
        for attempt in range(self.attempts):
            try:
                return parse_depth(self.transport(ENDPOINT, self.timeout))
            except urllib.error.HTTPError as exc:
                last = exc
                if exc.code not in (408, 425, 429, 500, 502, 503, 504):
                    raise MicrostructureError(f"INDODAX depth HTTP {exc.code}") from None
            except (urllib.error.URLError, TimeoutError, OSError, MicrostructureError) as exc:
                last = exc
                if isinstance(exc, MicrostructureError) and "invalid JSON" not in str(exc):
                    raise
            if attempt + 1 < self.attempts:
                time.sleep(0.25 * (2 ** attempt))
        raise MicrostructureError(f"depth request failed after {self.attempts} attempt(s): {type(last).__name__ if last else 'unknown'}") from None


class DepthStore:
    """SQLite append-only snapshot and level journal; no historical backfill inference."""
    def __init__(self, path: str | Path):
        self.path = str(Path(path).expanduser())
        if self.path == ":memory:":
            raise MicrostructureError("use a file-backed database for durable depth collection")
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=5)) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
            CREATE TABLE IF NOT EXISTS observations(
              id TEXT PRIMARY KEY, observed_at TEXT NOT NULL, exchange_ts_ms INTEGER,
              best_bid TEXT NOT NULL, best_bid_qty TEXT NOT NULL, best_ask TEXT NOT NULL,
              best_ask_qty TEXT NOT NULL, midpoint TEXT NOT NULL, spread_bps TEXT NOT NULL,
              bid_levels INTEGER NOT NULL, ask_levels INTEGER NOT NULL, book_hash TEXT NOT NULL,
              endpoint TEXT NOT NULL, collector_version TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS observations_time ON observations(observed_at);
            CREATE TABLE IF NOT EXISTS levels(
              observation_id TEXT NOT NULL REFERENCES observations(id), side TEXT NOT NULL,
              level_index INTEGER NOT NULL, price TEXT NOT NULL, quantity TEXT NOT NULL,
              PRIMARY KEY(observation_id,side,level_index));
            CREATE TABLE IF NOT EXISTS attempts(
              id INTEGER PRIMARY KEY AUTOINCREMENT, observed_at TEXT NOT NULL,
              status TEXT NOT NULL CHECK(status IN ('OK','ERROR')), duration_ms INTEGER NOT NULL,
              error_class TEXT, error_detail TEXT);
            """)

    def append(self, snapshot: Snapshot) -> tuple[str, bool]:
        observed = snapshot.observed_at.isoformat(timespec="microseconds")
        key = hashlib.sha256(f"INDODAX|SOL/IDR|{observed}|{snapshot.book_hash}|{VERSION}".encode()).hexdigest()
        with closing(sqlite3.connect(self.path, timeout=5)) as db:
            db.execute("PRAGMA foreign_keys=ON")
            with db:
                cur = db.execute("""INSERT OR IGNORE INTO observations VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                    key, observed, snapshot.exchange_timestamp_ms, str(snapshot.bid), str(snapshot.bids[0].quantity),
                    str(snapshot.ask), str(snapshot.asks[0].quantity), str(snapshot.mid), str(snapshot.spread_bps),
                    len(snapshot.bids), len(snapshot.asks), snapshot.book_hash, ENDPOINT, VERSION))
                inserted = cur.rowcount == 1
                if inserted:
                    db.executemany("INSERT INTO levels VALUES(?,?,?,?,?)", [
                        (key, side, i, str(level.price), str(level.quantity))
                        for side, levels in (("bid", snapshot.bids), ("ask", snapshot.asks)) for i, level in enumerate(levels)])
        return key, inserted

    def attempt(self, observed_at: datetime, duration_ms: int, error: Exception | None = None) -> None:
        with closing(sqlite3.connect(self.path, timeout=5)) as db, db:
            db.execute("INSERT INTO attempts(observed_at,status,duration_ms,error_class,error_detail) VALUES(?,?,?,?,?)", (
                utc(observed_at).isoformat(timespec="microseconds"), "ERROR" if error else "OK", max(0, duration_ms),
                type(error).__name__ if error else None, str(error)[:240] if error else None))

    @staticmethod
    def _dist(db, query: str, params=()):
        n, mean, mean_sq = db.execute(f"SELECT COUNT(*),AVG(v),AVG(v*v) FROM ({query})", params).fetchone()
        if not n:
            return {"mean": None, "median": None, "p95_nearest_rank": None, "population_stddev": None}
        mid = (n - 1) // 2
        pair = db.execute(f"SELECT v FROM ({query}) ORDER BY v LIMIT 2 OFFSET ?", (*params, mid)).fetchall()
        median = float(pair[0][0]) if n % 2 else (float(pair[0][0]) + float(pair[1][0])) / 2
        p95 = db.execute(f"SELECT v FROM ({query}) ORDER BY v LIMIT 1 OFFSET ?", (*params, math.ceil(.95*n)-1)).fetchone()[0]
        return {"mean": float(mean), "median": median, "p95_nearest_rank": float(p95), "population_stddev": math.sqrt(max(0.0, float(mean_sq)-float(mean)**2))}

    def summary(self, band_bps: float = 25.0) -> dict[str, Any]:
        if isinstance(band_bps, bool) or not isinstance(band_bps, (int, float)) or not math.isfinite(band_bps) or band_bps < 0:
            raise MicrostructureError("band_bps must be finite and non-negative")
        with closing(sqlite3.connect(self.path, timeout=5)) as db:
            n, first, last = db.execute("SELECT COUNT(*),MIN(observed_at),MAX(observed_at) FROM observations").fetchone()
            spread = self._dist(db, "SELECT CAST(spread_bps AS REAL) v FROM observations")
            depth = """WITH mids AS (SELECT id,(CAST(best_bid AS REAL)+CAST(best_ask AS REAL))/2 m FROM observations),
              vals AS (SELECT l.observation_id,
                SUM(CASE WHEN l.side='bid' AND CAST(l.price AS REAL)>=mids.m*(1-?/10000) THEN CAST(l.price AS REAL)*CAST(l.quantity AS REAL) ELSE 0 END) bid_v,
                SUM(CASE WHEN l.side='ask' AND CAST(l.price AS REAL)<=mids.m*(1+?/10000) THEN CAST(l.price AS REAL)*CAST(l.quantity AS REAL) ELSE 0 END) ask_v
                FROM levels l JOIN mids ON mids.id=l.observation_id GROUP BY l.observation_id) """
            bid = self._dist(db, depth + "SELECT bid_v v FROM vals", (band_bps, band_bps))
            ask = self._dist(db, depth + "SELECT ask_v v FROM vals", (band_bps, band_bps))
            total, ok, errors, latency = db.execute("SELECT COUNT(*),SUM(status='OK'),SUM(status='ERROR'),AVG(duration_ms) FROM attempts").fetchone()
        return {"schema_version":"sol-sentinel-depth-summary-v1", "venue":"INDODAX", "pair":"SOL/IDR", "collector_version":VERSION,
          "observation_count":n, "first_observed_at_utc":first, "last_observed_at_utc":last, "spread_bps":spread,
          "depth_band_bps":band_bps, "bid_quote_depth_idr":bid, "ask_quote_depth_idr":ask,
          "attempts":{"total":total or 0,"ok":ok or 0,"errors":errors or 0,"mean_duration_ms":latency},
          "historical_depth_warning":"Statistics include only snapshots actually captured by this collector; candle history is not used to infer past order-book depth."}


def collect(client: DepthClient, store: DepthStore, *, samples: int, interval_seconds: float, max_errors: int = 3) -> dict[str, Any]:
    if isinstance(samples, bool) or not isinstance(samples, int) or not 1 <= samples <= MAX_SAMPLES:
        raise MicrostructureError(f"samples must be 1..{MAX_SAMPLES}")
    if isinstance(interval_seconds, bool) or not isinstance(interval_seconds, (int,float)) or interval_seconds < 1:
        raise MicrostructureError("interval_seconds must be >= 1")
    if isinstance(max_errors, bool) or not isinstance(max_errors, int) or max_errors < 1:
        raise MicrostructureError("max_errors must be positive integer")
    started = time.monotonic(); inserted = duplicates = errors = 0
    for i in range(samples):
        observed = datetime.now(timezone.utc); begin = time.monotonic()
        try:
            snap = client.snapshot()
            _, was_new = store.append(snap)
            store.attempt(observed, int((time.monotonic()-begin)*1000))
            inserted += int(was_new); duplicates += int(not was_new); errors = 0
        except (MicrostructureError, OSError, TimeoutError, urllib.error.URLError) as exc:
            store.attempt(observed, int((time.monotonic()-begin)*1000), exc)
            errors += 1
            if errors >= max_errors:
                break
        if i + 1 < samples:
            time.sleep(max(0.0, interval_seconds - (time.monotonic()-begin)))
    result = store.summary()
    result.update({"requested_samples":samples,"new_observations_this_run":inserted,"duplicate_observations_this_run":duplicates,
       "consecutive_errors_at_end":errors,"collection_duration_seconds":round(time.monotonic()-started,3),
       "read_only":True,"authenticated":False,"orders_enabled":False,"network_accessed":True})
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=int, default=30)
    p.add_argument("--interval-seconds", type=float, default=10.0)
    p.add_argument("--timeout", type=float, default=5.0)
    p.add_argument("--database", default="~/.sol-sentinel/indodax_depth.sqlite3")
    p.add_argument("--max-errors", type=int, default=3)
    p.add_argument("--summary-only", action="store_true")
    a = p.parse_args(argv)
    try:
        store = DepthStore(a.database)
        if a.summary_only:
            result = store.summary(); result.update({"read_only":True,"authenticated":False,"orders_enabled":False,"network_accessed":False})
        else:
            result = collect(DepthClient(timeout=a.timeout, interval_seconds=max(1.0,a.interval_seconds)), store,
                             samples=a.samples, interval_seconds=a.interval_seconds, max_errors=a.max_errors)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["observation_count"] else 2
    except (MicrostructureError, OSError, sqlite3.Error) as exc:
        print(json.dumps({"status":"ERROR","error":f"{type(exc).__name__}: {str(exc)[:240]}","read_only":True,"orders_enabled":False}), file=__import__('sys').stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
