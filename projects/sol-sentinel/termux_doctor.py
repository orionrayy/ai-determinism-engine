#!/usr/bin/env python3
"""Lightweight, read-only SOL Sentinel diagnostics for Android Termux and desktop Python.

The only network operation is an unauthenticated GET to INDODAX public pair metadata.
The script never calls a trading, wallet, signing, transfer, deposit, or withdrawal API.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PAIR_URL = "https://indodax.com/api/pairs"
USER_AGENT = "SOL-Sentinel-Termux-Doctor/1.0"
CORE_MODULES = (
    "sol_sentinel.py",
    "capture_history.py",
    "paper_engine.py",
    "evaluate_splits.py",
    "cost_sensitivity.py",
    "execution_adapters.py",
    "capture_indodax_history.py",
    "jupiter_quote_probe.py",
    "indodax_microstructure.py",
)


def check(status: str, name: str, detail: str) -> dict[str, str]:
    return {"status": status, "name": name, "detail": detail}


def parse_indodax_sol_idr_pairs(payload: Any) -> dict[str, str]:
    """Return compact public SOL/IDR market metadata; reject ambiguous matches."""
    if not isinstance(payload, list):
        raise ValueError("INDODAX /api/pairs response is not a JSON array")
    matches: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        pair_id = str(item.get("id", "")).strip().lower()
        symbol = str(item.get("symbol", "")).replace("/", "").replace("_", "").upper()
        description = str(item.get("description", "")).replace("/", "").replace("_", "").upper()
        if pair_id == "solidr" or symbol == "SOLIDR" or description == "SOLIDR":
            matches.append(item)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one SOL/IDR market; found {len(matches)}")
    item = matches[0]
    base_asset = str(item.get("traded_currency_unit") or item.get("traded_currency") or "").upper()
    quote_asset = str(item.get("base_currency") or "").upper()
    pair_id = str(item.get("id") or "").lower()
    if (base_asset, quote_asset, pair_id) != ("SOL", "IDR", "solidr"):
        raise ValueError("SOL/IDR pair metadata does not match canonical SOL, IDR, solidr")
    return {
        "pair_id": pair_id,
        "base_asset": base_asset,
        "quote_asset": quote_asset,
        "price_tick": str(item.get("price_precision") or item.get("pricescale") or ""),
        "quantity_step": str(item.get("quantity_increment") or ""),
        "minimum_quote_notional": str(item.get("trade_min_base_currency") or ""),
        "minimum_base_quantity": str(item.get("trade_min_traded_currency") or ""),
    }


def verify_native_dataset(csv_path: str | Path, expected_sha256: str | None = None) -> dict[str, Any]:
    """Stream-check native 1m OHLCV, timestamps, identity metadata, and SHA-256."""
    path = Path(csv_path).expanduser()
    if not path.is_file():
        raise ValueError(f"CSV file does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual_sha = digest.hexdigest()
    if expected_sha256 and actual_sha.lower() != expected_sha256.strip().lower():
        raise ValueError(f"CSV SHA-256 mismatch: got {actual_sha}")
    sidecar_path = Path(str(path) + ".meta.json")
    metadata: dict[str, Any] | None = None
    if sidecar_path.is_file():
        try:
            metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise ValueError("Dataset sidecar is unreadable JSON") from None
        if not isinstance(metadata, dict):
            raise ValueError("Dataset sidecar must be a JSON object")
        if metadata.get("data_status") != "OK":
            raise ValueError("Dataset sidecar data_status is not OK")
        if metadata.get("dataset_sha256") != actual_sha:
            raise ValueError("Dataset SHA-256 does not match sidecar")
        if (
            metadata.get("venue") != "INDODAX"
            or metadata.get("pair") != "SOL/IDR"
            or metadata.get("input_price_currency") != "IDR"
        ):
            raise ValueError("Dataset sidecar is not native INDODAX SOL/IDR")
        if metadata.get("fx_conversion_applied") is not False:
            raise ValueError("Native IDR sidecar must explicitly disable FX conversion")

    required = {"open_time_ms", "close_time_ms", "open", "high", "low", "close", "volume"}
    count = 0
    previous_open: int | None = None
    first_open: int | None = None
    last_close: int | None = None
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError("CSV is missing required OHLCV/time columns")
        for line_number, row in enumerate(reader, start=2):
            try:
                open_ms = int(row["open_time_ms"])
                close_ms = int(row["close_time_ms"])
                o, h, low, c, volume = (
                    float(row[key]) for key in ("open", "high", "low", "close", "volume")
                )
            except (TypeError, ValueError, KeyError, OverflowError):
                raise ValueError(f"Invalid numeric value at CSV line {line_number}") from None
            if close_ms != open_ms + 59_999:
                raise ValueError(f"Invalid 1m close boundary at CSV line {line_number}")
            if not all(math.isfinite(value) for value in (o, h, low, c, volume)):
                raise ValueError(f"Non-finite OHLCV value at CSV line {line_number}")
            if min(o, h, low, c) <= 0 or volume < 0:
                raise ValueError(f"Non-positive price or negative volume at CSV line {line_number}")
            if low > min(o, c) or max(o, c) > h or low > h:
                raise ValueError(f"OHLC ordering invariant failed at CSV line {line_number}")
            if previous_open is not None and open_ms - previous_open != 60_000:
                raise ValueError(f"Timestamp gap or duplicate at CSV line {line_number}")
            if first_open is None:
                first_open = open_ms
            previous_open = open_ms
            last_close = close_ms
            count += 1
    if count < 2:
        raise ValueError("Dataset must contain at least two contiguous candles")
    if metadata and metadata.get("count") not in (None, count):
        raise ValueError("CSV row count does not match sidecar")
    return {
        "path": str(path),
        "rows": count,
        "sha256": actual_sha,
        "first_open_time_ms": first_open,
        "last_close_time_ms": last_close,
        "sidecar_verified": metadata is not None,
        "venue": metadata.get("venue") if metadata else None,
        "pair": metadata.get("pair") if metadata else None,
        "fx_conversion_applied": metadata.get("fx_conversion_applied") if metadata else None,
    }


def sqlite_recovery_check(directory: str | Path | None = None) -> dict[str, Any]:
    """Exercise SQLite write, duplicate replay, close/reopen, and integrity checks."""
    temp_dir = tempfile.TemporaryDirectory(prefix="sol-sentinel-doctor-", dir=directory)
    db_path = Path(temp_dir.name) / "doctor.sqlite3"
    try:
        conn = sqlite3.connect(str(db_path), timeout=5.0)
        conn.execute(
            "CREATE TABLE events (event_id TEXT PRIMARY KEY, event_hash TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT OR IGNORE INTO events(event_id, event_hash) VALUES(?, ?)",
            ("doctor-run", "sha256:fixture"),
        )
        conn.commit()
        conn.execute(
            "INSERT OR IGNORE INTO events(event_id, event_hash) VALUES(?, ?)",
            ("doctor-run", "sha256:fixture"),
        )
        conn.commit()
        conn.close()

        reopened = sqlite3.connect(str(db_path), timeout=5.0)
        try:
            integrity = reopened.execute("PRAGMA integrity_check").fetchone()[0]
            rows = reopened.execute(
                "SELECT event_id, event_hash FROM events ORDER BY event_id"
            ).fetchall()
        finally:
            reopened.close()
        if integrity != "ok" or rows != [("doctor-run", "sha256:fixture")]:
            raise RuntimeError("SQLite recovery/idempotency invariant failed")
        return {"sqlite_version": sqlite3.sqlite_version, "integrity": integrity, "rows_after_replay": len(rows)}
    finally:
        temp_dir.cleanup()


def public_indodax_probe(timeout: float = 5.0) -> dict[str, str]:
    """Read only the unauthenticated public pair catalogue with bounded I/O."""
    if not math.isfinite(timeout) or timeout <= 0 or timeout > 15:
        raise ValueError("Network timeout must be between 0 and 15 seconds")
    request = urllib.request.Request(
        PAIR_URL,
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError(f"INDODAX public metadata returned HTTP {response.status}")
            raw = response.read(1_000_001)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"INDODAX public metadata returned HTTP {exc.code}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"INDODAX public metadata request failed: {type(exc).__name__}") from None
    if len(raw) > 1_000_000:
        raise RuntimeError("INDODAX public metadata response exceeded 1 MB")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RuntimeError("INDODAX public metadata returned invalid JSON") from None
    return parse_indodax_sol_idr_pairs(payload)


def inspect_runtime(project_root: Path, *, network: bool, timeout: float) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    version_ok = sys.version_info >= (3, 10)
    checks.append(check(
        "PASS" if version_ok else "FAIL",
        "python_version",
        f"{platform.python_version()} ({platform.python_implementation()}); required >= 3.10",
    ))
    is_termux = "com.termux" in os.environ.get("PREFIX", "") or Path("/data/data/com.termux").exists()
    checks.append(check(
        "PASS" if is_termux else "WARN",
        "termux_environment",
        f"Termux detected={is_termux}; platform={platform.system()} {platform.machine()}; "
        "a non-Termux result does not validate an Android device",
    ))
    required_stdlib = ("csv", "hashlib", "json", "sqlite3", "ssl", "urllib.request")
    missing = [name for name in required_stdlib if importlib.util.find_spec(name) is None]
    checks.append(check(
        "PASS" if not missing else "FAIL",
        "standard_library",
        "All required modules available" if not missing else f"Missing: {', '.join(missing)}",
    ))
    available_mb: float | None = None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                available_mb = int(line.split()[1]) / 1024
                break
    except (OSError, ValueError, IndexError):
        pass
    checks.append(check(
        "PASS" if available_mb is not None and available_mb >= 256 else "WARN",
        "memory",
        f"MemAvailable={available_mb:.0f} MiB" if available_mb is not None else "MemAvailable not exposed by this runtime",
    ))
    try:
        free_mb = shutil.disk_usage(Path.home()).free / (1024 * 1024)
        checks.append(check(
            "PASS" if free_mb >= 256 else "WARN",
            "disk_space",
            f"{free_mb:.0f} MiB free under home directory",
        ))
    except OSError as exc:
        checks.append(check("WARN", "disk_space", f"Cannot inspect home disk space: {type(exc).__name__}"))
    try:
        sql = sqlite_recovery_check()
        checks.append(check("PASS", "sqlite_recovery", json.dumps(sql, sort_keys=True)))
    except (OSError, sqlite3.Error, RuntimeError) as exc:
        checks.append(check("FAIL", "sqlite_recovery", f"{type(exc).__name__}: {exc}"))
    source_dir = project_root / "projects" / "sol-sentinel"
    compile_errors: list[str] = []
    for filename in CORE_MODULES + ("termux_doctor.py",):
        source_path = source_dir / filename
        if not source_path.is_file():
            compile_errors.append(f"missing:{filename}")
            continue
        try:
            compile(source_path.read_bytes(), str(source_path), "exec")
        except (OSError, SyntaxError, ValueError) as exc:
            compile_errors.append(f"{filename}:{type(exc).__name__}")
    checks.append(check(
        "PASS" if not compile_errors else "FAIL",
        "project_compile",
        "Core project modules compile in-memory" if not compile_errors else "; ".join(compile_errors),
    ))
    if network:
        try:
            market = public_indodax_probe(timeout)
            checks.append(check("PASS", "indodax_public_read_only", json.dumps(market, sort_keys=True)))
        except (OSError, RuntimeError, ValueError) as exc:
            checks.append(check("WARN", "indodax_public_read_only", f"{type(exc).__name__}: {exc}"))
    else:
        checks.append(check("SKIP", "indodax_public_read_only", "Skipped by --skip-network"))
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--data", default="", help="optional native INDODAX SOL/IDR CSV to stream-validate")
    parser.add_argument("--expected-sha256", default="", help="optional expected CSV SHA-256")
    parser.add_argument("--skip-network", action="store_true", help="do not contact public INDODAX metadata")
    parser.add_argument("--timeout", type=float, default=5.0, help="public GET timeout, maximum 15 seconds")
    parser.add_argument("--run-tests", action="store_true", help="run the offline unittest suite (maximum 180 seconds)")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)
    root = Path(args.project_root).expanduser().resolve()
    checks = inspect_runtime(root, network=not args.skip_network, timeout=args.timeout)
    if args.data:
        try:
            result = verify_native_dataset(args.data, args.expected_sha256 or None)
            checks.append(check("PASS", "native_dataset", json.dumps(result, sort_keys=True)))
        except (OSError, ValueError, csv.Error) as exc:
            checks.append(check("FAIL", "native_dataset", f"{type(exc).__name__}: {exc}"))
    if args.run_tests:
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "unittest", "discover", "-s", str(root / "projects" / "sol-sentinel"), "-p", "test_*.py"],
                cwd=str(root), capture_output=True, text=True, timeout=180, check=False,
            )
            tail = "\n".join((completed.stdout + "\n" + completed.stderr).splitlines()[-12:])
            checks.append(check(
                "PASS" if completed.returncode == 0 else "FAIL",
                "offline_unit_tests",
                f"exit={completed.returncode}; {tail[-1600:]}",
            ))
        except (OSError, subprocess.TimeoutExpired) as exc:
            checks.append(check("FAIL", "offline_unit_tests", f"{type(exc).__name__}: {exc}"))
    failed = any(item["status"] == "FAIL" for item in checks)
    summary = {
        "schema_version": "sol-sentinel-termux-doctor-v1",
        "mode": "READ_ONLY_DIAGNOSTICS",
        "execution_enabled": False,
        "network_effects": ["unauthenticated GET https://indodax.com/api/pairs"] if not args.skip_network else [],
        "checks": checks,
        "passed": sum(item["status"] == "PASS" for item in checks),
        "warnings": sum(item["status"] == "WARN" for item in checks),
        "failed": sum(item["status"] == "FAIL" for item in checks),
        "note": "A pass on another host does not prove compatibility on the user's Android/Termux device.",
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print("SOL Sentinel diagnostics — READ_ONLY_DIAGNOSTICS")
        for item in checks:
            print(f"[{item['status']}] {item['name']}: {item['detail']}")
        print(f"Summary: passed={summary['passed']} warnings={summary['warnings']} failed={summary['failed']}")
        print(summary["note"])
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
