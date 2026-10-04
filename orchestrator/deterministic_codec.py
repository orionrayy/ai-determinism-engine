"""Canonical serialization and digest primitives.

All deterministic protocol hashes use one serializer so formatting changes cannot
silently produce different semantic digests across modules.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any, *, default: Any = None) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=default,
    ).encode("utf-8")


def digest(value: Any, *, default: Any = None) -> str:
    return hashlib.sha256(canonical_json(value, default=default)).hexdigest()
