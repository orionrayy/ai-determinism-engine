#!/usr/bin/env python3
"""Compatibility facade for the canonical orchestrator.private_input module."""
from __future__ import annotations

from orchestrator.private_input import (
    DEFAULT_TTL_SECONDS,
    DIGEST_RE,
    EXECUTION_ID_RE,
    FINGERPRINT_RE,
    MAX_BODY_BYTES,
    MAX_INPUT_BYTES,
    MAX_TTL_SECONDS,
    PROTOCOL,
    REF_RE,
    PrivateInputError,
    canonical_json,
    delete_private_input,
    derive_input_ref,
    fetch_private_input,
    input_digest,
    private_input_config,
    request_signature,
    store_private_input,
)

__all__ = [
    "DEFAULT_TTL_SECONDS",
    "DIGEST_RE",
    "EXECUTION_ID_RE",
    "FINGERPRINT_RE",
    "MAX_BODY_BYTES",
    "MAX_INPUT_BYTES",
    "MAX_TTL_SECONDS",
    "PROTOCOL",
    "REF_RE",
    "PrivateInputError",
    "canonical_json",
    "delete_private_input",
    "derive_input_ref",
    "fetch_private_input",
    "input_digest",
    "private_input_config",
    "request_signature",
    "store_private_input",
]
