#!/usr/bin/env python3
"""Private input transport protocol for the orchestration control plane.

The repository/Actions boundary carries only an opaque reference and a digest.
Raw structured input is stored/fetched through a separately authenticated HTTPS
private-input service.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

PROTOCOL = "ai-orchestrator.private-input/v2"
MAX_BODY_BYTES = 128 * 1024
MAX_INPUT_BYTES = 96 * 1024
MAX_TTL_SECONDS = 7 * 24 * 60 * 60
DEFAULT_TTL_SECONDS = 24 * 60 * 60
REF_RE = re.compile(r"^[0-9a-f]{64}$")
DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
EXECUTION_ID_RE = re.compile(r"^[0-9a-f]{64}$")
FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_FETCH_404_RETRIES = 2


class PrivateInputError(RuntimeError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def input_digest(payload: dict[str, Any]) -> str:
    encoded = canonical_json(payload)
    if len(encoded) > MAX_INPUT_BYTES:
        raise PrivateInputError("private input exceeds 96 KiB")
    return hashlib.sha256(encoded).hexdigest()


def derive_input_ref(
    execution_id: str,
    digest: str,
    secret: str,
) -> str:
    execution = str(execution_id or "").strip()
    if not EXECUTION_ID_RE.fullmatch(execution):
        raise PrivateInputError("invalid private input execution identity")
    if not DIGEST_RE.fullmatch(str(digest or "")):
        raise PrivateInputError("invalid private input digest")
    message = f"{PROTOCOL}\n{execution}\n{digest}".encode("utf-8")
    return hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()


def private_input_config() -> tuple[str, str]:
    url = os.environ.get("ORCHESTRATOR_PRIVATE_INPUT_URL", "").strip()
    secret = os.environ.get("ORCHESTRATOR_PRIVATE_INPUT_SECRET", "")
    if not url:
        raise PrivateInputError("ORCHESTRATOR_PRIVATE_INPUT_URL is not configured")
    if not secret:
        raise PrivateInputError("ORCHESTRATOR_PRIVATE_INPUT_SECRET is not configured")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        raise PrivateInputError("private input transport HTTPS is required")
    if parsed.query or parsed.fragment:
        raise PrivateInputError("private input URL must not contain query or fragment")
    return url.rstrip("/"), secret


def request_signature(
    *,
    method: str,
    path: str,
    timestamp: int,
    body: bytes,
    secret: str,
) -> str:
    normalized_method = str(method or "GET").upper()
    normalized_path = str(path or "/")
    message = (
        normalized_method.encode("utf-8")
        + b"\n"
        + normalized_path.encode("utf-8")
        + b"\n"
        + str(timestamp).encode("utf-8")
        + b"\n"
        + body
    )
    return hmac.new(
        secret.encode("utf-8"),
        message,
        hashlib.sha256,
    ).hexdigest()


def _signed_headers(
    *,
    method: str,
    path: str,
    timestamp: int,
    body: bytes,
    secret: str,
) -> dict[str, str]:
    signature = request_signature(
        method=method,
        path=path,
        timestamp=timestamp,
        body=body,
        secret=secret,
    )
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "ai-orchestrator-private-input/1.0",
        "X-Orchestrator-Protocol": PROTOCOL,
        "X-Orchestrator-Timestamp": str(timestamp),
        "X-Orchestrator-Signature": "sha256=" + signature,
    }


def store_private_input(
    *,
    execution_id: str,
    intent_fingerprint: str,
    payload: dict[str, Any],
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> str:
    url, secret = private_input_config()
    if not FINGERPRINT_RE.fullmatch(str(intent_fingerprint or "")):
        raise PrivateInputError("invalid private input intent fingerprint")
    digest = input_digest(payload)
    ref = derive_input_ref(execution_id, digest, secret)
    try:
        ttl = int(ttl_seconds)
    except (TypeError, ValueError) as exc:
        raise PrivateInputError("private input TTL must be an integer") from exc
    if ttl < 300 or ttl > MAX_TTL_SECONDS:
        raise PrivateInputError(
            f"private input TTL must be between 300 and {MAX_TTL_SECONDS} seconds"
        )
    envelope = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "input_ref": ref,
        "execution_id": str(execution_id),
        "intent_fingerprint": str(intent_fingerprint),
        "input_digest": digest,
        "expires_at": int(time.time()) + ttl,
        "payload": payload,
    }
    body = canonical_json(envelope)
    if len(body) > MAX_BODY_BYTES:
        raise PrivateInputError("private input envelope exceeds 128 KiB")
    request_path = urllib.parse.urlparse(url + "/v1/inputs").path or "/"
    timestamp = int(time.time())
    request = urllib.request.Request(
        url + "/v1/inputs",
        data=body,
        headers={
            **_signed_headers(
                method="POST",
                path=request_path,
                timestamp=timestamp,
                body=body,
                secret=secret,
            ),
            "Idempotency-Key": ref,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read(MAX_BODY_BYTES + 1)
            status = response.status
    except urllib.error.HTTPError as exc:
        raise PrivateInputError(f"private input store returned HTTP {exc.code}") from exc
    except Exception as exc:
        raise PrivateInputError("private input store transport unavailable") from exc
    if not 200 <= status < 300:
        raise PrivateInputError(f"private input store returned HTTP {status}")
    try:
        result = json.loads(raw.decode("utf-8", "replace")) if raw else {}
    except json.JSONDecodeError as exc:
        raise PrivateInputError("private input store returned invalid JSON") from exc
    if not isinstance(result, dict) or result.get("ok") is not True:
        raise PrivateInputError("private input store rejected the input")
    returned_ref = str(result.get("input_ref") or "").strip()
    if returned_ref != ref:
        raise PrivateInputError("private input store returned a different input reference")
    for field, expected in (
        ("execution_id", str(execution_id)),
        ("input_digest", digest),
        ("intent_fingerprint", str(intent_fingerprint)),
    ):
        actual = str(result.get(field) or "").strip()
        if actual and actual != expected:
            raise PrivateInputError(f"private input store {field} mismatch")
    return ref


def fetch_private_input(
    *,
    input_ref: str,
    execution_id: str,
    expected_digest: str,
    expected_intent_fingerprint: str | None = None,
) -> dict[str, Any]:
    url, secret = private_input_config()
    ref = str(input_ref or "").strip()
    if not REF_RE.fullmatch(ref):
        raise PrivateInputError("invalid private input reference")
    if not DIGEST_RE.fullmatch(str(expected_digest or "")):
        raise PrivateInputError("invalid private input digest")
    expected_ref = derive_input_ref(execution_id, expected_digest, secret)
    if not hmac.compare_digest(ref, expected_ref):
        raise PrivateInputError("private input reference does not match execution identity")
    request_url = url + "/v1/inputs/" + urllib.parse.quote(ref, safe="")
    request_path = urllib.parse.urlparse(request_url).path or "/"
    last_not_found = False
    for attempt in range(MAX_FETCH_404_RETRIES + 1):
        timestamp = int(time.time())
        body = b""
        request = urllib.request.Request(
            request_url,
            headers=_signed_headers(
                method="GET",
                path=request_path,
                timestamp=timestamp,
                body=body,
                secret=secret,
            ),
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read(MAX_BODY_BYTES + 1)
                status = response.status
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                last_not_found = True
                if attempt < MAX_FETCH_404_RETRIES:
                    time.sleep(0.25 * (attempt + 1))
                    continue
                break
            raise PrivateInputError(f"private input fetch returned HTTP {exc.code}") from exc
        except Exception as exc:
            raise PrivateInputError("private input transport unavailable") from exc
        break
    if last_not_found:
        raise PrivateInputError("private input was not found")
    if not 200 <= status < 300:
        raise PrivateInputError(f"private input fetch returned HTTP {status}")
    if len(raw) > MAX_BODY_BYTES:
        raise PrivateInputError("private input response exceeds 128 KiB")
    try:
        result = json.loads(raw.decode("utf-8", "replace")) if raw else {}
    except json.JSONDecodeError as exc:
        raise PrivateInputError("private input fetch returned invalid JSON") from exc
    if not isinstance(result, dict) or result.get("ok") is not True:
        raise PrivateInputError("private input fetch was rejected")
    payload = result.get("payload")
    if not isinstance(payload, dict):
        raise PrivateInputError("private input payload must be an object")
    returned_ref = str(result.get("input_ref") or "").strip()
    if returned_ref != ref:
        raise PrivateInputError("private input reference mismatch")
    returned_digest = str(result.get("input_digest") or "").strip()
    actual_digest = input_digest(payload)
    if returned_digest != expected_digest or actual_digest != expected_digest:
        raise PrivateInputError("private input digest mismatch")
    returned_execution_id = str(result.get("execution_id") or "").strip()
    if returned_execution_id != str(execution_id):
        raise PrivateInputError("private input execution identity mismatch")
    returned_fp = str(result.get("intent_fingerprint") or "").strip()
    if expected_intent_fingerprint:
        if returned_fp != expected_intent_fingerprint:
            raise PrivateInputError("private input intent fingerprint mismatch")
    elif not FINGERPRINT_RE.fullmatch(returned_fp):
        raise PrivateInputError("private input intent fingerprint is missing or invalid")
    expires_at_raw = result.get("expires_at")
    if not isinstance(expires_at_raw, int) or expires_at_raw <= int(time.time()):
        raise PrivateInputError("private input has expired")
    return payload


def delete_private_input(input_ref: str) -> bool:
    url, secret = private_input_config()
    ref = str(input_ref or "").strip()
    if not REF_RE.fullmatch(ref):
        raise PrivateInputError("invalid private input reference")
    request_url = url + "/v1/inputs/" + urllib.parse.quote(ref, safe="")
    request_path = urllib.parse.urlparse(request_url).path or "/"
    body = b""
    timestamp = int(time.time())
    request = urllib.request.Request(
        request_url,
        headers=_signed_headers(
            method="DELETE",
            path=request_path,
            timestamp=timestamp,
            body=body,
            secret=secret,
        ),
        method="DELETE",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            status = response.status
            response.read(MAX_BODY_BYTES + 1)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return True
        raise PrivateInputError(f"private input delete returned HTTP {exc.code}") from exc
    except Exception as exc:
        raise PrivateInputError("private input delete transport unavailable") from exc
    if not 200 <= status < 300:
        raise PrivateInputError(f"private input delete returned HTTP {status}")
    return True
