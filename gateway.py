#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.parse
import urllib.request

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", "10000"))
EXECUTION_ID_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
DOMAIN_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
OPERATION_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def intent_fingerprint(domain: str, operation: str, payload: dict) -> str:
    return hashlib.sha256(
        canonical_json({
            "domain": str(domain).strip(),
            "operation": str(operation).strip().lower(),
            "input": payload,
        })
    ).hexdigest()


def derive_execution_id(
    event_id: str,
    domain: str,
    operation: str,
    fingerprint: str,
) -> str:
    return hashlib.sha256(
        canonical_json({
            "schema_version": 1,
            "event_id": str(event_id).strip(),
            "domain": str(domain).strip(),
            "operation": str(operation).strip().lower(),
            "intent_fingerprint": fingerprint,
        })
    ).hexdigest()


def parse_content_length(value: str | None) -> int:
    try:
        length = int(value or "0")
    except (TypeError, ValueError) as exc:
        raise ValueError("content_length_invalid") from exc
    if length <= 0 or length > 128 * 1024:
        raise ValueError("payload size must be between 1 byte and 128 KiB")
    return length


def build_execution_event(payload: dict) -> tuple[str, dict, str]:
    domain = str(payload.get("domain") or "").strip()
    operation = str(payload.get("operation") or "").strip().lower()
    if not DOMAIN_RE.fullmatch(domain):
        raise ValueError("domain_invalid")
    if not OPERATION_RE.fullmatch(operation):
        raise ValueError("operation_invalid")

    request_input = payload.get("payload")
    if request_input is None:
        request_input = payload.get("input") or {}
    if not isinstance(request_input, dict):
        raise ValueError("payload/input must be an object")
    if len(request_input) > 256:
        raise ValueError("input_has_too_many_properties")

    idempotency_key = str(
        payload.get("idempotency_key")
        or payload.get("event_id")
        or payload.get("request_id")
        or ""
    ).strip()
    if not idempotency_key:
        idempotency_key = hashlib.sha256(
            canonical_json({
                "domain": domain,
                "operation": operation,
                "input": request_input,
            })
        ).hexdigest()
    if len(idempotency_key) > 128:
        raise ValueError("idempotency_key_too_long")

    event_id = str(payload.get("event_id") or idempotency_key).strip()
    if not event_id or len(event_id) > 128:
        raise ValueError("event_id_invalid")

    expected_fp = intent_fingerprint(domain, operation, request_input)
    supplied_fp = str(payload.get("intent_fingerprint") or "").strip()
    if supplied_fp:
        if not FINGERPRINT_RE.fullmatch(supplied_fp):
            raise ValueError("intent_fingerprint_invalid")
        if supplied_fp != expected_fp:
            raise ValueError("intent_fingerprint_mismatch")

    expected_digest = hashlib.sha256(canonical_json(request_input)).hexdigest()
    supplied_digest = str(payload.get("input_digest") or "").strip()
    if supplied_digest:
        if not FINGERPRINT_RE.fullmatch(supplied_digest):
            raise ValueError("input_digest_invalid")
        if supplied_digest != expected_digest:
            raise ValueError("input_digest_mismatch")

    derived_execution_id = derive_execution_id(
        event_id,
        domain,
        operation,
        expected_fp,
    )
    supplied_execution_id = str(payload.get("execution_id") or "").strip()
    execution_id = supplied_execution_id or derived_execution_id
    if not EXECUTION_ID_RE.fullmatch(execution_id):
        raise ValueError("execution_id_invalid")

    workflow_id = str(payload.get("workflow_id") or execution_id).strip()
    if not SAFE_ID_RE.fullmatch(workflow_id):
        raise ValueError("workflow_id_invalid")

    parent_execution_id = payload.get("parent_execution_id")
    if parent_execution_id is not None:
        parent_execution_id = str(parent_execution_id).strip() or None
        if parent_execution_id and (
            len(parent_execution_id) > 128
            or not SAFE_ID_RE.fullmatch(parent_execution_id)
        ):
            raise ValueError("parent_execution_id_invalid")

    try:
        attempt = int(payload.get("attempt") or 1)
    except (TypeError, ValueError):
        raise ValueError("attempt_invalid")
    if attempt < 1 or attempt > 1000:
        raise ValueError("attempt_out_of_range")

    requested_mode = str(
        payload.get("requested_mode") or "dry-run"
    ).strip().lower()
    if requested_mode not in {"dry-run", "live"}:
        raise ValueError("requested_mode_invalid")
    if requested_mode == "live":
        raise ValueError("live_structured_requires_private_input_channel")

    source = str(payload.get("source") or "automation-core").strip()[:128]
    if not source:
        source = "automation-core"

    metadata = {
        "execution_id": execution_id,
        "parent_execution_id": parent_execution_id,
        "workflow_id": workflow_id,
        "domain": domain,
        "operation": operation,
        "intent_fingerprint": expected_fp,
        "input_digest": expected_digest,
        "attempt": attempt,
        "source": source,
        "requested_mode": requested_mode,
        "idempotency_key": idempotency_key,
    }
    goal = f"Execute orchestration operation {domain}.{operation}"
    return goal, metadata, event_id


def github_dispatch(goal: str, metadata: dict, event_id: str | None = None) -> dict:
    token = os.environ.get("GITHUB_GATEWAY_TOKEN")
    repository = os.environ.get(
        "GITHUB_REPOSITORY",
        "orionrayy/ai-determinism-engine",
    )
    if not token:
        raise RuntimeError("GITHUB_GATEWAY_TOKEN is not configured")
    url = f"https://api.github.com/repos/{repository}/dispatches"
    client_payload = {"goal": goal, "metadata": metadata}
    if event_id:
        client_payload["event_id"] = event_id
    for field in (
        "execution_id",
        "parent_execution_id",
        "workflow_id",
        "domain",
        "operation",
        "intent_fingerprint",
        "input_digest",
        "attempt",
        "requested_mode",
        "idempotency_key",
    ):
        if field in metadata and metadata[field] not in (None, ""):
            client_payload[field] = metadata[field]
    payload = json.dumps({
        "event_type": "orchestrator.event",
        "client_payload": client_payload,
    }).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
            "User-Agent": "ai-orchestrator-gateway/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return {"github_status": response.status}


def normalized_headers(headers: dict[str, str]) -> dict[str, str]:
    return {
        str(key).lower(): str(value).strip()
        for key, value in headers.items()
    }


def authorized(
    headers: dict[str, str],
    raw_body: bytes | None = None,
) -> bool:
    configured = os.environ.get("GATEWAY_SHARED_SECRET")
    if not configured:
        return False
    normalized = normalized_headers(headers)
    supplied = normalized.get("authorization", "")
    expected = "Bearer " + configured
    if secrets.compare_digest(supplied, expected):
        return True

    if raw_body is None:
        return False
    timestamp = normalized.get("x-orchestrator-timestamp", "")
    signature = normalized.get("x-orchestrator-signature", "")
    try:
        ts = int(timestamp)
    except ValueError:
        return False
    if abs(int(time.time()) - ts) > 300:
        return False
    signed = timestamp.encode("utf-8") + b"\n" + raw_body
    expected_sig = hmac.new(
        configured.encode("utf-8"),
        signed,
        hashlib.sha256,
    ).hexdigest()
    return secrets.compare_digest(signature, "sha256=" + expected_sig)


class Handler(BaseHTTPRequestHandler):
    server_version = "AIOrchestratorGateway/1.0"

    def _send(self, code: int, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        print("%s - %s" % (self.address_string(), format % args), flush=True)

    def do_GET(self):
        path = urllib.parse.urlsplit(self.path).path
        if path == "/health":
            self._send(200, {
                "ok": True,
                "service": "ai-orchestrator-gateway",
                "configured": {
                    "github_token": bool(os.environ.get("GITHUB_GATEWAY_TOKEN")),
                    "shared_secret": bool(os.environ.get("GATEWAY_SHARED_SECRET")),
                },
            })
            return
        self._send(404, {"ok": False, "error": "not_found"})

    def do_POST(self):
        path = urllib.parse.urlsplit(self.path).path
        if path != "/event":
            self._send(404, {"ok": False, "error": "not_found"})
            return

        try:
            length = parse_content_length(self.headers.get("Content-Length"))
            raw = self.rfile.read(length)
            if not authorized(dict(self.headers.items()), raw):
                self._send(401, {"ok": False, "error": "unauthorized"})
                return

            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("request body must be an object")

            structured_call = any(
                key in payload
                for key in (
                    "domain",
                    "operation",
                    "execution_id",
                    "intent_fingerprint",
                    "idempotency_key",
                    "requested_mode",
                    "input_digest",
                )
            )
            if structured_call:
                goal, metadata, event_id = build_execution_event(payload)
            else:
                goal = str(payload.get("goal", "")).strip()
                if not goal:
                    raise ValueError("goal is required")
                if len(goal) > 4000:
                    raise ValueError("goal too long")
                metadata = payload.get("metadata", {})
                if not isinstance(metadata, dict):
                    metadata = {"value": str(metadata)}
                event_id = (
                    normalized_headers(dict(self.headers.items())).get("idempotency-key")
                    or str(payload.get("event_id") or "").strip()
                    or None
                )

            result = github_dispatch(goal, metadata, event_id=event_id)
            receipt = {"ok": True, "queued": True, **result}
            for field in (
                "request_id",
                "execution_id",
                "parent_execution_id",
                "workflow_id",
                "intent_fingerprint",
                "input_digest",
                "attempt",
                "requested_mode",
                "idempotency_key",
            ):
                if isinstance(metadata, dict) and field in metadata and metadata[field] not in (None, ""):
                    receipt[field] = metadata[field]
            if "request_id" not in receipt:
                receipt["request_id"] = str(event_id or "")
            self._send(202, receipt)
        except ValueError as exc:
            self._send(400, {"ok": False, "error": str(exc)})
        except Exception as exc:
            print(f"dispatch error: {exc}", flush=True)
            self._send(503, {"ok": False, "error": "dispatch_unavailable"})


if __name__ == "__main__":
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
