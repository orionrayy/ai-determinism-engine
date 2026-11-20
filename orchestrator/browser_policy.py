"""Policy preflight and evidence contracts for Playwright browser tasks.

This module is stdlib-only. It does not execute browser actions. A trusted
adapter must call it before dispatching each MCP action.
"""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping
from urllib.parse import urlsplit

PROTOCOL = "ai-orchestrator.browser/v1"
MAX_INPUT_BYTES = 64 * 1024
MAX_URL_LENGTH = 4096
IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
SHA256_RE = re.compile(r"^[a-f0-9]{64}$")

READ_ONLY_ACTIONS = frozenset({
    "navigate", "navigate_back", "navigate_forward", "snapshot", "find",
    "take_screenshot", "wait_for", "tabs",
})
INTERACTIVE_ACTIONS = frozenset({
    "click", "hover", "drag", "type", "fill_form", "select_option",
    "check", "uncheck", "press_key", "handle_dialog", "file_upload",
    "close",
})
ALLOWED_ACTIONS = READ_ONLY_ACTIONS | INTERACTIVE_ACTIONS
HIGH_RISK_EFFECTS = frozenset({
    "external_write", "financial", "destructive", "account_admin", "unknown",
})
EFFECTS = frozenset({
    "read_only", "reversible", "external_write", "financial",
    "destructive", "account_admin", "unknown",
})


class BrowserPolicyError(ValueError):
    """A browser task violates the local policy contract."""


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise BrowserPolicyError(f"input must be finite JSON data: {exc}") from exc


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_action(value: Any) -> str:
    action = str(value or "").strip().lower()
    if action.startswith("browser_"):
        action = action[len("browser_"):]
    if action not in ALLOWED_ACTIONS:
        raise BrowserPolicyError(f"unsupported Playwright MCP action: {action or '<empty>'}")
    return action


def validate_target_url(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str) or len(value) > MAX_URL_LENGTH:
        raise BrowserPolicyError("target_url must be a string no longer than 4096 characters")
    if any(ch.isspace() or ord(ch) < 32 for ch in value):
        raise BrowserPolicyError("target_url must not contain whitespace or control characters")
    try:
        parsed = urlsplit(value.strip())
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise BrowserPolicyError(f"malformed target_url: {exc}") from exc
    if parsed.scheme not in {"https", "http"} or not hostname:
        raise BrowserPolicyError("target_url must use HTTPS (HTTP is limited to loopback)")
    if parsed.username is not None or parsed.password is not None:
        raise BrowserPolicyError("embedded URL credentials are not permitted")
    if parsed.scheme == "http":
        loopback = hostname.lower() == "localhost"
        try:
            loopback = loopback or ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            pass
        if not loopback:
            raise BrowserPolicyError("plaintext HTTP is permitted only for localhost/loopback")
    if port is not None and not (1 <= port <= 65535):
        raise BrowserPolicyError("target_url port is outside the valid range")
    return value.strip()


@dataclass(frozen=True, slots=True)
class BrowserTask:
    workflow_id: str
    node_id: str
    action: str
    effect: str
    target_url: str | None = None
    input: Mapping[str, Any] | None = None
    protocol: str = PROTOCOL

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BrowserTask":
        if not isinstance(payload, Mapping):
            raise BrowserPolicyError("browser task must be an object")
        allowed = {
            "protocol", "workflow_id", "node_id", "action",
            "effect", "target_url", "input",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise BrowserPolicyError(
                f"unknown browser task fields: {', '.join(sorted(map(str, unknown)))}"
            )

        protocol = str(payload.get("protocol") or PROTOCOL)
        if protocol != PROTOCOL:
            raise BrowserPolicyError(f"unsupported browser task protocol: {protocol}")
        workflow_id = str(payload.get("workflow_id") or "").strip()
        node_id = str(payload.get("node_id") or "").strip()
        if not IDENTIFIER_RE.fullmatch(workflow_id):
            raise BrowserPolicyError("workflow_id is missing or invalid")
        if not IDENTIFIER_RE.fullmatch(node_id):
            raise BrowserPolicyError("node_id is missing or invalid")

        action = normalize_action(payload.get("action"))
        effect = str(payload.get("effect") or "").strip().lower()
        if effect not in EFFECTS:
            raise BrowserPolicyError("effect must be an explicit supported effect class")
        target_url = validate_target_url(payload.get("target_url"))
        if action == "navigate" and target_url is None:
            raise BrowserPolicyError("navigate requires target_url")

        task_input = payload.get("input", {})
        if not isinstance(task_input, Mapping):
            raise BrowserPolicyError("input must be an object")
        encoded_input = _canonical_json(dict(task_input)).encode("utf-8")
        if len(encoded_input) > MAX_INPUT_BYTES:
            raise BrowserPolicyError("browser input exceeds 64 KiB")
        return cls(
            workflow_id=workflow_id,
            node_id=node_id,
            action=action,
            effect=effect,
            target_url=target_url,
            input=dict(task_input),
            protocol=protocol,
        )

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "workflow_id": self.workflow_id,
            "node_id": self.node_id,
            "action": self.action,
            "effect": self.effect,
            "target_url": self.target_url,
            "input": dict(self.input or {}),
        }


def browser_task_digest(task: BrowserTask) -> str:
    """Digest exact task semantics; raw task input is not returned as evidence."""
    return _sha256(_canonical_json(task.semantic_payload()))


def evaluate_browser_task(
    task: BrowserTask,
    *,
    approved_intent_sha256: str | None = None,
) -> dict[str, Any]:
    """Return a fail-closed dispatch decision.

    approved_intent_sha256 must originate from the trusted supervisor's
    verified approval receipt, never from page content or a task payload.
    """
    digest = browser_task_digest(task)
    approval_required = task.effect in HIGH_RISK_EFFECTS
    approval_matches = (
        isinstance(approved_intent_sha256, str)
        and SHA256_RE.fullmatch(approved_intent_sha256) is not None
        and hmac.compare_digest(approved_intent_sha256, digest)
    )
    action_class = (
        "read_only" if task.action in READ_ONLY_ACTIONS
        else "file_io" if task.action == "file_upload"
        else "interactive"
    )
    allowed = not approval_required or approval_matches
    return {
        "protocol": PROTOCOL,
        "allowed": allowed,
        "workflow_id": task.workflow_id,
        "node_id": task.node_id,
        "action": task.action,
        "action_class": action_class,
        "effect": task.effect,
        "approval_required": approval_required,
        "approval_matches": approval_matches,
        "semantic_sha256": digest,
        "target_origin": target_origin(task.target_url),
        "reason": (
            "approved_intent_match" if approval_required and allowed
            else "trusted_approval_required" if approval_required
            else "policy_preflight_passed"
        ),
    }


def target_origin(value: str | None) -> str | None:
    """Return origin only; never retain path, query, or fragment in evidence."""
    if not value:
        return None
    parsed = urlsplit(value)
    host = (parsed.hostname or "").lower()
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    default_port = (
        (parsed.scheme == "https" and parsed.port == 443)
        or (parsed.scheme == "http" and parsed.port == 80)
    )
    netloc = host if parsed.port is None or default_port else f"{host}:{parsed.port}"
    return f"{parsed.scheme}://{netloc}"


def build_browser_evidence(
    task: BrowserTask,
    *,
    observed_at: str,
    outcome: str,
    output_sha256: str | None = None,
    screenshot_sha256: str | None = None,
) -> dict[str, Any]:
    """Build minimal metadata without page content or secret-bearing URLs."""
    if outcome not in {"succeeded", "failed", "uncertain"}:
        raise BrowserPolicyError("outcome must be succeeded, failed, or uncertain")
    try:
        parsed_timestamp = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise BrowserPolicyError("observed_at must be an ISO-8601 timestamp") from exc
    if parsed_timestamp.tzinfo is None:
        raise BrowserPolicyError("observed_at must include a timezone")

    for label, digest in (
        ("output_sha256", output_sha256),
        ("screenshot_sha256", screenshot_sha256),
    ):
        if digest is not None and (
            not isinstance(digest, str) or not SHA256_RE.fullmatch(digest)
        ):
            raise BrowserPolicyError(f"{label} must be a lowercase SHA-256 hex digest")

    body = {
        "protocol": PROTOCOL,
        "workflow_id": task.workflow_id,
        "node_id": task.node_id,
        "action": task.action,
        "target_origin": target_origin(task.target_url),
        "observed_at": observed_at,
        "outcome": outcome,
        "semantic_sha256": browser_task_digest(task),
        "output_sha256": output_sha256,
        "screenshot_sha256": screenshot_sha256,
    }
    return {"record_id": _sha256(_canonical_json(body)), **body}
