from __future__ import annotations

import hashlib
import secrets
from typing import Any

PROTOCOL = "ai-orchestrator.native-worker/v1"


def execution_id(workflow_id: str, node_id: str) -> str:
    return hashlib.sha256(
        f"{workflow_id}:{node_id}".encode("utf-8")
    ).hexdigest()


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def verify_token(token: str, token_hash: str) -> bool:
    return secrets.compare_digest(hash_token(token), str(token_hash))


def create_task(
    workflow_id: str,
    node_id: str,
    connector: str,
    action: str,
    goal: str,
    node_input: dict[str, Any],
    callback_url: str,
    expires_at: int,
) -> tuple[dict[str, Any], str]:
    if not node_input.get("public_safe"):
        raise ValueError("native worker tasks require public_safe=true")
    if not callback_url.startswith("https://"):
        raise ValueError("native worker callback must use HTTPS")
    if expires_at <= 0:
        raise ValueError("expires_at must be positive")

    token = secrets.token_urlsafe(32)
    safe_input = {key: node_input[key] for key in ("public_safe", "instruction", "query", "payload") if key in node_input}
    task = {
        "protocol": PROTOCOL,
        "workflow_id": workflow_id,
        "node_id": node_id,
        "execution_id": execution_id(workflow_id, node_id),
        "connector": str(connector).strip().lower(),
        "action": str(action).strip().lower(),
        "goal": str(goal)[:4000],
        "input": safe_input,
        "callback": callback_url,
        "expires_at": int(expires_at),
        "token_hash": hash_token(token),
    }
    return task, token
