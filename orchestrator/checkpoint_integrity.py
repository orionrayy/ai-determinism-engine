#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class CheckpointIntegrityError(RuntimeError):
    pass


def _safe_path(root: Path, value: str) -> Path:
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise CheckpointIntegrityError("checkpoint path escapes repository root") from exc
    return candidate


def verify_checkpoint(
    root: Path,
    node: dict[str, Any],
    *,
    expected_workflow_id: str | None = None,
) -> dict[str, Any]:
    output = node.get("output")
    checkpoint = output.get("checkpoint") if isinstance(output, dict) else None
    if not isinstance(checkpoint, dict):
        return {
            "verified": False,
            "legacy": True,
            "reason": "checkpoint_digest_not_present",
        }

    path_value = str(checkpoint.get("path") or "").strip()
    expected_hash = str(checkpoint.get("sha256") or "").strip().lower()
    if not path_value or len(expected_hash) != 64:
        raise CheckpointIntegrityError("checkpoint metadata is incomplete")

    path = _safe_path(root, path_value.lstrip("/"))
    if not path.is_file():
        raise CheckpointIntegrityError("checkpoint file is missing")

    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise CheckpointIntegrityError("checkpoint SHA-256 mismatch")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointIntegrityError("checkpoint is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise CheckpointIntegrityError("checkpoint payload must be an object")

    checkpoint_node = payload.get("node")
    if not isinstance(checkpoint_node, dict):
        raise CheckpointIntegrityError("checkpoint node payload is missing")
    if str(checkpoint_node.get("id") or "") != str(node.get("id") or ""):
        raise CheckpointIntegrityError("checkpoint node binding mismatch")
    if checkpoint_node.get("status") != "completed":
        raise CheckpointIntegrityError("checkpoint node is not completed")
    if expected_workflow_id is not None:
        if str(payload.get("workflow") or "") != str(expected_workflow_id):
            raise CheckpointIntegrityError("checkpoint workflow binding mismatch")

    return {
        "verified": True,
        "legacy": False,
        "path": path_value,
        "sha256": actual_hash,
    }
