#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from typing import Any

_VOLATILE_INPUT_KEYS = {
    "workflow_id",
    "context",
    "repair_feedback",
    "approval_issue",
    "approval_granted",
    "approval_fingerprint",
    "approval_actor",
    "approval_approved_at",
    "retry_jitter_seed",
    "previous_tool",
    "provider_resolution_change",
    "provider_resolution_fingerprint",
    "plan_intent_fingerprint",
}


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def node_definition(node: Any) -> dict[str, Any]:
    if hasattr(node, "input"):
        input_value = dict(node.input or {})
        node_id = str(node.id)
        capability = str(node.capability)
        tool = str(node.tool)
        depends_on = list(node.depends_on or [])
        risk = str(node.risk)
        contract = dict(node.contract or {})
        agent_role = str(getattr(node, "agent_role", "") or "")
    else:
        input_value = dict(node.get("input") or {})
        node_id = str(node.get("id") or "")
        capability = str(node.get("capability") or "")
        tool = str(node.get("tool") or "")
        depends_on = list(node.get("depends_on") or [])
        risk = str(node.get("risk") or "low")
        contract = dict(node.get("contract") or {})
        agent_role = str(node.get("agent_role") or "")
    for key in _VOLATILE_INPUT_KEYS:
        input_value.pop(key, None)
    return {
        "id": node_id,
        "capability": capability,
        "tool": tool,
        "depends_on": sorted(str(item) for item in depends_on),
        "risk": risk,
        "agent_role": agent_role,
        "input": input_value,
        "contract": contract,
    }


def fingerprint_nodes(nodes: list[Any]) -> str:
    definitions = [
        node_definition(node)
        for node in sorted(nodes, key=lambda item: str(item.id if hasattr(item, "id") else item.get("id")))
    ]
    return hashlib.sha256(canonical_json(definitions)).hexdigest()


def intent_definition(node: Any) -> dict[str, Any]:
    """Return semantic plan intent without binding it to a concrete provider."""
    definition = node_definition(node)
    definition.pop("tool", None)
    return definition


def provider_resolution_definition(node: Any) -> dict[str, Any]:
    """Return the concrete provider binding separately from semantic intent."""
    if hasattr(node, "input"):
        return {
            "id": str(node.id),
            "capability": str(node.capability),
            "tool": str(node.tool),
            "provider_binding": dict(
                node.input.get("provider_binding") or {}
            ) if isinstance(node.input.get("provider_binding"), dict) else {},
        }
    binding = node.get("provider_binding") or {}
    return {
        "id": str(node.get("id") or ""),
        "capability": str(node.get("capability") or ""),
        "tool": str(node.get("tool") or ""),
        "provider_binding": dict(binding) if isinstance(binding, dict) else {},
    }


def fingerprint_intent(nodes: list[Any]) -> str:
    definitions = [
        intent_definition(node)
        for node in sorted(
            nodes,
            key=lambda item: str(item.id if hasattr(item, "id") else item.get("id")),
        )
    ]
    return hashlib.sha256(canonical_json(definitions)).hexdigest()


def fingerprint_provider_resolution(nodes: list[Any]) -> str:
    definitions = [
        provider_resolution_definition(node)
        for node in sorted(
            nodes,
            key=lambda item: str(item.id if hasattr(item, "id") else item.get("id")),
        )
    ]
    return hashlib.sha256(canonical_json(definitions)).hexdigest()
