#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from typing import Any

POLICY_SNAPSHOT_VERSION = 1

_TOOL_POLICY_KEYS = (
    "required_env",
    "secret_env",
    "side_effects",
    "free_tier",
    "risk",
    "allowed_actions",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _tool_policy_spec(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        key: spec.get(key)
        for key in _TOOL_POLICY_KEYS
        if key in spec
    }


def _candidate_tools(registry: dict[str, dict[str, Any]], capability: str) -> list[str]:
    spec = registry.get(f"capability:{capability}", {})
    values = [spec.get("default_tool"), *(spec.get("fallback_tools") or [])]
    return list(dict.fromkeys(str(value) for value in values if value))


def build_policy_snapshot(
    registry: dict[str, dict[str, Any]],
    nodes: list[Any],
    *,
    live: bool,
) -> dict[str, Any]:
    routes = []
    seen_tools: set[str] = set()

    for node in sorted(
        nodes,
        key=lambda item: str(item.id if hasattr(item, "id") else item.get("id")),
    ):
        node_id = str(
            node.id if hasattr(node, "id")
            else node.get("id") or ""
        )
        capability = str(
            node.capability if hasattr(node, "capability")
            else node.get("capability") or ""
        )
        tool = str(
            node.tool if hasattr(node, "tool")
            else node.get("tool") or ""
        )
        risk = str(
            node.risk if hasattr(node, "risk")
            else node.get("risk") or "low"
        )
        input_value = node.input if hasattr(node, "input") else node.get("input") or {}
        action = (
            str(input_value.get("action") or "")
            if isinstance(input_value, dict)
            else ""
        )
        candidates = _candidate_tools(registry, capability)
        selected_tools = [tool, *candidates]
        for candidate in selected_tools:
            if candidate:
                seen_tools.add(candidate)

        routes.append({
            "node_id": node_id,
            "capability": capability,
            "tool": tool,
            "risk": risk,
            "action": action,
            "candidates": candidates,
        })

    tool_specs = {
        tool: _tool_policy_spec(registry.get(tool, {}))
        for tool in sorted(seen_tools)
    }

    return {
        "version": POLICY_SNAPSHOT_VERSION,
        "live": bool(live),
        "free_only": os.environ.get("ORCHESTRATOR_FREE_ONLY", "true").lower() == "true",
        "routes": routes,
        "tool_specs": tool_specs,
    }


def fingerprint_policy(snapshot: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(snapshot)).hexdigest()
