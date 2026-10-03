#!/usr/bin/env python3
"""Free, dependency-free cross-module control-plane evaluation harness."""
from __future__ import annotations

import argparse
import json
from typing import Any, Callable

from blueprint_compiler import build_compilation_manifest
from context_budget import pack_node_context
from orchestrator import (
    TRACE_SCHEMA_VERSION,
    _trace_envelope,
    build_ingress_intent_digest,
    free_only,
    tool_available,
)
from state_schema import CURRENT_WORKFLOW_SCHEMA_VERSION


def _case_blueprint_repeatability() -> dict[str, Any]:
    blueprint = {
        "blueprint_id": "eval-blueprint",
        "version": "1",
        "requirements": [
            {"id": "a", "summary": "A"},
            {"id": "b", "summary": "B", "depends_on": ["a"]},
        ],
    }
    first = build_compilation_manifest(blueprint)
    second = build_compilation_manifest(blueprint)
    passed = (
        first["manifest_digest"] == second["manifest_digest"]
        and first["blueprint"]["blueprint_digest"]
        == second["blueprint"]["blueprint_digest"]
    )
    return {
        "name": "blueprint_repeatability",
        "passed": passed,
        "manifest_digest": first["manifest_digest"],
    }


def _case_context_budget() -> dict[str, Any]:
    deps = {
        f"n{i:02d}": {
            "capability": "research",
            "tool": "wikipedia",
            "status": "completed",
            "output": {"body": "x" * 9000},
            "error": {},
            "evidence_sha256": "a" * 64,
        }
        for i in range(20)
    }
    packed = pack_node_context(
        goal="cross-module context evaluation",
        dependencies=deps,
        contract={"required_fields": ["result"]},
        repair_feedback={},
    )
    raw = json.dumps(
        packed, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "name": "context_budget",
        "passed": (
            len(raw) <= 48 * 1024
            and packed["context_digest"]
            and (
                packed["context_budget"]["omitted_dependencies"]
                or any(
                    isinstance(item, dict)
                    and (item.get("output_truncated") or item.get("omitted"))
                    for item in packed["dependencies"].values()
                )
            )
        ),
        "bytes": len(raw),
        "context_digest": packed["context_digest"],
    }


def _case_ingress_binding() -> dict[str, Any]:
    base = {
        "live": True,
        "external_workflow_id": "eval-1",
        "external_domain": "generic",
        "external_operation": "write",
        "intent_fingerprint": "a" * 64,
        "input_digest": "b" * 64,
        "private_input_ref": "c" * 64,
    }
    first = build_ingress_intent_digest(goal="same", **base)
    same = build_ingress_intent_digest(goal="same", **base)
    changed = build_ingress_intent_digest(goal="changed", **base)
    return {
        "name": "ingress_semantic_binding",
        "passed": first == same and first != changed,
    }


def _case_free_policy() -> dict[str, Any]:
    registry = {
        "paid-tool": {"free_tier": False},
        "free-tool": {"free_tier": True, "required_env": None},
    }
    return {
        "name": "free_only_policy",
        "passed": free_only()
        and not tool_available("paid-tool", registry)
        and tool_available("free-tool", registry),
    }


def _case_wave_ordering() -> dict[str, Any]:
    manifest = build_compilation_manifest(
        {
            "blueprint_id": "eval-waves",
            "requirements": [
                {"id": "a", "summary": "A"},
                {"id": "b", "summary": "B", "depends_on": ["a"], "workstream": "left"},
                {"id": "c", "summary": "C", "depends_on": ["a"], "workstream": "right"},
            ]
        },
        max_requirements_per_unit=1,
    )
    waves = manifest["waves"]
    by_id = {wave["wave_id"]: wave for wave in waves}
    valid = True
    for wave in waves:
        for dependency in wave["depends_on_waves"]:
            valid = valid and by_id[dependency]["ordinal"] < wave["ordinal"]
    return {
        "name": "wave_dependency_order",
        "passed": bool(waves) and valid and any(
            wave["parallel_candidate"] and len(wave["unit_ids"]) > 1
            for wave in waves
        ),
        "wave_count": len(waves),
    }


def _case_trace_contract() -> dict[str, Any]:
    workflow = _trace_envelope("workflow.started", {"workflow_id": "eval-wf"})
    node = _trace_envelope(
        "node.started",
        {"workflow_id": "eval-wf", "node_id": "n01"},
    )
    agent = _trace_envelope(
        "agent.started",
        {
            "workflow_id": "eval-wf",
            "node_id": "n01",
            "agent_id": "agent-1",
        },
    )
    expected_node_span = _trace_envelope(
        "node.completed",
        {"workflow_id": "eval-wf", "node_id": "n01"},
    )["span_id"]
    return {
        "name": "trace_contract",
        "passed": (
            workflow["schema_version"] == TRACE_SCHEMA_VERSION
            and workflow["span_kind"] == "workflow"
            and node["parent_span_id"] == workflow["span_id"]
            and agent["parent_span_id"] == expected_node_span
        ),
    }


CASES: tuple[Callable[[], dict[str, Any]], ...] = (
    _case_blueprint_repeatability,
    _case_context_budget,
    _case_ingress_binding,
    _case_free_policy,
    _case_wave_ordering,
    _case_trace_contract,
)


def run_evaluations() -> dict[str, Any]:
    results = []
    for case in CASES:
        try:
            result = case()
        except Exception as exc:
            result = {
                "name": case.__name__,
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        results.append(result)
    passed = sum(1 for item in results if item.get("passed"))
    return {
        "harness": "orchestrator-control-plane-eval",
        "schema_version": 1,
        "workflow_schema_version": CURRENT_WORKFLOW_SCHEMA_VERSION,
        "free_only": free_only(),
        "passed": passed == len(results),
        "passed_cases": passed,
        "total_cases": len(results),
        "cases": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run_evaluations()
    if args.json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        for case in result["cases"]:
            print(("PASS" if case.get("passed") else "FAIL") + " " + case["name"])
        print(f"{result['passed_cases']}/{result['total_cases']} cases passed")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
