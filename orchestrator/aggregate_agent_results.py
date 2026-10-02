#!/usr/bin/env python3
"""Aggregate and validate federated agent result artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .agent_protocol import (
        FederationProtocolError,
        MAX_RESULT_JSON_BYTES,
        canonical_json,
        digest,
        task_from_dict,
        validate_result,
        AgentResult,
    )
except ImportError:
    from agent_protocol import (
        FederationProtocolError,
        MAX_RESULT_JSON_BYTES,
        canonical_json,
        digest,
        task_from_dict,
        validate_result,
        AgentResult,
    )


MAX_RESULTS = 8
MAX_AGGREGATE_BYTES = 96 * 1024


def _load(path: Path, limit: int) -> dict[str, Any]:
    raw = path.read_bytes()
    if len(raw) > limit:
        raise FederationProtocolError(f"{path} exceeds size limit")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise FederationProtocolError(f"{path} must contain an object")
    return value


def aggregate(manifest_path: Path, results_root: Path) -> dict[str, Any]:
    manifest = _load(manifest_path, 48 * 1024)
    tasks_raw = manifest.get("tasks")
    if not isinstance(tasks_raw, list) or not 1 <= len(tasks_raw) <= MAX_RESULTS:
        raise FederationProtocolError("manifest tasks must contain 1..8 tasks")
    tasks = [task_from_dict(item) for item in tasks_raw]
    expected = {task.task_id: task for task in tasks}
    files = sorted(results_root.glob("**/result.json"))
    if len(files) != len(expected):
        raise FederationProtocolError(
            f"expected {len(expected)} results, found {len(files)}"
        )

    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in files:
        value = _load(path, MAX_RESULT_JSON_BYTES)
        result = AgentResult(
            federation_id=str(value.get("federation_id") or ""),
            workflow_id=str(value.get("workflow_id") or ""),
            task_id=str(value.get("task_id") or ""),
            agent_id=str(value.get("agent_id") or ""),
            role=str(value.get("role") or ""),
            capability=str(value.get("capability") or ""),
            tool=str(value.get("tool") or ""),
            attempt=int(value.get("attempt") or 0),
            input_digest=str(value.get("input_digest") or ""),
            status=str(value.get("status") or ""),
            output=dict(value.get("output") or {}),
            output_sha256=str(value.get("output_sha256") or ""),
            error=dict(value["error"]) if isinstance(value.get("error"), dict) else None,
            worker_run_id=str(value.get("worker_run_id") or ""),
            protocol_version=int(value.get("protocol_version") or 0),
        )
        if result.task_id in seen:
            raise FederationProtocolError("duplicate task result")
        task = expected.get(result.task_id)
        if task is None:
            raise FederationProtocolError(f"unexpected task result: {result.task_id}")
        validate_result(result, expected_task=task)
        seen.add(result.task_id)
        results.append(result.to_dict())

    results.sort(key=lambda item: item["task_id"])
    completed = sum(1 for item in results if item["status"] == "completed")
    failed = len(results) - completed
    aggregate = {
        "protocol_version": manifest["protocol_version"],
        "federation_id": manifest["federation_id"],
        "workflow_id": manifest["workflow_id"],
        "status": "completed" if failed == 0 else "partial_failure",
        "task_count": len(results),
        "completed": completed,
        "failed": failed,
        "results": results,
    }
    encoded = canonical_json(aggregate)
    if len(encoded) > MAX_AGGREGATE_BYTES:
        raise FederationProtocolError("aggregate exceeds size limit")
    aggregate["aggregate_sha256"] = digest(aggregate)
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output", default="aggregate.json")
    args = parser.parse_args()
    aggregate_value = aggregate(Path(args.manifest), Path(args.results_root))
    Path(args.output).write_text(
        json.dumps(aggregate_value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "federation_id": aggregate_value["federation_id"],
        "workflow_id": aggregate_value["workflow_id"],
        "status": aggregate_value["status"],
        "aggregate_sha256": aggregate_value["aggregate_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
