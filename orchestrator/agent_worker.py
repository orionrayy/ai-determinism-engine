#!/usr/bin/env python3
"""Execute one safe federated agent task on an isolated runner."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from . import orchestrator as core
    from .agent_protocol import build_result, task_from_dict
except ImportError:
    import orchestrator as core
    from agent_protocol import build_result, task_from_dict


MAX_WORKER_RESULT_FILE_BYTES = 16 * 1024


def _write_result(path: Path, result: dict) -> None:
    payload = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    encoded = payload.encode("utf-8")
    if len(encoded) > MAX_WORKER_RESULT_FILE_BYTES:
        raise RuntimeError("worker result exceeds file size limit")
    path.write_bytes(encoded)


def run(task_payload: dict) -> dict:
    task = task_from_dict(task_payload)
    registry = core.load_registry()
    spec = registry.get(task.tool, {})
    if core.free_only() and not bool(spec.get("free_tier", False)) and task.tool not in core.BUILTIN_FREE_TOOLS:
        raise RuntimeError(f"tool {task.tool} is disabled by ORCHESTRATOR_FREE_ONLY=true")
    if core.side_effecting(
        core.Node(
            id=task.task_id,
            capability=task.capability,
            tool=task.tool,
            risk=task.risk,
            input={"action": task.context.get("action")},
            agent_role=task.role,
        ),
        registry,
    ):
        raise RuntimeError("federated worker refuses side-effecting tool")
    node = core.Node(
        id=task.task_id,
        capability=task.capability,
        tool=task.tool,
        risk=task.risk,
        input={
            "workflow_id": task.workflow_id,
            "instruction": task.instruction,
            "context": task.context,
            "artifacts": task.context.get("artifacts", []),
        },
        contract=task.contract,
        agent_role=task.role,
    )
    core.enforce_node_policy([node], registry, live=False)
    output = core.execute_node(
        node,
        str(task.context.get("goal") or task.instruction),
        dry_run=False,
    )
    return build_result(
        task,
        status="completed",
        output=output,
        worker_run_id=os.environ.get("GITHUB_RUN_ID", ""),
    ).to_dict()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-json", required=True)
    parser.add_argument("--output", default="result.json")
    args = parser.parse_args()
    try:
        task = json.loads(args.task_json)
        if not isinstance(task, dict):
            raise ValueError("task payload must be an object")
        result = run(task)
        _write_result(Path(args.output), result)
        print(json.dumps({
            "task_id": result["task_id"],
            "agent_id": result["agent_id"],
            "status": result["status"],
            "output_sha256": result["output_sha256"],
        }, sort_keys=True))
        return 0
    except Exception as exc:
        try:
            task = task_from_dict(json.loads(args.task_json))
            failed = build_result(
                task,
                status="failed",
                output={},
                error={"type": type(exc).__name__, "message": str(exc)},
                worker_run_id=os.environ.get("GITHUB_RUN_ID", ""),
            ).to_dict()
            _write_result(Path(args.output), failed)
        except Exception:
            return 2
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
