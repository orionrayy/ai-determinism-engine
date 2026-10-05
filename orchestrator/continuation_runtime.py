#!/usr/bin/env python3
"""GitHub Actions continuation decision runtime.

Keep workflow YAML as a thin transport adapter. The decision logic is importable,
unit-testable, and compiled by the normal Python CI path.
"""
from __future__ import annotations

import argparse
import os
from typing import Any

try:
    from .orchestrator import load_state
except ImportError:
    from orchestrator import load_state


def find_workflow_for_run(
    state: dict[str, Any],
    *,
    run_id: str,
    run_attempt: str,
) -> dict[str, Any] | None:
    for item in (state.get("workflows") or {}).values():
        if not isinstance(item, dict):
            continue
        if (
            str(item.get("github_run_id") or "") == str(run_id)
            and str(item.get("github_run_attempt") or "") == str(run_attempt)
        ):
            return item
    return None


def decide_continuation(
    state: dict[str, Any],
    *,
    run_id: str,
    run_attempt: str,
) -> dict[str, str]:
    workflow = find_workflow_for_run(
        state,
        run_id=run_id,
        run_attempt=run_attempt,
    )
    status = str(workflow.get("status") or "missing") if workflow else "missing"
    return {
        "workflow_id": str(workflow.get("id") or "") if workflow else "",
        "continue": "true" if status == "running" else "false",
        "trigger_issue": str(workflow.get("trigger_issue") or "") if workflow else "",
        "status": status,
    }


def write_github_output(values: dict[str, str]) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT", "").strip()
    if not output_path:
        raise RuntimeError("GITHUB_OUTPUT is not configured")
    with open(output_path, "a", encoding="utf-8") as handle:
        for key, value in values.items():
            value = str(value)
            if "\n" not in value and "\r" not in value:
                handle.write(f"{key}={value}\n")
                continue
            delimiter = f"ORCHESTRATOR_OUTPUT_{os.urandom(16).hex()}"
            handle.write(f"{key}<<{delimiter}\n{value}\n{delimiter}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=os.environ.get("WORKFLOW_RUN_ID", ""))
    parser.add_argument(
        "--run-attempt",
        default=os.environ.get("WORKFLOW_RUN_ATTEMPT", ""),
    )
    args = parser.parse_args()
    run_id = str(args.run_id).strip()
    run_attempt = str(args.run_attempt).strip()
    if not run_id or not run_attempt:
        raise SystemExit("workflow run identity is required")
    decision = decide_continuation(
        load_state(),
        run_id=run_id,
        run_attempt=run_attempt,
    )
    write_github_output(decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
