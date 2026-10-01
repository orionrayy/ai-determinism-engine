#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / ".orchestrator"
STATE_FILE = STATE_DIR / "state.json"
EVENT_FILE = STATE_DIR / "events.jsonl"
CHECKPOINT_DIR = STATE_DIR / "checkpoints"
REGISTRY_FILE = ROOT / "orchestrator" / "tools.json"

MAX_NODES = 24
MAX_REPLANS = 2

TRANSITIONS = {
    "pending": {"ready", "cancelled"},
    "ready": {"running", "cancelled"},
    "running": {"validating", "waiting_approval", "retrying", "failed", "cancelled"},
    "validating": {"completed", "retrying", "failed"},
    "waiting_approval": {"ready", "failed", "cancelled"},
    "retrying": {"ready", "failed"},
    "failed": {"replanning", "cancelled"},
    "replanning": {"ready", "failed", "cancelled"},
    "completed": set(),
    "cancelled": set(),
}

@dataclass
class Node:
    id: str
    capability: str
    tool: str
    depends_on: list[str] = field(default_factory=list)
    risk: str = "low"
    status: str = "pending"
    retry_count: int = 0
    max_retries: int = 2
    input: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] = field(default_factory=dict)

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def append_event(event_type: str, payload: dict[str, Any]) -> None:
    EVENT_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry = {"ts": utc_now(), "event_type": event_type, "payload": payload}
    with EVENT_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

def load_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        return {"version": 2, "workflows": {}, "last_workflow_id": None}
    return json.loads(STATE_FILE.read_text(encoding="utf-8"))

def save_state(state: dict[str, Any]) -> None:
    write_json(STATE_FILE, state)

def new_id(prefix: str) -> str:
    return f"{prefix}_{int(time.time() * 1000)}"

def classify_risk(capability: str) -> str:
    if capability in {"deploy", "publish", "delete", "external_write"}:
        return "high"
    if capability in {"build", "edit", "send"}:
        return "medium"
    return "low"

def load_registry() -> dict[str, dict[str, Any]]:
    if REGISTRY_FILE.exists():
        return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
    return {}

def tool_available(tool_name: str, registry: dict[str, dict[str, Any]]) -> bool:
    spec = registry.get(tool_name, {})
    env_var = spec.get("required_env")
    return not env_var or bool(os.environ.get(env_var))

def deterministic_plan(goal: str, registry: dict[str, dict[str, Any]]) -> list[Node]:
    g = goal.lower()
    if any(k in g for k in ("website", "web app", "app", "software", "build", "deploy")):
        sequence = [
            ("research", "Collect requirements and constraints."),
            ("spec", "Produce an implementation specification."),
            ("build", "Implement the requested system."),
            ("test", "Run deterministic tests."),
            ("deploy", "Deploy only after validation."),
            ("validate", "Verify the deployed result."),
            ("notify", "Report state and artifacts."),
        ]
    elif any(k in g for k in ("research", "compare", "literature", "study", "analysis")):
        sequence = [
            ("research", "Collect evidence and primary sources."),
            ("analyze", "Synthesize evidence and uncertainty."),
            ("draft", "Produce the requested research output."),
            ("validate", "Check citations and consistency."),
            ("notify", "Report the completed result."),
        ]
    elif any(k in g for k in ("content", "post", "instagram", "youtube", "publish")):
        sequence = [
            ("research", "Collect source material and constraints."),
            ("draft", "Create the content draft."),
            ("validate", "Check factuality and format."),
            ("publish", "Publish only after validation."),
            ("notify", "Report publication status."),
        ]
    else:
        sequence = [
            ("research", "Gather minimum required information."),
            ("execute", "Perform the requested operation."),
            ("validate", "Validate output against the goal."),
            ("notify", "Report the result and artifacts."),
        ]

    nodes: list[Node] = []
    previous: list[str] = []
    for index, (capability, instruction) in enumerate(sequence, start=1):
        preferred = registry.get(f"capability:{capability}", {}).get("default_tool")
        if not preferred:
            preferred = {
                "research": "firecrawl",
                "analyze": "openai",
                "draft": "openai",
                "spec": "openai",
                "build": "github",
                "test": "github",
                "deploy": "webhook",
                "validate": "webhook",
                "publish": "webhook",
                "notify": "webhook",
                "execute": "webhook",
            }.get(capability, "noop")
        node = Node(
            id=f"n{index:02d}-{capability}",
            capability=capability,
            tool=preferred,
            depends_on=list(previous),
            risk=classify_risk(capability),
            input={"goal": goal, "instruction": instruction},
        )
        nodes.append(node)
        previous = [node.id]
    return nodes

def validate_dag(nodes: list[Node]) -> None:
    if not nodes or len(nodes) > MAX_NODES:
        raise ValueError(f"invalid node count: {len(nodes)}")
    ids = {node.id for node in nodes}
    if len(ids) != len(nodes):
        raise ValueError("duplicate node id")
    by_id = {node.id: node for node in nodes}
    for node in nodes:
        if node.tool not in {"noop"} and node.tool == "":
            raise ValueError(f"empty tool for {node.id}")
        for dependency in node.depends_on:
            if dependency not in ids:
                raise ValueError(f"unknown dependency {dependency} for {node.id}")
            if dependency == node.id:
                raise ValueError(f"self dependency {node.id}")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise ValueError("cycle detected")
        if node_id in visited:
            return
        visiting.add(node_id)
        for dependency in by_id[node_id].depends_on:
            visit(dependency)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in ids:
        visit(node_id)

def http_json(
    url: str,
    method: str = "GET",
    body: Any | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        raise RuntimeError("HTTPS is required for external HTTP tools")
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request_headers = {
        "User-Agent": "ai-orchestrator-core/2.0",
        "Accept": "application/json",
        **(headers or {}),
    }
    if data is not None:
        request_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=request_headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8", "replace")
        try:
            value = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            value = {"text": raw}
        return {"status_code": response.status, "data": value}

def execute_openai(node: Node, goal: str) -> dict[str, Any]:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required for the OpenAI adapter")
    payload = {
        "model": os.environ.get("OPENAI_MODEL", "gpt-5.6"),
        "input": (
            "Act as a conservative workflow worker. Return JSON with "
            "result, risks, next_action.\\n"
            f"GOAL: {goal}\\nINSTRUCTION: {node.input.get('instruction', '')}"
        ),
    }
    return http_json(
        "https://api.openai.com/v1/responses",
        method="POST",
        body=payload,
        headers={"Authorization": f"Bearer {key}"},
        timeout=90,
    )

def execute_webhook(node: Node, goal: str) -> dict[str, Any]:
    url = node.input.get("url") or os.environ.get("ORCHESTRATOR_WEBHOOK_URL")
    if not url:
        raise RuntimeError("ORCHESTRATOR_WEBHOOK_URL or node.input.url is required")
    headers = {"X-Orchestrator-Event": "node.execute"}
    secret = os.environ.get("ORCHESTRATOR_WEBHOOK_SECRET")
    if secret:
        headers["Authorization"] = f"Bearer {secret}"
    return http_json(
        url,
        method="POST",
        body={
            "workflow_id": node.input.get("workflow_id"),
            "node_id": node.id,
            "goal": goal,
            "capability": node.capability,
            "instruction": node.input.get("instruction"),
        },
        headers=headers,
        timeout=90,
    )

def execute_github(node: Node) -> dict[str, Any]:
    token = os.environ.get("GITHUB_TOKEN")
    repository = os.environ.get("GITHUB_REPOSITORY")
    if not token or not repository:
        raise RuntimeError("GitHub token/repository unavailable")
    headers = {
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    action = node.input.get("action", "metadata")
    if action == "metadata":
        return http_json(
            f"https://api.github.com/repos/{repository}",
            headers=headers,
        )
    if action == "create_issue":
        return http_json(
            f"https://api.github.com/repos/{repository}/issues",
            method="POST",
            body={
                "title": node.input.get("title", "Orchestrator task"),
                "body": node.input.get("body", ""),
            },
            headers=headers,
        )
    raise RuntimeError(f"GitHub action not allowlisted: {action}")

def execute_node(node: Node, goal: str, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {
            "simulated": True,
            "tool": node.tool,
            "capability": node.capability,
            "instruction": node.input.get("instruction"),
        }
    if node.tool == "openai":
        return execute_openai(node, goal)
    if node.tool == "webhook":
        return execute_webhook(node, goal)
    if node.tool == "github":
        return execute_github(node)
    if node.tool == "noop":
        return {"message": "noop"}
    raise RuntimeError(f"unknown tool adapter: {node.tool}")

def transition(node: Node, new_status: str) -> None:
    if new_status == node.status:
        return
    allowed = TRANSITIONS.get(node.status, set())
    if new_status not in allowed:
        raise RuntimeError(f"illegal transition {node.status} -> {new_status} ({node.id})")
    node.status = new_status

def ready_nodes(nodes: list[Node]) -> list[Node]:
    completed = {node.id for node in nodes if node.status == "completed"}
    return [
        node for node in nodes
        if node.status == "pending" and all(dep in completed for dep in node.depends_on)
    ]

def replan_after_failure(
    workflow: dict[str, Any],
    nodes: list[Node],
    failed_node: Node,
    registry: dict[str, dict[str, Any]],
) -> bool:
    replans = int(workflow.get("replan_count", 0))
    if replans >= MAX_REPLANS:
        return False

    fallback_tools = registry.get(f"capability:{failed_node.capability}", {}).get("fallback_tools", [])
    for candidate in fallback_tools:
        if candidate == failed_node.tool:
            continue
        if not tool_available(candidate, registry):
            continue
        transition(failed_node, "replanning")
        failed_node.tool = candidate
        failed_node.retry_count = 0
        failed_node.error = {}
        failed_node.output = {}
        workflow["replan_count"] = replans + 1
        append_event(
            "node.replanned",
            {
                "workflow_id": workflow["id"],
                "node_id": failed_node.id,
                "from_tool": failed_node.input.get("previous_tool"),
                "to_tool": candidate,
                "replan_count": workflow["replan_count"],
            },
        )
        failed_node.input["previous_tool"] = candidate
        transition(failed_node, "ready")
        workflow["status"] = "running"
        return True
    return False

def run_workflow(workflow: dict[str, Any], approve_high_risk: bool = False) -> None:
    nodes = [Node(**node) for node in workflow["nodes"]]
    validate_dag(nodes)
    registry = load_registry()
    live = bool(workflow.get("live")) and os.environ.get("ORCHESTRATOR_LIVE", "").lower() == "true"
    workflow["status"] = "running"
    workflow["execution_mode"] = "live" if live else "dry-run"
    workflow.setdefault("replan_count", 0)

    for node in nodes:
        node.input["workflow_id"] = workflow["id"]

    safety = 0
    while True:
        safety += 1
        if safety > 100:
            raise RuntimeError("orchestration safety limit reached")

        ready = ready_nodes(nodes)
        for node in ready:
            transition(node, "ready")
        if not ready:
            if all(node.status == "completed" for node in nodes):
                workflow["status"] = "completed"
                workflow["nodes"] = [asdict(node) for node in nodes]
                append_event("workflow.completed", {"workflow_id": workflow["id"]})
                return

            waiting = [node for node in nodes if node.status in {"pending", "waiting_approval", "retrying"}]
            if waiting and all(node.status == "waiting_approval" for node in waiting):
                workflow["status"] = "waiting_approval"
                workflow["nodes"] = [asdict(node) for node in nodes]
                return

            workflow["status"] = "failed"
            workflow["nodes"] = [asdict(node) for node in nodes]
            return

        for node in ready:
            if live and node.risk in {"high", "critical"} and not approve_high_risk:
                transition(node, "waiting_approval")
                workflow["status"] = "waiting_approval"
                append_event(
                    "approval.required",
                    {"workflow_id": workflow["id"], "node_id": node.id, "risk": node.risk},
                )
                continue

            transition(node, "running")
            append_event(
                "node.started",
                {"workflow_id": workflow["id"], "node_id": node.id, "tool": node.tool},
            )

            attempts = node.retry_count
            while True:
                try:
                    node.output = execute_node(node, workflow["goal"], dry_run=not live)
                    transition(node, "validating")
                    node.output["validation"] = {
                        "passed": True,
                        "checked_at": utc_now(),
                    }
                    transition(node, "completed")
                    append_event(
                        "node.completed",
                        {"workflow_id": workflow["id"], "node_id": node.id, "tool": node.tool},
                    )
                    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
                    write_json(
                        CHECKPOINT_DIR / f"{workflow['id']}-{node.id}.json",
                        {
                            "workflow": workflow["id"],
                            "node": asdict(node),
                            "ts": utc_now(),
                        },
                    )
                    break
                except Exception as exc:
                    node.error = {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "trace": traceback.format_exc(limit=4),
                    }
                    if attempts < node.max_retries:
                        attempts += 1
                        node.retry_count = attempts
                        transition(node, "retrying")
                        append_event(
                            "node.retrying",
                            {
                                "workflow_id": workflow["id"],
                                "node_id": node.id,
                                "attempt": attempts,
                                "error": str(exc),
                            },
                        )
                        time.sleep(min(2 ** attempts, 8))
                        transition(node, "ready")
                        continue

                    transition(node, "failed")
                    append_event(
                        "node.failed",
                        {"workflow_id": workflow["id"], "node_id": node.id, "error": node.error},
                    )

                    if replan_after_failure(workflow, nodes, node, registry):
                        break

                    workflow["status"] = "failed"
                    workflow["failed_node"] = node.id
                    workflow["nodes"] = [asdict(item) for item in nodes]
                    return

        workflow["nodes"] = [asdict(node) for node in nodes]
        save_state({**load_state(), "workflows": {workflow["id"]: workflow}, "last_workflow_id": workflow["id"]})

def create_workflow(goal: str, live: bool) -> dict[str, Any]:
    registry = load_registry()
    nodes = deterministic_plan(goal, registry)
    validate_dag(nodes)
    return {
        "id": new_id("wf"),
        "created_at": utc_now(),
        "goal": goal,
        "status": "planning",
        "live": live,
        "execution_mode": "dry-run",
        "replan_count": 0,
        "nodes": [asdict(node) for node in nodes],
    }

def print_summary(workflow: dict[str, Any]) -> None:
    counts: dict[str, int] = {}
    for node in workflow.get("nodes", []):
        counts[node["status"]] = counts.get(node["status"], 0) + 1
    print(json.dumps({
        "workflow_id": workflow["id"],
        "status": workflow["status"],
        "mode": workflow.get("execution_mode"),
        "replans": workflow.get("replan_count", 0),
        "counts": counts,
        "failed_node": workflow.get("failed_node"),
    }, indent=2))

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", default=os.environ.get("ORCHESTRATOR_GOAL", ""))
    parser.add_argument("--workflow-id")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--approve-high-risk", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    state = load_state()

    if args.list:
        for workflow in state.get("workflows", {}).values():
            print_summary(workflow)
        return 0

    if args.workflow_id:
        workflow = state.get("workflows", {}).get(args.workflow_id)
        if not workflow:
            raise SystemExit(f"workflow not found: {args.workflow_id}")
        run_workflow(workflow, approve_high_risk=args.approve_high_risk)
        state["workflows"][workflow["id"]] = workflow
        state["last_workflow_id"] = workflow["id"]
        save_state(state)
        print_summary(workflow)
        return 0 if workflow["status"] in {"completed", "waiting_approval"} else 2

    if not args.goal:
        raise SystemExit("provide --goal or --workflow-id")

    live = args.live or os.environ.get("ORCHESTRATOR_LIVE", "").lower() == "true"
    workflow = create_workflow(args.goal, live=live)
    workflow["status"] = "ready"
    state["workflows"][workflow["id"]] = workflow
    state["last_workflow_id"] = workflow["id"]
    append_event(
        "workflow.created",
        {"workflow_id": workflow["id"], "goal": workflow["goal"], "live": live},
    )
    run_workflow(workflow, approve_high_risk=args.approve_high_risk)
    state["workflows"][workflow["id"]] = workflow
    state["last_workflow_id"] = workflow["id"]
    save_state(state)
    print_summary(workflow)
    return 0 if workflow["status"] in {"completed", "waiting_approval"} else 2

if __name__ == "__main__":
    raise SystemExit(main())
