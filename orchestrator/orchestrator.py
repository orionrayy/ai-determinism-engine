#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
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
    "ready": {"running", "waiting_approval", "cancelled"},
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

def persist_workflow(workflow: dict[str, Any]) -> None:
    state = load_state()
    workflow["updated_at"] = utc_now()
    state.setdefault("workflows", {})[workflow["id"]] = workflow
    state["last_workflow_id"] = workflow["id"]
    save_state(state)

def new_id(prefix: str) -> str:
    return f"{prefix}_{int(time.time() * 1000)}"

def classify_risk(capability: str) -> str:
    if capability in {"deploy", "publish", "delete", "external_write"}:
        return "high"
    if capability in {"build", "edit", "send"}:
        return "medium"
    return "low"

RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}
BUILTIN_TOOLS = {
    "gemini", "openai", "firecrawl", "research_bundle",
    "wikipedia", "webhook", "github", "noop",
}
BUILTIN_FREE_TOOLS = {"gemini", "research_bundle", "wikipedia", "github", "noop"}

def required_risk(node: Node, registry: dict[str, dict[str, Any]]) -> str:
    floor = classify_risk(node.capability)
    side_effects = set(registry.get(node.tool, {}).get("side_effects", []))
    if "external_request" in side_effects:
        floor = "high"
    if node.tool == "github" and node.input.get("action") in {"create_issue", "create_or_update_file", "delete_file", "dispatch_workflow"}:
        floor = "high"
    return floor

def enforce_node_policy(nodes: list[Node], registry: dict[str, dict[str, Any]]) -> None:
    for node in nodes:
        if node.tool not in BUILTIN_TOOLS and node.tool not in registry:
            raise ValueError(f"unregistered tool for {node.id}: {node.tool}")
        floor = required_risk(node, registry)
        if RISK_ORDER.get(node.risk, 0) < RISK_ORDER[floor]:
            node.risk = floor

def load_registry() -> dict[str, dict[str, Any]]:
    if REGISTRY_FILE.exists():
        return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
    return {}

def free_only() -> bool:
    return os.environ.get("ORCHESTRATOR_FREE_ONLY", "true").lower() == "true"

def tool_available(
    tool_name: str,
    registry: dict[str, dict[str, Any]],
    require_env: bool = True,
    enforce_free: bool = True,
) -> bool:
    spec = registry.get(tool_name, {})
    if enforce_free and free_only():
        is_free = bool(spec.get("free_tier", False))
        if not spec and tool_name in BUILTIN_FREE_TOOLS:
            is_free = True
        if not is_free:
            return False
    if not require_env:
        return True
    env_var = spec.get("required_env")
    return not env_var or bool(os.environ.get(env_var))

def deterministic_plan(goal: str, registry: dict[str, dict[str, Any]], live: bool = False) -> list[Node]:
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
        cap_spec = registry.get(f"capability:{capability}", {})
        candidates = [cap_spec.get("default_tool")] + cap_spec.get("fallback_tools", [])
        candidates = [item for item in candidates if item]
        preferred = next(
            (item for item in candidates if tool_available(item, registry, require_env=live, enforce_free=True)),
            None,
        )
        if preferred is None:
            if cap_spec and not live:
                preferred = candidates[0]
            elif cap_spec:
                raise ValueError(
                    f"no available tool for capability {capability} under current policy"
                )
            else:
                preferred = {
                    "research": "research_bundle",
                    "analyze": "gemini",
                    "draft": "gemini",
                    "spec": "gemini",
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
            input={
                "goal": goal,
                "instruction": instruction,
                "query": goal if capability == "research" else "",
            },
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

def execute_gemini(node: Node, goal: str) -> dict[str, Any]:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is required for the Gemini adapter")
    model = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
    payload = {
        "contents": [{
            "parts": [{
                "text": (
                    "Act as a conservative workflow worker. Return JSON with "
                    "result, risks, next_action.\n"
                    f"GOAL: {goal}\n"
                    f"INSTRUCTION: {node.input.get('instruction', '')}"
                )
            }]
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
        },
    }
    return http_json(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        method="POST",
        body=payload,
        headers={"x-goog-api-key": key},
        timeout=90,
    )

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

def execute_firecrawl(node: Node, goal: str) -> dict[str, Any]:
    key = os.environ.get("FIRECRAWL_API_KEY")
    if not key:
        raise RuntimeError("FIRECRAWL_API_KEY is required for the Firecrawl adapter")
    url = node.input.get("url") or os.environ.get("ORCHESTRATOR_RESEARCH_URL")
    if not url:
        raise RuntimeError(
            "Firecrawl research requires node.input.url or ORCHESTRATOR_RESEARCH_URL"
        )
    return http_json(
        "https://api.firecrawl.dev/v2/scrape",
        method="POST",
        body={
            "url": url,
            "formats": ["markdown"],
            "onlyMainContent": True,
        },
        headers={"Authorization": f"Bearer {key}"},
        timeout=120,
    )

def execute_research_bundle(node: Node, goal: str) -> dict[str, Any]:
    from research_bundle import research_bundle
    query = str(node.input.get("query") or goal).strip()
    if not query:
        raise RuntimeError("research bundle requires a query")
    return research_bundle(query)

def execute_wikipedia(node: Node, goal: str) -> dict[str, Any]:
    query = str(node.input.get("query") or goal).strip()
    if not query:
        raise RuntimeError("Wikipedia research requires a query")
    params = urllib.parse.urlencode({
        "action": "query",
        "list": "search",
        "srsearch": query[:250],
        "srlimit": "8",
        "format": "json",
        "utf8": "1",
    })
    return http_json(
        f"https://en.wikipedia.org/w/api.php?{params}",
        headers={"User-Agent": "ai-orchestrator-core/2.0 research"},
        timeout=30,
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


def github_headers() -> dict[str, str]:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN is required for GitHub operations")
    return {
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

def github_repository() -> str:
    repository = os.environ.get("GITHUB_REPOSITORY")
    if not repository:
        raise RuntimeError("GITHUB_REPOSITORY is not available")
    return repository

def create_approval_issue(workflow: dict[str, Any], node: Node) -> int:
    result = http_json(
        f"https://api.github.com/repos/{github_repository()}/issues",
        method="POST",
        body={
            "title": f"[ORCHESTRATOR APPROVAL] {workflow['id']} / {node.id}",
            "body": (
                "High-risk orchestration action is waiting for explicit approval.\n\n"
                f"Workflow: {workflow['id']}\nNode: {node.id}\n"
                f"Capability: {node.capability}\nTool: {node.tool}\n"
                f"Goal: {workflow['goal']}\n\n"
                "Add label 'orchestrator-approved' to approve this action. "
                "Add label 'orchestrator-rejected' to reject it."
            ),
        },
        headers=github_headers(),
    )
    issue_number = result.get("data", {}).get("number")
    if not isinstance(issue_number, int):
        raise RuntimeError("GitHub did not return an approval issue number")
    return issue_number

def get_issue_labels(issue_number: int) -> set[str]:
    result = http_json(
        f"https://api.github.com/repos/{github_repository()}/issues/{issue_number}",
        headers=github_headers(),
    )
    labels = result.get("data", {}).get("labels", [])
    return {str(label.get("name")) for label in labels if isinstance(label, dict)}

def refresh_approvals(workflow: dict[str, Any], nodes: list[Node]) -> None:
    for node in nodes:
        if node.status != "waiting_approval":
            continue
        issue_number = node.input.get("approval_issue")
        if not issue_number:
            continue
        labels = get_issue_labels(int(issue_number))
        if "orchestrator-rejected" in labels:
            node.input["approval_granted"] = False
            transition(node, "failed")
            node.error = {"type": "approval_rejected", "issue": issue_number}
            workflow["status"] = "failed"
            workflow["failed_node"] = node.id
            append_event(
                "approval.rejected",
                {"workflow_id": workflow["id"], "node_id": node.id, "issue": issue_number},
            )
        elif "orchestrator-approved" in labels:
            node.input["approval_granted"] = True
            transition(node, "ready")
            workflow["status"] = "running"
            append_event(
                "approval.approved",
                {"workflow_id": workflow["id"], "node_id": node.id, "issue": issue_number},
            )

def _github_path(path: str) -> str:
    value = str(path or "").strip().lstrip("/")
    if not value or value.startswith("../") or "/../" in value or value == "..":
        raise RuntimeError("invalid repository path")
    return value

def execute_github(node: Node) -> dict[str, Any]:
    headers = github_headers()
    repository = github_repository()
    action = node.input.get("action", "metadata")
    if action == "metadata":
        return http_json(
            f"https://api.github.com/repos/{repository}",
            headers=headers,
        )
    if action == "read_file":
        path = _github_path(node.input.get("path"))
        branch = str(node.input.get("branch") or "main")
        result = http_json(
            f"https://api.github.com/repos/{repository}/contents/{urllib.parse.quote(path, safe='/')}"
            f"?ref={urllib.parse.quote(branch, safe='')}",
            headers=headers,
        )
        data = result.get("data", {})
        if isinstance(data, dict) and isinstance(data.get("content"), str):
            import base64
            try:
                data["decoded_content"] = base64.b64decode(
                    data["content"].replace("\n", "")
                ).decode("utf-8")
            except (ValueError, UnicodeDecodeError):
                pass
        return result
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
    if action == "create_or_update_file":
        path = _github_path(node.input.get("path"))
        content = str(node.input.get("content", ""))
        if len(content.encode("utf-8")) > 512 * 1024:
            raise RuntimeError("file content exceeds 512 KiB safety limit")
        branch = str(node.input.get("branch") or "main")
        message = str(node.input.get("message") or f"orchestrator: update {path}")
        encoded = __import__("base64").b64encode(content.encode("utf-8")).decode("ascii")
        body = {"message": message, "content": encoded, "branch": branch}
        try:
            existing = http_json(
                f"https://api.github.com/repos/{repository}/contents/{urllib.parse.quote(path, safe='/')}"
                f"?ref={urllib.parse.quote(branch, safe='')}",
                headers=headers,
            )
            existing_sha = existing.get("data", {}).get("sha")
            if existing_sha:
                body["sha"] = existing_sha
        except Exception as exc:
            if "404" not in str(exc):
                raise
        return http_json(
            f"https://api.github.com/repos/{repository}/contents/{urllib.parse.quote(path, safe='/')}",
            method="PUT",
            body=body,
            headers=headers,
        )
    if action == "delete_file":
        path = _github_path(node.input.get("path"))
        branch = str(node.input.get("branch") or "main")
        message = str(node.input.get("message") or f"orchestrator: delete {path}")
        current = http_json(
            f"https://api.github.com/repos/{repository}/contents/{urllib.parse.quote(path, safe='/')}"
            f"?ref={urllib.parse.quote(branch, safe='')}",
            headers=headers,
        )
        sha = current.get("data", {}).get("sha")
        if not sha:
            raise RuntimeError("GitHub did not return file sha")
        return http_json(
            f"https://api.github.com/repos/{repository}/contents/{urllib.parse.quote(path, safe='/')}",
            method="DELETE",
            body={"message": message, "sha": sha, "branch": branch},
            headers=headers,
        )
    if action == "dispatch_workflow":
        workflow = str(node.input.get("workflow") or "").strip()
        ref = str(node.input.get("ref") or "main")
        if not workflow:
            raise RuntimeError("workflow is required")
        inputs = node.input.get("inputs", {})
        if not isinstance(inputs, dict):
            raise RuntimeError("workflow inputs must be an object")
        return http_json(
            f"https://api.github.com/repos/{repository}/actions/workflows/{urllib.parse.quote(workflow, safe='')}/dispatches",
            method="POST",
            body={"ref": ref, "inputs": {str(k): str(v) for k, v in inputs.items()}},
            headers=headers,
        )
    raise RuntimeError(f"GitHub action not allowlisted: {action}")

def execute_node(node: Node, goal: str, dry_run: bool) -> dict[str, Any]:
    registry = load_registry()
    spec = registry.get(node.tool, {})
    if free_only() and not dry_run:
        is_free = bool(spec.get("free_tier", False))
        if not spec and node.tool in BUILTIN_FREE_TOOLS:
            is_free = True
        if not is_free:
            raise RuntimeError(
                f"tool {node.tool} is disabled by ORCHESTRATOR_FREE_ONLY=true"
            )
    if dry_run:
        return {
            "simulated": True,
            "tool": node.tool,
            "capability": node.capability,
            "instruction": node.input.get("instruction"),
        }
    if node.tool == "gemini":
        return execute_gemini(node, goal)
    if node.tool == "openai":
        return execute_openai(node, goal)
    if node.tool == "firecrawl":
        return execute_firecrawl(node, goal)
    if node.tool == "research_bundle":
        return execute_research_bundle(node, goal)
    if node.tool == "wikipedia":
        return execute_wikipedia(node, goal)
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
        old_tool = failed_node.tool
        if candidate == old_tool:
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
                "from_tool": old_tool,
                "to_tool": candidate,
                "replan_count": workflow["replan_count"],
            },
        )
        failed_node.input["previous_tool"] = old_tool
        transition(failed_node, "ready")
        workflow["status"] = "running"
        return True
    return False

def run_one_step(workflow: dict[str, Any], approve_high_risk: bool = False) -> str:
    nodes = [Node(**node) for node in workflow['nodes']]
    validate_dag(nodes)
    registry = load_registry()
    live = bool(workflow.get('live'))
    enforce_node_policy(nodes, registry)
    workflow['status'] = 'running'
    workflow['execution_mode'] = 'live' if live else 'dry-run'
    workflow.setdefault('replan_count', 0)
    for node in nodes:
        node.input['workflow_id'] = workflow['id']

    refresh_approvals(workflow, nodes)
    if workflow.get('status') == 'failed':
        workflow['nodes'] = [asdict(node) for node in nodes]
        persist_workflow(workflow)
        return 'failed'

    eligible = ready_nodes(nodes)
    for node in eligible:
        transition(node, 'ready')
    if not eligible:
        if all(node.status == 'completed' for node in nodes):
            workflow['status'] = 'completed'
            workflow['nodes'] = [asdict(node) for node in nodes]
            append_event('workflow.completed', {'workflow_id': workflow['id']})
            persist_workflow(workflow)
            notify_issue(workflow, 'Orchestrator: workflow ' + workflow['id'] + ' completed.')
            return 'completed'
        workflow['status'] = 'waiting_approval' if any(node.status == 'waiting_approval' for node in nodes) else 'failed'
        workflow['nodes'] = [asdict(node) for node in nodes]
        persist_workflow(workflow)
        return workflow['status']

    node = sorted(eligible, key=lambda item: item.id)[0]
    if live and node.risk in {'high', 'critical'} and not approve_high_risk and not node.input.get('approval_granted'):
        transition(node, 'waiting_approval')
        workflow['status'] = 'waiting_approval'
        try:
            if not node.input.get('approval_issue'):
                node.input['approval_issue'] = create_approval_issue(workflow, node)
        except Exception as exc:
            node.error = {'type': type(exc).__name__, 'message': str(exc)}
        append_event('approval.required', {
            'workflow_id': workflow['id'],
            'node_id': node.id,
            'risk': node.risk,
            'issue': node.input.get('approval_issue'),
        })
        notify_issue(
            workflow,
            'Orchestrator: workflow ' + workflow['id'] + ' is waiting for approval on node ' + node.id + '.'
        )
        workflow['nodes'] = [asdict(item) for item in nodes]
        persist_workflow(workflow)
        return 'waiting_approval'

    transition(node, 'running')
    append_event('node.started', {'workflow_id': workflow['id'], 'node_id': node.id, 'tool': node.tool})
    attempts = node.retry_count
    while True:
        try:
            node.output = execute_node(node, workflow['goal'], dry_run=not live)
            transition(node, 'validating')
            node.output['validation'] = {'passed': True, 'checked_at': utc_now()}
            transition(node, 'completed')
            append_event('node.completed', {'workflow_id': workflow['id'], 'node_id': node.id, 'tool': node.tool})
            notify_issue(
                workflow,
                'Orchestrator: node ' + node.id + ' completed using ' + node.tool + '.'
            )
            workflow['nodes'] = [asdict(item) for item in nodes]
            persist_workflow(workflow)
            checkpoint_payload = {
                'workflow': workflow['id'],
                'node': asdict(node),
                'ts': utc_now(),
            }
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            checkpoint_path = CHECKPOINT_DIR / f"{workflow['id']}-{node.id}.json"
            write_json(checkpoint_path, checkpoint_payload)
            node.output['checkpoint'] = {
                'path': str(checkpoint_path.relative_to(ROOT)) if checkpoint_path.is_relative_to(ROOT) else str(checkpoint_path),
                'sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            }
            workflow['nodes'] = [asdict(item) for item in nodes]
            persist_workflow(workflow)
            if all(item.status == 'completed' for item in nodes):
                workflow['status'] = 'completed'
                persist_workflow(workflow)
                append_event('workflow.completed', {'workflow_id': workflow['id']})
                return 'completed'
            return 'completed_step'
        except Exception as exc:
            node.error = {
                'type': type(exc).__name__,
                'message': str(exc),
                'trace': traceback.format_exc(limit=4),
            }
            if attempts < node.max_retries:
                attempts += 1
                node.retry_count = attempts
                transition(node, 'retrying')
                append_event('node.retrying', {
                    'workflow_id': workflow['id'],
                    'node_id': node.id,
                    'attempt': attempts,
                    'error': str(exc),
                })
                time.sleep(min(2 ** attempts, 8))
                transition(node, 'ready')
                transition(node, 'running')
                continue
            transition(node, 'failed')
            append_event('node.failed', {'workflow_id': workflow['id'], 'node_id': node.id, 'error': node.error})
            notify_issue(
                workflow,
                'Orchestrator: node ' + node.id + ' failed: ' + node.error.get('message', 'unknown error')
            )
            if replan_after_failure(workflow, nodes, node, registry):
                workflow['nodes'] = [asdict(item) for item in nodes]
                persist_workflow(workflow)
                return 'replanned'
            workflow['status'] = 'failed'
            workflow['failed_node'] = node.id
            workflow['nodes'] = [asdict(item) for item in nodes]
            persist_workflow(workflow)
            return 'failed'
def run_workflow(workflow: dict[str, Any], approve_high_risk: bool = False) -> None:
    nodes = [Node(**node) for node in workflow["nodes"]]
    validate_dag(nodes)
    registry = load_registry()
    live = bool(workflow.get("live"))
    enforce_node_policy(nodes, registry)
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

        refresh_approvals(workflow, nodes)
        if workflow.get("status") == "failed":
            workflow["nodes"] = [asdict(node) for node in nodes]
            persist_workflow(workflow)
            return
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
            if live and node.risk in {"high", "critical"} and not approve_high_risk and not node.input.get("approval_granted"):
                transition(node, "waiting_approval")
                workflow["status"] = "waiting_approval"
                try:
                    if not node.input.get("approval_issue"):
                        node.input["approval_issue"] = create_approval_issue(workflow, node)
                except Exception as approval_exc:
                    node.error = {
                        "type": type(approval_exc).__name__,
                        "message": str(approval_exc),
                    }
                append_event(
                    "approval.required",
                    {
                        "workflow_id": workflow["id"],
                        "node_id": node.id,
                        "risk": node.risk,
                        "issue": node.input.get("approval_issue"),
                    },
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
                    checkpoint = {
                        "workflow": workflow["id"],
                        "node": asdict(node),
                        "ts": utc_now(),
                    }
                    checkpoint_path = CHECKPOINT_DIR / f"{workflow['id']}-{node.id}.json"
                    write_json(checkpoint_path, checkpoint)
                    checkpoint_bytes = checkpoint_path.read_bytes()
                    try:
                        checkpoint_ref = str(checkpoint_path.relative_to(ROOT))
                    except ValueError:
                        checkpoint_ref = str(checkpoint_path)
                    node.output["checkpoint"] = {
                        "path": checkpoint_ref,
                        "sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
                    }
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
                        transition(node, "running")
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
        persist_workflow(workflow)

def notify_issue(workflow: dict[str, Any], message: str) -> None:
    try:
        from issue_notify import post_issue_status
        post_issue_status(workflow, message, http_json)
    except Exception:
        return

def create_workflow(goal: str, live: bool, trigger_issue: int | None = None) -> dict[str, Any]:
    registry = load_registry()
    nodes = None
    if os.environ.get("ORCHESTRATOR_LLM_PLANNER", "true").lower() == "true" and os.environ.get("GEMINI_API_KEY"):
        try:
            from llm_planner import plan_goal
            nodes = plan_goal(goal, registry, Node, validate_dag)
            append_event("planner.llm", {"goal": goal, "nodes": len(nodes)})
        except Exception as planner_exc:
            append_event("planner.fallback", {"goal": goal, "error": str(planner_exc)})
    if nodes is None:
        nodes = deterministic_plan(goal, registry, live=live)
    validate_dag(nodes)
    return {
        "id": new_id("wf"),
        "created_at": utc_now(),
        "goal": goal,
        "status": "planning",
        "live": live,
        "execution_mode": "dry-run",
        "replan_count": 0,
        "trigger_issue": trigger_issue,
        "github_run_id": os.environ.get("ORCHESTRATOR_GITHUB_RUN_ID"),
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


def resume_pending_workflows(state: dict[str, Any], approve_high_risk: bool = False, step: bool = False) -> int:
    resumed = 0
    candidates = [
        workflow for workflow in state.get("workflows", {}).values()
        if workflow.get("status") in {"waiting_approval", "running"}
    ]
    candidates.sort(key=lambda item: item.get("updated_at") or item.get("created_at") or "")
    for workflow in candidates:
        if step:
            run_one_step(workflow, approve_high_risk=approve_high_risk)
        else:
            run_workflow(workflow, approve_high_risk=approve_high_risk)
        state["workflows"][workflow["id"]] = workflow
        state["last_workflow_id"] = workflow["id"]
        resumed += 1
        if workflow.get("status") == "failed" or step:
            break
    save_state(state)
    return resumed

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--goal', default=os.environ.get('ORCHESTRATOR_GOAL', ''))
    parser.add_argument('--workflow-id')
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--approve-high-risk', action='store_true')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--step', action='store_true')
    args = parser.parse_args()

    state = load_state()

    if args.list:
        for workflow in state.get('workflows', {}).values():
            print_summary(workflow)
        return 0

    if args.resume:
        count = resume_pending_workflows(
            state, approve_high_risk=args.approve_high_risk, step=args.step
        )
        print(json.dumps({'resumed_workflows': count}, indent=2))
        return 0

    if args.workflow_id:
        workflow = state.get('workflows', {}).get(args.workflow_id)
        if not workflow:
            raise SystemExit(f'workflow not found: {args.workflow_id}')
        if args.step:
            result = run_one_step(workflow, approve_high_risk=args.approve_high_risk)
            state['workflows'][workflow['id']] = workflow
            state['last_workflow_id'] = workflow['id']
            save_state(state)
            print_summary(workflow)
            return 0 if result not in {'failed', 'continuation_failed'} else 2
        run_workflow(workflow, approve_high_risk=args.approve_high_risk)
        state['workflows'][workflow['id']] = workflow
        state['last_workflow_id'] = workflow['id']
        save_state(state)
        print_summary(workflow)
        return 0 if workflow['status'] in {'completed', 'waiting_approval'} else 2

    if not args.goal:
        raise SystemExit('provide --goal or --workflow-id')

    live = args.live or os.environ.get('ORCHESTRATOR_LIVE', '').lower() == 'true'
    trigger_issue_raw = os.environ.get('ORCHESTRATOR_TRIGGER_ISSUE', '').strip()
    trigger_issue = int(trigger_issue_raw) if trigger_issue_raw.isdigit() else None
    workflow = create_workflow(args.goal, live=live, trigger_issue=trigger_issue)
    workflow['status'] = 'ready'
    state['workflows'][workflow['id']] = workflow
    state['last_workflow_id'] = workflow['id']
    append_event(
        'workflow.created',
        {'workflow_id': workflow['id'], 'goal': workflow['goal'], 'live': live},
    )

    if args.step:
        result = run_one_step(workflow, approve_high_risk=args.approve_high_risk)
        state['workflows'][workflow['id']] = workflow
        state['last_workflow_id'] = workflow['id']
        save_state(state)
        print_summary(workflow)
        return 0 if result not in {'failed', 'continuation_failed'} else 2

    run_workflow(workflow, approve_high_risk=args.approve_high_risk)
    state['workflows'][workflow['id']] = workflow
    state['last_workflow_id'] = workflow['id']
    save_state(state)
    print_summary(workflow)
    return 0 if workflow['status'] in {'completed', 'waiting_approval'} else 2

if __name__ == '__main__':
    raise SystemExit(main())
