#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import hmac
import json
import os
import tempfile
import time
import traceback
import urllib.error
import urllib.parse
import re
import urllib.request
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .capability_graph import (
        QUARANTINED, effective_health, load_health, record_tool_result, route_capability, save_health
    )
    from .connector_bridge import (
        ConnectorReconciliationError,
        ConnectorRequestError,
        execute_connector_bridge,
        reconcile_connector_execution,
    )
    from .evidence import build_evidence
    from .failure_policy import classify_failure, deterministic_retry_delay, retry_allowed
    from .plan_integrity import fingerprint_nodes
    from .policy_integrity import build_policy_snapshot, fingerprint_policy
    from .state_schema import CURRENT_WORKFLOW_SCHEMA_VERSION, StateSchemaError, migrate_state
    from .checkpoint_integrity import CheckpointIntegrityError, verify_checkpoint
    from .durability_barrier import DurabilityBarrierError, commit_side_effect_start
except ImportError:
    from capability_graph import (
        QUARANTINED, effective_health, load_health, record_tool_result, route_capability, save_health
    )
    from connector_bridge import (
        ConnectorReconciliationError,
        ConnectorRequestError,
        execute_connector_bridge,
        reconcile_connector_execution,
    )
    from evidence import build_evidence
    from failure_policy import classify_failure, deterministic_retry_delay, retry_allowed
    from plan_integrity import fingerprint_nodes
    from policy_integrity import build_policy_snapshot, fingerprint_policy
    from state_schema import CURRENT_WORKFLOW_SCHEMA_VERSION, StateSchemaError, migrate_state
    from checkpoint_integrity import CheckpointIntegrityError, verify_checkpoint
    from durability_barrier import DurabilityBarrierError, commit_side_effect_start

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / ".orchestrator"
STATE_FILE = STATE_DIR / "state.json"
EVENT_FILE = STATE_DIR / "events.jsonl"
CHECKPOINT_DIR = STATE_DIR / "checkpoints"
REGISTRY_FILE = ROOT / "orchestrator" / "tools.json"

MAX_NODES = 24
MAX_REPLANS = 2
DEFAULT_MAX_PARALLEL = 4
DEFAULT_MAX_EXECUTION_STEPS = 96
MAX_MAX_EXECUTION_STEPS = 256
RUNNING_RECOVERY_GRACE_SECONDS = 300
MAX_CONTEXT_BYTES = 48 * 1024
MAX_GOAL_CHARS = 4000
MAX_NODE_INTENT_BYTES = 32 * 1024
MAX_NODE_STATE_BYTES = 128 * 1024
MAX_ARTIFACTS_PER_NODE = 32
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")

TRANSITIONS = {
    "pending": {"ready", "cancelled"},
    "ready": {"running", "waiting_approval", "cancelled", "failed"},
    "running": {"validating", "waiting_approval", "retrying", "failed", "cancelled", "ready"},
    "validating": {"completed", "retrying", "failed"},
    "waiting_approval": {"ready", "failed", "cancelled"},
    "retrying": {"ready", "failed"},
    "failed": {"replanning", "reconciling", "ready", "cancelled"},
    "replanning": {"ready", "failed", "cancelled"},
    "reconciling": {"completed", "ready", "failed", "cancelled"},
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
    contract: dict[str, Any] = field(default_factory=dict)

def normalize_goal(goal: Any) -> str:
    value = str(goal or "").strip()
    if not value:
        raise ValueError("goal is required")
    if len(value) > MAX_GOAL_CHARS:
        raise ValueError(f"goal exceeds {MAX_GOAL_CHARS} characters")
    return value


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        try:
            dir_fd = os.open(path.parent, os.O_DIRECTORY)
        except (AttributeError, OSError):
            dir_fd = None
        if dir_fd is not None:
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

def append_event(event_type: str, payload: dict[str, Any]) -> None:
    """Best-effort observability; event-log failure must not break durable state."""
    try:
        EVENT_FILE.parent.mkdir(parents=True, exist_ok=True)
        entry = {"ts": utc_now(), "event_type": event_type, "payload": payload}
        with EVENT_FILE.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            handle.flush()
    except (OSError, TypeError, ValueError):
        return

def execution_key(workflow: dict[str, Any], node: Node) -> str:
    raw = f"{workflow['id']}:{node.id}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def side_effecting(node: Node, registry: dict[str, dict[str, Any]]) -> bool:
    if node.tool == "github":
        action = str(node.input.get("action") or "metadata")
        return action in {
            "create_issue", "create_or_update_file", "delete_file", "dispatch_workflow"
        }
    if node.tool == "local_validator":
        return False
    return bool(registry.get(node.tool, {}).get("side_effects")) or node.capability in {
        "deploy", "publish", "delete", "external_write", "send"
    }

def mark_execution_prepared(workflow: dict[str, Any], node: Node) -> str:
    key = execution_key(workflow, node)
    history = workflow.setdefault("executions", {})
    record = history.get(key)
    if record:
        return key
    history[key] = {
        "node_id": node.id,
        "status": "prepared",
        "prepared_at": utc_now(),
    }
    return key

def mark_execution_started(workflow: dict[str, Any], node: Node, key: str) -> None:
    record = workflow.setdefault("executions", {}).setdefault(key, {})
    record.update({
        "node_id": node.id,
        "status": "started",
        "started_at": utc_now(),
    })

def mark_execution_completed(
    workflow: dict[str, Any], key: str, output: dict[str, Any] | None = None
) -> None:
    record = workflow.setdefault("executions", {}).setdefault(key, {})
    record.update({
        "status": "completed",
        "completed_at": utc_now(),
    })
    if output is not None:
        record["output"] = output


def mark_execution_not_applied(
    workflow: dict[str, Any], key: str
) -> None:
    record = workflow.setdefault("executions", {}).setdefault(key, {})
    record.update({
        "status": "not_applied",
        "reconciled_at": utc_now(),
    })


def load_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        return migrate_state({"version": 4, "workflows": {}, "last_workflow_id": None})
    try:
        raw = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return migrate_state(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, StateSchemaError) as exc:
        raise RuntimeError(f"invalid orchestrator state: {exc}") from exc

def save_state(state: dict[str, Any]) -> None:
    try:
        write_json(STATE_FILE, migrate_state(state))
    except StateSchemaError as exc:
        raise RuntimeError(f"invalid orchestrator state: {exc}") from exc

def persist_workflow(workflow: dict[str, Any]) -> None:
    state = load_state()
    current_run_id = os.environ.get("ORCHESTRATOR_GITHUB_RUN_ID", "").strip()
    current_run_attempt_raw = os.environ.get("ORCHESTRATOR_GITHUB_RUN_ATTEMPT", "").strip()
    current_run_attempt = (
        int(current_run_attempt_raw) if current_run_attempt_raw.isdigit() else None
    )
    if current_run_id:
        previous_run_id = str(workflow.get("github_run_id") or "").strip() or None
        previous_run_attempt = workflow.get("github_run_attempt")
        origin_run_id = (
            str(workflow.get("origin_github_run_id") or "").strip()
            or previous_run_id
            or current_run_id
        )
        origin_run_attempt = workflow.get("origin_github_run_attempt")
        if origin_run_attempt is None:
            origin_run_attempt = (
                previous_run_attempt
                if previous_run_attempt is not None
                else current_run_attempt
            )
        workflow["origin_github_run_id"] = origin_run_id
        workflow["origin_github_run_attempt"] = origin_run_attempt
        workflow["github_run_id"] = current_run_id
        workflow["github_run_attempt"] = current_run_attempt
        if (
            previous_run_id != current_run_id
            or previous_run_attempt != current_run_attempt
        ):
            append_event("workflow.worker_run_rebound", {
                "workflow_id": workflow["id"],
                "previous_run_id": previous_run_id,
                "previous_run_attempt": previous_run_attempt,
                "current_run_id": current_run_id,
                "current_run_attempt": current_run_attempt,
                "origin_run_id": origin_run_id,
                "origin_run_attempt": origin_run_attempt,
            })
    workflow["updated_at"] = utc_now()
    state.setdefault("workflows", {})[workflow["id"]] = workflow
    state["last_workflow_id"] = workflow["id"]
    save_state(state)

class ExecutionBudgetExceeded(RuntimeError):
    pass


def configured_max_execution_steps() -> int:
    raw = os.environ.get("ORCHESTRATOR_MAX_EXECUTION_STEPS", "").strip()
    if not raw:
        return DEFAULT_MAX_EXECUTION_STEPS
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("ORCHESTRATOR_MAX_EXECUTION_STEPS must be an integer") from exc
    if value < 1 or value > MAX_MAX_EXECUTION_STEPS:
        raise ValueError(
            f"ORCHESTRATOR_MAX_EXECUTION_STEPS must be between 1 and {MAX_MAX_EXECUTION_STEPS}"
        )
    return value


def execution_budget(workflow: dict[str, Any]) -> dict[str, Any]:
    budget = workflow.setdefault("execution_budget", {})
    try:
        maximum = int(budget.get("max_steps", DEFAULT_MAX_EXECUTION_STEPS))
    except (TypeError, ValueError):
        maximum = DEFAULT_MAX_EXECUTION_STEPS
    try:
        used = int(budget.get("used_steps", 0))
    except (TypeError, ValueError):
        used = 0
    budget["max_steps"] = max(1, min(maximum, MAX_MAX_EXECUTION_STEPS))
    budget["used_steps"] = max(0, min(used, MAX_MAX_EXECUTION_STEPS))
    return budget


def reserve_execution_steps(
    workflow: dict[str, Any],
    node_ids: list[str],
) -> dict[str, int]:
    """Atomically admit a batch against the remaining durable execution budget."""
    ids = [str(node_id) for node_id in node_ids]
    if not ids:
        return {}
    budget = execution_budget(workflow)
    available = budget["max_steps"] - budget["used_steps"]
    if len(ids) > available:
        raise ExecutionBudgetExceeded(
            f"execution step budget cannot admit batch ({len(ids)} requested, {available} available)"
        )
    start = budget["used_steps"]
    reservations = {
        node_id: start + offset
        for offset, node_id in enumerate(ids, start=1)
    }
    budget["used_steps"] = start + len(ids)
    budget["last_node_id"] = ids[-1]
    budget["last_reserved_at"] = utc_now()
    for node_id, used in reservations.items():
        try:
            append_event("workflow.budget_step_reserved", {
                "workflow_id": workflow["id"],
                "node_id": node_id,
                "used_steps": used,
                "max_steps": budget["max_steps"],
                "batch": True,
            })
        except OSError:
            pass
    return reservations


def reserve_execution_step(workflow: dict[str, Any], node_id: str) -> int:
    budget = execution_budget(workflow)
    if budget["used_steps"] >= budget["max_steps"]:
        raise ExecutionBudgetExceeded(
            f"execution step budget exhausted ({budget['used_steps']}/{budget['max_steps']})"
        )
    budget["used_steps"] += 1
    budget["last_node_id"] = str(node_id)
    budget["last_reserved_at"] = utc_now()
    append_event("workflow.budget_step_reserved", {
        "workflow_id": workflow["id"],
        "node_id": str(node_id),
        "used_steps": budget["used_steps"],
        "max_steps": budget["max_steps"],
    })
    return budget["used_steps"]


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
    "wikipedia", "webhook", "github", "connector_bridge", "local_validator", "artifact_verifier", "noop",
}
BUILTIN_FREE_TOOLS = {"gemini", "research_bundle", "wikipedia", "github", "connector_bridge", "local_validator", "artifact_verifier", "noop"}

def required_risk(node: Node, registry: dict[str, dict[str, Any]]) -> str:
    floor = classify_risk(node.capability)
    side_effects = set(registry.get(node.tool, {}).get("side_effects", []))
    if "external_request" in side_effects:
        floor = "high"
    if node.tool == "github" and node.input.get("action") in {"create_issue", "create_or_update_file", "delete_file", "dispatch_workflow"}:
        floor = "high"
    return floor

def enforce_node_policy(
    nodes: list[Node],
    registry: dict[str, dict[str, Any]],
    live: bool = False,
    *,
    route: bool = True,
) -> None:
    for node in nodes:
        if node.tool not in BUILTIN_TOOLS and node.tool not in registry:
            raise ValueError(f"unregistered tool for {node.id}: {node.tool}")
        floor = required_risk(node, registry)
        if RISK_ORDER.get(node.risk, 0) < RISK_ORDER[floor]:
            node.risk = floor
        if route and registry.get(f"capability:{node.capability}"):
            node.tool = route_tool(
                node.capability,
                registry,
                live=live,
                preferred=node.tool,
            )
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

def tool_health_path() -> Path:
    return STATE_DIR / "tool_health.json"


def load_tool_health() -> dict[str, Any]:
    return load_health(tool_health_path())


def update_tool_health(node: Node, success: bool, registry: dict[str, dict[str, Any]]) -> dict[str, Any]:
    health = load_tool_health()
    updated = record_tool_result(
        health,
        node.tool,
        success=success,
        side_effecting=side_effecting(node, registry),
    )
    save_health(tool_health_path(), health)
    append_event("tool.health", {
        "tool": node.tool,
        "status": updated.get("status"),
        "failure_streak": updated.get("failure_streak", 0),
    })
    return updated


def route_tool(
    capability: str,
    registry: dict[str, dict[str, Any]],
    *,
    live: bool,
    preferred: str | None = None,
    exclude: set[str] | None = None,
) -> str:
    return route_capability(
        capability,
        registry,
        load_tool_health(),
        live=live,
        preferred=preferred,
        exclude=exclude,
    )


def preflight_node(
    node: Node,
    registry: dict[str, dict[str, Any]],
    *,
    live: bool,
) -> None:
    """Validate local execution prerequisites before crossing any side-effect barrier."""
    if node.tool not in BUILTIN_TOOLS and node.tool not in registry:
        raise RuntimeError(f"tool {node.tool} is not registered")
    if not live:
        return

    spec = registry.get(node.tool, {})
    is_free = bool(spec.get("free_tier", False))
    if not spec and node.tool in BUILTIN_FREE_TOOLS:
        is_free = True
    if free_only() and not is_free:
        raise RuntimeError(f"tool {node.tool} is disabled by ORCHESTRATOR_FREE_ONLY=true")

    env_name = spec.get("required_env")
    if env_name and not os.environ.get(env_name):
        if not (node.tool == "webhook" and node.input.get("url")):
            raise RuntimeError(f"{env_name} is required for tool {node.tool}")
    secret_env = spec.get("secret_env")
    if secret_env and not os.environ.get(secret_env):
        raise RuntimeError(f"{secret_env} is required for tool {node.tool}")

    health = load_tool_health()
    if effective_health(health, node.tool) == QUARANTINED:
        raise RuntimeError(f"tool {node.tool} is quarantined by the capability health policy")

    if node.tool == "webhook":
        url = str(node.input.get("url") or os.environ.get("ORCHESTRATOR_WEBHOOK_URL") or "").strip()
        if urllib.parse.urlparse(url).scheme != "https":
            raise RuntimeError("webhook execution requires an HTTPS URL")
    if node.tool == "connector_bridge":
        url = str(os.environ.get("ORCHESTRATOR_CONNECTOR_BRIDGE_URL") or "").strip()
        if urllib.parse.urlparse(url).scheme != "https":
            raise RuntimeError("connector bridge execution requires an HTTPS URL")


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
        if cap_spec:
            preferred = route_tool(capability, registry, live=live)
        else:
            preferred = {
                "research": "research_bundle",
                "analyze": "gemini",
                "draft": "gemini",
                "spec": "gemini",
                "build": "github",
                "test": "github",
                "deploy": "webhook",
                "validate": "local_validator",
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

def _node_intent_input(input_value: dict[str, Any]) -> dict[str, Any]:
    volatile = {
        "workflow_id",
        "context",
        "repair_feedback",
        "approval_issue",
        "approval_granted",
        "approval_fingerprint",
        "approval_actor",
        "approval_approved_at",
    }
    return {
        key: value
        for key, value in input_value.items()
        if key not in volatile
    }


def validate_dag(nodes: list[Node]) -> None:
    if not nodes or len(nodes) > MAX_NODES:
        raise ValueError(f"invalid node count: {len(nodes)}")
    ids = {node.id for node in nodes}
    if len(ids) != len(nodes):
        raise ValueError("duplicate node id")
    by_id = {node.id: node for node in nodes}
    for node in nodes:
        intent_serialized = json.dumps(
            {
                "id": node.id,
                "capability": node.capability,
                "tool": node.tool,
                "depends_on": list(node.depends_on),
                "risk": node.risk,
                "input": _node_intent_input(node.input),
                "contract": node.contract,
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
        if len(intent_serialized.encode("utf-8")) > MAX_NODE_INTENT_BYTES:
            raise ValueError(
                f"node {node.id} intent exceeds {MAX_NODE_INTENT_BYTES} bytes"
            )
        state_serialized = json.dumps(
            asdict(node),
            ensure_ascii=False,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
        if len(state_serialized.encode("utf-8")) > MAX_NODE_STATE_BYTES:
            raise ValueError(
                f"node {node.id} runtime state exceeds {MAX_NODE_STATE_BYTES} bytes"
            )
        if len(node.input.get("artifacts", [])) > MAX_ARTIFACTS_PER_NODE:
            raise ValueError(
                f"node {node.id} exceeds {MAX_ARTIFACTS_PER_NODE} artifacts"
            )
        if not SAFE_ID_RE.fullmatch(str(node.id or "")):
            raise ValueError(f"unsafe node id: {node.id!r}")
        if not SAFE_ID_RE.fullmatch(str(node.capability or "")):
            raise ValueError(f"unsafe capability name: {node.capability!r}")
        if not SAFE_ID_RE.fullmatch(str(node.tool or "")):
            raise ValueError(f"unsafe tool name: {node.tool!r}")
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
    if node.capability == "validate":
        instruction = (
            "Act as a strict workflow validator. Return only JSON with "
            "passed (boolean), checks (array), findings (array), and next_action. "
            "Set passed=true only when the dependency evidence satisfies the goal."
        )
    else:
        instruction = "Act as a conservative workflow worker. Return JSON with result, risks, next_action."
    payload = {
        "contents": [{
            "parts": [{
                "text": (
                    instruction + "\n"
                    f"GOAL: {goal}\n"
                    f"INSTRUCTION: {node.input.get('instruction', '')}\n"
                    f"CONTEXT: {compact_json(node.input.get('context', {}), limit=24 * 1024)}"
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
            f"GOAL: {goal}\\nINSTRUCTION: {node.input.get('instruction', '')}\\n"
            f"CONTEXT: {compact_json(node.input.get('context', {}), limit=24 * 1024)}"
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

def execute_artifact_verifier(node: Node, goal: str) -> dict[str, Any]:
    artifacts = node.input.get("artifacts") or []
    if not isinstance(artifacts, list):
        raise RuntimeError("artifacts must be an array")
    checks = []
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            raise RuntimeError(f"artifact {index} must be an object")
        kind = str(artifact.get("type") or "").strip().lower()
        if kind == "url":
            url = str(artifact.get("url") or "").strip()
            if not url:
                raise RuntimeError(f"artifact {index} url is required")
            result = http_json(url, timeout=30)
            status = result.get("status_code")
            ok = isinstance(status, int) and 200 <= status < 300
            contains = artifact.get("contains")
            if ok and contains:
                data = result.get("data")
                text_blob = compact_json(data, limit=24 * 1024)
                ok = str(contains) in text_blob
            checks.append({
                "index": index,
                "type": kind,
                "status_code": status,
                "passed": ok,
            })
            if not ok:
                raise RuntimeError(f"artifact {index} URL verification failed")
        elif kind == "github_file":
            path = _github_path(artifact.get("path"))
            read_node = Node(
                id=f"{node.id}-artifact-{index}",
                capability="execute",
                tool="github",
                input={
                    "action": "read_file",
                    "path": path,
                    "branch": str(artifact.get("branch") or "main"),
                },
            )
            result = execute_github(read_node)
            status = result.get("status_code")
            ok = isinstance(status, int) and 200 <= status < 300
            checks.append({
                "index": index,
                "type": kind,
                "path": path,
                "status_code": status,
                "passed": ok,
            })
            if not ok:
                raise RuntimeError(f"artifact {index} GitHub file verification failed")
        elif kind == "local_file":
            path = _safe_local_path(str(artifact.get("path") or ""))
            ok = path.is_file()
            checks.append({
                "index": index,
                "type": kind,
                "path": str(path.relative_to(ROOT)),
                "passed": ok,
            })
            if not ok:
                raise RuntimeError(f"artifact {index} local file does not exist")
        else:
            raise RuntimeError(f"unsupported artifact type: {kind or 'missing'}")
    return {
        "passed": True,
        "checks": checks,
        "verified_count": len(checks),
        "goal": goal,
        "verifier": "artifact_verifier",
    }


def _safe_local_path(value: str) -> Path:
    candidate = (ROOT / value.lstrip("/")).resolve()
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("invalid local artifact path") from exc
    return candidate


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
    approval_fingerprint = fingerprint_nodes([node])
    node.input["approval_fingerprint"] = approval_fingerprint
    node.input["approval_actor"] = None
    node.input["approval_approved_at"] = None
    result = http_json(
        f"https://api.github.com/repos/{github_repository()}/issues",
        method="POST",
        body={
            "title": f"[ORCHESTRATOR APPROVAL] {workflow['id']} / {node.id}",
            "body": (
                "High-risk orchestration action is waiting for explicit approval.\n\n"
                f"Workflow: {workflow['id']}\nNode: {node.id}\n"
                f"Capability: {node.capability}\nTool: {node.tool}\n"
                f"Goal: {workflow['goal']}\n"
                f"Approval fingerprint: {approval_fingerprint}\n\n"
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
    approval_event = os.environ.get("ORCHESTRATOR_APPROVAL_EVENT", "").lower() == "true"
    approval_issue_raw = os.environ.get("ORCHESTRATOR_APPROVAL_ISSUE", "").strip()
    approval_issue = int(approval_issue_raw) if approval_issue_raw.isdigit() else None
    for node in nodes:
        if node.status != "waiting_approval":
            continue
        issue_number = node.input.get("approval_issue")
        if not issue_number:
            continue
        if approval_event and approval_issue != int(issue_number):
            continue
        if not approval_event:
            append_event(
                "approval.unverified_label",
                {
                    "workflow_id": workflow["id"],
                    "node_id": node.id,
                    "issue": issue_number,
                },
            )
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
            approved_fingerprint = str(node.input.get("approval_fingerprint") or "").strip()
            current_fingerprint = fingerprint_nodes([node])
            if not approved_fingerprint or approved_fingerprint != current_fingerprint:
                node.input["approval_granted"] = False
                node.input["approval_actor"] = None
                node.input["approval_approved_at"] = None
                node.input["approval_issue"] = None
                node.input["approval_fingerprint"] = None
                transition(node, "ready")
                workflow["status"] = "running"
                append_event(
                    "approval.stale",
                    {
                        "workflow_id": workflow["id"],
                        "node_id": node.id,
                        "issue": issue_number,
                    },
                )
                continue
            node.input["approval_granted"] = True
            node.input["approval_actor"] = os.environ.get("GITHUB_ACTOR") or None
            node.input["approval_approved_at"] = utc_now()
            transition(node, "ready")
            workflow["status"] = "running"
            append_event(
                "approval.approved",
                {
                    "workflow_id": workflow["id"],
                    "node_id": node.id,
                    "issue": issue_number,
                    "approval_fingerprint": approved_fingerprint,
                    "actor": node.input["approval_actor"],
                },
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
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
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

def compact_json(value: Any, limit: int = MAX_CONTEXT_BYTES) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if len(raw.encode("utf-8")) <= limit:
        return raw
    clipped = raw.encode("utf-8")[:limit].decode("utf-8", "ignore")
    return clipped + "...[truncated]"


def build_node_context(nodes: list[Node], node: Node) -> dict[str, Any]:
    by_id = {item.id: item for item in nodes}
    dependencies: dict[str, Any] = {}
    for dep_id in node.depends_on:
        dep = by_id[dep_id]
        evidence = dep.output.get("evidence", {}) if isinstance(dep.output, dict) else {}
        dependencies[dep_id] = {
            "capability": dep.capability,
            "tool": dep.tool,
            "status": dep.status,
            "output": compact_json(dep.output, limit=12 * 1024),
            "error": dep.error,
            "evidence_sha256": evidence.get("evidence_sha256"),
        }
    return {
        "goal": node.input.get("goal", ""),
        "dependencies": dependencies,
        "contract": node.contract,
        "repair_feedback": node.input.get("repair_feedback", {}),
    }


def execute_local_validator(node: Node, goal: str) -> dict[str, Any]:
    context = node.input.get("context") or {}
    dependencies = context.get("dependencies", {}) if isinstance(context, dict) else {}
    checks = []
    passed = True
    for dep_id, dep in dependencies.items():
        status = dep.get("status") if isinstance(dep, dict) else None
        output = dep.get("output") if isinstance(dep, dict) else ""
        ok = status == "completed" and bool(output)
        checks.append({
            "node": dep_id,
            "completed": status == "completed",
            "has_output": bool(output),
            "passed": ok,
        })
        passed = passed and ok
    if not dependencies:
        checks.append({
            "node": "workflow",
            "completed": True,
            "has_output": True,
            "passed": True,
        })
    contract = node.contract or {}
    required_fields = contract.get("required_fields", [])
    if required_fields:
        for field_name in required_fields:
            found = False
            for dep in dependencies.values():
                candidate = dep.get("output") if isinstance(dep, dict) else None
                try:
                    parsed = json.loads(candidate) if isinstance(candidate, str) else candidate
                except json.JSONDecodeError:
                    parsed = candidate
                current: Any = parsed
                for part in str(field_name).split("."):
                    if isinstance(current, dict) and part in current:
                        current = current[part]
                    else:
                        current = None
                        break
                if current is not None:
                    found = True
                    break
            checks.append({"check": f"contract:{field_name}", "passed": found})
            passed = passed and found
    min_sources = contract.get("min_sources")
    if min_sources is not None:
        source_count = 0
        for dep in dependencies.values():
            candidate = dep.get("output") if isinstance(dep, dict) else None
            try:
                parsed = json.loads(candidate) if isinstance(candidate, str) else candidate
            except json.JSONDecodeError:
                parsed = candidate
            if isinstance(parsed, dict) and isinstance(parsed.get("sources"), dict):
                source_count = max(source_count, len(parsed["sources"]))
        ok = source_count >= int(min_sources)
        checks.append({
            "check": "contract:min_sources",
            "actual": source_count,
            "required": int(min_sources),
            "passed": ok,
        })
        passed = passed and ok
    return {
        "passed": passed,
        "checks": checks,
        "goal": goal,
        "validator": "local_validator",
    }


def extract_first_llm_json(output: dict[str, Any]) -> dict[str, Any] | None:
    candidates = output.get("candidates")
    if not isinstance(candidates, list):
        return None
    for candidate in candidates:
        parts = candidate.get("content", {}).get("parts", []) if isinstance(candidate, dict) else []
        for part in parts:
            text_value = part.get("text") if isinstance(part, dict) else None
            if not text_value:
                continue
            try:
                value = json.loads(str(text_value))
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    return None


def validate_node_output(node: Node, output: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(output, dict):
        raise RuntimeError("node output must be an object")
    contract = node.contract or {}
    if output.get("simulated") is True and not contract:
        return {
            "passed": True,
            "checks": [{"check": "dry_run_simulation", "passed": True}],
            "checked_at": utc_now(),
        }

    checks = []
    required_fields = contract.get("required_fields", [])
    if required_fields:
        if not isinstance(required_fields, list):
            raise RuntimeError("contract.required_fields must be an array")
        for field_name in required_fields:
            current: Any = output
            for part in str(field_name).split("."):
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    current = None
                    break
            ok = current is not None
            checks.append({"check": f"required:{field_name}", "passed": ok})
            if not ok:
                raise RuntimeError(f"contract field missing: {field_name}")

    min_sources = contract.get("min_sources")
    if min_sources is not None:
        sources = output.get("sources")
        count = len(sources) if isinstance(sources, dict) else 0
        ok = count >= int(min_sources)
        checks.append({"check": "min_sources", "actual": count, "required": int(min_sources), "passed": ok})
        if not ok:
            raise RuntimeError(f"contract requires at least {min_sources} sources")

    for key in ("status_code",):
        value = output.get(key)
        if isinstance(value, int):
            ok = 200 <= value < 300
            checks.append({"check": key, "value": value, "passed": ok})
            if not ok:
                raise RuntimeError(f"{key}={value} is not successful")

    response = output.get("response")
    if isinstance(response, dict):
        nested = response.get("status_code")
        if isinstance(nested, int):
            ok = 200 <= nested < 300
            checks.append({"check": "response.status_code", "value": nested, "passed": ok})
            if not ok:
                raise RuntimeError(f"response.status_code={nested} is not successful")

    if node.tool == "research_bundle":
        sources = output.get("sources")
        ok = isinstance(sources, dict) and bool(sources)
        checks.append({"check": "research_sources", "passed": ok})
        if not ok:
            raise RuntimeError("research bundle returned no sources")

    if node.tool in {"gemini", "openai"}:
        has_payload = bool(
            output.get("candidates")
            or output.get("output")
            or output.get("data")
            or output.get("text")
        )
        checks.append({"check": "llm_payload", "passed": has_payload})
        if not has_payload:
            raise RuntimeError("LLM adapter returned no usable payload")
        if node.capability == "validate" and node.tool == "gemini":
            verdict = extract_first_llm_json(output)
            if not isinstance(verdict, dict):
                raise RuntimeError("semantic validator returned no JSON verdict")
            passed = verdict.get("passed")
            checks.append({"check": "semantic_verdict", "passed": passed is True})
            if passed is not True:
                raise RuntimeError("semantic validator rejected dependency evidence")

    if node.tool == "connector_bridge":
        bridge_response = output.get("response")
        if isinstance(bridge_response, dict) and bridge_response.get("ok") is False:
            raise RuntimeError(str(bridge_response.get("error") or "connector bridge rejected request"))

    if node.tool == "local_validator":
        ok = output.get("passed") is True
        checks.append({"check": "semantic_validation", "passed": ok})
        if not ok:
            raise RuntimeError("semantic validation failed")

    postconditions = contract.get("postconditions", [])
    if postconditions:
        if not isinstance(postconditions, list):
            raise RuntimeError("contract.postconditions must be an array")
        for index, condition in enumerate(postconditions):
            if not isinstance(condition, dict):
                raise RuntimeError(f"contract.postconditions[{index}] must be an object")
            kind = str(condition.get("type") or "").strip().lower()
            field = str(condition.get("field") or "").strip()
            current: Any = output
            if field:
                for part in field.split("."):
                    if isinstance(current, dict) and part in current:
                        current = current[part]
                    else:
                        current = None
                        break
            if kind == "field_exists":
                passed = bool(field) and current is not None
            elif kind == "field_equals":
                passed = bool(field) and current == condition.get("value")
            elif kind == "field_in":
                values = condition.get("values")
                if not isinstance(values, list):
                    raise RuntimeError(f"contract.postconditions[{index}].values must be an array")
                passed = bool(field) and current in values
            elif kind == "non_empty":
                passed = bool(field) and bool(current)
            elif kind == "http_status":
                if not isinstance(current, int):
                    raise RuntimeError(
                        f"postcondition {index} field {field or '<root>'} is not an integer HTTP status"
                    )
                minimum = int(condition.get("min", 200))
                maximum = int(condition.get("max", 299))
                passed = minimum <= current <= maximum
            else:
                raise RuntimeError(f"unsupported postcondition type: {kind or 'missing'}")
            check = {
                "check": f"postcondition:{index}:{kind}",
                "field": field,
                "passed": passed,
            }
            if kind == "http_status":
                check["value"] = current
            checks.append(check)
            if not passed:
                raise RuntimeError(f"postcondition {index} failed: {kind}")
    return {
        "passed": True,
        "checks": checks,
        "acceptance": {
            "postconditions": postconditions,
            "passed": True,
        },
        "checked_at": utc_now(),
    }


def node_success_checkpoint(workflow: dict[str, Any], node: Node) -> None:
    evidence = build_evidence(
        workflow["id"],
        node.id,
        node.capability,
        node.tool,
        node.output,
        node.output.get("validation", {}),
        node.input.get("artifacts"),
    )
    workflow.setdefault("evidence", {})[node.id] = evidence
    node.output["evidence"] = evidence
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "workflow": workflow["id"],
        "node": asdict(node),
        "ts": utc_now(),
    }
    if not SAFE_ID_RE.fullmatch(str(workflow.get("id") or "")):
        raise RuntimeError("unsafe workflow id for checkpoint path")
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


def verify_completed_checkpoints(
    workflow: dict[str, Any],
    nodes: list[Node],
) -> bool:
    statuses = []
    for node in nodes:
        if node.status != "completed":
            continue
        try:
            result = verify_checkpoint(
                ROOT,
                asdict(node),
                expected_workflow_id=workflow["id"],
                allowed_dir=CHECKPOINT_DIR,
            )
            statuses.append(result)
        except CheckpointIntegrityError as exc:
            workflow["checkpoint_integrity"] = "failed"
            workflow["checkpoint_error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "node_id": node.id,
            }
            append_event("workflow.checkpoint_drift", {
                "workflow_id": workflow["id"],
                "node_id": node.id,
                "error": str(exc),
            })
            workflow["status"] = "failed"
            return False
    workflow["checkpoint_integrity"] = (
        "verified" if statuses and all(item.get("verified") for item in statuses)
        else "legacy_unverified"
    )
    return True


def ensure_plan_integrity(workflow: dict[str, Any], nodes: list[Node]) -> bool:
    fingerprint = fingerprint_nodes(nodes)
    previous = str(workflow.get("plan_fingerprint") or "").strip()
    if not previous:
        workflow["plan_fingerprint"] = fingerprint
        workflow["plan_integrity"] = "initialized"
        return True
    if previous == fingerprint:
        workflow["plan_integrity"] = "verified"
        return True
    workflow["plan_integrity"] = "drift_detected"
    workflow["plan_drift"] = {
        "expected": previous,
        "actual": fingerprint,
        "detected_at": utc_now(),
    }
    workflow["status"] = "failed"
    append_event("workflow.plan_drift", {
        "workflow_id": workflow["id"],
        "expected": previous,
        "actual": fingerprint,
    })
    return False


def connector_failure_policy(node: Node, exc: Exception) -> tuple[bool, bool]:
    if not isinstance(exc, ConnectorRequestError) or not exc.uncertain:
        return True, False
    return bool(exc.retry_allowed), True


def refresh_policy_snapshot(
    workflow: dict[str, Any],
    nodes: list[Node],
    registry: dict[str, dict[str, Any]],
    *,
    status: str = "refreshed",
) -> None:
    snapshot = build_policy_snapshot(
        registry,
        nodes,
        live=bool(workflow.get("live")),
    )
    workflow["route_snapshot"] = snapshot
    workflow["policy_fingerprint"] = fingerprint_policy(snapshot)
    workflow["policy_integrity"] = status


def ensure_policy_integrity(
    workflow: dict[str, Any],
    nodes: list[Node],
    registry: dict[str, dict[str, Any]],
) -> bool:
    snapshot = build_policy_snapshot(
        registry,
        nodes,
        live=bool(workflow.get("live")),
    )
    fingerprint = fingerprint_policy(snapshot)
    previous = str(workflow.get("policy_fingerprint") or "").strip()
    if not previous:
        workflow["route_snapshot"] = snapshot
        workflow["policy_fingerprint"] = fingerprint
        workflow["policy_integrity"] = "initialized"
        append_event("workflow.policy_initialized", {
            "workflow_id": workflow["id"],
            "policy_fingerprint": fingerprint,
        })
        return True
    if previous != fingerprint:
        workflow["policy_integrity"] = "drift_detected"
        workflow["policy_drift"] = {
            "expected": previous,
            "actual": fingerprint,
            "detected_at": utc_now(),
        }
        workflow["status"] = "failed"
        append_event("workflow.policy_drift", {
            "workflow_id": workflow["id"],
            "expected": previous,
            "actual": fingerprint,
        })
        return False
    workflow["route_snapshot"] = snapshot
    workflow["policy_integrity"] = "verified"
    return True


def execution_failure_policy(
    node: Node,
    exc: Exception,
    registry: dict[str, dict[str, Any]],
    *,
    dry_run: bool,
) -> tuple[str, bool, bool]:
    failure_class = classify_failure(exc)
    explicit_retry = getattr(exc, "retry_allowed", None)
    can_retry = retry_allowed(
        failure_class,
        explicitly_retryable=explicit_retry,
    )
    connector_can_retry, uncertain = connector_failure_policy(node, exc)
    if uncertain:
        failure_class = "uncertain"
        can_retry = connector_can_retry
        node.error["execution_uncertain"] = True
        node.error["reconciliation_required"] = True

    if side_effecting(node, registry) and not dry_run:
        connector_safe_retry = (
            isinstance(exc, ConnectorRequestError)
            and uncertain
            and connector_can_retry
        )
        if not connector_safe_retry:
            node.error["post_start_side_effect_failure"] = True
            node.error["retry_blocked_after_side_effect_start"] = True
            node.error["replan_blocked_after_side_effect_start"] = True
            can_retry = False
            if failure_class in {"transient", "intermittent", "dependency"}:
                failure_class = "uncertain"
                uncertain = True
                node.error["execution_uncertain"] = True
                node.error["reconciliation_required"] = True

    node.error["failure_class"] = failure_class
    node.error["retry_allowed"] = can_retry
    return failure_class, can_retry, uncertain

def execute_with_retries(node: Node, goal: str, dry_run: bool) -> tuple[bool, dict[str, Any] | None]:
    registry = load_registry()
    attempts = node.retry_count
    while True:
        try:
            output = execute_node(node, goal, dry_run=dry_run)
            node.output = output
            output["validation"] = validate_node_output(node, output)
            transition(node, "validating")
            transition(node, "completed")
            return True, None
        except Exception as exc:
            node.error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "trace": traceback.format_exc(limit=4),
            }
            failure_class, can_retry, uncertain = execution_failure_policy(
                node,
                exc,
                registry,
                dry_run=dry_run,
            )
            if attempts < node.max_retries and can_retry:
                attempts += 1
                node.retry_count = attempts
                transition(node, "retrying")
                delay = deterministic_retry_delay(
                    str(node.input.get("workflow_id") or ""),
                    node.id,
                    attempts,
                )
                append_event("node.retrying", {
                    "workflow_id": node.input.get("workflow_id"),
                    "node_id": node.id,
                    "attempt": attempts,
                    "error": str(exc),
                    "failure_class": failure_class,
                    "retry_delay": delay,
                })
                time.sleep(delay)
                transition(node, "ready")
                transition(node, "running")
                continue
            transition(node, "failed")
            return False, node.error


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
    if dry_run and node.tool not in {"local_validator"}:
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
    if node.tool == "connector_bridge":
        return execute_connector_bridge(node, goal, dry_run)
    if node.tool == "artifact_verifier":
        return execute_artifact_verifier(node, goal)
    if node.tool == "local_validator":
        return execute_local_validator(node, goal)
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
        if node.status in {"pending", "ready"}
        and all(dep in completed for dep in node.depends_on)
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

    old_tool = failed_node.tool
    try:
        candidate = route_tool(
            failed_node.capability,
            registry,
            live=bool(workflow.get("live")),
            exclude={old_tool},
        )
    except ValueError:
        return False
    if candidate == old_tool:
        return False

    transition(failed_node, "replanning")
    old_error = dict(failed_node.error)
    old_output = dict(failed_node.output)
    old_next_action = (
        old_output.get("next_action") if isinstance(old_output, dict) else None
    )

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
    workflow.setdefault("repair_feedback", {})[failed_node.id] = {
        "tool": old_tool,
        "error": old_error,
        "output": compact_json(old_output, limit=12 * 1024),
        "next_action": old_next_action,
    }
    append_event("node.repair_feedback", {
        "workflow_id": workflow["id"],
        "node_id": failed_node.id,
        "previous_tool": old_tool,
    })
    workflow["plan_fingerprint"] = fingerprint_nodes(nodes)
    workflow["plan_integrity"] = "replanned"
    refresh_policy_snapshot(workflow, nodes, registry, status="replanned")
    transition(failed_node, "ready")
    workflow["status"] = "running"
    return True


def recover_barrier_failed_side_effects(
    workflow: dict[str, Any],
    nodes: list[Node],
    registry: dict[str, dict[str, Any]],
) -> bool:
    """Rearm side effects whose durability barrier failed before the effect ran.

    The barrier invokes no external side effect itself. A barrier failure leaves the
    node safe to retry, but execution should resume on a fresh worker.
    """
    recovered = False
    for node in nodes:
        if node.status != "failed" or not side_effecting(node, registry):
            continue
        execution_id = execution_key(workflow, node)
        record = workflow.setdefault("executions", {}).get(execution_id)
        if not isinstance(record, dict) or record.get("status") != "barrier_failed":
            continue
        record["status"] = "prepared"
        record.pop("barrier_error", None)
        record["rearmed_at"] = utc_now()
        node.error = {}
        transition(node, "ready")
        workflow["status"] = "running"
        append_event("node.barrier_failure_rearmed", {
            "workflow_id": workflow["id"],
            "node_id": node.id,
            "execution_id": execution_id,
        })
        recovered = True
    return recovered

def recover_inflight_side_effects(
    workflow: dict[str, Any],
    nodes: list[Node],
    registry: dict[str, dict[str, Any]],
) -> None:
    """Turn durable START records with a still-running node into explicit in-doubt failures.

    This covers a worker interruption after the v19 barrier committed but before the
    node reached a terminal state. Connector nodes can then enter the existing
    reconciliation path; opaque side effects remain fail-closed.
    """
    for node in nodes:
        if node.status != "running" or not side_effecting(node, registry):
            continue
        execution_id = execution_key(workflow, node)
        record = workflow.setdefault("executions", {}).get(execution_id)
        if not isinstance(record, dict):
            continue
        if record.get("status") == "prepared":
            transition(node, "ready")
            workflow["status"] = "running"
            append_event("node.pre_start_execution_rearmed", {
                "workflow_id": workflow["id"],
                "node_id": node.id,
                "execution_id": execution_id,
            })
            continue
        if record.get("status") != "started":
            continue
        node.error = {
            "type": "execution_uncertain",
            "message": "A durable START record exists for an interrupted side-effecting node; external outcome must be reconciled before replay.",
            "execution_id": execution_id,
            "execution_uncertain": True,
            "reconciliation_required": True,
            "inflight_recovered": True,
        }
        transition(node, "failed")
        workflow["status"] = "failed"
        workflow["failed_node"] = node.id
        append_event("node.inflight_execution_recovered", {
            "workflow_id": workflow["id"],
            "node_id": node.id,
            "execution_id": execution_id,
        })

def recover_inflight_non_side_effects(
    workflow: dict[str, Any],
    nodes: list[Node],
    registry: dict[str, dict[str, Any]],
) -> bool:
    """Rearm durable non-side-effect executions after a runner interruption."""
    recovered = False
    for node in nodes:
        if node.status != "running" or side_effecting(node, registry):
            continue
        transition(node, "ready")
        append_event("node.non_side_effect_execution_rearmed", {
            "workflow_id": workflow["id"],
            "node_id": node.id,
        })
        recovered = True
    if recovered:
        workflow["status"] = "running"
    return recovered


def reconcile_first_uncertain(
    workflow: dict[str, Any],
    nodes: list[Node],
    registry: dict[str, dict[str, Any]],
) -> str | None:
    if not workflow.get("live"):
        return None
    candidates = [
        node for node in nodes
        if node.status == "failed"
        and node.error.get("execution_uncertain")
        and node.tool == "connector_bridge"
    ]
    if not candidates:
        return None
    node = sorted(candidates, key=lambda item: item.id)[0]
    transition(node, "reconciling")
    execution_id = execution_key(workflow, node)
    try:
        result = reconcile_connector_execution(
            node,
            workflow["goal"],
            dry_run=False,
        )
    except ConnectorReconciliationError as exc:
        node.error = {
            **node.error,
            "type": type(exc).__name__,
            "message": str(exc),
            "reconciliation_state": "unknown",
            "reconciliation_required": True,
        }
        transition(node, "failed")
        workflow["status"] = "failed"
        workflow["failed_node"] = node.id
        workflow.setdefault("reconciliations", {})[node.id] = {
            "state": "unknown",
            "error": str(exc),
            "checked_at": utc_now(),
        }
        append_event("node.reconciliation_failed", {
            "workflow_id": workflow["id"],
            "node_id": node.id,
            "execution_id": execution_id,
            "error": str(exc),
        })
        return "failed"

    state = str(result.get("state") or "unknown").lower()
    workflow.setdefault("reconciliations", {})[node.id] = result
    if state == "applied":
        node.output = {
            "reconciled": True,
            "reconciliation": result,
            "request_id": result.get("request_id"),
            "execution_id": execution_id,
        }
        node.error["reconciled"] = True
        node.error["reconciliation_state"] = "applied"
        node.error.pop("reconciliation_required", None)
        node.output["validation"] = {
            "passed": True,
            "checks": [{"check": "reconciliation_applied", "passed": True}],
            "checked_at": utc_now(),
        }
        transition(node, "completed")
        mark_execution_completed(workflow, execution_id, node.output)
        node_success_checkpoint(workflow, node)
        update_tool_health(node, True, registry)
        append_event("node.reconciled", {
            "workflow_id": workflow["id"],
            "node_id": node.id,
            "state": state,
        })
        workflow["status"] = (
            "completed" if all(item.status == "completed" for item in nodes) else "running"
        )
        return "reconciled"
    if state == "not_applied":
        mark_execution_not_applied(workflow, execution_id)
        node.error = {
            **node.error,
            "reconciled": True,
            "reconciliation_state": "not_applied",
        }
        node.error.pop("reconciliation_required", None)
        node.retry_count = 0
        transition(node, "ready")
        workflow["status"] = "running"
        append_event("node.reconciled", {
            "workflow_id": workflow["id"],
            "node_id": node.id,
            "state": state,
        })
        return "reconciled_ready"

    node.error["reconciliation_state"] = "unknown"
    node.error["reconciliation_required"] = True
    transition(node, "failed")
    workflow["status"] = "failed"
    workflow["failed_node"] = node.id
    append_event("node.reconciliation_unknown", {
        "workflow_id": workflow["id"],
        "node_id": node.id,
        "state": state,
    })
    return "failed"


def run_one_step(workflow: dict[str, Any], approve_high_risk: bool = False) -> str:
    if workflow.get('status') in {'completed', 'cancelled'}:
        return str(workflow['status'])
    nodes = [Node(**node) for node in workflow['nodes']]
    validate_dag(nodes)
    registry = load_registry()
    live = bool(workflow.get('live'))
    enforce_node_policy(nodes, registry, live=live, route=False)
    if not ensure_plan_integrity(workflow, nodes):
        workflow['nodes'] = [asdict(node) for node in nodes]
        persist_workflow(workflow)
        return 'failed'
    if not ensure_policy_integrity(workflow, nodes, registry):
        workflow['nodes'] = [asdict(node) for node in nodes]
        persist_workflow(workflow)
        return 'failed'
    if not verify_completed_checkpoints(workflow, nodes):
        workflow['nodes'] = [asdict(node) for node in nodes]
        persist_workflow(workflow)
        return 'failed'
    workflow['status'] = 'running'
    workflow['execution_mode'] = 'live' if live else 'dry-run'
    workflow.setdefault('replan_count', 0)
    workflow.setdefault('repair_feedback', {})
    workflow.setdefault('evidence', {})
    workflow.setdefault('reconciliations', {})
    execution_budget(workflow)
    for node in nodes:
        node.input['workflow_id'] = workflow['id']
        node.input['repair_feedback'] = workflow.get('repair_feedback', {}).get(node.id, {})

    if recover_barrier_failed_side_effects(workflow, nodes, registry):
        workflow['nodes'] = [asdict(node) for node in nodes]
        persist_workflow(workflow)
        return 'rearmed_pre_side_effect'
    recover_inflight_side_effects(workflow, nodes, registry)
    recover_inflight_non_side_effects(workflow, nodes, registry)
    reconciliation = reconcile_first_uncertain(workflow, nodes, registry)
    if reconciliation is not None:
        workflow['nodes'] = [asdict(node) for node in nodes]
        persist_workflow(workflow)
        if reconciliation == 'failed':
            return 'failed'
        return reconciliation

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
    try:
        preflight_node(node, registry, live=live)
    except Exception as preflight_exc:
        node.error = {
            'type': type(preflight_exc).__name__,
            'message': str(preflight_exc),
            'failure_class': 'dependency',
            'preflight_failed': True,
        }
        transition(node, 'failed')
        workflow['status'] = 'failed'
        workflow['failed_node'] = node.id
        append_event('node.preflight_failed', {
            'workflow_id': workflow['id'],
            'node_id': node.id,
            'tool': node.tool,
            'error': str(preflight_exc),
        })
        if replan_after_failure(workflow, nodes, node, registry):
            workflow['nodes'] = [asdict(item) for item in nodes]
            persist_workflow(workflow)
            return 'replanned'
        workflow['nodes'] = [asdict(item) for item in nodes]
        persist_workflow(workflow)
        return 'failed'
    if live and node.risk in {'high', 'critical'} and not approve_high_risk and not node.input.get('approval_granted'):
        transition(node, 'waiting_approval')
        workflow['status'] = 'waiting_approval'
        try:
            if not node.input.get('approval_issue'):
                node.input['approval_issue'] = create_approval_issue(workflow, node)
        except Exception as exc:
            node.error = {'type': type(exc).__name__, 'message': str(exc)}
            transition(node, 'failed')
            workflow['status'] = 'failed'
            workflow['failed_node'] = node.id
            workflow['nodes'] = [asdict(item) for item in nodes]
            persist_workflow(workflow)
            return 'failed'
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

    try:
        reserve_execution_step(workflow, node.id)
    except ExecutionBudgetExceeded as budget_exc:
        node.error = {
            'type': type(budget_exc).__name__,
            'message': str(budget_exc),
            'failure_class': 'dependency',
            'budget_exhausted': True,
        }
        transition(node, 'failed')
        workflow['status'] = 'failed'
        workflow['failed_node'] = node.id
        workflow['budget_exhausted'] = {
            'node_id': node.id,
            'used_steps': execution_budget(workflow)['used_steps'],
            'max_steps': execution_budget(workflow)['max_steps'],
        }
        workflow['nodes'] = [asdict(item) for item in nodes]
        persist_workflow(workflow)
        return 'failed'
    workflow['nodes'] = [asdict(item) for item in nodes]
    persist_workflow(workflow)
    node.input["context"] = build_node_context(nodes, node)
    transition(node, 'running')
    execution_id = execution_key(workflow, node)
    if side_effecting(node, registry):
        record = workflow.setdefault('executions', {}).get(execution_id)
        if record and record.get('status') == 'started':
            node.error = {
                'type': 'execution_uncertain',
                'message': 'A prior run may have completed an external side effect before state persistence.',
                'execution_id': execution_id,
            }
            transition(node, 'failed')
            workflow['status'] = 'failed'
            workflow['failed_node'] = node.id
            append_event('node.execution_uncertain', {
                'workflow_id': workflow['id'],
                'node_id': node.id,
                'execution_id': execution_id,
            })
            workflow['nodes'] = [asdict(item) for item in nodes]
            persist_workflow(workflow)
            return 'failed'
        mark_execution_prepared(workflow, node)
        workflow['nodes'] = [asdict(item) for item in nodes]
        persist_workflow(workflow)
        mark_execution_started(workflow, node, execution_id)
        workflow['nodes'] = [asdict(item) for item in nodes]
        persist_workflow(workflow)
        if live and side_effecting(node, registry):
            try:
                commit_side_effect_start(
                    ROOT,
                    execution_id=execution_id,
                )
            except DurabilityBarrierError as barrier_exc:
                record = workflow.setdefault('executions', {}).setdefault(execution_id, {})
                record['status'] = 'barrier_failed'
                record['barrier_error'] = str(barrier_exc)
                node.error = {
                    'type': type(barrier_exc).__name__,
                    'message': str(barrier_exc),
                    'failure_class': 'dependency',
                    'durability_barrier_failed': True,
                    'execution_id': execution_id,
                }
                transition(node, 'failed')
                workflow['status'] = 'failed'
                workflow['failed_node'] = node.id
                workflow['nodes'] = [asdict(item) for item in nodes]
                persist_workflow(workflow)
                append_event('node.durability_barrier_failed', {
                    'workflow_id': workflow['id'],
                    'node_id': node.id,
                    'execution_id': execution_id,
                    'error': str(barrier_exc),
                })
                return 'failed'
    append_event('node.started', {'workflow_id': workflow['id'], 'node_id': node.id, 'tool': node.tool})
    attempts = node.retry_count
    while True:
        try:
            node.output = execute_node(node, workflow['goal'], dry_run=not live)
            node.output['validation'] = validate_node_output(node, node.output)
            transition(node, 'validating')
            transition(node, 'completed')
            if side_effecting(node, registry):
                mark_execution_completed(workflow, execution_id, node.output)
            update_tool_health(node, True, registry)
            node_success_checkpoint(workflow, node)
            append_event('node.completed', {'workflow_id': workflow['id'], 'node_id': node.id, 'tool': node.tool})
            notify_issue(
                workflow,
                'Orchestrator: node ' + node.id + ' completed using ' + node.tool + '.'
            )
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
            failure_class, can_retry, uncertain = execution_failure_policy(
                node,
                exc,
                registry,
                dry_run=not live,
            )
            if attempts < node.max_retries and can_retry:
                attempts += 1
                node.retry_count = attempts
                transition(node, 'retrying')
                delay = deterministic_retry_delay(workflow['id'], node.id, attempts)
                append_event('node.retrying', {
                    'workflow_id': workflow['id'],
                    'node_id': node.id,
                    'attempt': attempts,
                    'error': str(exc),
                    'execution_uncertain': uncertain,
                    'failure_class': failure_class,
                    'retry_delay': delay,
                })
                time.sleep(delay)
                transition(node, 'ready')
                transition(node, 'running')
                continue
            transition(node, 'failed')
            update_tool_health(node, False, registry)
            if uncertain:
                append_event('node.execution_uncertain', {
                    'workflow_id': workflow['id'],
                    'node_id': node.id,
                    'error': node.error,
                })
            append_event('node.failed', {'workflow_id': workflow['id'], 'node_id': node.id, 'error': node.error})
            notify_issue(
                workflow,
                'Orchestrator: node ' + node.id + ' failed: ' + node.error.get('message', 'unknown error')
            )
            if not uncertain and not node.error.get("replan_blocked_after_side_effect_start") and replan_after_failure(workflow, nodes, node, registry):
                workflow['nodes'] = [asdict(item) for item in nodes]
                persist_workflow(workflow)
                return 'replanned'
            workflow['status'] = 'failed'
            workflow['failed_node'] = node.id
            workflow['nodes'] = [asdict(item) for item in nodes]
            persist_workflow(workflow)
            return 'failed'
def run_workflow(workflow: dict[str, Any], approve_high_risk: bool = False) -> None:
    if workflow.get("status") in {"completed", "cancelled"}:
        return
    nodes = [Node(**node) for node in workflow["nodes"]]
    validate_dag(nodes)
    registry = load_registry()
    live = bool(workflow.get("live"))
    enforce_node_policy(nodes, registry, live=live, route=False)
    if not ensure_plan_integrity(workflow, nodes):
        workflow["nodes"] = [asdict(node) for node in nodes]
        persist_workflow(workflow)
        return
    if not ensure_policy_integrity(workflow, nodes, registry):
        workflow["nodes"] = [asdict(node) for node in nodes]
        persist_workflow(workflow)
        return
    if not verify_completed_checkpoints(workflow, nodes):
        workflow["nodes"] = [asdict(node) for node in nodes]
        persist_workflow(workflow)
        return
    workflow["status"] = "running"
    workflow["execution_mode"] = "live" if live else "dry-run"
    workflow.setdefault("replan_count", 0)
    workflow.setdefault("repair_feedback", {})
    workflow.setdefault("evidence", {})
    workflow.setdefault("reconciliations", {})
    execution_budget(workflow)
    workflow.setdefault("max_parallel", int(os.environ.get("ORCHESTRATOR_MAX_PARALLEL", DEFAULT_MAX_PARALLEL)))
    workflow["max_parallel"] = max(1, min(int(workflow["max_parallel"]), 8))

    for node in nodes:
        node.input["workflow_id"] = workflow["id"]
        node.input["repair_feedback"] = workflow.get("repair_feedback", {}).get(node.id, {})

    safety = 0
    while True:
        safety += 1
        if safety > 200:
            raise RuntimeError("orchestration safety limit reached")

        if recover_barrier_failed_side_effects(workflow, nodes, registry):
            workflow['nodes'] = [asdict(node) for node in nodes]
            persist_workflow(workflow)
            return
        recover_inflight_side_effects(workflow, nodes, registry)
        recover_inflight_non_side_effects(workflow, nodes, registry)
        reconciliation = reconcile_first_uncertain(workflow, nodes, registry)
        if reconciliation == "failed":
            workflow["nodes"] = [asdict(node) for node in nodes]
            persist_workflow(workflow)
            return
        if reconciliation in {"reconciled", "reconciled_ready"}:
            workflow["nodes"] = [asdict(node) for node in nodes]
            persist_workflow(workflow)

        refresh_approvals(workflow, nodes)
        if workflow.get("status") == "failed":
            workflow["nodes"] = [asdict(node) for node in nodes]
            persist_workflow(workflow)
            return

        ready = ready_nodes(nodes)
        for node in ready:
            if node.status == "pending":
                transition(node, "ready")

        if not ready:
            if all(node.status == "completed" for node in nodes):
                workflow["status"] = "completed"
                workflow["nodes"] = [asdict(node) for node in nodes]
                persist_workflow(workflow)
                append_event("workflow.completed", {"workflow_id": workflow["id"]})
                notify_issue(workflow, "Orchestrator: workflow " + workflow["id"] + " completed.")
                return

            waiting = [node for node in nodes if node.status in {"waiting_approval", "retrying", "pending"}]
            workflow["status"] = (
                "waiting_approval"
                if waiting and all(node.status == "waiting_approval" for node in waiting)
                else "failed"
            )
            workflow["nodes"] = [asdict(node) for node in nodes]
            persist_workflow(workflow)
            return

        # Side effects remain serialized; independent read/compute nodes may run in parallel.
        safe_ready = [
            node for node in ready
            if not side_effecting(node, registry) and node.risk not in {"high", "critical"}
        ]
        unsafe_ready = [node for node in ready if node not in safe_ready]
        batch = (
            [sorted(unsafe_ready, key=lambda item: item.id)[0]]
            if unsafe_ready
            else sorted(safe_ready, key=lambda item: item.id)[:workflow["max_parallel"]]
        )

        # Admission control is performed before any node is persisted as running.
        # Never partially activate a parallel batch and then fail a later reservation:
        # otherwise an earlier safe node could be stranded as running in a failed workflow.
        budget = execution_budget(workflow)
        available_steps = budget["max_steps"] - budget["used_steps"]
        if available_steps <= 0:
            node = sorted(batch, key=lambda item: item.id)[0]
            budget_exc = ExecutionBudgetExceeded(
                f"execution step budget exhausted ({budget['used_steps']}/{budget['max_steps']})"
            )
            node.error = {
                "type": type(budget_exc).__name__,
                "message": str(budget_exc),
                "failure_class": "dependency",
                "budget_exhausted": True,
            }
            transition(node, "failed")
            workflow["status"] = "failed"
            workflow["failed_node"] = node.id
            workflow["budget_exhausted"] = {
                "node_id": node.id,
                "used_steps": budget["used_steps"],
                "max_steps": budget["max_steps"],
            }
            workflow["nodes"] = [asdict(item) for item in nodes]
            persist_workflow(workflow)
            return
        if len(batch) > available_steps:
            batch = batch[:available_steps]

        # Phase 1: preflight the entire batch before any node is persisted as running.
        # If a preflight/replan/approval gate blocks, no sibling node is stranded.
        executable = []
        admission_blocked = False
        for node in batch:
            try:
                preflight_node(node, registry, live=live)
            except Exception as preflight_exc:
                node.error = {
                    "type": type(preflight_exc).__name__,
                    "message": str(preflight_exc),
                    "failure_class": "dependency",
                    "preflight_failed": True,
                }
                transition(node, "failed")
                append_event("node.preflight_failed", {
                    "workflow_id": workflow["id"],
                    "node_id": node.id,
                    "tool": node.tool,
                    "error": str(preflight_exc),
                })
                if replan_after_failure(workflow, nodes, node, registry):
                    workflow["nodes"] = [asdict(item) for item in nodes]
                    persist_workflow(workflow)
                    admission_blocked = True
                    break
                workflow["status"] = "failed"
                workflow["failed_node"] = node.id
                workflow["nodes"] = [asdict(item) for item in nodes]
                persist_workflow(workflow)
                return
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
                    transition(node, "failed")
                    workflow["status"] = "failed"
                    workflow["failed_node"] = node.id
                    workflow["nodes"] = [asdict(item) for item in nodes]
                    persist_workflow(workflow)
                    return
                append_event("approval.required", {
                    "workflow_id": workflow["id"],
                    "node_id": node.id,
                    "risk": node.risk,
                    "issue": node.input.get("approval_issue"),
                })
                workflow["nodes"] = [asdict(item) for item in nodes]
                persist_workflow(workflow)
                return
        if admission_blocked:
            continue

        try:
            reserve_execution_steps(workflow, [node.id for node in batch])
        except ExecutionBudgetExceeded as budget_exc:
            node = sorted(batch, key=lambda item: item.id)[0]
                node.error = {
                    "type": type(budget_exc).__name__,
                    "message": str(budget_exc),
                    "failure_class": "dependency",
                    "budget_exhausted": True,
                }
                transition(node, "failed")
                workflow["status"] = "failed"
                workflow["failed_node"] = node.id
                workflow["budget_exhausted"] = {
                    "node_id": node.id,
                    "used_steps": execution_budget(workflow)["used_steps"],
                    "max_steps": execution_budget(workflow)["max_steps"],
                }
                workflow["nodes"] = [asdict(item) for item in nodes]
                persist_workflow(workflow)
                return

            # Phase 2: only now cross runtime activation and side-effect barriers.
            executable = []
            if all(not side_effecting(node, registry) for node in batch):
                # Safe independent nodes share one durable activation record. This avoids
                # persisting a reserved budget before any node is actually activated.
                for node in batch:
                    node.input["context"] = build_node_context(nodes, node)
                    transition(node, "running")
                workflow["nodes"] = [asdict(item) for item in nodes]
                persist_workflow(workflow)
                for node in batch:
                    append_event("node.started", {
                        "workflow_id": workflow["id"],
                        "node_id": node.id,
                        "tool": node.tool,
                    })
                    executable.append((node, execution_key(workflow, node)))
            else:
                for node in batch:
                    workflow["nodes"] = [asdict(item) for item in nodes]
                    persist_workflow(workflow)
                    node.input["context"] = build_node_context(nodes, node)
                    transition(node, "running")

                    execution_id = execution_key(workflow, node)
                    if side_effecting(node, registry):
                        record = workflow.setdefault("executions", {}).get(execution_id)
                        if record and record.get("status") == "started":
                            node.error = {
                                "type": "execution_uncertain",
                                "message": "A prior run may have completed an external side effect before state persistence.",
                                "execution_id": execution_id,
                            }
                            transition(node, "failed")
                            workflow["status"] = "failed"
                            workflow["failed_node"] = node.id
                            append_event("node.execution_uncertain", {
                                "workflow_id": workflow["id"],
                                "node_id": node.id,
                                "execution_id": execution_id,
                            })
                            workflow["nodes"] = [asdict(item) for item in nodes]
                            persist_workflow(workflow)
                            return

                        mark_execution_prepared(workflow, node)
                        workflow["nodes"] = [asdict(item) for item in nodes]
                        persist_workflow(workflow)
                        mark_execution_started(workflow, node, execution_id)
                        workflow["nodes"] = [asdict(item) for item in nodes]
                        persist_workflow(workflow)
                        if live and side_effecting(node, registry):
                            try:
                                commit_side_effect_start(
                                    ROOT,
                                    execution_id=execution_id,
                                )
                            except DurabilityBarrierError as barrier_exc:
                                record = workflow.setdefault("executions", {}).setdefault(execution_id, {})
                                record["status"] = "barrier_failed"
                                record["barrier_error"] = str(barrier_exc)
                                node.error = {
                                    "type": type(barrier_exc).__name__,
                                    "message": str(barrier_exc),
                                    "failure_class": "dependency",
                                    "durability_barrier_failed": True,
                                    "execution_id": execution_id,
                                }
                                transition(node, "failed")
                                workflow["status"] = "failed"
                                workflow["failed_node"] = node.id
                                workflow["nodes"] = [asdict(item) for item in nodes]
                                persist_workflow(workflow)
                                append_event("node.durability_barrier_failed", {
                                    "workflow_id": workflow["id"],
                                    "node_id": node.id,
                                    "execution_id": execution_id,
                                    "error": str(barrier_exc),
                                })
                                return

                    append_event("node.started", {
                        "workflow_id": workflow["id"],
                        "node_id": node.id,
                        "tool": node.tool,
                    })
                    executable.append((node, execution_id))
        if not executable:
            workflow["nodes"] = [asdict(node) for node in nodes]
            persist_workflow(workflow)
            continue

        dry_run = not live
        results = []

        if len(executable) == 1 or any(side_effecting(node, registry) for node, _ in executable):
            for node, execution_id in executable:
                results.append((node, execution_id, *execute_with_retries(node, workflow["goal"], dry_run)))
        else:
            with ThreadPoolExecutor(
                max_workers=min(workflow["max_parallel"], len(executable)),
                thread_name_prefix="orchestrator-node",
            ) as pool:
                futures = {
                    pool.submit(execute_with_retries, node, workflow["goal"], dry_run): (node, execution_id)
                    for node, execution_id in executable
                }
                completed_futures = {}
                for future in as_completed(futures):
                    node, execution_id = futures[future]
                    completed_futures[node.id] = (node, execution_id, *future.result())
                results = [completed_futures[node.id] for node, _ in sorted(executable, key=lambda item: item[0].id)]

        replan_needed = False
        for node, execution_id, success, error in results:
            if success:
                if side_effecting(node, registry):
                    mark_execution_completed(workflow, execution_id, node.output)
                update_tool_health(node, True, registry)
                node_success_checkpoint(workflow, node)
                append_event("node.completed", {
                    "workflow_id": workflow["id"],
                    "node_id": node.id,
                    "tool": node.tool,
                })
                notify_issue(
                    workflow,
                    "Orchestrator: node " + node.id + " completed using " + node.tool + ".",
                )
            else:
                update_tool_health(node, False, registry)
                if not node.error.get("replan_blocked_after_side_effect_start") and replan_after_failure(workflow, nodes, node, registry):
                    replan_needed = True
                else:
                    workflow["status"] = "failed"
                    workflow["failed_node"] = node.id
                    workflow["nodes"] = [asdict(item) for item in nodes]
                    persist_workflow(workflow)
                    return

        workflow["nodes"] = [asdict(node) for node in nodes]
        persist_workflow(workflow)

        if replan_needed:
            continue

def notify_issue(workflow: dict[str, Any], message: str) -> None:
    try:
        from issue_notify import post_issue_status
        post_issue_status(workflow, message, http_json)
    except Exception:
        return

def callback_target_fingerprint(url: str) -> str | None:
    value = str(url or "").strip()
    if not value:
        return None
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        return None
    normalized = urllib.parse.urlunsplit((
        parsed.scheme.lower(),
        parsed.netloc.lower(),
        parsed.path or "/",
        parsed.query,
        "",
    ))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def notify_execution_callback(workflow: dict[str, Any]) -> bool:
    """Durable terminal callback outbox with stable request identity."""
    callback = workflow.setdefault("callback", {})
    if isinstance(callback, dict) and callback.get("status") in {"sent", "dead_letter", "disabled"}:
        return callback.get("status") == "sent"
    url = os.environ.get("ORCHESTRATOR_CALLBACK_URL", "").strip()
    secret = os.environ.get("ORCHESTRATOR_CALLBACK_SECRET", "")
    execution_id = str(workflow.get("execution_id") or "").strip()
    if not url or not secret or not execution_id:
        return False
    current_target_fingerprint = callback_target_fingerprint(url)
    if current_target_fingerprint is None:
        if isinstance(callback, dict):
            callback["status"] = "dead_letter"
            callback["last_error"] = "callback endpoint must use HTTPS"
        return False
    previous_target_fingerprint = (
        str(callback.get("target_fingerprint") or "").strip()
        if isinstance(callback, dict)
        else ""
    )
    if (
        previous_target_fingerprint
        and previous_target_fingerprint != current_target_fingerprint
    ):
        if isinstance(callback, dict):
            callback["status"] = "dead_letter"
            callback["last_error"] = "callback endpoint changed"
        append_event(
            "callback.endpoint_drift",
            {
                "workflow_id": workflow.get("id"),
                "execution_id": execution_id,
            },
        )
        return False
    if isinstance(callback, dict):
        callback["target_fingerprint"] = current_target_fingerprint
    if isinstance(callback, dict):
        callback["status"] = "pending"
        callback["last_attempt_at"] = utc_now()
        callback["attempts"] = int(callback.get("attempts", 0)) + 1
    status = str(workflow.get("status") or "failed")
    callback_status = "completed" if status == "completed" else "failed"
    if status != "completed":
        for item in workflow.get("nodes", []):
            error = item.get("error") if isinstance(item, dict) else {}
            if isinstance(error, dict) and error.get("execution_uncertain"):
                callback_status = "uncertain"
                break

    result = {
        "workflow_id": str(workflow.get("id") or ""),
        "external_workflow_id": str(workflow.get("external_workflow_id") or ""),
        "external_domain": str(workflow.get("external_domain") or ""),
        "external_operation": str(workflow.get("external_operation") or ""),
        "input_digest": str(workflow.get("input_digest") or ""),
        "attempt": int(workflow.get("external_attempt") or 1),
    }
    payload = {
        "schema_version": 1,
        "request_id": execution_id,
        "execution_id": execution_id,
        "intent_fingerprint": str(workflow.get("intent_fingerprint") or ""),
        "status": callback_status,
        "result": result if callback_status == "completed" else {},
        "error": (
            str(workflow.get("failed_node") or "engine_failed")
            if callback_status != "completed"
            else ""
        ),
    }
    body = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    attempts = int(callback.get("attempts", 0)) if isinstance(callback, dict) else 0
    remaining_attempts = min(3, max(0, 12 - attempts))
    if remaining_attempts == 0:
        if isinstance(callback, dict):
            callback["status"] = "dead_letter"
            callback["last_error"] = "callback retry budget exhausted"
        return False
    for _ in range(remaining_attempts):
        timestamp = str(int(time.time()))
        signature = "sha256=" + hmac.new(
            secret.encode("utf-8"),
            timestamp.encode("utf-8") + b"\n" + body,
            hashlib.sha256,
        ).hexdigest()
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-Engine-Timestamp": timestamp,
                "X-Engine-Signature": signature,
                "Idempotency-Key": execution_id,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                if 200 <= int(response.status) < 300:
                    callback["status"] = "sent"
                    callback["sent_at"] = utc_now()
                    callback.pop("last_error", None)
                    append_event(
                        "callback.sent",
                        {
                            "workflow_id": workflow.get("id"),
                            "execution_id": execution_id,
                            "status": callback_status,
                        },
                    )
                    return True
        except Exception:
            pass
        time.sleep(1)
    callback["status"] = (
        "dead_letter"
        if int(callback.get("attempts", 0)) >= 12
        else "pending"
    )
    callback["last_error"] = "callback delivery failed after bounded retries"
    append_event(
        "callback.failed",
        {
            "workflow_id": workflow.get("id"),
            "execution_id": execution_id,
            "status": callback_status,
        },
    )
    return False


def create_workflow(
    goal: str,
    live: bool,
    trigger_issue: int | None = None,
    event_id: str | None = None,
    execution_id: str | None = None,
    parent_execution_id: str | None = None,
    external_workflow_id: str | None = None,
    external_domain: str | None = None,
    external_operation: str | None = None,
    intent_fingerprint: str | None = None,
    input_digest: str | None = None,
    idempotency_key: str | None = None,
    external_attempt: int | None = None,
) -> dict[str, Any]:
    goal = normalize_goal(goal)
    registry = load_registry()
    nodes = None
    if os.environ.get("ORCHESTRATOR_LLM_PLANNER", "true").lower() == "true" and os.environ.get("GEMINI_API_KEY"):
        try:
            from llm_planner import plan_goal
            nodes = plan_goal(goal, registry, Node, validate_dag, live=live)
            append_event("planner.llm", {"goal": goal, "nodes": len(nodes)})
        except Exception as planner_exc:
            append_event("planner.fallback", {"goal": goal, "error": str(planner_exc)})
    if nodes is None:
        nodes = deterministic_plan(goal, registry, live=live)
    validate_dag(nodes)
    enforce_node_policy(nodes, registry, live=live, route=True)
    route_snapshot = build_policy_snapshot(registry, nodes, live=live)
    return {
        "id": new_id("wf"),
        "created_at": utc_now(),
        "goal": goal,
        "status": "planning",
        "live": live,
        "execution_mode": "dry-run",
        "replan_count": 0,
        "repair_feedback": {},
        "evidence": {},
        "reconciliations": {},
        "schema_version": CURRENT_WORKFLOW_SCHEMA_VERSION,
        "plan_fingerprint": None,
        "policy_fingerprint": fingerprint_policy(route_snapshot),
        "route_snapshot": route_snapshot,
        "policy_integrity": "initialized",
        "execution_budget": {"max_steps": configured_max_execution_steps(), "used_steps": 0},
        "callback": {"status": "pending"} if execution_id else {"status": "disabled"},
        "checkpoint_integrity": "pending",
        "plan_integrity": "pending",
        "max_parallel": max(1, min(int(os.environ.get("ORCHESTRATOR_MAX_PARALLEL", DEFAULT_MAX_PARALLEL)), 8)),
        "trigger_issue": trigger_issue,
        "event_id": event_id,
        "execution_id": execution_id,
        "parent_execution_id": parent_execution_id,
        "external_workflow_id": external_workflow_id,
        "external_domain": external_domain,
        "external_operation": external_operation,
        "intent_fingerprint": intent_fingerprint,
        "input_digest": input_digest,
        "idempotency_key": idempotency_key,
        "external_attempt": int(external_attempt or 1),
        "origin_github_run_id": os.environ.get("ORCHESTRATOR_GITHUB_RUN_ID") or None,
        "github_run_id": os.environ.get("ORCHESTRATOR_GITHUB_RUN_ID") or None,
        "github_run_attempt": (
            int(os.environ["ORCHESTRATOR_GITHUB_RUN_ATTEMPT"])
            if os.environ.get("ORCHESTRATOR_GITHUB_RUN_ATTEMPT", "").isdigit()
            else None
        ),
        "origin_github_run_attempt": (
            int(os.environ["ORCHESTRATOR_GITHUB_RUN_ATTEMPT"])
            if os.environ.get("ORCHESTRATOR_GITHUB_RUN_ATTEMPT", "").isdigit()
            else None
        ),
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
        "execution_id": workflow.get("execution_id"),
        "external_operation": workflow.get("external_operation"),
    }, indent=2))


def running_recovery_due(
    workflow: dict[str, Any],
    *,
    now: datetime | None = None,
    grace_seconds: int = RUNNING_RECOVERY_GRACE_SECONDS,
) -> bool:
    if workflow.get("status") != "running":
        return True
    if not str(workflow.get("github_run_id") or "").strip():
        return True
    raw_updated = str(workflow.get("updated_at") or "").strip()
    if not raw_updated:
        return True
    try:
        updated = datetime.fromisoformat(raw_updated.replace("Z", "+00:00"))
    except ValueError:
        return True
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=timezone.utc)
    current = now or datetime.now(timezone.utc)
    age = (current - updated).total_seconds()
    return age >= max(0, int(grace_seconds))


def resume_pending_workflows(
    state: dict[str, Any],
    approve_high_risk: bool = False,
    step: bool = False,
    *,
    scheduled_recovery: bool = False,
) -> int:
    resumed = 0
    candidates = []
    for workflow in state.get("workflows", {}).values():
        status = workflow.get("status")
        executions = workflow.get("executions", {})
        barrier_failed = any(
            isinstance(node, dict)
            and node.get("status") == "failed"
            and isinstance(executions, dict)
            and isinstance(node.get("id"), str)
            and isinstance(executions.get(hashlib.sha256(f"{workflow.get('id')}:{node['id']}".encode("utf-8")).hexdigest()), dict)
            and executions[hashlib.sha256(f"{workflow.get('id')}:{node['id']}".encode("utf-8")).hexdigest()].get("status") == "barrier_failed"
            for node in workflow.get("nodes", [])
        )
        uncertain = any(
            isinstance(node, dict)
            and node.get("status") == "failed"
            and isinstance(node.get("error"), dict)
            and node["error"].get("execution_uncertain")
            and node.get("tool") == "connector_bridge"
            for node in workflow.get("nodes", [])
        )
        if status == "running":
            if not scheduled_recovery or running_recovery_due(workflow):
                candidates.append(workflow)
        elif status == "waiting_approval" or (
            status == "failed" and (uncertain or barrier_failed)
        ):
            candidates.append(workflow)
    candidates.sort(key=lambda item: item.get("updated_at") or item.get("created_at") or "")
    for workflow in candidates:
        callback = workflow.get("callback")
        if (
            isinstance(callback, dict)
            and callback.get("status") == "pending"
            and workflow.get("status") in {"completed", "failed"}
            and workflow.get("execution_id")
        ):
            notify_execution_callback(workflow)
            state["workflows"][workflow["id"]] = workflow
            state["last_workflow_id"] = workflow["id"]
            resumed += 1
            continue
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

def find_existing_replay(
    state: dict[str, Any],
    *,
    event_id: str | None,
    execution_id: str | None,
    idempotency_key: str | None,
) -> dict[str, Any] | None:
    identifiers = (
        ("event_id", event_id),
        ("execution_id", execution_id),
        ("idempotency_key", idempotency_key),
    )
    matches: dict[str, dict[str, Any]] = {}
    for workflow in state.get("workflows", {}).values():
        if not isinstance(workflow, dict):
            continue
        for field_name, value in identifiers:
            normalized = str(value or "").strip()
            if normalized and normalized == str(workflow.get(field_name) or "").strip():
                matches[str(workflow.get("id") or "")] = workflow
                break
    if len(matches) > 1:
        raise SystemExit("replay identifiers map to multiple persisted workflows")
    return next(iter(matches.values()), None) if matches else None



def validate_event_replay_identity(
    workflow: dict[str, Any],
    *,
    execution_id: str | None,
    intent_fingerprint: str | None,
    input_digest: str | None,
    idempotency_key: str | None,
) -> None:
    comparisons = (
        ("execution_id", execution_id, workflow.get("execution_id")),
        ("intent_fingerprint", intent_fingerprint, workflow.get("intent_fingerprint")),
        ("input_digest", input_digest, workflow.get("input_digest")),
        ("idempotency_key", idempotency_key, workflow.get("idempotency_key")),
    )
    for field_name, incoming, stored in comparisons:
        incoming_value = str(incoming or "").strip()
        stored_value = str(stored or "").strip()
        if incoming_value and stored_value and incoming_value != stored_value:
            raise SystemExit(
                f"event_id conflicts with persisted {field_name} for workflow {workflow.get('id')}"
            )


def finalize_workflow_state(
    state: dict[str, Any],
    workflow: dict[str, Any],
) -> None:
    state["workflows"][workflow["id"]] = workflow
    state["last_workflow_id"] = workflow["id"]
    if workflow.get("status") in {"completed", "failed"}:
        notify_execution_callback(workflow)
    save_state(state)



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
            state,
            approve_high_risk=args.approve_high_risk,
            step=args.step,
            scheduled_recovery=os.environ.get("ORCHESTRATOR_RESUME", "").lower() == "true",
        )
        print(json.dumps({'resumed_workflows': count}, indent=2))
        return 0

    if args.workflow_id:
        workflow = state.get('workflows', {}).get(args.workflow_id)
        if not workflow:
            raise SystemExit(f'workflow not found: {args.workflow_id}')
        if args.step:
            result = run_one_step(workflow, approve_high_risk=args.approve_high_risk)
            finalize_workflow_state(state, workflow)
            print_summary(workflow)
            return 0 if result not in {'failed', 'continuation_failed'} else 2
        run_workflow(workflow, approve_high_risk=args.approve_high_risk)
        finalize_workflow_state(state, workflow)
        print_summary(workflow)
        return 0 if workflow['status'] in {'completed', 'waiting_approval'} else 2

    if not args.goal:
        raise SystemExit('provide --goal or --workflow-id')

    live = args.live or os.environ.get('ORCHESTRATOR_LIVE', '').lower() == 'true'
    requested_mode = os.environ.get("ORCHESTRATOR_REQUESTED_MODE", "").strip().lower()
    if requested_mode:
        if requested_mode not in {"dry-run", "live"}:
            raise SystemExit("invalid requested execution mode")
        live = requested_mode == "live"
    trigger_issue_raw = os.environ.get('ORCHESTRATOR_TRIGGER_ISSUE', '').strip()
    trigger_issue = int(trigger_issue_raw) if trigger_issue_raw.isdigit() else None
    event_id = os.environ.get("ORCHESTRATOR_EVENT_ID", "").strip() or None
    execution_id_env = os.environ.get("ORCHESTRATOR_EXECUTION_ID", "").strip() or None
    parent_execution_id = os.environ.get("ORCHESTRATOR_PARENT_EXECUTION_ID", "").strip() or None
    external_workflow_id = os.environ.get("ORCHESTRATOR_EXTERNAL_WORKFLOW_ID", "").strip() or None
    external_domain = os.environ.get("ORCHESTRATOR_EXTERNAL_DOMAIN", "").strip() or None
    external_operation = os.environ.get("ORCHESTRATOR_EXTERNAL_OPERATION", "").strip() or None
    intent_fingerprint_env = os.environ.get("ORCHESTRATOR_INTENT_FINGERPRINT", "").strip() or None
    input_digest = os.environ.get("ORCHESTRATOR_INPUT_DIGEST", "").strip() or None
    idempotency_key = os.environ.get("ORCHESTRATOR_IDEMPOTENCY_KEY", "").strip() or None
    try:
        external_attempt = int(os.environ.get("ORCHESTRATOR_EXTERNAL_ATTEMPT", "1"))
    except ValueError:
        raise SystemExit("invalid external attempt")
    if event_id or execution_id_env or idempotency_key:
        existing = find_existing_replay(
            state,
            event_id=event_id,
            execution_id=execution_id_env,
            idempotency_key=idempotency_key,
        )
        if existing:
            validate_event_replay_identity(
                existing,
                execution_id=execution_id_env,
                intent_fingerprint=intent_fingerprint_env,
                input_digest=input_digest,
                idempotency_key=idempotency_key,
            )
            if existing.get("status") in {"completed", "failed"}:
                finalize_workflow_state(state, existing)
            print_summary(existing)
            return 0 if existing.get("status") in {"completed", "waiting_approval", "running"} else 2
    workflow = create_workflow(
        args.goal,
        live=live,
        trigger_issue=trigger_issue,
        event_id=event_id,
        execution_id=execution_id_env,
        parent_execution_id=parent_execution_id,
        external_workflow_id=external_workflow_id,
        external_domain=external_domain,
        external_operation=external_operation,
        intent_fingerprint=intent_fingerprint_env,
        input_digest=input_digest,
        idempotency_key=idempotency_key,
        external_attempt=external_attempt,
    )
    workflow['status'] = 'ready'
    state['workflows'][workflow['id']] = workflow
    state['last_workflow_id'] = workflow['id']
    append_event(
        'workflow.created',
        {'workflow_id': workflow['id'], 'goal': workflow['goal'], 'live': live},
    )

    if args.step:
        result = run_one_step(workflow, approve_high_risk=args.approve_high_risk)
        finalize_workflow_state(state, workflow)
        print_summary(workflow)
        return 0 if result not in {'failed', 'continuation_failed'} else 2

    run_workflow(workflow, approve_high_risk=args.approve_high_risk)
    finalize_workflow_state(state, workflow)
    print_summary(workflow)
    return 0 if workflow['status'] in {'completed', 'waiting_approval'} else 2

if __name__ == '__main__':
    raise SystemExit(main())
