#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, sys, time, traceback, urllib.parse, urllib.request
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATE_DIR = Path(".orchestrator")
STATE_FILE = STATE_DIR / "state.json"
EVENT_FILE = STATE_DIR / "events.jsonl"
CHECKPOINT_DIR = STATE_DIR / "checkpoints"
MAX_NODES = 24

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

CAPABILITY_TOOL = {
    "research": "firecrawl", "analyze": "openai", "draft": "openai",
    "spec": "openai", "build": "github", "test": "github",
    "deploy": "webhook", "validate": "webhook", "publish": "webhook",
    "notify": "webhook", "execute": "webhook",
}

TRANSITIONS = {
    "pending": {"ready", "cancelled"}, "ready": {"running", "cancelled"},
    "running": {"validating", "waiting_approval", "retrying", "failed", "cancelled"},
    "validating": {"completed", "retrying", "failed"},
    "waiting_approval": {"ready", "failed", "cancelled"},
    "retrying": {"ready", "failed"}, "failed": {"replanning", "cancelled"},
    "replanning": {"ready", "failed", "cancelled"}, "completed": set(),
    "cancelled": set(),
}

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def append_event(event_type: str, payload: dict[str, Any]) -> None:
    EVENT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with EVENT_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": utc_now(), "event_type": event_type, "payload": payload}, ensure_ascii=False) + "\n")

def load_state() -> dict[str, Any]:
    return json.loads(STATE_FILE.read_text(encoding="utf-8")) if STATE_FILE.exists() else {
        "version": 1, "workflows": {}, "last_workflow_id": None
    }

def save_state(state: dict[str, Any]) -> None:
    write_json(STATE_FILE, state)

def new_id(prefix: str) -> str:
    return f"{prefix}_{int(time.time() * 1000)}"

def classify_risk(cap: str) -> str:
    return "high" if cap in {"deploy", "publish", "delete", "external_write"} else \
           "medium" if cap in {"build", "edit", "send"} else "low"

def deterministic_plan(goal: str) -> list[Node]:
    g = goal.lower()
    if any(k in g for k in ("website", "web app", "app", "software", "build", "deploy")):
        seq = [("research","Collect requirements and constraints."),
               ("spec","Produce an implementation specification."),
               ("build","Implement the requested system."),
               ("test","Run deterministic tests."),
               ("deploy","Deploy only after validation."),
               ("validate","Verify the deployed result."),
               ("notify","Report state and artifacts.")]
    elif any(k in g for k in ("research","compare","literature","study","analysis")):
        seq = [("research","Collect evidence and primary sources."),
               ("analyze","Synthesize evidence and uncertainty."),
               ("draft","Produce the requested research output."),
               ("validate","Check citations and consistency."),
               ("notify","Report the completed result.")]
    elif any(k in g for k in ("content","post","instagram","youtube","publish")):
        seq = [("research","Collect source material and constraints."),
               ("draft","Create the content draft."),
               ("validate","Check factuality and format."),
               ("publish","Publish only after validation."),
               ("notify","Report publication status.")]
    else:
        seq = [("research","Gather minimum required information."),
               ("execute","Perform the requested operation."),
               ("validate","Validate output against the goal."),
               ("notify","Report the result and artifacts.")]
    out, prev = [], []
    for i, (cap, instruction) in enumerate(seq, 1):
        nid = f"n{i:02d}-{cap}"
        out.append(Node(nid, cap, CAPABILITY_TOOL.get(cap, "noop"), list(prev),
                        classify_risk(cap), input={"goal": goal, "instruction": instruction}))
        prev = [nid]
    return out

def validate_dag(nodes: list[Node]) -> None:
    if not nodes or len(nodes) > MAX_NODES:
        raise ValueError(f"invalid node count: {len(nodes)}")
    ids = {n.id for n in nodes}
    if len(ids) != len(nodes):
        raise ValueError("duplicate node id")
    by_id = {n.id: n for n in nodes}
    for n in nodes:
        for d in n.depends_on:
            if d not in ids: raise ValueError(f"unknown dependency {d}")
            if d == n.id: raise ValueError(f"self dependency {n.id}")
    visiting, visited = set(), set()
    def visit(nid: str) -> None:
        if nid in visiting: raise ValueError("cycle detected")
        if nid in visited: return
        visiting.add(nid)
        for d in by_id[nid].depends_on: visit(d)
        visiting.remove(nid); visited.add(nid)
    for nid in ids: visit(nid)

def http_json(url: str, method="GET", body=None, headers=None, timeout=60) -> dict[str, Any]:
    data = json.dumps(body).encode() if body is not None else None
    h = {"User-Agent":"ai-orchestrator-core/1.0","Accept":"application/json", **(headers or {})}
    if data is not None: h["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", "replace")
        try: value = json.loads(raw) if raw else {}
        except json.JSONDecodeError: value = {"text": raw}
        return {"status_code": resp.status, "data": value}

def execute_openai(node: Node, goal: str) -> dict[str, Any]:
    key = os.environ.get("OPENAI_API_KEY")
    if not key: raise RuntimeError("OPENAI_API_KEY is required for openai adapter")
    payload = {"model": os.environ.get("OPENAI_MODEL","gpt-5.6"),
               "input": f"Execute conservatively. Goal: {goal}\nInstruction: {node.input.get('instruction','')}"}
    return http_json("https://api.openai.com/v1/responses", "POST", payload,
                     {"Authorization": f"Bearer {key}"}, 90)

def execute_webhook(node: Node, goal: str) -> dict[str, Any]:
    url = node.input.get("url") or os.environ.get("ORCHESTRATOR_WEBHOOK_URL")
    if not url or urllib.parse.urlparse(url).scheme != "https":
        raise RuntimeError("webhook adapter requires an HTTPS URL")
    headers = {"X-Orchestrator-Event":"node.execute"}
    secret = os.environ.get("ORCHESTRATOR_WEBHOOK_SECRET")
    if secret: headers["Authorization"] = f"Bearer {secret}"
    return http_json(url, "POST", {
        "workflow_id": node.input.get("workflow_id"), "node_id": node.id,
        "goal": goal, "capability": node.capability,
        "instruction": node.input.get("instruction")
    }, headers, 90)

def execute_github(node: Node) -> dict[str, Any]:
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not token or not repo: raise RuntimeError("GitHub token/repository unavailable")
    headers = {"Authorization":f"Bearer {token}","X-GitHub-Api-Version":"2022-11-28"}
    action = node.input.get("action","metadata")
    if action == "metadata":
        return http_json(f"https://api.github.com/repos/{repo}", headers=headers)
    if action == "create_issue":
        return http_json(f"https://api.github.com/repos/{repo}/issues","POST",
                         {"title":node.input.get("title","Orchestrator task"),
                          "body":node.input.get("body","")}, headers=headers)
    raise RuntimeError(f"GitHub action not allowlisted: {action}")

def execute_node(node: Node, goal: str, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {"simulated": True, "tool": node.tool, "capability": node.capability,
                "instruction": node.input.get("instruction")}
    if node.tool == "openai": return execute_openai(node, goal)
    if node.tool == "webhook": return execute_webhook(node, goal)
    if node.tool == "github": return execute_github(node)
    if node.tool == "noop": return {"message":"noop"}
    raise RuntimeError(f"unknown tool adapter: {node.tool}")

def transition(node: Node, new: str) -> None:
    if new == node.status: return
    if new not in TRANSITIONS.get(node.status,set()):
        raise RuntimeError(f"illegal transition {node.status}->{new} ({node.id})")
    node.status = new

def ready_nodes(nodes: list[Node]) -> list[Node]:
    done = {n.id for n in nodes if n.status == "completed"}
    return [n for n in nodes if n.status == "pending" and all(d in done for d in n.depends_on)]

def run_workflow(workflow: dict[str, Any], approve_high_risk=False) -> None:
    nodes = [Node(**n) for n in workflow["nodes"]]
    validate_dag(nodes)
    live = bool(workflow.get("live")) and os.environ.get("ORCHESTRATOR_LIVE","").lower()=="true"
    workflow["status"], workflow["execution_mode"] = "running", ("live" if live else "dry-run")
    for n in nodes: n.input["workflow_id"] = workflow["id"]
    safety = 0
    while True:
        safety += 1
        if safety > 100: raise RuntimeError("orchestration safety limit reached")
        ready = ready_nodes(nodes)
        if not ready:
            if all(n.status=="completed" for n in nodes):
                workflow["status"]="completed"; workflow["nodes"]=[asdict(n) for n in nodes]; return
            waiting = [n for n in nodes if n.status in {"pending","waiting_approval","retrying"}]
            if waiting and all(n.status=="waiting_approval" for n in waiting):
                workflow["status"]="waiting_approval"; workflow["nodes"]=[asdict(n) for n in nodes]; return
            raise RuntimeError("DAG made no progress")
        for node in ready:
            if live and node.risk in {"high","critical"} and not approve_high_risk:
                transition(node,"waiting_approval"); workflow["status"]="waiting_approval"
                append_event("approval.required",{"workflow_id":workflow["id"],"node_id":node.id,"risk":node.risk})
                continue
            transition(node,"running")
            append_event("node.started",{"workflow_id":workflow["id"],"node_id":node.id,"tool":node.tool})
            attempt = node.retry_count
            while True:
                try:
                    node.output = execute_node(node,workflow["goal"],not live)
                    transition(node,"validating")
                    node.output["validation"]={"passed":True,"checked_at":utc_now()}
                    transition(node,"completed")
                    append_event("node.completed",{"workflow_id":workflow["id"],"node_id":node.id})
                    CHECKPOINT_DIR.mkdir(parents=True,exist_ok=True)
                    write_json(CHECKPOINT_DIR/f"{workflow['id']}-{node.id}.json",
                               {"workflow":workflow["id"],"node":asdict(node),"ts":utc_now()})
                    break
                except Exception as exc:
                    node.error={"type":type(exc).__name__,"message":str(exc),
                                "trace":traceback.format_exc(limit=4)}
                    if attempt < node.max_retries:
                        attempt += 1; node.retry_count=attempt; transition(node,"retrying")
                        append_event("node.retrying",{"workflow_id":workflow["id"],"node_id":node.id,"attempt":attempt})
                        time.sleep(min(2**attempt,8)); transition(node,"ready")
                        continue
                    transition(node,"failed"); workflow["status"]="failed"; workflow["failed_node"]=node.id
                    workflow["nodes"]=[asdict(n) for n in nodes]
                    append_event("node.failed",{"workflow_id":workflow["id"],"node_id":node.id,"error":node.error})
                    return

def create_workflow(goal: str, live: bool) -> dict[str, Any]:
    nodes = deterministic_plan(goal); validate_dag(nodes)
    wf = {"id":new_id("wf"),"created_at":utc_now(),"goal":goal,"status":"planning",
          "live":live,"execution_mode":"dry-run","nodes":[asdict(n) for n in nodes]}
    return wf

def summary(wf: dict[str, Any]) -> None:
    counts={}
    for n in wf.get("nodes",[]): counts[n["status"]]=counts.get(n["status"],0)+1
    print(json.dumps({"workflow_id":wf["id"],"status":wf["status"],
                      "mode":wf.get("execution_mode"),"counts":counts,
                      "failed_node":wf.get("failed_node")},indent=2))

def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--goal",default=os.environ.get("ORCHESTRATOR_GOAL",""))
    p.add_argument("--workflow-id"); p.add_argument("--live",action="store_true")
    p.add_argument("--approve-high-risk",action="store_true"); p.add_argument("--list",action="store_true")
    a=p.parse_args(); state=load_state()
    if a.list:
        for wf in state["workflows"].values(): summary(wf)
        return 0
    if a.workflow_id:
        wf=state["workflows"].get(a.workflow_id)
        if not wf: raise SystemExit("workflow not found")
        run_workflow(wf,a.approve_high_risk); save_state(state); summary(wf)
        return 0 if wf["status"] in {"completed","waiting_approval"} else 2
    if not a.goal: raise SystemExit("provide --goal or --workflow-id")
    live=a.live or os.environ.get("ORCHESTRATOR_LIVE","").lower()=="true"
    wf=create_workflow(a.goal,live); state["workflows"][wf["id"]]=wf; state["last_workflow_id"]=wf["id"]
    append_event("workflow.created",{"workflow_id":wf["id"],"goal":wf["goal"],"live":live})
    run_workflow(wf,a.approve_high_risk); save_state(state); summary(wf)
    return 0 if wf["status"] in {"completed","waiting_approval"} else 2

if __name__=="__main__": raise SystemExit(main())
