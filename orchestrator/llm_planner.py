from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

try:
    from .capability_graph import free_only_enabled
    from .connector_bridge import (
        ConnectorBridgeError,
        discover_capabilities,
        validate_discovered_payload,
    )
except ImportError:
    from capability_graph import free_only_enabled
    from connector_bridge import (
        ConnectorBridgeError,
        discover_capabilities,
        validate_discovered_payload,
    )


def _post(url: str, payload: dict, api_key: str) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers={
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'x-goog-api-key': api_key,
            'User-Agent': 'ai-orchestrator-planner/1.0',
        },
        method='POST',
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.loads(response.read().decode('utf-8'))


def _object_from_text(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith('```'):
        cleaned = cleaned.replace('```json', '', 1).replace('```', '', 1).strip()
    left = cleaned.find('{')
    right = cleaned.rfind('}')
    if left < 0 or right <= left:
        raise ValueError('planner returned no JSON object')
    value = json.loads(cleaned[left:right + 1])
    if not isinstance(value, dict):
        raise ValueError('planner result is not an object')
    return value


def plan_goal(goal: str, registry: dict, Node, validate_dag, live: bool = False) -> list:
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise RuntimeError('GEMINI_API_KEY is not configured')

    capabilities = sorted(
        key.split(':', 1)[1] for key in registry if key.startswith('capability:')
    )
    tools = sorted(
        key for key in registry if not key.startswith('capability:')
    )
    bridge_inventory = {}
    if live:
        bridge_url = os.environ.get('ORCHESTRATOR_CONNECTOR_BRIDGE_URL', '').strip()
        if bridge_url:
            try:
                bridge_inventory = discover_capabilities(bridge_url)
            except ConnectorBridgeError:
                bridge_inventory = {}

    inventory_text = json.dumps(
        bridge_inventory,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    )[:12000]

    prompt = (
        'Create a minimal executable workflow DAG for this goal. Return only JSON with '
        'nodes[]. Each node has id, capability, tool, depends_on, risk, instruction, contract, artifacts. '
        'Connector-bridge nodes may additionally set connector, action, and payload. '
        'Maximum 24 nodes. Dependencies must reference node ids. Build a true DAG: maximize independent nodes that can run in parallel when dependencies allow. '
        'Use only these capabilities: ' + ', '.join(capabilities) + '. '
        'Use only these tools: ' + ', '.join(tools) + '. '
        'Use low/medium/high/critical risk and mark external side effects high or critical. '
        'Prefer tools that require no credentials. Include a final validate node whose dependencies cover the outputs it must verify. For validation nodes, define contract.required_fields and/or contract.min_sources when deterministically checkable; declare artifacts as a list of expected deliverables. '
        'Live connector inventory (sanitized; empty means unavailable): ' + inventory_text + '. '
        'Only use connector/action pairs present in that live inventory when tool=connector_bridge. '
        'For action_specs, honor required fields and declared primitive types exactly; do not invent connector fields. '
        'Goal: ' + goal
    )
    model = os.environ.get('GEMINI_MODEL', 'gemini-3.8-flash')
    if free_only_enabled():
        configured = registry.get("gemini", {})
        allowed_models = {
            str(value).strip() for value in (configured.get("free_models") or [])
        }
        if allowed_models and model not in allowed_models:
            raise RuntimeError("configured Gemini planner model is not free-tier allowlisted")
    else:
        model = os.environ.get('GEMINI_PLANNER_MODEL', model)
    endpoint = (
        'https://generativelanguage.googleapis.com/v1beta/models/'
        + urllib.parse.quote(model, safe='')
        + ':generateContent'
    )
    response = _post(
        endpoint,
        {
            'contents': [{'parts': [{'text': prompt}]}],
            'generationConfig': {
                'temperature': 0.1,
                'responseMimeType': 'application/json',
            },
        },
        api_key,
    )
    candidates = response.get('candidates', [])
    if not candidates:
        raise ValueError('planner returned no candidates')
    parts = candidates[0].get('content', {}).get('parts', [])
    text = next(
        (part.get('text') for part in parts if isinstance(part, dict) and part.get('text')),
        None,
    )
    if not text:
        raise ValueError('planner returned no text')

    raw = _object_from_text(text)
    items = raw.get('nodes')
    if not isinstance(items, list):
        raise ValueError('planner JSON must contain nodes[]')
    allowed_caps = set(capabilities) | {'execute'}
    allowed_tools = set(tools) | {'noop'}
    nodes = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError('planner node is not an object')
        node_id = str(item.get('id', '')).strip()
        capability = str(item.get('capability', '')).strip()
        tool = str(item.get('tool', '')).strip()
        risk = str(item.get('risk', 'low')).strip()
        instruction = str(item.get('instruction', '')).strip()
        contract = item.get('contract', {})
        artifacts = item.get('artifacts', [])
        deps = item.get('depends_on', [])
        connector = str(item.get('connector', '')).strip().lower()
        action = str(item.get('action', '')).strip().lower()
        payload = item.get('payload', {})
        if not node_id or capability not in allowed_caps or tool not in allowed_tools:
            raise ValueError('planner produced unsupported node fields')
        if not isinstance(deps, list):
            raise ValueError('depends_on must be a list')
        if not isinstance(contract, dict):
            raise ValueError('contract must be an object')
        if not isinstance(artifacts, list):
            raise ValueError('artifacts must be an array')
        if not isinstance(payload, dict):
            raise ValueError('payload must be an object')
        if tool == 'connector_bridge':
            if not connector or not action:
                raise ValueError('connector_bridge nodes require connector and action')
            if live:
                spec = bridge_inventory.get(connector)
                if not isinstance(spec, dict) or spec.get('configured') is False:
                    raise ValueError('planner connector is not available in live inventory')
                if action not in spec.get('actions', []):
                    raise ValueError('planner connector action is not available in live inventory')
                try:
                    validate_discovered_payload(connector, action, payload, bridge_inventory)
                except ConnectorBridgeError as exc:
                    raise ValueError(str(exc)) from exc
        if risk not in {'low', 'medium', 'high', 'critical'}:
            raise ValueError('planner produced invalid risk')
        nodes.append(Node(
            id=node_id,
            capability=capability,
            tool=tool,
            depends_on=[str(dep) for dep in deps],
            risk=risk,
            input={
                'goal': goal,
                'instruction': instruction,
                'artifacts': artifacts,
                **({'connector': connector, 'action': action, 'payload': payload}
                   if tool == 'connector_bridge' else {}),
            },
            contract=contract,
        ))
    validate_dag(nodes)
    return nodes
