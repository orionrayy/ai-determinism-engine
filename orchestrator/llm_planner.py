from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request


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


def plan_goal(goal: str, registry: dict, Node, validate_dag) -> list:
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise RuntimeError('GEMINI_API_KEY is not configured')

    capabilities = sorted(
        key.split(':', 1)[1] for key in registry if key.startswith('capability:')
    )
    tools = sorted(
        key for key in registry if not key.startswith('capability:')
    )
    prompt = (
        'Create a minimal executable workflow DAG for this goal. Return only JSON with '
        'nodes[]. Each node has id, capability, tool, depends_on, risk, instruction, contract, artifacts. '
        'Maximum 24 nodes. Dependencies must reference node ids. Build a true DAG: maximize independent nodes that can run in parallel when dependencies allow. '
        'Use only these capabilities: ' + ', '.join(capabilities) + '. '
        'Use only these tools: ' + ', '.join(tools) + '. '
        'Use low/medium/high/critical risk and mark external side effects high or critical. '
        'Prefer tools that require no credentials. Include a final validate node whose dependencies cover the outputs it must verify. For validation nodes, define contract.required_fields and/or contract.min_sources when deterministically checkable; declare artifacts as a list of expected deliverables. Goal: ' + goal
    )
    model = os.environ.get('GEMINI_PLANNER_MODEL', os.environ.get('GEMINI_MODEL', 'gemini-3.8-flash'))
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
        if not node_id or capability not in allowed_caps or tool not in allowed_tools:
            raise ValueError('planner produced unsupported node fields')
        if not isinstance(deps, list):
            raise ValueError('depends_on must be a list')
        if not isinstance(contract, dict):
            raise ValueError('contract must be an object')
        if not isinstance(artifacts, list):
            raise ValueError('artifacts must be an array')
        if risk not in {'low', 'medium', 'high', 'critical'}:
            raise ValueError('planner produced invalid risk')
        nodes.append(Node(
            id=node_id,
            capability=capability,
            tool=tool,
            depends_on=[str(dep) for dep in deps],
            risk=risk,
            input={'goal': goal, 'instruction': instruction, 'artifacts': artifacts},
            contract=contract,
        ))
    validate_dag(nodes)
    return nodes
