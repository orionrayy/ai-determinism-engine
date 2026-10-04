#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

HEALTHY = "healthy"
DEGRADED = "degraded"
QUARANTINED = "quarantined"
PROBING = "probing"

HEALTH_RANK = {
    HEALTHY: 0,
    PROBING: 1,
    DEGRADED: 2,
    QUARANTINED: 3,
}
RISK_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}

DEFAULT_COOLDOWN_SECONDS = 300
SIDE_EFFECT_COOLDOWN_SECONDS = 900
MAX_FAILURE_STREAK = 2


def _now() -> int:
    return int(time.time())


def _risk(spec: dict[str, Any]) -> str:
    value = str(spec.get("risk") or "").lower()
    if value in RISK_RANK:
        return value
    return "high" if spec.get("side_effects") else "low"


def _candidates(registry: dict[str, dict[str, Any]], capability: str) -> list[str]:
    spec = registry.get(f"capability:{capability}", {})
    values = [spec.get("default_tool"), *(spec.get("fallback_tools") or [])]
    return list(dict.fromkeys(str(value) for value in values if value))


def _free(tool: str, registry: dict[str, dict[str, Any]]) -> bool:
    spec = registry.get(tool, {})
    return bool(spec.get("free_tier", False))


def _requires_env(tool: str, registry: dict[str, dict[str, Any]]) -> bool:
    env_name = registry.get(tool, {}).get("required_env")
    return bool(env_name)


def _env_available(tool: str, registry: dict[str, dict[str, Any]], live: bool) -> bool:
    if not live:
        return True
    env_name = registry.get(tool, {}).get("required_env")
    return not env_name or bool(os.environ.get(env_name))


def _free_allowed(tool: str, registry: dict[str, dict[str, Any]]) -> bool:
    if os.environ.get("ORCHESTRATOR_FREE_ONLY", "true").lower() != "true":
        return True
    return _free(tool, registry)



def configured_model(tool: str, registry: dict[str, dict[str, Any]]) -> str | None:
    if tool != "gemini":
        return None
    spec = registry.get("gemini", {})
    default_model = str(spec.get("default_model") or "").strip()
    if not default_model:
        return None
    return (
        os.environ.get("GEMINI_MODEL")
        or default_model
    ).strip()


def execution_eligible(
    tool: str,
    registry: dict[str, dict[str, Any]],
    health: dict[str, Any] | None = None,
    *,
    live: bool = False,
    model: str | None = None,
    now: int | None = None,
) -> bool:
    health = health or {}
    spec = registry.get(tool, {})
    if os.environ.get("ORCHESTRATOR_FREE_ONLY", "true").lower() == "true" and not bool(spec.get("free_tier", False)):
        return False
    if live and not _env_available(tool, registry, True):
        return False
    if effective_health(health, tool, now=now) == QUARANTINED:
        return False
    if tool == "gemini" and os.environ.get("ORCHESTRATOR_FREE_ONLY", "true").lower() == "true":
        selected_model = (model or configured_model(tool, registry) or "").strip()
        allowed = registry.get("gemini", {}).get("free_models")
        if not isinstance(allowed, list) or selected_model not in {str(item).strip() for item in allowed}:
            return False
    return True


def _health_record(health: dict[str, Any], tool: str) -> dict[str, Any]:
    value = health.get(tool)
    return value if isinstance(value, dict) else {}


def effective_health(
    health: dict[str, Any],
    tool: str,
    now: int | None = None,
) -> str:
    now = _now() if now is None else int(now)
    record = _health_record(health, tool)
    status = str(record.get("status") or HEALTHY)
    cooldown_until = int(record.get("cooldown_until") or 0)
    if status in {QUARANTINED, DEGRADED} and cooldown_until <= now:
        return PROBING
    if status not in HEALTH_RANK:
        return HEALTHY
    return status


def route_capability(
    capability: str,
    registry: dict[str, dict[str, Any]],
    health: dict[str, Any] | None = None,
    *,
    live: bool = False,
    preferred: str | None = None,
    exclude: set[str] | None = None,
    now: int | None = None,
) -> str:
    health = health or {}
    exclude = exclude or set()
    candidates = _candidates(registry, capability)
    if preferred and preferred not in candidates:
        candidates.insert(0, preferred)
    if not candidates:
        raise ValueError(f"no registered tools for capability {capability}")

    scored: list[tuple[tuple[Any, ...], str]] = []
    for tool in candidates:
        if tool in exclude:
            continue
        spec = registry.get(tool, {})
        free_ok = execution_eligible(tool, registry, health, live=live, now=now)
        env_ok = _env_available(tool, registry, live)
        status = effective_health(health, tool, now=now)
        if not free_ok or status == QUARANTINED:
            continue
        risk = _risk(spec)
        # Explicit preference is meaningful, but availability/free policy still
        # gates the candidate. Health, credential burden, risk, and lexical order
        # provide deterministic tie-breakers after the preference.
        score = (
            1 if tool == preferred else 0,
            1 if env_ok else 0,
            1 if free_ok else 0,
            1 if not _requires_env(tool, registry) else 0,
            HEALTH_RANK[status],
            -int(round(float(_health_record(health, tool).get("reliability_score") or 0.5) * 10000)),
            RISK_RANK[risk],
            tool,
        )
        scored.append((score, tool))

    if not scored:
        raise ValueError(f"no candidate tools remain for capability {capability}")

    available = [item for item in scored if item[0][1] == 1 and item[0][2] == 1]
    if live and not available:
        raise ValueError(
            f"no available tool for capability {capability} under current policy"
        )
    return min(
        scored,
        key=lambda item: (
            -item[0][0],
            -item[0][1],
            -item[0][2],
            -item[0][3],
            item[0][4],
            item[0][5],
            item[0][6],
            item[1],
        ),
    )[1]


def record_tool_result(
    health: dict[str, Any],
    tool: str,
    *,
    success: bool,
    side_effecting: bool = False,
    now: int | None = None,
) -> dict[str, Any]:
    now = _now() if now is None else int(now)
    current = dict(_health_record(health, tool))
    streak = int(current.get("failure_streak") or 0)

    successes = max(0, int(current.get("success_count") or 0))
    failures = max(0, int(current.get("failure_count") or 0))
    if success:
        successes += 1
        streak = 0
        updated = {
            "status": HEALTHY,
            "failure_streak": 0,
            "success_count": successes,
            "failure_count": failures,
            "reliability_score": round((successes + 1) / (successes + failures + 2), 4),
            "last_success_at": now,
            "cooldown_until": 0,
        }
    else:
        failures += 1
        streak += 1
        quarantine = side_effecting or streak >= MAX_FAILURE_STREAK
        updated = {
            "status": QUARANTINED if quarantine else DEGRADED,
            "failure_streak": streak,
            "success_count": successes,
            "failure_count": failures,
            "reliability_score": round((successes + 1) / (successes + failures + 2), 4),
            "last_failure_at": now,
            "cooldown_until": now + (
                SIDE_EFFECT_COOLDOWN_SECONDS if side_effecting else DEFAULT_COOLDOWN_SECONDS
            ),
        }

    health[tool] = updated
    return updated


def load_health(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def save_health(path: Path, health: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(health, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def available_tools(
    capability: str,
    registry: dict[str, dict[str, Any]],
    health: dict[str, Any] | None = None,
    *,
    live: bool = False,
    now: int | None = None,
) -> list[str]:
    health = health or {}
    candidates = _candidates(registry, capability)
    return [
        tool for tool in candidates
        if execution_eligible(tool, registry, health, live=live, now=now)
    ]