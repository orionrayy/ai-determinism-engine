#!/usr/bin/env python3
"""Zero-cost federation backpressure and fairness policy."""
from __future__ import annotations

import hashlib
from typing import Any

FEDERATION_SLOTS = 3
DEFAULT_MAX_BATCHES_PER_WORKFLOW = 4
MAX_BATCHES_PER_WORKFLOW = 8
DEFAULT_MAX_TASKS_PER_WORKFLOW = 16
MAX_TASKS_PER_WORKFLOW = 32
MAX_TASKS_PER_BATCH = 4


class FederationQuotaError(ValueError):
    pass


def federation_slot(federation_id: str) -> int:
    value = str(federation_id or "").strip()
    if not value:
        raise FederationQuotaError("federation_id is required")
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16) % FEDERATION_SLOTS


def configured_quota(workflow: dict[str, Any]) -> tuple[int, int]:
    try:
        batches = int(workflow.get("max_federation_batches", DEFAULT_MAX_BATCHES_PER_WORKFLOW))
        tasks = int(workflow.get("max_federation_tasks", DEFAULT_MAX_TASKS_PER_WORKFLOW))
    except (TypeError, ValueError) as exc:
        raise FederationQuotaError("invalid federation quota configuration") from exc
    return (
        max(0, min(batches, MAX_BATCHES_PER_WORKFLOW)),
        max(0, min(tasks, MAX_TASKS_PER_WORKFLOW)),
    )


def quota_usage(workflow: dict[str, Any]) -> tuple[int, int]:
    try:
        batches = max(0, int(workflow.get("federation_batches_used", 0)))
        tasks = max(0, int(workflow.get("federation_tasks_used", 0)))
    except (TypeError, ValueError) as exc:
        raise FederationQuotaError("invalid federation quota usage") from exc
    return batches, tasks


def can_reserve(
    workflow: dict[str, Any],
    task_count: int,
) -> tuple[bool, str]:
    if task_count < 1 or task_count > MAX_TASKS_PER_BATCH:
        return False, "batch_size"
    max_batches, max_tasks = configured_quota(workflow)
    used_batches, used_tasks = quota_usage(workflow)
    if used_batches >= max_batches:
        return False, "max_batches"
    if used_tasks + task_count > max_tasks:
        return False, "max_tasks"
    return True, ""


def reserve(
    workflow: dict[str, Any],
    task_count: int,
) -> None:
    allowed, reason = can_reserve(workflow, task_count)
    if not allowed:
        raise FederationQuotaError(f"federation quota exhausted: {reason}")
    used_batches, used_tasks = quota_usage(workflow)
    workflow["federation_batches_used"] = used_batches + 1
    workflow["federation_tasks_used"] = used_tasks + task_count


def refund(
    workflow: dict[str, Any],
    task_count: int,
) -> None:
    used_batches, used_tasks = quota_usage(workflow)
    workflow["federation_batches_used"] = max(0, used_batches - 1)
    workflow["federation_tasks_used"] = max(0, used_tasks - max(0, int(task_count)))
