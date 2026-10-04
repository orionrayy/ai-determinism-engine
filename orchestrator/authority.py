from __future__ import annotations

from typing import Any, Mapping

GIT_DURABLE = "git_durable"
DISTRIBUTED_CONTROL_PLANE = "distributed_control_plane"
VALID_AUTHORITY_MODES = frozenset({GIT_DURABLE, DISTRIBUTED_CONTROL_PLANE})


def authority_mode_for_workflow(
    live: bool,
    control_plane_available: bool,
) -> str:
    if bool(live) and bool(control_plane_available):
        return DISTRIBUTED_CONTROL_PLANE
    return GIT_DURABLE


def infer_legacy_authority(workflow: Mapping[str, Any]) -> str:
    explicit = str(workflow.get("authority_mode") or "").strip()
    if explicit in VALID_AUTHORITY_MODES:
        return explicit

    if workflow.get("control_plane", {}).get("enabled") is True if isinstance(
        workflow.get("control_plane"), Mapping
    ) else False:
        return DISTRIBUTED_CONTROL_PLANE

    executions = workflow.get("executions")
    if isinstance(executions, Mapping):
        if any(
            isinstance(record, Mapping)
            and record.get("durability_authority") == "control_plane"
            for record in executions.values()
        ):
            return DISTRIBUTED_CONTROL_PLANE

    return GIT_DURABLE


def require_runtime_authority(
    workflow: Mapping[str, Any],
    *,
    control_plane_configured: bool,
    control_plane_active: bool,
) -> None:
    if not workflow.get("live"):
        return
    mode = infer_legacy_authority(workflow)
    if mode == DISTRIBUTED_CONTROL_PLANE and (
        not control_plane_configured or not control_plane_active
    ):
        raise RuntimeError(
            "distributed control-plane authority is required for this live workflow"
        )


__all__ = [
    "DISTRIBUTED_CONTROL_PLANE",
    "GIT_DURABLE",
    "VALID_AUTHORITY_MODES",
    "authority_mode_for_workflow",
    "infer_legacy_authority",
    "require_runtime_authority",
]
