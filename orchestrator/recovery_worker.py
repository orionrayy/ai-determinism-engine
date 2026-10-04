from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import subprocess
from typing import Any, Callable, Mapping
import urllib.error
import urllib.parse
import urllib.request

try:
    from .authority import DISTRIBUTED_CONTROL_PLANE, infer_legacy_authority
    from .control_plane import ControlPlaneClient, ControlPlaneError
    from .orchestrator import compact_terminal_workflows, load_state
    from .recovery_policy import recovery_event_id, recovery_reasons
except ImportError:
    from authority import DISTRIBUTED_CONTROL_PLANE, infer_legacy_authority
    from control_plane import ControlPlaneClient, ControlPlaneError
    from orchestrator import compact_terminal_workflows, load_state
    from recovery_policy import recovery_event_id, recovery_reasons


MAX_RUN_RESPONSE_BYTES = 256 * 1024


def _github_run_status(
    repository: str,
    run_id: str,
    token: str,
) -> str:
    path = (
        "/repos/"
        + urllib.parse.quote(repository, safe="/")
        + "/actions/runs/"
        + urllib.parse.quote(str(run_id), safe="")
    )
    request = urllib.request.Request(
        "https://api.github.com" + path,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "ai-determinism-engine-recovery/1.0",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read(MAX_RUN_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RUN_RESPONSE_BYTES:
                raise RuntimeError("GitHub run response exceeds safety bound")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"GitHub run lookup failed with HTTP {exc.code}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"GitHub run lookup failed: {exc}") from exc

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("GitHub run response is not valid JSON") from exc
    status = payload.get("status")
    if not isinstance(status, str):
        raise RuntimeError("GitHub run status is missing")
    return status


def _remote_snapshot(
    workflow: Mapping[str, Any],
    control_plane: ControlPlaneClient | None,
) -> dict[str, Any]:
    if (
        control_plane is None
        or infer_legacy_authority(workflow) != DISTRIBUTED_CONTROL_PLANE
    ):
        return dict(workflow)

    remote = control_plane.get_workflow_state(str(workflow.get("id") or ""))
    if remote is None:
        return dict(workflow)
    value = dict(remote.state)
    value["control_plane_state_version"] = int(remote.state_version)
    return value


def collect_recovery_actions(
    state: Mapping[str, Any],
    *,
    now: datetime,
    run_status_lookup: Callable[[str, str], str] | None = None,
    control_plane: ControlPlaneClient | None = None,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    workflows = state.get("workflows")
    if not isinstance(workflows, Mapping):
        return actions

    for workflow_id, raw in workflows.items():
        if not isinstance(raw, Mapping):
            continue
        workflow = _remote_snapshot(raw, control_plane)
        current_id = str(workflow.get("id") or workflow_id).strip()
        if not current_id:
            continue

        active_status = None
        run_id = str(workflow.get("github_run_id") or "").strip()
        if run_id and run_status_lookup is not None:
            try:
                active_status = run_status_lookup(
                    run_id,
                    str(workflow.get("github_run_attempt") or ""),
                )
            except Exception:
                # A tracked run that cannot be inspected must not be duplicated.
                continue

        reasons = recovery_reasons(
            workflow,
            now=now,
            active_run_status=active_status,
        )
        generation = str(
            workflow.get("control_plane_state_version")
            or workflow.get("state_generation")
            or workflow.get("github_run_attempt")
            or workflow.get("updated_at")
            or "0"
        )
        for reason in reasons:
            actions.append(
                {
                    "workflow_id": current_id,
                    "reason": reason,
                    "generation": generation,
                    "event_id": recovery_event_id(
                        workflow,
                        reason,
                        generation,
                    ),
                    "scheduled_recovery": True,
                }
            )

    actions.sort(
        key=lambda item: (
            str(item.get("workflow_id") or ""),
            str(item.get("reason") or ""),
        )
    )
    return actions


def dispatch_recovery(
    repository: str,
    action: Mapping[str, Any],
) -> None:
    payload = {
        "event_type": "orchestrator.continue",
        "client_payload": dict(action),
    }
    subprocess.run(
        [
            "gh",
            "api",
            "--method",
            "POST",
            f"repos/{repository}/dispatches",
            "--input",
            "-",
        ],
        input=json.dumps(payload, ensure_ascii=False),
        text=True,
        check=True,
    )


def run_recovery(
    *,
    repository: str,
    schedule_run_id: str,
    token: str,
    state: dict[str, Any] | None = None,
    now: datetime | None = None,
    dispatcher: Callable[[str, Mapping[str, Any]], None] = dispatch_recovery,
    run_status_lookup: Callable[[str, str], str] | None = None,
) -> dict[str, Any]:
    current = now or datetime.now(timezone.utc)
    durable_state = state if state is not None else load_state()

    if run_status_lookup is None:
        run_status_lookup = (
            lambda run_id, _attempt: _github_run_status(
                repository,
                run_id,
                token,
            )
        )

    control_plane = None
    if os.environ.get("ORCHESTRATOR_CONTROL_PLANE_URL", "").strip():
        try:
            control_plane = ControlPlaneClient.from_env()
        except ControlPlaneError:
            control_plane = None

    actions = collect_recovery_actions(
        durable_state,
        now=current,
        run_status_lookup=run_status_lookup,
        control_plane=control_plane,
    )

    dispatched: list[dict[str, Any]] = []
    dispatch_errors: list[dict[str, Any]] = []
    for action in actions:
        payload = {
            **action,
            "schedule_run_id": schedule_run_id,
        }
        try:
            dispatcher(repository, payload)
            dispatched.append(payload)
        except Exception as exc:
            # One broken target must not suppress independent recovery targets.
            dispatch_errors.append(
                {
                    "workflow_id": payload["workflow_id"],
                    "reason": payload["reason"],
                    "error": str(exc),
                }
            )

    # Distributed workflows are compacted only by their control-plane authority;
    # Git-only snapshots are safe to compact locally here.
    compactable = {
        str(workflow_id): workflow
        for workflow_id, workflow in (durable_state.get("workflows") or {}).items()
        if isinstance(workflow, Mapping)
        and infer_legacy_authority(workflow) != DISTRIBUTED_CONTROL_PLANE
    }
    compacted: list[str] = []
    if compactable:
        compact_state = {"workflows": compactable}
        compacted = compact_terminal_workflows(compact_state, now=current)

    summary = {
        "schedule_run_id": schedule_run_id,
        "candidate_count": len(actions),
        "dispatched": dispatched,
        "dispatch_errors": dispatch_errors,
        "compacted": compacted,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def main() -> int:
    repository = str(os.environ.get("REPOSITORY") or "").strip()
    token = str(os.environ.get("GH_TOKEN") or "").strip()
    schedule_run_id = str(os.environ.get("SCHEDULE_RUN_ID") or "").strip()
    if not repository or not token or not schedule_run_id:
        raise SystemExit(
            "REPOSITORY, GH_TOKEN, and SCHEDULE_RUN_ID are required"
        )
    run_recovery(
        repository=repository,
        schedule_run_id=schedule_run_id,
        token=token,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
