#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
from pathlib import Path


class DurabilityBarrierError(RuntimeError):
    """Raised when an external side-effect start cannot be durably fenced."""


def _run_git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise DurabilityBarrierError(
            f"git {' '.join(args)} failed"
            + (f": {detail[:500]}" if detail else "")
        )
    return result


def commit_side_effect_start(
    root: Path,
    *,
    execution_id: str,
    state_dir: str = ".orchestrator",
) -> bool:
    """Commit the START record to the orchestration branch before an external side effect.

    The barrier is intentionally active only inside GitHub Actions and only when the
    workflow enables it. Local runs retain the previous behavior.
    """
    if os.environ.get("GITHUB_ACTIONS", "").lower() != "true":
        return False
    if os.environ.get("ORCHESTRATOR_DURABILITY_BARRIER", "").lower() != "true":
        raise DurabilityBarrierError(
            "live side-effect execution requires ORCHESTRATOR_DURABILITY_BARRIER=true"
        )

    _run_git(root, "config", "user.name", "github-actions[bot]")
    _run_git(
        root,
        "config",
        "user.email",
        "41898282+github-actions[bot]@users.noreply.github.com",
    )
    _run_git(root, "fetch", "--no-tags", "origin", "main")

    current = _run_git(root, "rev-parse", "HEAD").stdout.strip()
    remote = _run_git(root, "rev-parse", "origin/main").stdout.strip()
    if current != remote:
        raise DurabilityBarrierError(
            "main changed after checkout; refusing side-effect start until a fresh worker resumes"
        )

    _run_git(root, "add", "--", state_dir)
    staged = _run_git(
        root,
        "diff",
        "--cached",
        "--quiet",
        "--",
        state_dir,
        check=False,
    )
    if staged.returncode == 0:
        raise DurabilityBarrierError(
            "no durable START-state change is staged for the side-effect barrier"
        )
    if staged.returncode != 1:
        raise DurabilityBarrierError("unable to inspect staged orchestration state")

    _run_git(
        root,
        "commit",
        "-m",
        f"chore(orchestrator): persist execution start {execution_id}",
    )
    # The expected remote SHA becomes the CAS guard at the actual use point.
    # A remote change after the HEAD check therefore causes this push to fail
    # before any external side effect is allowed to proceed.
    try:
        _run_git(
            root,
            "push",
            f"--force-with-lease=refs/heads/main:{remote}",
            "origin",
            "HEAD:refs/heads/main",
        )
    except DurabilityBarrierError as exc:
        raise DurabilityBarrierError(
            "durability barrier CAS rejected: main changed after preflight"
        ) from exc
    return True
