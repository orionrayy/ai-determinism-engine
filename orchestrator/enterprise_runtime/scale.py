from __future__ import annotations

from dataclasses import dataclass
import time
import tempfile
from pathlib import Path
from typing import Any, Callable

from .contracts import TaskResultEnvelope
from .queue import SQLiteTaskQueue
from .state import FileObjectStore, SegmentedWorkflowStore
from .workers import LongLivedWorker


@dataclass(frozen=True, slots=True)
class CampaignResult:
    name: str
    passed: bool
    tasks: int
    elapsed_seconds: float
    details: dict[str, Any]


def _result_from_claim(payload: dict[str, Any], claim: Any) -> TaskResultEnvelope:
    trace = payload.get("trace", {"trace_id": "w1", "span_id": claim.task_id})
    return TaskResultEnvelope(
        protocol_version=str(payload.get("protocol_version", "1.0")),
        task_id=claim.task_id,
        workflow_id=str(payload.get("workflow_id", "wf-scale")),
        tenant_id=claim.tenant_id,
        attempt=claim.attempt,
        outcome="completed",
        retry_class="none",
        output={"ok": True},
        trace=trace,
        worker_id=claim.worker_id,
    )


def run_w1(*, task_count: int = 100, workers: int = 4) -> CampaignResult:
    """Deterministic local load/failure campaign; not an enterprise capacity claim."""
    task_count = max(1, min(int(task_count), 5000))
    workers = max(1, min(int(workers), 64))
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "w1.sqlite"
        queue = SQLiteTaskQueue(str(db))
        failure_once = {i for i in range(task_count) if i % 17 == 0}

        def execute(payload: dict[str, Any], claim: Any) -> TaskResultEnvelope:
            index = int(str(claim.task_id).rsplit("-", 1)[-1])
            if index in failure_once and claim.attempt == 1:
                return TaskResultEnvelope(
                    "1.0", claim.task_id, "wf-scale", claim.tenant_id, claim.attempt,
                    "failed", "safe", {}, {"trace_id": "w1", "span_id": claim.task_id},
                    error_code="injected_transient_failure", worker_id=claim.worker_id,
                )
            return _result_from_claim(payload, claim)

        try:
            for i in range(task_count):
                queue.enqueue(
                    f"w1-task-{i:05d}", "tenant-w1",
                    {
                        "workflow_id": "wf-scale",
                        "trace": {"trace_id": "w1", "span_id": f"task-{i}"},
                    },
                    dedupe_key=f"w1-dedupe-{i:05d}",
                    max_attempts=3,
                )
            workers_pool = [
                LongLivedWorker(f"worker-{index}", queue, execute)
                for index in range(workers)
            ]
            processed = 0
            stalled_rounds = 0
            while True:
                progress = 0
                for worker in workers_pool:
                    if worker.run_once():
                        processed += 1
                        progress += 1
                if progress == 0:
                    stalled_rounds += 1
                    if queue.depth() == 0:
                        break
                    if stalled_rounds > 2:
                        return CampaignResult(
                            "W1", False, task_count, time.perf_counter() - started,
                            {"processed": processed, "queue_depth": queue.depth(), "reason": "stalled"},
                        )
                else:
                    stalled_rounds = 0
            passed = queue.depth() == 0 and all(
                queue.status(f"w1-task-{i:05d}") == "completed" for i in range(task_count)
            )
            return CampaignResult("W1", passed, task_count, time.perf_counter() - started, {
                "claim_iterations": processed,
                "workers": workers,
                "injected_transient_failures": len(failure_once),
                "queue_depth": queue.depth(),
            })
        finally:
            queue.close()
def run_w2(*, task_count: int = 64) -> CampaignResult:
    """State/partition failure-injection campaign over the $0 file reference substrate."""
    task_count = max(1, min(int(task_count), 1000))
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = SegmentedWorkflowStore(FileObjectStore(root / "objects"))
        store.write_snapshot("wf-w2", "tenant-w2", {"status": "running", "completed": 0},
                             generation=1, watermark=0)
        for sequence in range(1, task_count + 1):
            store.append_event("wf-w2", "task.completed", {"n": 1}, sequence=sequence)
        def reducer(state: dict[str, Any], event: Any) -> dict[str, Any]:
            out = dict(state)
            if event.event_type == "task.completed":
                out["completed"] = int(out.get("completed", 0)) + int(event.payload["n"])
            return out
        reconstructed = store.reconstruct("wf-w2", reducer)
        passed = reconstructed.get("completed") == task_count and store.verify("wf-w2", reducer)
        return CampaignResult("W2", passed, task_count, time.perf_counter() - started, {
            "completed": reconstructed.get("completed"),
        })


def run_w3(*, task_count: int = 64) -> CampaignResult:
    """W3 is gated on W2; it remains a local reference campaign, not production proof."""
    w2 = run_w2(task_count=task_count)
    if not w2.passed:
        return CampaignResult("W3", False, task_count, w2.elapsed_seconds, {"blocked_by": "W2"})
    started = time.perf_counter()
    w3 = run_w1(task_count=task_count, workers=4)
    return CampaignResult("W3", w3.passed, task_count, time.perf_counter() - started, {
        "prerequisite_w2": True, "w1_passed": w3.passed
    })
