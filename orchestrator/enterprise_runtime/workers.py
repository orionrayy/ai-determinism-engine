from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any, Callable

from .contracts import TaskEnvelope, TaskResultEnvelope, WorkerHeartbeat, WorkerRegistration
from .queue import QueueClaimRecord, TaskQueue


@dataclass(slots=True)
class WorkerRecord:
    registration: WorkerRegistration
    last_heartbeat: float
    active_tasks: int = 0


class WorkerRegistry:
    def __init__(self, heartbeat_timeout: float = 90.0) -> None:
        self.heartbeat_timeout = max(5.0, float(heartbeat_timeout))
        self._workers: dict[str, WorkerRecord] = {}
        self._lock = threading.RLock()

    def register(self, registration: WorkerRegistration, now: float | None = None) -> None:
        now = time.time() if now is None else float(now)
        with self._lock:
            self._workers[registration.worker_id] = WorkerRecord(registration, now)

    def heartbeat(self, heartbeat: WorkerHeartbeat, now: float | None = None) -> None:
        now = time.time() if now is None else float(now)
        with self._lock:
            record = self._workers.get(heartbeat.worker_id)
            if record is None:
                raise KeyError("worker_not_registered")
            reported = float(heartbeat.heartbeat_at)
            if abs(now - reported) > self.heartbeat_timeout * 2:
                raise ValueError("heartbeat_timestamp_outside_tolerance")
            record.last_heartbeat = reported
            record.active_tasks = heartbeat.active_tasks
            if record.registration.health.get("status") == "quarantined":
                return

    def set_drain(self, worker_id: str, mode: str) -> None:
        if mode not in {"active","drain_requested","draining","retired"}:
            raise ValueError("invalid drain mode")
        with self._lock:
            record = self._workers[worker_id]
            data = record.registration.to_dict()
            data["drain_state"] = {"mode": mode, "requested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            record.registration = WorkerRegistration(**data)

    def healthy(self, worker_id: str, now: float | None = None) -> bool:
        now = time.time() if now is None else float(now)
        with self._lock:
            record = self._workers.get(worker_id)
            if record is None:
                return False
            return (
                now - record.last_heartbeat <= self.heartbeat_timeout
                and record.registration.health.get("status") in {"ready","degraded"}
                and record.registration.drain_state.get("mode") in {"active","drain_requested"}
            )

    def eligible(self, capability: str, *, now: float | None = None) -> list[WorkerRecord]:
        now = time.time() if now is None else float(now)
        with self._lock:
            return [
                r for r in self._workers.values()
                if self.healthy(r.registration.worker_id, now)
                and r.registration.drain_state.get("mode") == "active"
                and capability in r.registration.capabilities
                and int(r.registration.resources.get("max_concurrent", 0) or
                        r.registration.resources.get("available_slots", 0)) > r.active_tasks
            ]


class LongLivedWorker:
    """Replaceable stateless worker loop.

    Workflow truth remains in the queue/control-plane; this class owns only
    process-local execution counters and task handling.
    """

    def __init__(self, worker_id: str, queue: TaskQueue,
                 execute: Callable[[dict[str, Any], QueueClaimRecord], TaskResultEnvelope],
                 *, lease_seconds: int = 120) -> None:
        self.worker_id = worker_id
        self.queue = queue
        self.execute = execute
        self.lease_seconds = max(1, min(int(lease_seconds), 3600))
        self.running = False

    def run_once(self) -> bool:
        claim = self.queue.claim(self.worker_id, lease_seconds=self.lease_seconds)
        if claim is None:
            return False
        try:
            result = self.execute(claim.payload, claim)
            if (
                result.task_id != claim.task_id
                or result.tenant_id != claim.tenant_id
                or result.workflow_id != str(claim.payload.get("workflow_id", result.workflow_id))
                or int(result.attempt) != int(claim.attempt)
            ):
                self.queue.nack(claim, "result_identity_mismatch", dead_letter=True)
                return True
            if result.outcome == "completed":
                # A lost acknowledgement must not be turned into a blind retry:
                # an external effect may already have committed.
                self.queue.ack(claim)
            elif result.retry_class in {"safe","reconcile"}:
                self.queue.nack(claim, result.error_code or "task_failed")
            else:
                self.queue.nack(claim, result.error_code or "task_blocked", dead_letter=True)
        except Exception as exc:
            self.queue.nack(claim, str(exc))
        return True

    def run(self, *, max_iterations: int | None = None,
            idle_sleep: float = 0.25, stop: Callable[[], bool] | None = None) -> int:
        self.running = True
        count = 0
        try:
            while self.running and (max_iterations is None or count < max_iterations):
                if stop and stop():
                    break
                if self.run_once():
                    count += 1
                    continue
                time.sleep(max(0.0, idle_sleep))
        finally:
            self.running = False
        return count

    def shutdown(self) -> None:
        self.running = False
