import hashlib
import json
import tempfile
import time
import unittest
from pathlib import Path

from enterprise_runtime import (
    AdmissionDecision, EvidenceRef, FairAdmissionController, FileObjectStore,
    LongLivedWorker, MemoryTelemetrySink, NATSJetStreamAdapter, PolicySnapshot,
    SegmentedWorkflowStore, SkillManifest, SkillRegistry, SQLiteTaskQueue,
    TaskEnvelope, TaskResultEnvelope, TraceContext, TelemetryRecorder,
    WorkerHeartbeat, WorkerRegistration, WorkerRegistry, run_w1, run_w2, run_w3,
)
from enterprise_runtime.contracts import ContractError
from enterprise_runtime.state import StateIntegrityError


SHA0 = "sha256:" + "0" * 64


def make_task(task_id="task-1", tenant_id="tenant-1", *, skill=None, provider=""):
    return TaskEnvelope(
        protocol_version="1.0",
        task_id=task_id,
        workflow_id="wf-1",
        tenant_id=tenant_id,
        capability="analyze",
        task_kind="activity",
        attempt=1,
        input={"mode": "inline", "value": "x"},
        contract={"name": "test"},
        routing={"provider": provider} if provider else {},
        deadline={},
        trace={"trace_id": "trace-1", "span_id": task_id},
        skill=skill or {},
    )


def make_skill(*, billing="free_public", trust="standard"):
    base = dict(
        skill_id="test.skill",
        version="1.0.0",
        digest=SHA0,
        description="test skill",
        capabilities=("analyze",),
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        trust_level=trust,
        billing_class=billing,
        side_effecting=False,
        required_approval=False,
        permissions={"network_class": "public"},
        resource_profile={"max_concurrent": 1},
        adapters={"http": "reference"},
    )
    probe = SkillManifest(**base)
    base["digest"] = SkillRegistry.expected_digest(probe)
    return SkillManifest(**base)


class QueueTests(unittest.TestCase):
    def test_idempotent_enqueue_claim_ack_and_expiry_dead_letters(self):
        with tempfile.TemporaryDirectory() as tmp:
            q = SQLiteTaskQueue(str(Path(tmp) / "queue.sqlite"))
            try:
                self.assertTrue(q.enqueue("t1", "tenant", {"workflow_id": "wf"}, dedupe_key="d1", max_attempts=2))
                self.assertFalse(q.enqueue("t1-copy", "tenant", {"workflow_id": "wf"}, dedupe_key="d1"))
                claim = q.claim("worker", lease_seconds=1)
                self.assertIsNotNone(claim)
                self.assertTrue(q.ack(claim))
                self.assertFalse(q.ack(claim))

                self.assertTrue(q.enqueue("t2", "tenant", {"workflow_id": "wf"}, dedupe_key="d2", max_attempts=2))
                c2 = q.claim("worker", lease_seconds=1)
                self.assertEqual(q.nack(c2, "retryable"), "queued")
                c3 = q.claim("worker", lease_seconds=1)
                self.assertEqual(c3.attempt, 2)
                self.assertEqual(q.nack(c3, "retryable"), "dead")

                self.assertTrue(q.enqueue("t3", "tenant", {"workflow_id": "wf"}, dedupe_key="d3", max_attempts=1))
                c4 = q.claim("worker", lease_seconds=1)
                self.assertEqual(q.reclaim_expired(now=c4.lease_expires_at), 1)
                self.assertEqual(q.status("t3"), "dead")
            finally:
                q.close()

    def test_partition_filtering(self):
        q = SQLiteTaskQueue()
        try:
            q.enqueue("a", "tenant", {}, dedupe_key="a", partition="p1")
            q.enqueue("b", "tenant", {}, dedupe_key="b", partition="p2")
            self.assertEqual(q.claim("w", partition="p2").task_id, "b")
            self.assertEqual(q.claim("w", partition="p3"), None)
        finally:
            q.close()

    def test_nats_adapter_preserves_idempotent_message_identity(self):
        calls = []
        adapter = NATSJetStreamAdapter(lambda op, payload: calls.append((op, payload)) or True)
        self.assertTrue(adapter.enqueue("S", "tasks.a", {"x": 1}, msg_id="dedupe-1"))
        adapter.ack("C", 7)
        adapter.nak("C", 8, delay_seconds=3)
        adapter.term("C", 9)
        self.assertEqual([c[0] for c in calls], ["publish", "ack", "nak", "term"])
        self.assertEqual(calls[0][1]["msg_id"], "dedupe-1")


class StateTests(unittest.TestCase):
    @staticmethod
    def reducer(state, event):
        state = dict(state)
        state["count"] = int(state.get("count", 0)) + int(event.payload["n"])
        return state

    def test_snapshot_tail_replay_and_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SegmentedWorkflowStore(FileObjectStore(Path(tmp) / "objects"))
            manifest = store.write_snapshot("wf", "tenant", {"count": 2}, generation=1, watermark=10)
            self.assertEqual(manifest.snapshot_watermark, 10)
            store.append_event("wf", "task.completed", {"n": 1}, sequence=11)
            store.append_event("wf", "task.completed", {"n": 2}, sequence=12)
            self.assertEqual(store.reconstruct("wf", self.reducer)["count"], 5)
            self.assertTrue(store.verify("wf", self.reducer))

            event_path = Path(tmp) / "objects" / manifest.event_file
            event_path.write_text(event_path.read_text(encoding="utf-8").replace('"n":2', '"n":9'), encoding="utf-8")
            self.assertFalse(store.verify("wf", self.reducer))

    def test_snapshot_and_object_key_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            objects = FileObjectStore(Path(tmp) / "objects")
            with self.assertRaises(StateIntegrityError):
                objects.put_bytes("../escape.txt", b"x")
            with self.assertRaises(StateIntegrityError):
                objects.put_bytes("", b"x")

    def test_sequence_gap_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SegmentedWorkflowStore(FileObjectStore(Path(tmp) / "objects"))
            store.write_snapshot("wf", "tenant", {"count": 0}, generation=1, watermark=0)
            with self.assertRaises(StateIntegrityError):
                store.append_event("wf", "task.completed", {"n": 1}, sequence=2)


class WorkerTests(unittest.TestCase):
    def registration(self):
        return WorkerRegistration(
            protocol_version="1.0", worker_id="w1", build_id="b1",
            capabilities=("analyze",), skill_versions={"test.skill": "1.0.0"},
            trust_level="standard", network_class="public",
            resources={"max_concurrent": 2}, health={"status": "ready"},
            drain_state={"mode": "active"},
        )

    def test_drain_and_heartbeat_rules(self):
        registry = WorkerRegistry(heartbeat_timeout=10)
        registry.register(self.registration(), now=100)
        registry.heartbeat(
            WorkerHeartbeat("w1", "pool", ("analyze",), 0, 2, 100), now=100
        )
        self.assertEqual(len(registry.eligible("analyze", now=105)), 1)
        registry.set_drain("w1", "drain_requested")
        self.assertEqual(registry.eligible("analyze", now=105), [])
        with self.assertRaises(ValueError):
            registry.heartbeat(
                WorkerHeartbeat("w1", "pool", ("analyze",), 0, 2, 50), now=100
            )

    def test_worker_rejects_result_attempt_mismatch(self):
        q = SQLiteTaskQueue()
        try:
            q.enqueue("t", "tenant", {"workflow_id": "wf"}, dedupe_key="d", max_attempts=2)
            def execute(payload, claim):
                return TaskResultEnvelope(
                    "1.0", claim.task_id, "wf", claim.tenant_id, claim.attempt + 1,
                    "completed", "none", {}, {"trace_id": "t", "span_id": "s"}
                )
            worker = LongLivedWorker("w", q, execute)
            worker.run_once()
            self.assertEqual(q.status("t"), "dead")
        finally:
            q.close()

    def test_stale_success_ack_is_not_requeued(self):
        q = SQLiteTaskQueue()
        try:
            q.enqueue("t", "tenant", {"workflow_id": "wf"}, dedupe_key="d")
            def execute(payload, claim):
                q.ack(claim)
                return TaskResultEnvelope(
                    "1.0", claim.task_id, "wf", claim.tenant_id, claim.attempt,
                    "completed", "none", {}, {"trace_id": "t", "span_id": "s"}
                )
            LongLivedWorker("w", q, execute).run_once()
            self.assertEqual(q.status("t"), "completed")
            self.assertEqual(q.depth(), 0)
        finally:
            q.close()


class AdmissionAndSkillTests(unittest.TestCase):
    def test_hierarchical_admission_and_hard_free(self):
        controller = FairAdmissionController()
        controller.configure_limit("tenant", "tenant-1", 1)
        task = make_task()
        decision = controller.admit(task)
        self.assertEqual(decision.status, "accepted")
        self.assertEqual(controller.admit(task).status, "quota_blocked")
        controller.release(task)
        self.assertEqual(controller.admit(task).status, "accepted")

        policy = PolicySnapshot("1", SHA0, "tenant-1", True)
        paid = make_task(skill={"skill_id": "x", "version": "1", "digest": SHA0, "billing_class": "paid"})
        self.assertEqual(controller.admit(paid, policy=policy).status, "quota_blocked")

    def test_skill_digest_trust_permissions_and_quarantine(self):
        registry = SkillRegistry()
        skill = make_skill()
        registry.register(skill)
        task = make_task(skill={
            "skill_id": skill.skill_id, "version": skill.version, "digest": skill.digest,
            "billing_class": skill.billing_class
        })
        authorized = registry.authorize(
            task, trust_level="standard", network_class="public",
            free_only=True, granted_permissions={"network_class": "public"}
        )
        self.assertEqual(authorized.digest, skill.digest)
        registry.quarantine(skill.skill_id, skill.version, "bad health")
        with self.assertRaises(PermissionError):
            registry.get(skill.skill_id, skill.version)


class TelemetryAndScaleTests(unittest.TestCase):
    def test_telemetry_preserves_trace_identity_and_audit_severity(self):
        sink = MemoryTelemetrySink()
        rec = TelemetryRecorder(sink)
        ctx = TraceContext("trace-1", "span-1", "parent-1")
        rec.lifecycle("task.completed", context=ctx, workflow_id="wf", task_id="t", tenant_id="tenant", attempt=1, worker_id="w", audit=True)
        rec.metric("queue_wait_seconds", 0.25, unit="s")
        self.assertEqual(sink.events[0]["trace_id"], "trace-1")
        self.assertEqual(sink.events[0]["severity"], "AUDIT")
        self.assertEqual(sink.events[1]["metric_name"], "queue_wait_seconds")

    def test_w1_w2_w3_are_gated_and_reference_only(self):
        self.assertTrue(run_w1(task_count=12, workers=3).passed)
        self.assertTrue(run_w2(task_count=12).passed)
        self.assertTrue(run_w3(task_count=12).passed)


if __name__ == "__main__":
    unittest.main()
