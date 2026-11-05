#!/usr/bin/env python3
import unittest

from runtime_contracts import (
    ArtifactRef,
    ContractError,
    EvidenceRef,
    PolicySnapshot,
    QueueClaim,
    SkillManifest,
    TaskEnvelope,
    TaskResultEnvelope,
    WorkerHeartbeat,
    WorkflowCommand,
    contract_digest,
)


def digest(ch: str) -> str:
    return ch * 64


def policy() -> PolicySnapshot:
    return PolicySnapshot(
        version="1",
        digest=digest("a"),
        free_only=True,
        tenant_id="tenant-a",
        limits={"max_tasks": 10},
    )


def skill() -> SkillManifest:
    return SkillManifest(
        skill_id="research.claims",
        version="1.0.0",
        digest=digest("b"),
        capabilities=("research", "claim_verification"),
        input_schema="schema://research/input",
        output_schema="schema://research/output",
        runtime="python",
        entrypoint="orchestrator.skills.research:run",
        trust_class="reviewed",
        cost_class="free_public",
        risk="low",
    )


def artifact() -> ArtifactRef:
    return ArtifactRef(
        artifact_id="artifact-a",
        sha256=digest("c"),
        schema_version="1",
        workflow_id="wf-a",
        producer_task_id="task-a",
        tenant_id="tenant-a",
        location="r2://example/artifact-a",
        media_type="application/json",
    )


def evidence() -> EvidenceRef:
    return EvidenceRef(
        evidence_id="evidence-a",
        work_id="doi:10.1234/example",
        source_uri="https://example.org/source",
        content_sha256=digest("d"),
        retrieved_at="2026-10-05T00:00:00Z",
        access_verification="verified",
        passage_ids=("passage-a",),
    )


class PortableContractTests(unittest.TestCase):
    def test_digest_is_stable(self):
        command = WorkflowCommand(
            workflow_id="wf-a",
            tenant_id="tenant-a",
            goal="research",
            input_digest=digest("1"),
            intent_digest=digest("2"),
            policy=policy(),
        )
        self.assertEqual(command.command_digest, command.command_digest)
        self.assertEqual(contract_digest(command), command.command_digest)

    def test_artifact_reference_binds_identity(self):
        first = artifact()
        second = ArtifactRef(
            artifact_id=first.artifact_id,
            sha256=digest("e"),
            schema_version=first.schema_version,
            workflow_id=first.workflow_id,
            producer_task_id=first.producer_task_id,
            tenant_id=first.tenant_id,
            location=first.location,
        )
        self.assertNotEqual(first.digest, second.digest)

    def test_task_envelope_contains_portable_boundary(self):
        envelope = TaskEnvelope(
            workflow_id="wf-a",
            task_id="task-a",
            tenant_id="tenant-a",
            attempt=1,
            capability="research",
            skill=skill(),
            input_ref=artifact(),
            policy=policy(),
            idempotency_key="wf-a:task-a",
            fence_epoch=7,
            trace_id="trace-a",
            partition="p-01",
        )
        encoded = envelope.to_dict()
        self.assertEqual(encoded["schema_version"], 1)
        self.assertEqual(encoded["skill"]["skill_id"], "research.claims")
        self.assertEqual(encoded["input_ref"]["workflow_id"], "wf-a")
        self.assertEqual(len(envelope.envelope_digest), 64)

    def test_result_envelope_bounds_evidence_and_result(self):
        result = TaskResultEnvelope(
            workflow_id="wf-a",
            task_id="task-a",
            tenant_id="tenant-a",
            attempt=1,
            outcome="completed",
            retry_class="none",
            output_ref=artifact(),
            evidence_refs=(evidence(),),
            worker_id="worker-a",
            trace_id="trace-a",
        )
        self.assertEqual(result.to_dict()["evidence_refs"][0]["access_verification"], "verified")
        self.assertEqual(len(result.result_digest), 64)

    def test_queue_claim_is_portable(self):
        claim = QueueClaim(
            queue_id="research",
            task_id="task-a",
            worker_id="worker-a",
            claim_id="claim-a",
            attempt=1,
            lease_expires_at=100,
            fence_epoch=9,
        )
        self.assertEqual(claim.to_dict()["fence_epoch"], 9)

    def test_worker_heartbeat_rejects_over_capacity(self):
        with self.assertRaises(ContractError):
            WorkerHeartbeat(
                worker_id="worker-a",
                pool="default",
                capabilities=("research",),
                active_tasks=5,
                capacity=4,
                emitted_at_epoch=100,
            )

    def test_invalid_sha_is_rejected(self):
        with self.assertRaises(ContractError):
            ArtifactRef(
                artifact_id="artifact-a",
                sha256="not-a-digest",
                schema_version="1",
                workflow_id="wf-a",
                producer_task_id="task-a",
                location="r2://example/artifact-a",
            )

    def test_duplicate_capabilities_are_rejected(self):
        with self.assertRaises(ContractError):
            SkillManifest(
                skill_id="bad",
                version="1",
                digest=digest("b"),
                capabilities=("research", "research"),
                input_schema="in",
                output_schema="out",
                runtime="python",
                entrypoint="x",
            )

    def test_ownership_is_bound_to_tenant_and_workflow(self):
        bad_input = ArtifactRef(
            artifact_id="artifact-b",
            sha256=digest("e"),
            schema_version="1",
            workflow_id="wf-other",
            producer_task_id="task-a",
            tenant_id="tenant-b",
            location="r2://example/artifact-b",
        )
        with self.assertRaises(ContractError):
            TaskEnvelope(
                workflow_id="wf-a",
                task_id="task-a",
                tenant_id="tenant-a",
                attempt=1,
                capability="research",
                skill=skill(),
                input_ref=bad_input,
                policy=policy(),
                idempotency_key="wf-a:task-a",
            )

    def test_policy_boolean_is_strict(self):
        with self.assertRaises(ContractError):
            PolicySnapshot(
                version="1",
                digest=digest("a"),
                free_only="false",
                tenant_id="tenant-a",
            )

    def test_schema_version_is_fail_closed(self):
        with self.assertRaises(ContractError):
            QueueClaim(
                queue_id="q",
                task_id="t",
                worker_id="w",
                claim_id="c",
                attempt=1,
                lease_expires_at=100,
                schema_version=99,
            )


if __name__ == "__main__":
    unittest.main()
