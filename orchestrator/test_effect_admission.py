from __future__ import annotations

import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from control_plane import Lease
import orchestrator as o
from orchestrator import Node


class EffectAdmissionTests(unittest.TestCase):
    def test_insufficient_lease_horizon_is_rejected(self):
        lease = SimpleNamespace(
            workflow_id="wf-1",
            owner="worker",
            fence_epoch=4,
            expires_at=1010,
        )
        with self.assertRaises(o.EffectLeaseAdmissionError):
            o.ensure_effect_lease_horizon(
                lease,
                expected_seconds=120,
                safety_seconds=30,
            )

    def test_sufficient_lease_horizon_is_admitted(self):
        lease = SimpleNamespace(
            workflow_id="wf-1",
            owner="worker",
            fence_epoch=4,
            expires_at=1151,
        )
        self.assertIsNone(
            o.ensure_effect_lease_horizon(
                lease,
                expected_seconds=120,
                safety_seconds=30,
            )
        )

    def test_invalid_horizon_configuration_is_bounded(self):
        with self.assertRaises(ValueError):
            o.parse_positive_seconds("0", "x", minimum=1, maximum=3600)
        with self.assertRaises(ValueError):
            o.parse_positive_seconds("4000", "x", minimum=1, maximum=3600)
        self.assertEqual(
            o.parse_positive_seconds("120", "x", minimum=1, maximum=3600),
            120,
        )

    def test_side_effect_path_renews_and_does_not_reacquire_workflow_lease(self):
        node = Node(
            id="n1",
            capability="execute",
            tool="github",
            depends_on=[],
            risk="medium",
            input={"action": "create_issue", "title": "x", "body": "y"},
        )
        workflow = {
            "id": "wf-effect",
            "goal": "effect",
            "status": "running",
            "live": True,
            "nodes": [o.asdict(node)],
            "retry_jitter_seed": "seed",
            "max_attempts": 10,
        }

        class FakeCP:
            owner = "worker"

            def __init__(self):
                self.acquire_calls = 0
                self.renew_calls = 0
                self.claim_epochs = []
                self.complete_epochs = []
                self._lease = Lease(
                    "wf-effect",
                    self.owner,
                    7,
                    int(time.time()) + 900,
                )

            def acquire_lease(self, workflow_id):
                self.acquire_calls += 1
                raise AssertionError("workflow lease must not be reacquired inside an active session")

            def renew_lease(self, workflow_id, fence_epoch):
                self.renew_calls += 1
                self._lease = Lease(
                    workflow_id,
                    self.owner,
                    fence_epoch,
                    int(time.time()) + 900,
                )
                return self._lease

            def claim_effect(self, workflow_id, effect_id, semantic_digest, fence_epoch):
                self.claim_epochs.append(fence_epoch)
                return SimpleNamespace(
                    status="claimed",
                    effect_id=effect_id,
                    semantic_digest=semantic_digest,
                )

            def complete_effect(
                self,
                workflow_id,
                effect_id,
                semantic_digest,
                fence_epoch,
                output_sha256="",
            ):
                self.complete_epochs.append(fence_epoch)

        cp = FakeCP()

        def fake_execute(node_obj, **callbacks):
            callbacks["before_attempt"]()
            callbacks["on_success"](node_obj)
            node_obj.status = "completed"
            node_obj.output = {"ok": True}
            return True, None

        with patch.object(o, "load_registry", return_value={
            "github": {
                "side_effects": ["repository_write"],
                "free_tier": True,
                "effect_contracts": {
                    "create_issue": {
                        "retry": "reconcile",
                        "reconciliation": "deterministic",
                    }
                },
            }
        }), patch.object(o, "validate_dag"),              patch.object(o, "enforce_node_policy"),              patch.object(o, "ensure_plan_integrity", return_value=True),              patch.object(o, "verify_completed_checkpoints", return_value=True),              patch.object(o, "delegate_ready_agents", return_value=False),              patch.object(o, "refresh_approvals"),              patch.object(o, "build_node_context", return_value={}),              patch.object(o, "reserve_llm_call", return_value=True),              patch.object(o, "persist_workflow"),              patch.object(o, "append_event"),              patch.object(o, "notify_issue"),              patch.object(o, "node_success_checkpoint"),              patch.object(o, "update_tool_health"),              patch.object(o, "acquire_node_resource_locks", return_value=[]),              patch.object(o, "release_node_resource_locks"),              patch.object(o, "execute_with_retries", side_effect=lambda node_obj, goal, dry_run, **kw: (
                 kw["before_attempt"](),
                 kw["on_success"](node_obj),
                 (setattr(node_obj, "status", "completed"), setattr(node_obj, "output", {"ok": True})),
                 True,
                 None,
             )[-2:]):
            result = o._run_one_step_inner(
                workflow,
                control_plane=cp,
                control_plane_lease=cp._lease,
            )

        self.assertEqual(result, "completed")
        self.assertEqual(cp.acquire_calls, 0)
        self.assertGreaterEqual(cp.renew_calls, 2)
        self.assertTrue(cp.claim_epochs)
        self.assertTrue(cp.complete_epochs)
        self.assertEqual(cp.claim_epochs[-1], 7)
        self.assertEqual(cp.complete_epochs[-1], 7)


if __name__ == "__main__":
    unittest.main()
