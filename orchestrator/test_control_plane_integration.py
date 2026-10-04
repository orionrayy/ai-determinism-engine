from __future__ import annotations

import unittest
from unittest.mock import patch

import orchestrator as o

class ControlPlaneIntegrationTests(unittest.TestCase):
    def test_effect_digest_ignores_runtime_fields(self):
        n = o.Node("n1","publish","github",[],risk="high",input={
            "action":"create_issue","title":"X","body":"Y","workflow_id":"wf",
            "context":{"a":1},"repair_feedback":{"a":1},
            "retry_jitter_seed":"a","approval_granted":True,
        })
        a = o.effect_semantic_digest(n)
        n.input["context"]={"a":2}
        n.input["repair_feedback"]={"a":2}
        n.input["retry_jitter_seed"]="b"
        self.assertEqual(a, o.effect_semantic_digest(n))

    def test_uncertain_effect_blocks_replan(self):
        n = o.Node("n1","publish","github",[],risk="high")
        n.status = "failed"
        n.error = {"execution_uncertain":True}
        wf = {"id":"wf","live":True,"replan_count":0,"attempts_used":0,"max_attempts":8}
        registry = {"github":{"free_tier":True},"capability:publish":{"default_tool":"github","fallback_tools":[]}}
        with patch.object(o, "route_tool", side_effect=AssertionError("must not route")):
            self.assertFalse(o.replan_after_failure(wf,[n],n,registry))

    def test_record_effect_claim(self):
        n = o.Node("n1","publish","github",[],input={"action":"x"})
        wf = {"id":"wf"}
        effect_id = o.execution_key(wf,n)
        digest = o.effect_semantic_digest(n)
        o.record_effect_claim(wf,effect_id,digest,9)
        self.assertEqual(wf["executions"][effect_id]["fence_epoch"],9)
        self.assertEqual(wf["executions"][effect_id]["effect_semantic_digest"],digest)

    def test_resource_keys_are_sorted_and_deduped(self):
        n = o.Node(
            "n1",
            "publish",
            "github",
            [],
            input={"resource_keys": ["b", "a", "b"]},
            resources=["c", "a"],
        )
        self.assertEqual(o.node_resource_keys(n), ["a", "b", "c"])

    def test_hot_state_persist_uses_control_plane_cas(self):
        n = o.Node("n1", "research", "noop", [], input={})
        workflow = {
            "id": "wf-hot",
            "live": True,
            "status": "running",
            "nodes": [o.asdict(n)],
            "control_plane_state_version": 2,
        }

        class Stub:
            owner = "worker"

            def put_workflow_state(self, *args, **kwargs):
                self.kwargs = kwargs
                return 3

        stub = Stub()
        lease = type("Lease", (), {"fence_epoch": 8})()
        token1 = o.ACTIVE_CONTROL_PLANE.set(stub)
        token2 = o.ACTIVE_CONTROL_PLANE_LEASE.set(lease)
        try:
            o.persist_workflow(workflow)
        finally:
            o.ACTIVE_CONTROL_PLANE.reset(token1)
            o.ACTIVE_CONTROL_PLANE_LEASE.reset(token2)

        self.assertEqual(workflow["control_plane_state_version"], 3)
        self.assertEqual(stub.kwargs["expected_state_version"], 2)


    def test_resource_lock_release_order_is_reverse_sorted(self):
        released = []

        class Stub:
            def release_resource(self, resource_key, *, workflow_id, fence_epoch):
                released.append((resource_key, workflow_id, fence_epoch))

        leases = [
            type("Lease", (), {"resource_key": "a", "fence_epoch": 1})(),
            type("Lease", (), {"resource_key": "c", "fence_epoch": 3})(),
        ]
        lock_map = {"node-b": leases}
        o.release_node_resource_lock_map(
            {"id": "wf"},
            lock_map,
            Stub(),
        )
        self.assertEqual(released, [
            ("c", "wf", 3),
            ("a", "wf", 1),
        ])
        self.assertEqual(lock_map, {})

    def test_remote_event_path_is_opt_in(self):
        class Stub:
            owner = "worker"

            def append_outbox_event(self, *args, **kwargs):
                self.called = True

        stub = Stub()
        lease = type("Lease", (), {"fence_epoch": 1})()
        token1 = o.ACTIVE_CONTROL_PLANE.set(stub)
        token2 = o.ACTIVE_CONTROL_PLANE_LEASE.set(lease)
        try:
            with patch.dict(
                o.os.environ,
                {"ORCHESTRATOR_CONTROL_PLANE_REMOTE_EVENTS": "false"},
                clear=False,
            ):
                stub.called = False
                o.append_event("node.completed", {"workflow_id": "wf", "node_id": "n1"})
                self.assertFalse(stub.called)

            with patch.dict(
                o.os.environ,
                {"ORCHESTRATOR_CONTROL_PLANE_REMOTE_EVENTS": "true"},
                clear=False,
            ):
                stub.called = False
                o.append_event("node.completed", {"workflow_id": "wf", "node_id": "n1"})
                self.assertTrue(stub.called)
        finally:
            o.ACTIVE_CONTROL_PLANE.reset(token1)
            o.ACTIVE_CONTROL_PLANE_LEASE.reset(token2)


    def test_effect_contracts_are_action_aware(self):
        registry = {
            "github": {
                "effect_contracts": {
                    "create_issue": {
                        "identity": "engine",
                        "retry": "reconcile",
                        "reconciliation": "deterministic",
                        "fencing": "engine",
                    },
                    "dispatch_workflow": {
                        "identity": "engine",
                        "retry": "blocked",
                        "reconciliation": "none",
                        "fencing": "engine",
                    },
                },
            }
        }
        issue = o.Node(
            "n1",
            "publish",
            "github",
            [],
            input={"action": "create_issue"},
        )
        dispatch = o.Node(
            "n2",
            "publish",
            "github",
            [],
            input={"action": "dispatch_workflow"},
        )
        issue_contract = o.resolve_effect_contract(issue, registry)
        dispatch_contract = o.resolve_effect_contract(dispatch, registry)
        self.assertEqual(issue_contract.retry, "reconcile")
        self.assertEqual(dispatch_contract.retry, "blocked")
        self.assertTrue(issue_contract.can_reconcile)
        self.assertFalse(dispatch_contract.can_reconcile)

    def test_missing_live_effect_contract_fails_closed(self):
        n = o.Node(
            "n1",
            "publish",
            "github",
            [],
            input={"action": "unreviewed_new_action"},
        )
        registry = {
            "github": {
                "side_effects": ["repo_write"],
                "free_tier": True,
            },
        }
        with self.assertRaises(RuntimeError):
            o.require_live_effect_contract(n, registry, dry_run=False)


    def test_connector_bridge_contract_is_wildcard(self):
        registry = {
            "connector_bridge": {
                "effect_contracts": {
                    "*": {
                        "identity": "provider",
                        "retry": "reconcile",
                        "reconciliation": "provider",
                        "fencing": "engine",
                    }
                }
            }
        }
        n = o.Node(
            "n1",
            "publish",
            "connector_bridge",
            [],
            input={"action": "notion_create_page"},
        )
        contract = o.resolve_effect_contract(n, registry)
        self.assertEqual(contract.identity, "provider")
        self.assertEqual(contract.retry, "reconcile")
        self.assertTrue(contract.can_reconcile)


    def test_noop_never_becomes_side_effect_from_capability_name(self):
        n = o.Node(
            "n1",
            "deploy",
            "noop",
            [],
            risk="high",
        )
        self.assertFalse(
            o.side_effecting(
                n,
                {"noop": {"free_tier": True}},
            )
        )


if __name__ == "__main__":
    unittest.main()
