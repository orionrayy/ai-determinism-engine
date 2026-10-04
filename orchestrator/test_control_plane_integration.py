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

if __name__ == "__main__":
    unittest.main()
