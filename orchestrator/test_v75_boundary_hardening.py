import os
import unittest
from unittest.mock import patch

import capability_graph as cg
import epistemic_deliberation as ed
import epistemic_validation as ev
import evidence_records as er
import orchestrator as o
import source_authority as sa


class V75BoundaryHardeningTests(unittest.TestCase):
    def test_control_plane_configuration_requires_both_coordinates(self):
        with patch.dict(
            o.os.environ,
            {"ORCHESTRATOR_CONTROL_PLANE_URL": "https://cp.example", "ORCHESTRATOR_CONTROL_PLANE_SECRET": ""},
            clear=True,
        ):
            self.assertFalse(o.control_plane_configured())
        with patch.dict(
            o.os.environ,
            {"ORCHESTRATOR_CONTROL_PLANE_URL": "", "ORCHESTRATOR_CONTROL_PLANE_SECRET": "secret"},
            clear=True,
        ):
            self.assertFalse(o.control_plane_configured())

    def test_git_authority_does_not_open_control_plane_session(self):
        workflow = {"id": "wf-git", "live": True, "authority_mode": "git_durable"}
        with patch.object(o.ControlPlaneClient, "from_env", side_effect=AssertionError("unexpected control plane")),              patch.object(o, "persist_workflow"):
            with patch.dict(o.os.environ, {}, clear=True):
                with o.control_plane_session(workflow) as context:
                    self.assertEqual(context, (None, None))

    def test_distributed_authority_fails_closed_when_control_plane_missing(self):
        workflow = {"id": "wf-dist", "live": True, "authority_mode": "distributed_control_plane"}
        with patch.object(o, "persist_workflow"), patch.object(o, "append_event"):
            with patch.dict(o.os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "distributed control-plane authority"):
                    with o.control_plane_session(workflow):
                        pass

    def test_create_workflow_persists_immutable_authority_mode(self):
        with patch.dict(o.os.environ, {}, clear=True),              patch.object(o, "load_registry", return_value={}),              patch.object(o, "tool_available", return_value=False):
            git_workflow = o.create_workflow("do work", live=True)
        self.assertEqual(git_workflow["authority_mode"], "git_durable")

        with patch.dict(
            o.os.environ,
            {
                "ORCHESTRATOR_CONTROL_PLANE_URL": "https://cp.example",
                "ORCHESTRATOR_CONTROL_PLANE_SECRET": "secret",
            },
            clear=True,
        ), patch.object(o, "load_registry", return_value={}),              patch.object(o, "tool_available", return_value=False):
            distributed_workflow = o.create_workflow("do work", live=True)
        self.assertEqual(
            distributed_workflow["authority_mode"],
            "distributed_control_plane",
        )

    def test_preferred_capability_tool_is_a_real_preference(self):
        registry = {
            "capability:analyze": {
                "default_tool": "free",
                "fallback_tools": ["free_keyed"],
            },
            "free": {"free_tier": True, "required_env": None, "risk": "low"},
            "free_keyed": {"free_tier": True, "required_env": "FREE_KEY", "risk": "low"},
        }
        with patch.dict(
            os.environ,
            {"ORCHESTRATOR_FREE_ONLY": "true", "FREE_KEY": "x"},
            clear=True,
        ):
            self.assertEqual(
                cg.route_capability(
                    "analyze",
                    registry,
                    live=True,
                    preferred="free_keyed",
                ),
                "free_keyed",
            )

    def test_provider_reliability_is_recorded_and_used(self):
        health = {}
        cg.record_tool_result(health, "free", success=True, now=10)
        cg.record_tool_result(health, "free", success=False, now=20)
        self.assertEqual(health["free"]["success_count"], 1)
        self.assertEqual(health["free"]["failure_count"], 1)
        self.assertEqual(health["free"]["reliability_score"], 0.5)

    def test_llm_evidence_is_bound_to_trusted_records_and_cannot_forge_authority(self):
        output = {
            "confidence": 0.95,
            "claims": [{
                "claim_id": "c1",
                "statement": "supported",
                "material": True,
                "status": "SUPPORTED_DIRECT",
                "evidence_refs": ["source:a"],
            }],
            "evidence_records": [{
                "canonical_id": "source:a",
                "authority_score": 0.99,
                "authority_class": "official_primary",
            }],
        }
        trusted = [{
            "canonical_id": "source:a",
            "authority_score": 0.52,
            "authority_class": "index",
        }]
        result = ev.validate_epistemic_output(
            output,
            trusted_evidence_records=trusted,
        )
        self.assertTrue(result["passed"] is False)
        self.assertEqual(
            result["bound_evidence_records"][0]["authority_score"],
            0.52,
        )
        self.assertIn("source_authority", result["selective"].get("failed_checks", []))

    def test_llm_cannot_cross_evidence_trust_boundary(self):
        output = {
            "confidence": 0.9,
            "claims": [{
                "claim_id": "c1",
                "statement": "invented",
                "material": True,
                "status": "SUPPORTED_DIRECT",
                "evidence_refs": ["fake"],
            }],
            "evidence_records": [{
                "canonical_id": "fake",
                "authority_score": 1.0,
            }],
        }
        result = ev.validate_epistemic_output(
            output,
            trusted_evidence_records=[{"canonical_id": "real"}],
        )
        self.assertFalse(result["passed"])
        self.assertEqual(
            result["reason"],
            "LLM attempted to cite evidence outside trusted dependency records",
        )

    def test_selective_gate_uses_canonical_coverage_and_blocks_high_confidence_mismatch(self):
        output = {
            "confidence": 0.95,
            "claims": [
                {
                    "claim_id": "c1",
                    "statement": "supported",
                    "material": True,
                    "status": "SUPPORTED_DIRECT",
                    "evidence_refs": ["a"],
                },
                {
                    "claim_id": "c2",
                    "statement": "contested",
                    "material": True,
                    "status": "CONTESTED",
                    "evidence_refs": ["b"],
                },
            ],
            "evidence_records": [
                {"canonical_id": "a"},
                {"canonical_id": "b"},
            ],
        }
        result = ev.validate_epistemic_output(output)
        self.assertFalse(result["passed"])
        self.assertTrue(result["selective"]["abstain"])
        self.assertIn("supported_coverage", result["selective"]["failed_checks"])

    def test_selective_abstention_is_not_telemetry_only(self):
        output = {
            "confidence": 0.95,
            "claims": [
                {
                    "claim_id": "c1",
                    "statement": "supported",
                    "material": True,
                    "status": "SUPPORTED_DIRECT",
                    "evidence_refs": ["a"],
                },
                {
                    "claim_id": "c2",
                    "statement": "unsupported",
                    "material": True,
                    "status": "UNSUPPORTED",
                    "evidence_refs": [],
                },
            ],
            "evidence_records": [{"canonical_id": "a"}],
        }
        result = ev.validate_epistemic_output(output)
        self.assertFalse(result["passed"])
        self.assertTrue(result["selective_abstention"])

    def test_source_authority_is_explainable_and_dedupe_keeps_strongest_profile(self):
        peer = er.normalize_source(
            "semantic_scholar",
            {
                "title": "Paper",
                "year": 2026,
                "provider": "semantic_scholar",
                "source_type": "journal-article",
                "peer_reviewed": True,
                "paperId": "p1",
            },
        )
        preprint = er.normalize_source(
            "arxiv",
            {
                "title": "Paper",
                "year": 2026,
                "provider": "arxiv",
                "arxiv_id": "2601.12345",
            },
        )
        self.assertEqual(peer["authority_class"], "peer_reviewed")
        self.assertGreater(peer["authority_score"], preprint["authority_score"])
        merged = er.deduplicate_sources([
            dict(peer, canonical_id="doi:10.1/x"),
            dict(preprint, canonical_id="doi:10.1/x"),
        ])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["authority_score"], peer["authority_score"])
        self.assertEqual(
            sa.source_authority("semantic_scholar", {"peer_reviewed": True})["authority_tier"],
            "tier1",
        )

    def test_debate_ignores_self_reported_independence_count(self):
        proposals = [
            {
                "agent_id": "a1",
                "answer": "A",
                "confidence": 0.9,
                "evidence_refs": ["x"],
                "independent_source_count": 99,
            },
            {
                "agent_id": "a2",
                "answer": "A",
                "confidence": 0.9,
                "evidence_refs": ["x"],
                "independent_source_count": 99,
            },
        ]
        result = ed.debate_decision(proposals)
        self.assertTrue(result["required"])
        self.assertEqual(result["reason"], "insufficient_evidence")

    def test_debate_accepts_computed_independence(self):
        proposals = [
            {
                "agent_id": "a1",
                "answer": "A",
                "confidence": 0.9,
                "evidence_refs": ["x"],
                "independence": {"distinct_work_count": 2},
            },
            {
                "agent_id": "a2",
                "answer": "A",
                "confidence": 0.9,
                "evidence_refs": ["y"],
                "independence": {"distinct_work_count": 2},
            },
        ]
        result = ed.debate_decision(proposals)
        self.assertFalse(result["required"])
        self.assertEqual(result["reason"], "stable_consensus")

    def test_planner_has_no_deprecated_generation_parameters(self):
        import llm_planner as lp

        registry = {
            "gemini": {
                "free_tier": True,
                "default_model": "gemini-3.8-flash",
                "free_models": ["gemini-3.8-flash"],
            },
            "capability:analyze": {
                "default_tool": "gemini",
                "fallback_tools": [],
            },
        }
        response = {
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": '{"nodes":[{"id":"n1","capability":"analyze","tool":"gemini","depends_on":[],"risk":"low","instruction":"analyze","contract":{},"artifacts":[]}]}'
                    }]
                }
            }]
        }
        with patch.dict(
            o.os.environ,
            {"ORCHESTRATOR_FREE_ONLY": "true", "GEMINI_API_KEY": "x"},
            clear=True,
        ), patch.object(lp, "_post", return_value=response) as post:
            lp.plan_goal("analyze", registry, o.Node, o.validate_dag)
        generation = post.call_args.args[1]["generationConfig"]
        self.assertNotIn("temperature", generation)
        self.assertNotIn("candidateCount", generation)
        self.assertEqual(
            generation["thinkingConfig"]["thinkingLevel"],
            "medium",
        )

    def test_gemini_uses_thinking_config_and_authoritative_packed_context(self):
        context = {
            "dependencies": {
                f"d{i}": {"output": {"text": "x" * 9000}}
                for i in range(1, 6)
            }
        }
        node = o.Node(
            "n1",
            "analyze",
            "gemini",
            [],
            input={"workflow_id": "wf", "instruction": "analyze", "context": context},
        )
        captured = {}
        registry = {
            "gemini": {
                "free_tier": True,
                "required_env": "GEMINI_API_KEY",
                "default_model": "gemini-3.8-flash",
                "free_models": ["gemini-3.8-flash"],
            }
        }
        def fake_http(url, **kwargs):
            captured["payload"] = kwargs["body"]
            return {"candidates": [{"content": {"parts": [{"text": '{"result":"ok"}'}]}}]}

        with patch.dict(
            o.os.environ,
            {
                "ORCHESTRATOR_FREE_ONLY": "true",
                "GEMINI_API_KEY": "x",
            },
            clear=True,
        ), patch.object(o, "load_registry", return_value=registry),              patch.object(o, "http_json", side_effect=fake_http):
            o.execute_gemini(node, "analyze")

        generation = captured["payload"]["generationConfig"]
        self.assertNotIn("temperature", generation)
        self.assertNotIn("candidateCount", generation)
        self.assertEqual(
            generation["thinkingConfig"]["thinkingLevel"],
            "medium",
        )
        context_text = captured["payload"]["contents"][0]["parts"][0]["text"]
        self.assertIn('"text":', context_text)
        self.assertGreater(len(context_text.encode("utf-8")), 24 * 1024)
        self.assertIn('"d5"', context_text)


if __name__ == "__main__":
    unittest.main()
