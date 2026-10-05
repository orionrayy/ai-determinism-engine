import json
import os
import unittest
from unittest.mock import patch

import gateway


class Response:
    def __init__(self, status=200, payload=None):
        self.status = status
        self._payload = payload
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def read(self):
        return json.dumps(self._payload or {}).encode('utf-8')


class TelegramGatewayContractTests(unittest.TestCase):
    def test_approval_payload_is_node_scoped(self):
        goal, metadata, _ = gateway.build_execution_event({
            'domain': 'orchestration',
            'operation': 'approve',
            'input': {'workflow_id': 'wf:1', 'node_id': 'n07-deploy'},
            'event_id': 'event-1',
            'idempotency_key': 'event-1',
        })
        self.assertEqual(goal, 'Execute orchestration operation orchestration.approve')
        self.assertEqual(metadata['workflow_id'], 'wf:1')
        self.assertEqual(metadata['domain'], 'orchestration')
        self.assertEqual(metadata['operation'], 'approve')

    def test_approval_does_not_change_generic_intent_contract(self):
        _, first, _ = gateway.build_execution_event({
            'domain': 'orchestration',
            'operation': 'approve',
            'input': {'workflow_id': 'wf:1', 'node_id': 'n07-deploy'},
            'event_id': 'event-a',
            'idempotency_key': 'event-a',
        })
        _, second, _ = gateway.build_execution_event({
            'domain': 'orchestration',
            'operation': 'approve',
            'input': {'workflow_id': 'wf:1', 'node_id': 'n07-deploy'},
            'event_id': 'event-b',
            'idempotency_key': 'event-b',
        })
        self.assertEqual(first['intent_fingerprint'], second['intent_fingerprint'])

    def test_github_dispatch_returns_workflow_identity_after_204(self):
        captured = {}
        class DispatchResponse:
            status = 204
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
        def fake_urlopen(request, timeout=30):
            captured["body"] = request.data.decode("utf-8")
            return DispatchResponse()
        metadata = {"workflow_id": "wf:telegram", "execution_id": "e" * 64}
        with patch.dict("os.environ", {"GITHUB_GATEWAY_TOKEN": "secret", "GITHUB_REPOSITORY": "repo/test"}), patch.object(gateway.urllib.request, "urlopen", side_effect=fake_urlopen):
            result = gateway.github_dispatch("goal", metadata, event_id="event-1")
        self.assertEqual(result["github_status"], 204)
        self.assertEqual(result["workflow_id"], "wf:telegram")
        self.assertEqual(result["execution_id"], "e" * 64)
        self.assertIn("wf:telegram", captured["body"])

    def test_github_approval_finds_exact_open_issue_and_applies_idempotent_label(self):
        calls = []
        responses = iter([
            Response(200, {'items': [{'number': 42, 'title': '[ORCHESTRATOR APPROVAL] wf:1 / n07-deploy'}]}),
            Response(200, {'labels': [{'name': 'orchestrator-approved'}]}),
        ])
        def fake_urlopen(request, timeout=30):
            calls.append((request.get_method(), request.full_url, request.data.decode('utf-8') if request.data else ''))
            return next(responses)
        metadata = {'workflow_id': 'wf:1', 'node_id': 'n07-deploy'}
        with patch.dict(os.environ, {'GITHUB_GATEWAY_TOKEN': 'secret', 'GITHUB_REPOSITORY': 'repo/test'}), patch.object(gateway.urllib.request, 'urlopen', side_effect=fake_urlopen):
            result = gateway.github_approve_workflow_node(metadata)
        self.assertEqual(result['approval_issue'], 42)
        self.assertTrue(result['approved'])
        self.assertEqual(calls[0][0], 'GET')
        self.assertTrue(calls[0][1].startswith('https://api.github.com/search/issues?'))
        self.assertIn('/repos/repo/test/issues/42/labels', calls[1][1])
        self.assertEqual(calls[1][0], 'POST')
        self.assertIn('orchestrator-approved', calls[1][2])

    def test_github_approval_fails_closed_on_zero_or_duplicate_matches(self):
        for items, expected in [
            ([], 'approval_issue_not_found'),
            ([{'number': 1, 'title': '[ORCHESTRATOR APPROVAL] wf:1 / n07-deploy'}, {'number': 2, 'title': '[ORCHESTRATOR APPROVAL] wf:1 / n07-deploy'}], 'multiple_open_approval_issues'),
        ]:
            def fake_urlopen(request, timeout=30):
                return Response(200, {'items': items})
            with self.subTest(expected=expected), patch.dict(os.environ, {'GITHUB_GATEWAY_TOKEN': 'secret', 'GITHUB_REPOSITORY': 'repo/test'}), patch.object(gateway.urllib.request, 'urlopen', side_effect=fake_urlopen):
                with self.assertRaisesRegex(RuntimeError, expected):
                    gateway.github_approve_workflow_node({'workflow_id': 'wf:1', 'node_id': 'n07-deploy'})


if __name__ == '__main__':
    unittest.main()