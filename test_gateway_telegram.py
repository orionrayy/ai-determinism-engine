import unittest
from unittest.mock import patch

import gateway


class TelegramGatewayContractTests(unittest.TestCase):
    def test_approval_flag_is_bound_to_approval_operation(self):
        goal, metadata, _ = gateway.build_execution_event({
            'domain': 'orchestration',
            'operation': 'approve',
            'input': {'workflow_id': 'wf:1'},
            'event_id': 'event-1',
            'idempotency_key': 'event-1',
            'approve_high_risk': True,
        })
        self.assertEqual(goal, 'Execute orchestration operation orchestration.approve')
        self.assertIs(metadata['approve_high_risk'], True)
        self.assertEqual(metadata['workflow_id'], 'wf:1')

    def test_approval_flag_cannot_be_attached_to_other_operation(self):
        with self.assertRaisesRegex(ValueError, 'approve_high_risk_operation_invalid'):
            gateway.build_execution_event({
                'domain': 'telegram',
                'operation': 'prompt',
                'input': {'goal': 'hello'},
                'event_id': 'event-2',
                'idempotency_key': 'event-2',
                'approve_high_risk': True,
            })

    def test_approval_changes_intent_digest(self):
        _, normal, _ = gateway.build_execution_event({
            'domain': 'orchestration',
            'operation': 'approve',
            'input': {'workflow_id': 'wf:1'},
            'event_id': 'event-3',
            'idempotency_key': 'event-3',
        })
        _, approved, _ = gateway.build_execution_event({
            'domain': 'orchestration',
            'operation': 'approve',
            'input': {'workflow_id': 'wf:1'},
            'event_id': 'event-4',
            'idempotency_key': 'event-4',
            'approve_high_risk': True,
        })
        self.assertNotEqual(normal['intent_fingerprint'], approved['intent_fingerprint'])

    def test_github_dispatch_payload_propagates_approval_flag(self):
        captured = {}
        class Response:
            status = 204
            def __enter__(self): return self
            def __exit__(self, *args): return False

        def fake_urlopen(request, timeout=30):
            captured['body'] = request.data.decode('utf-8')
            return Response()

        metadata = {'approve_high_risk': True, 'workflow_id': 'wf:1'}
        with patch.dict('os.environ', {'GITHUB_GATEWAY_TOKEN': 'secret', 'GITHUB_REPOSITORY': 'repo/test'}), patch.object(gateway.urllib.request, 'urlopen', side_effect=fake_urlopen):
            gateway.github_dispatch('Execute orchestration operation orchestration.approve', metadata, event_id='event-5')
        self.assertIn('approve_high_risk', captured['body'])


if __name__ == '__main__':
    unittest.main()