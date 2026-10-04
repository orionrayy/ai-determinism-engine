from __future__ import annotations

import hashlib
import hmac
import os
import unittest
from unittest.mock import patch

from control_plane import ControlPlaneClient, ControlPlaneConfigurationError, canonical_json

class ControlPlaneClientTests(unittest.TestCase):
    def test_canonical_json(self):
        self.assertEqual(canonical_json({"b":2,"a":1}), b'{"a":2}' if False else b'{"a":1,"b":2}')

    def test_signature(self):
        c = ControlPlaneClient("https://control.example", "s", owner="w")
        body = canonical_json({"owner":"w"})
        self.assertEqual(
            c._signature("10","POST","/x",body),
            hmac.new(b"s", b"\n".join([b"10",b"POST",b"/x",body]), hashlib.sha256).hexdigest(),
        )

    def test_partial_configuration_fails(self):
        with patch.dict(os.environ, {"ORCHESTRATOR_CONTROL_PLANE_URL":"https://control.example"}, clear=False):
            os.environ.pop("ORCHESTRATOR_CONTROL_PLANE_SECRET", None)
            with self.assertRaises(ControlPlaneConfigurationError):
                ControlPlaneClient.from_env()

    def test_https_is_required(self):
        with self.assertRaises(ControlPlaneConfigurationError):
            ControlPlaneClient("http://control.example", "s")

    def test_resource_lock_path(self):
        client = ControlPlaneClient("https://control.example", "s", owner="w")
        with patch.object(
            client,
            "_request",
            return_value={
                "status": "acquired",
                "fence_epoch": 4,
                "expires_at": 12345,
            },
        ) as request:
            lease = client.acquire_resource("repo:file", workflow_id="wf")
        self.assertEqual(lease.resource_key, "repo:file")
        self.assertEqual(lease.fence_epoch, 4)
        self.assertIn(
            "/v1/resources/repo%3Afile/lease/acquire",
            request.call_args.args[1],
        )

    def test_recovery_alarm_paths(self):
        client = ControlPlaneClient("https://control.example", "s", owner="w")
        with patch.object(
            client,
            "_request",
            side_effect=[
                {"status": "armed", "due_at": 200, "event_id": "recovery:1"},
                {"status": "acknowledged", "workflow_id": "wf", "event_id": "recovery:1"},
            ],
        ) as request:
            armed = client.arm_recovery(
                "wf",
                owner="w",
                fence_epoch=7,
                due_at=200,
                event_id="recovery:1",
            )
            ack = client.ack_recovery("wf", "recovery:1")
        self.assertEqual(armed["status"], "armed")
        self.assertEqual(ack["status"], "acknowledged")
        self.assertEqual(request.call_args_list[0].args[1], "/v1/workflows/wf/recovery/arm")
        self.assertEqual(request.call_args_list[1].args[1], "/v1/workflows/wf/recovery/ack")

    def test_workflow_state_parses_recovery_envelope(self):
        client = ControlPlaneClient("https://control.example", "s", owner="w")
        with patch.object(
            client,
            "_request",
            return_value={
                "status": "stored",
                "state_version": 8,
                "updated_at": 100,
                "state": {"id": "wf", "status": "running"},
                "recovery": {
                    "due": True,
                    "event_id": "recovery:1",
                    "due_at": 200,
                },
            },
        ):
            state = client.get_workflow_state("wf")
        self.assertTrue(state.recovery_due)
        self.assertEqual(state.recovery_event_id, "recovery:1")
        self.assertEqual(state.recovery_due_at, 200)

    def test_hot_state_cas_and_outbox_paths(self):
        client = ControlPlaneClient("https://control.example", "s", owner="w")
        with patch.object(
            client,
            "_request",
            side_effect=[
                {"status": "stored", "state_version": 3},
                {"status": "appended", "sequence": 9},
            ],
        ) as request:
            version = client.put_workflow_state(
                "wf",
                owner="w",
                fence_epoch=2,
                expected_state_version=2,
                state={"id": "wf", "status": "running"},
            )
            sequence = client.append_outbox_event(
                "wf",
                owner="w",
                fence_epoch=2,
                event_type="node.completed",
                payload={"node_id": "n1"},
            )
        self.assertEqual(version, 3)
        self.assertEqual(sequence, 9)
        self.assertEqual(request.call_args_list[0].args[0], "PUT")
        self.assertEqual(request.call_args_list[1].args[0], "POST")


if __name__ == "__main__":
    unittest.main()
