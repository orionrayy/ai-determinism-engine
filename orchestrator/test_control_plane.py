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

if __name__ == "__main__":
    unittest.main()
