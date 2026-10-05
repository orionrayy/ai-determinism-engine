from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


class ControlPlaneWorkerContractTests(unittest.TestCase):
    def test_worker_contains_identity_and_replay_guards(self):
        source = (ROOT / "control-plane" / "cloudflare" / "src" / "worker.js").read_text(
            encoding="utf-8"
        )
        for marker in (
            "X-Control-Plane-Request-ID",
            "request_dedupe",
            "request_replay",
            "workflow_identity_mismatch",
            "resource_identity_mismatch",
            "E.encode(stateJson).byteLength",
            "E.encode(raw).byteLength",
            "recovery_claim_invalid",
        ):
            self.assertIn(marker, source)

    def test_worker_signs_request_identity(self):
        source = (ROOT / "control-plane" / "cloudflare" / "src" / "worker.js").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            '[ts, request.method.toUpperCase(), path, String(requestId || "")]',
            source,
        )


if __name__ == "__main__":
    unittest.main()
