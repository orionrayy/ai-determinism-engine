import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from checkpoint_integrity import (
    CheckpointIntegrityError,
    verify_checkpoint,
)


class CheckpointIntegrityTests(unittest.TestCase):
    def write_checkpoint(self, root: Path, workflow: str, node_id: str) -> tuple[Path, str]:
        path = root / ".orchestrator" / "checkpoints" / f"{workflow}-{node_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        content = {
            "workflow": workflow,
            "node": {
                "id": node_id,
                "status": "completed",
                "output": {"evidence": {"evidence_sha256": "abc"}},
            },
        }
        path.write_text(json.dumps(content), encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return path, digest

    def test_checkpoint_digest_and_bindings_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, digest = self.write_checkpoint(root, "wf", "n01")
            node = {
                "id": "n01",
                "status": "completed",
                "output": {
                    "checkpoint": {
                        "path": str(path),
                        "sha256": digest,
                    }
                },
            }
            result = verify_checkpoint(
                root,
                node,
                expected_workflow_id="wf",
                allowed_dir=root / ".orchestrator" / "checkpoints",
            )
            self.assertTrue(result["verified"])

    def test_checksum_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, _ = self.write_checkpoint(root, "wf", "n01")
            path.write_text("{}" , encoding="utf-8")
            node = {
                "id": "n01",
                "status": "completed",
                "output": {
                    "checkpoint": {
                        "path": str(path),
                        "sha256": "0" * 64,
                    }
                },
            }
            with self.assertRaises(CheckpointIntegrityError):
                verify_checkpoint(root, node, expected_workflow_id="wf", allowed_dir=path.parent)

    def test_checkpoint_binding_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, digest = self.write_checkpoint(root, "wf-other", "n01")
            node = {
                "id": "n01",
                "status": "completed",
                "output": {
                    "checkpoint": {
                        "path": str(path),
                        "sha256": digest,
                    }
                },
            }
            with self.assertRaises(CheckpointIntegrityError):
                verify_checkpoint(
                    root,
                    node,
                    expected_workflow_id="wf",
                    allowed_dir=path.parent,
                )

    def test_missing_checkpoint_digest_is_legacy_unverified(self):
        result = verify_checkpoint(
            Path("/tmp"),
            {
                "id": "n01",
                "status": "completed",
                "output": {},
            },
        )
        self.assertTrue(result["legacy"])
        self.assertFalse(result["verified"])


if __name__ == "__main__":
    unittest.main()
