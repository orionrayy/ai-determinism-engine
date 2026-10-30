from __future__ import annotations

import unittest
from pathlib import Path


class ForwardPaperWorkflowContractTests(unittest.TestCase):
    def setUp(self):
        self.workflow = (
            Path(__file__).resolve().parents[2]
            / ".github"
            / "workflows"
            / "sol-sentinel-forward-paper.yml"
        ).read_text(encoding="utf-8")

    def test_durable_status_is_generated_before_artifact_upload(self):
        capture = self.workflow.index("- name: Capture durable operational status")
        upload = self.workflow.index("- name: Upload forward-paper evidence")
        summary = self.workflow.index("- name: Publish operational summary")
        self.assertLess(
            capture,
            upload,
            "status.json must exist before actions/upload-artifact snapshots forward-output/",
        )
        self.assertLess(upload, summary)

    def test_artifact_upload_includes_forward_output_directory(self):
        upload = self.workflow.index("- name: Upload forward-paper evidence")
        summary = self.workflow.index("- name: Publish operational summary")
        block = self.workflow[upload:summary]
        self.assertIn("uses: actions/upload-artifact@v4", block)
        self.assertIn("path: forward-output/", block)
        self.assertIn("retention-days: 30", block)


if __name__ == "__main__":
    unittest.main()
