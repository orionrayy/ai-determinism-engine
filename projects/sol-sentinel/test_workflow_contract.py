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

    def test_persistence_is_gated_by_successful_sqlite_integrity_check(self):
        checkpoint = self.workflow.index("- name: Checkpoint and verify SQLite state")
        persist = self.workflow.index("- name: Persist paper state")
        fail = self.workflow.index("- name: Fail when operation was unsuccessful")
        checkpoint_block = self.workflow[checkpoint:persist]
        persist_block = self.workflow[persist:fail]
        self.assertIn("id: state_integrity", checkpoint_block)
        self.assertIn("id: persist_state", persist_block)
        self.assertIn(
            "if: always() && steps.state_integrity.outcome == 'success'",
            persist_block,
        )

    def test_failed_integrity_uses_safe_status_and_summary_flags_it(self):
        capture = self.workflow.index("- name: Capture durable operational status")
        upload = self.workflow.index("- name: Upload forward-paper evidence")
        summary = self.workflow.index("- name: Publish operational summary")
        capture_block = self.workflow[capture:upload]
        summary_block = self.workflow[summary:]
        self.assertIn("STATE_INTEGRITY: ${{ steps.state_integrity.outcome }}", capture_block)
        self.assertIn('if [ "$STATE_INTEGRITY" = "success" ]', capture_block)
        self.assertIn('"consistent": False', capture_block)
        self.assertIn("STATE_PERSISTENCE: ${{ steps.persist_state.outcome }}", summary_block)
        self.assertIn("STATUS_CAPTURE: ${{ steps.capture_status.outcome }}", summary_block)
        self.assertIn('os.environ.get("STATE_INTEGRITY") != "success"', summary_block)
        self.assertIn('os.environ.get("STATE_PERSISTENCE") != "success"', summary_block)

    def test_artifact_upload_includes_forward_output_directory(self):
        upload = self.workflow.index("- name: Upload forward-paper evidence")
        summary = self.workflow.index("- name: Publish operational summary")
        block = self.workflow[upload:summary]
        self.assertIn("uses: actions/upload-artifact@v4", block)
        self.assertIn("path: forward-output/", block)
        self.assertIn("retention-days: 30", block)


if __name__ == "__main__":
    unittest.main()
