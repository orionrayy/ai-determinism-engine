"""Contract tests for the active Serevex project identity and paths."""
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]


class ProjectIdentityContractTests(unittest.TestCase):
    def test_project_uses_new_canonical_directory(self):
        self.assertTrue(PROJECT_ROOT.is_dir())
        self.assertEqual(PROJECT_ROOT.name, "serevex-quant")
        self.assertFalse((REPO_ROOT / "projects" / ("aur" + "um-intelligence")).exists())

    def test_benchmark_workflow_uses_new_canonical_path(self):
        workflow = REPO_ROOT / ".github" / "workflows" / "serevex-research-benchmark.yml"
        self.assertTrue(workflow.is_file())
        self.assertFalse((workflow.parent / ("aur" + "um-online-dataset-benchmark.yml")).exists())

    def test_active_labels_and_code_do_not_use_retired_brand(self):
        retired = "aur" + "um"
        candidates = [
            PROJECT_ROOT / "README.md",
            PROJECT_ROOT / "IMPLEMENTATION_STATUS.json",
            PROJECT_ROOT / "ROADMAP.md",
            PROJECT_ROOT / "online-benchmark" / "ci_benchmark.py",
            PROJECT_ROOT / "online-benchmark" / "tick_bidask_benchmark_v1_0.py",
            PROJECT_ROOT / "online-benchmark" / "research_readiness_gate_v1.py",
            REPO_ROOT / ".github" / "workflows" / "serevex-research-benchmark.yml",
        ]
        for path in candidates:
            with self.subTest(path=str(path)):
                self.assertTrue(path.is_file(), f"missing_active_file:{path}")
                self.assertNotIn(retired, path.read_text(encoding="utf-8").lower())


if __name__ == "__main__":
    unittest.main()
