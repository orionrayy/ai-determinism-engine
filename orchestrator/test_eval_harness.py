import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orchestrator.eval_harness import run_evaluations


class EvaluationHarnessTests(unittest.TestCase):
    def test_all_control_plane_evaluations_pass(self):
        result = run_evaluations()
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["passed_cases"], result["total_cases"])
        self.assertGreaterEqual(result["total_cases"], 6)


if __name__ == "__main__":
    unittest.main()
