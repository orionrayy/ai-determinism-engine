import unittest

from eval_harness import run_evaluations


class EvaluationHarnessTests(unittest.TestCase):
    def test_all_control_plane_evaluations_pass(self):
        result = run_evaluations()
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["passed_cases"], result["total_cases"])
        self.assertGreaterEqual(result["total_cases"], 6)


if __name__ == "__main__":
    unittest.main()
