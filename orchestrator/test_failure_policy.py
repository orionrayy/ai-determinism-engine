import unittest
import urllib.error

from failure_policy import (
    classify_failure,
    deterministic_retry_delay,
    retry_class_allowed,
)


class FailurePolicyTests(unittest.TestCase):
    def test_transient_and_intermittent_are_retryable(self):
        self.assertEqual(classify_failure(RuntimeError("transient")), "transient")
        self.assertEqual(classify_failure(TimeoutError("timed out")), "transient")
        self.assertEqual(classify_failure(RuntimeError("rate limit 429")), "intermittent")
        self.assertTrue(retry_class_allowed("transient"))
        self.assertTrue(retry_class_allowed("intermittent"))
        self.assertTrue(retry_class_allowed("dependency"))

    def test_contract_policy_semantic_and_unknown_are_not_retryable(self):
        self.assertEqual(classify_failure(ValueError("invalid payload")), "contract")
        self.assertEqual(classify_failure(RuntimeError("tool disabled by ORCHESTRATOR_FREE_ONLY=true")), "policy")
        self.assertEqual(classify_failure(RuntimeError("semantic validation failed")), "semantic")
        self.assertEqual(classify_failure(RuntimeError("unclassified terminal failure")), "permanent")
        for failure_class in ("contract", "policy", "semantic", "permanent", "uncertain"):
            self.assertFalse(retry_class_allowed(failure_class))

    def test_http_failures_have_typed_classes(self):
        timeout = urllib.error.HTTPError(
            "https://example.test",
            503,
            "service unavailable",
            {},
            None,
        )
        forbidden = urllib.error.HTTPError(
            "https://example.test",
            403,
            "forbidden",
            {},
            None,
        )
        self.assertEqual(classify_failure(timeout), "transient")
        self.assertEqual(classify_failure(forbidden), "policy")

    def test_retry_class_signal_never_overrides_non_retryable_classes(self):
        self.assertTrue(retry_class_allowed("transient"))
        self.assertFalse(retry_class_allowed("uncertain"))
        self.assertFalse(retry_class_allowed("semantic"))

    def test_deterministic_retry_delay_is_stable_and_bounded(self):
        a = deterministic_retry_delay("wf-1", "n-1", 1, retry_seed="seed-a")
        b = deterministic_retry_delay("wf-1", "n-1", 1, retry_seed="seed-a")
        c = deterministic_retry_delay("wf-1", "n-1", 2, retry_seed="seed-a")
        d = deterministic_retry_delay("wf-2", "n-1", 1, retry_seed="seed-b")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertNotEqual(a, d)
        self.assertGreaterEqual(a, 1.0)
        self.assertLessEqual(a, 2.0)
        self.assertGreaterEqual(c, 2.0)
        self.assertLessEqual(c, 4.0)


if __name__ == "__main__":
    unittest.main()
