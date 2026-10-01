import unittest
import urllib.error

from failure_policy import (
    classify_failure,
    deterministic_retry_delay,
    retry_allowed,
)


class FailurePolicyTests(unittest.TestCase):
    def test_transient_and_intermittent_are_retryable(self):
        self.assertEqual(classify_failure(RuntimeError("transient")), "transient")
        self.assertEqual(classify_failure(TimeoutError("timed out")), "transient")
        self.assertEqual(classify_failure(RuntimeError("rate limit 429")), "intermittent")
        self.assertTrue(retry_allowed("transient"))
        self.assertTrue(retry_allowed("intermittent"))
        self.assertTrue(retry_allowed("dependency"))

    def test_contract_policy_semantic_and_unknown_are_not_retryable(self):
        self.assertEqual(classify_failure(ValueError("invalid payload")), "contract")
        self.assertEqual(classify_failure(RuntimeError("tool disabled by ORCHESTRATOR_FREE_ONLY=true")), "policy")
        self.assertEqual(classify_failure(RuntimeError("semantic validation failed")), "semantic")
        self.assertEqual(classify_failure(RuntimeError("unclassified terminal failure")), "permanent")
        for failure_class in ("contract", "policy", "semantic", "permanent", "uncertain"):
            self.assertFalse(retry_allowed(failure_class))

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

    def test_explicit_retry_override_is_honored_only_for_retryable_classes(self):
        self.assertTrue(retry_allowed("transient", explicitly_retryable=True))
        self.assertFalse(retry_allowed("transient", explicitly_retryable=False))
        self.assertFalse(retry_allowed("uncertain", explicitly_retryable=True))

    def test_deterministic_retry_delay_is_stable_and_bounded(self):
        a = deterministic_retry_delay("wf-1", "n-1", 1)
        b = deterministic_retry_delay("wf-1", "n-1", 1)
        c = deterministic_retry_delay("wf-1", "n-1", 2)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertGreaterEqual(a, 2.0)
        self.assertLessEqual(a, 2.250)
        self.assertLessEqual(c, 4.250)


if __name__ == "__main__":
    unittest.main()
