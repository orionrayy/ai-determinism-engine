import unittest
import urllib.error

from failure_policy import (
    classify_failure,
    decide_retry,
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

    def test_central_retry_decision_blocks_uncertain_non_idempotent_side_effect(self):
        decision = decide_retry(
            "transient",
            uncertain=True,
            side_effect_started=True,
            idempotent=False,
        )
        self.assertEqual(decision["failure_class"], "uncertain")
        self.assertFalse(decision["retry_allowed"])
        self.assertTrue(decision["uncertain"])

    def test_central_retry_decision_allows_uncertain_idempotent_side_effect(self):
        decision = decide_retry(
            "transient",
            uncertain=True,
            side_effect_started=True,
            idempotent=True,
        )
        self.assertTrue(decision["retry_allowed"])
        self.assertTrue(decision["uncertain"])

    def test_retry_jitter_is_stable_per_seed_and_spreads_nodes(self):
        a = deterministic_retry_delay("wf-1", "n-1", 1, jitter_seed="seed-a")
        b = deterministic_retry_delay("wf-1", "n-1", 1, jitter_seed="seed-a")
        c = deterministic_retry_delay("wf-1", "n-1", 1, jitter_seed="seed-b")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertGreaterEqual(a, 2.0)
        self.assertLessEqual(a, 3.0)
        self.assertLessEqual(
            deterministic_retry_delay("wf-1", "n-1", 2, jitter_seed="seed-a"),
            6.0,
        )


if __name__ == "__main__":
    unittest.main()
