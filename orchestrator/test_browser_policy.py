import unittest

from orchestrator.browser_policy import (
    BrowserPolicyError,
    BrowserTask,
    browser_task_digest,
    build_browser_evidence,
    evaluate_browser_task,
    target_origin,
    validate_target_url,
)


def make_task(**overrides):
    payload = {
        "workflow_id": "wf-2026-001",
        "node_id": "browser-read",
        "action": "browser_navigate",
        "effect": "read_only",
        "target_url": "https://example.com/path?access_token=do-not-log#fragment",
        "input": {},
    }
    payload.update(overrides)
    return BrowserTask.from_mapping(payload)


class BrowserPolicyTests(unittest.TestCase):
    def test_normalizes_official_tool_names(self):
        self.assertEqual(make_task(action="browser_snapshot").action, "snapshot")

    def test_navigate_requires_target(self):
        with self.assertRaises(BrowserPolicyError):
            make_task(target_url=None)

    def test_rejects_unknown_action(self):
        with self.assertRaises(BrowserPolicyError):
            make_task(action="browser_evaluate")

    def test_rejects_unknown_fields(self):
        with self.assertRaises(BrowserPolicyError):
            make_task(unreviewed_option=True)

    def test_rejects_embedded_url_credentials(self):
        with self.assertRaises(BrowserPolicyError):
            validate_target_url("https://user:password@example.com/")

    def test_rejects_non_loopback_plain_http(self):
        with self.assertRaises(BrowserPolicyError):
            validate_target_url("http://example.com")

    def test_allows_localhost_http_for_dev(self):
        self.assertEqual(
            validate_target_url("http://localhost:3000/health"),
            "http://localhost:3000/health",
        )

    def test_allows_loopback_ip_http(self):
        self.assertEqual(
            validate_target_url("http://127.0.0.1:8080/"),
            "http://127.0.0.1:8080/",
        )

    def test_rejects_unsupported_url_scheme(self):
        with self.assertRaises(BrowserPolicyError):
            validate_target_url("file:///etc/passwd")

    def test_rejects_whitespace_url(self):
        with self.assertRaises(BrowserPolicyError):
            validate_target_url("https://example.com/a b")

    def test_rejects_non_object_input(self):
        with self.assertRaises(BrowserPolicyError):
            make_task(input=["not", "an", "object"])

    def test_rejects_oversized_input(self):
        with self.assertRaises(BrowserPolicyError):
            make_task(input={"payload": "x" * (64 * 1024)})

    def test_read_only_task_passes_without_approval(self):
        decision = evaluate_browser_task(make_task())
        self.assertTrue(decision["allowed"])
        self.assertFalse(decision["approval_required"])

    def test_unknown_effect_fails_closed(self):
        decision = evaluate_browser_task(make_task(effect="unknown"))
        self.assertFalse(decision["allowed"])
        self.assertTrue(decision["approval_required"])

    def test_high_risk_requires_exact_semantic_digest(self):
        item = make_task(action="browser_click", effect="external_write")
        digest = browser_task_digest(item)
        self.assertFalse(evaluate_browser_task(item)["allowed"])
        self.assertTrue(
            evaluate_browser_task(item, approved_intent_sha256=digest)["allowed"]
        )
        changed = make_task(
            action="browser_click",
            effect="external_write",
            input={"button": "different"},
        )
        self.assertFalse(
            evaluate_browser_task(changed, approved_intent_sha256=digest)["allowed"]
        )

    def test_evidence_redacts_path_query_and_fragment(self):
        item = make_task()
        evidence = build_browser_evidence(
            item,
            observed_at="2026-11-20T10:30:00+07:00",
            outcome="succeeded",
        )
        rendered = str(evidence)
        self.assertEqual(evidence["target_origin"], "https://example.com")
        self.assertNotIn("access_token", rendered)
        self.assertNotIn("/path", rendered)
        self.assertNotIn("fragment", rendered)

    def test_evidence_accepts_hashes_and_is_repeatable(self):
        item = make_task()
        digest = "a" * 64
        first = build_browser_evidence(
            item,
            observed_at="2026-11-20T10:30:00Z",
            outcome="succeeded",
            screenshot_sha256=digest,
        )
        second = build_browser_evidence(
            item,
            observed_at="2026-11-20T10:30:00Z",
            outcome="succeeded",
            screenshot_sha256=digest,
        )
        self.assertEqual(first, second)

    def test_invalid_timestamp_without_timezone_rejected(self):
        with self.assertRaises(BrowserPolicyError):
            build_browser_evidence(
                make_task(), observed_at="2026-11-20T10:30:00",
                outcome="succeeded",
            )

    def test_invalid_evidence_hash_rejected(self):
        with self.assertRaises(BrowserPolicyError):
            build_browser_evidence(
                make_task(), observed_at="2026-11-20T10:30:00Z",
                outcome="succeeded", output_sha256="not-a-sha",
            )

    def test_target_origin_handles_ipv6_and_ports(self):
        self.assertEqual(
            target_origin("https://[::1]:8443/a?x=1"),
            "https://[::1]:8443",
        )


if __name__ == "__main__":
    unittest.main()
