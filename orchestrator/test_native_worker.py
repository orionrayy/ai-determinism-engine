import hashlib
import unittest

import native_worker as nw


class NativeWorkerTests(unittest.TestCase):
    def test_execution_id_is_deterministic(self):
        expected = hashlib.sha256(b"wf1:n1").hexdigest()
        self.assertEqual(nw.execution_id("wf1", "n1"), expected)

    def test_create_task_requires_public_safe(self):
        with self.assertRaises(ValueError):
            nw.create_task(
                "wf1", "n1", "notion", "create_page", "goal",
                {"public_safe": False}, "https://bridge.example/native-result",
                2000000000,
            )

    def test_create_task_returns_one_time_token_and_envelope(self):
        task, token = nw.create_task(
            "wf1", "n1", "notion", "create_page", "goal",
            {"public_safe": True, "payload": {"title": "Hello"}},
            "https://bridge.example/native-result",
            2000000000,
        )
        self.assertEqual(task["protocol"], nw.PROTOCOL)
        self.assertEqual(task["execution_id"], nw.execution_id("wf1", "n1"))
        self.assertEqual(task["connector"], "notion")
        self.assertEqual(task["action"], "create_page")
        self.assertTrue(token)
        self.assertNotIn(token, task["token_hash"])

    def test_token_hash_verification(self):
        token = "secret-token"
        digest = nw.hash_token(token)
        self.assertTrue(nw.verify_token(token, digest))
        self.assertFalse(nw.verify_token("wrong", digest))


if __name__ == "__main__":
    unittest.main()
