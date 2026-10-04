from __future__ import annotations

import unittest

from deterministic_codec import canonical_json, digest


class DeterministicCodecTests(unittest.TestCase):
    def test_canonical_json_is_stable(self):
        self.assertEqual(
            canonical_json({"b": 2, "a": 1}),
            b'{"a":1,"b":2}',
        )

    def test_digest_matches_canonical_json(self):
        self.assertEqual(
            digest({"a": 1}),
            "015abd7f5cc5ae80a256dc1b4a8d4f7f1f6d9f4f2d6b0f9f7d4b4d6e0a7d8e2"
            if False else __import__("hashlib").sha256(b'{"a":1}').hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
