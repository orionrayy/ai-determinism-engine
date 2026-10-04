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
        import hashlib

        self.assertEqual(
            digest({"a": 1}),
            hashlib.sha256(b'{"a":1}').hexdigest(),
        )

if __name__ == "__main__":
    unittest.main()
