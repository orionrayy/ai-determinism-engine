import io
import unittest

from http_safety import read_response_limited


class HttpSafetyTests(unittest.TestCase):
    def test_reads_within_limit(self):
        value = read_response_limited(io.BytesIO(b"hello"), 5)
        self.assertEqual(value, b"hello")

    def test_rejects_body_over_limit(self):
        with self.assertRaisesRegex(RuntimeError, "HTTP response exceeds configured safety limit"):
            read_response_limited(io.BytesIO(b"abcdef"), 5)

    def test_rejects_non_positive_limit(self):
        with self.assertRaises(ValueError):
            read_response_limited(io.BytesIO(b"x"), 0)


if __name__ == "__main__":
    unittest.main()
