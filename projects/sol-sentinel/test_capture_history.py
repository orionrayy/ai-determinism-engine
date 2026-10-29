import unittest

import capture_history


class CaptureValidationTests(unittest.TestCase):
    def test_capture_rejects_days_outside_bound_without_network(self):
        with self.assertRaisesRegex(ValueError, "days must be"):
            capture_history.capture("SOLUSDT", "1m", 0, 60)

    def test_capture_rejects_unsupported_interval_without_network(self):
        with self.assertRaisesRegex(ValueError, "Unsupported interval"):
            capture_history.capture("SOLUSDT", "2m", 3, 60)

    def test_capture_rejects_unbounded_page_count_without_network(self):
        with self.assertRaisesRegex(ValueError, "max-pages must be"):
            capture_history.capture("SOLUSDT", "1m", 3, 61)


if __name__ == "__main__":
    unittest.main(verbosity=2)
