import json
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import capture_history


class FakeResponse:
    def __init__(self, payload):
        self.status = 200
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, _limit):
        return self.payload


class CaptureValidationTests(unittest.TestCase):
    def test_fx_reference_retries_transient_timeout(self):
        payload = json.dumps({
            "base": "USD",
            "quote": "IDR",
            "rate": 18158.0,
            "date": datetime.now(timezone.utc).date().isoformat(),
        }).encode("utf-8")
        with patch.object(
            capture_history.urllib.request,
            "urlopen",
            side_effect=[TimeoutError("temporary timeout"), FakeResponse(payload)],
        ) as mocked_urlopen, patch.object(capture_history.time, "sleep") as mocked_sleep:
            result = capture_history.fetch_quote_to_idr_reference(attempts=3, pause_seconds=0.0)
        self.assertEqual(result["quote_to_idr_rate"], 18158.0)
        self.assertFalse(result["quote_to_idr_is_trading_quote"])
        self.assertEqual(mocked_urlopen.call_count, 2)
        mocked_sleep.assert_called_once_with(0.0)

    def test_fx_reference_fails_after_bounded_timeouts(self):
        with patch.object(
            capture_history.urllib.request,
            "urlopen",
            side_effect=TimeoutError("temporary timeout"),
        ) as mocked_urlopen, patch.object(capture_history.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "failed after 3 attempts"):
                capture_history.fetch_quote_to_idr_reference(attempts=3, pause_seconds=0.0)
        self.assertEqual(mocked_urlopen.call_count, 3)

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
