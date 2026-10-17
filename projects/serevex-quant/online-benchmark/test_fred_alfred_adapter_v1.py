import json
import os
import unittest
from datetime import date, datetime, timezone
from unittest.mock import patch

import fred_alfred_adapter_v1 as fred
from external_context_schema_v1 import SourceManifest, manifest_sha256


def manifest(raw=b"fixture"):
    return SourceManifest("fred-alfred","CPI","https://api.stlouisfed.org/fred/series/observations",
        datetime(2026,3,1,tzinfo=timezone.utc),manifest_sha256(raw),len(raw),0,0,
        "FRED observations","official API;license-review-required","official-host")


def payload(rows=None):
    return json.dumps({"count":len(rows or [{"date":"2025-12-01","value":"100.0",
        "realtime_start":"2026-01-15","realtime_end":"9999-12-31"}]),
        "observations": rows or [{"date":"2025-12-01","value":"100.0",
        "realtime_start":"2026-01-15","realtime_end":"9999-12-31"}]}).encode()


class FredAlfredTests(unittest.TestCase):
    def test_missing_key_returns_typed_skip(self):
        with patch.dict(os.environ,{},clear=True):
            result=fred.acquire_fred({"series_id":"CPIAUCSL","observation_start":"2020-01-01",
                "observation_end":"2025-12-31"})
        self.assertEqual(result.status,"SKIPPED_CREDENTIAL_REQUIRED")

    def test_key_never_appears_in_report_or_error(self):
        with patch.dict(os.environ,{"FRED_API_KEY":"test-secret-do-not-leak"}):
            with patch("fred_alfred_adapter_v1.urllib.request.urlopen",
                       side_effect=OSError("test-secret-do-not-leak")) as mocked:
                result=fred.acquire_fred({"series_id":"NOT_REAL","observation_start":"2020-01-01","observation_end":"2025-12-31"})
                mocked.assert_called_once()
        self.assertNotIn("test-secret-do-not-leak",repr(result))
        self.assertNotIn("test-secret-do-not-leak",result.error_code or "")

    def test_latest_revision_does_not_rewrite_older_vintage(self):
        data=payload([
            {"date":"2025-12-01","value":"100.0","realtime_start":"2026-01-15","realtime_end":"2026-02-04"},
            {"date":"2025-12-01","value":"101.0","realtime_start":"2026-02-05","realtime_end":"9999-12-31"}])
        rows=fred.parse_fred_observations(data,manifest(data))
        asof_jan=fred.observation_as_of_vintage(rows,date(2026,1,20))
        asof_mar=fred.observation_as_of_vintage(rows,date(2026,3,1))
        self.assertEqual(asof_jan[0].value,100.0)
        self.assertEqual(asof_mar[0].value,101.0)
        self.assertNotEqual(asof_jan[0].source_revision,asof_mar[0].source_revision)

    def test_nonfinite_and_missing_values_are_quarantined(self):
        data=payload([
            {"date":"2025-12-01","value":"NaN","realtime_start":"2026-01-15","realtime_end":"9999-12-31"},
            {"date":"2025-12-02","value":".","realtime_start":"2026-01-15","realtime_end":"9999-12-31"}])
        self.assertEqual(fred.parse_fred_observations(data,manifest(data)),[])

    def test_pagination_and_result_caps_are_enforced(self):
        with self.assertRaises(ValueError):
            fred.validate_fred_pagination(offset=-1,limit=100,records_received=0,total_count=0)
        with self.assertRaises(ValueError):
            fred.validate_fred_pagination(offset=0,limit=fred.MAX_PAGE_SIZE+1,records_received=0,total_count=0)
        self.assertEqual(fred.validate_fred_pagination(offset=0,limit=100,records_received=100,total_count=300),100)


if __name__=="__main__":
    unittest.main()
