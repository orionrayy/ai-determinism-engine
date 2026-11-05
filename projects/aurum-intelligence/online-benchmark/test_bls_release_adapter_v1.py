import json
import unittest
from datetime import datetime, timezone

import bls_release_adapter_v1 as bls
from external_context_schema_v1 import SourceManifest, manifest_sha256


def manifest(raw=b"fixture"):
    return SourceManifest("bls-public-api-v1", "CPI", "https://api.bls.gov/publicAPI/v1/timeseries/data/",
        datetime(2026,3,1,tzinfo=timezone.utc), manifest_sha256(raw),len(raw),0,0,
        "BLS public API v1","public-data-rights-review-required","official-host")


def response(status="REQUEST_SUCCEEDED", value="321.123"):
    return json.dumps({"status":status,"Results":{"series":[{"seriesID":"CUUR0000SA0",
        "data":[{"year":"2025","period":"M12","periodName":"December","value":value,
        "footnotes":[{}]}]}]}}).encode()


class BlsReleaseTests(unittest.TestCase):
    def test_observation_period_is_not_published_at(self):
        rows=bls.parse_bls_api_response(response(),manifest(response()))
        self.assertEqual(rows[0].attributes["observation_period"],"2025-M12")
        self.assertIsNone(rows[0].published_at)
        self.assertIsNone(rows[0].available_at)
        self.assertEqual(rows[0].availability_quality,"unknown")

    def test_public_api_v1_limits_are_enforced(self):
        with self.assertRaises(ValueError):
            bls.validate_bls_request(["S1"]*26,2020,2025)
        with self.assertRaises(ValueError):
            bls.validate_bls_request(["S1"],2010,2025)
        with self.assertRaises(ValueError):
            bls.validate_bls_request(["S1"],2020,2021,requests_today=25)

    def test_error_status_and_rate_limit_are_typed(self):
        with self.assertRaises(bls.BLSAPIError) as ctx:
            bls.parse_bls_api_response(response("REQUEST_NOT_PROCESSED"),manifest())
        self.assertEqual(ctx.exception.status,"REQUEST_NOT_PROCESSED")
        self.assertEqual(bls.classify_bls_error("daily_limit"),"FAILED")
        self.assertEqual(bls.classify_bls_error("partial_series"),"PARTIAL")

    def test_missing_release_calendar_means_descriptive_only(self):
        self.assertIsNone(bls.map_release_availability("CUUR0000SA0","2025-M12",{}))
        release=datetime(2026,1,13,13,30,tzinfo=timezone.utc)
        self.assertEqual(bls.map_release_availability("CUUR0000SA0","2025-M12",
            {"CUUR0000SA0|2025-M12":release}),release)

    def test_consensus_is_not_invented(self):
        rows=bls.parse_bls_api_response(response(),manifest())
        self.assertNotIn("consensus",rows[0].attributes)
        self.assertNotIn("surprise",rows[0].attributes)


if __name__=="__main__":
    unittest.main()
