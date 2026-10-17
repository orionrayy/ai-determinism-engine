import csv
import io
import unittest
from datetime import date, datetime, timezone

import cftc_cot_adapter_v1 as cot
from external_context_schema_v1 import SourceManifest, manifest_sha256


def manifest(payload=b"fixture"):
    return SourceManifest("cftc-cot", "disaggregated-futures-only",
        "https://www.cftc.gov/files/dea/history/fut_disagg_txt_2025.zip",
        datetime(2026, 2, 1, tzinfo=timezone.utc), manifest_sha256(payload),
        len(payload), 1, 1, "CFTC historical compressed",
        "review-required", "official-host; independence-unverified")


def payload():
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=[
        "Market_and_Exchange_Names", "FutOnly_or_Combined", "Report_Date_as_YYYY-MM-DD",
        "Open_Interest_All", "Lev_Money_Positions_Long_All", "Lev_Money_Positions_Short_All",
        "NonRept_Positions_Long_All", "NonRept_Positions_Short_All"])
    w.writeheader()
    w.writerow({"Market_and_Exchange_Names":"GOLD - COMMODITY EXCHANGE INC.",
        "FutOnly_or_Combined":"FutOnly", "Report_Date_as_YYYY-MM-DD":"2026-01-06",
        "Open_Interest_All":"500000", "Lev_Money_Positions_Long_All":"120000",
        "Lev_Money_Positions_Short_All":"75000", "NonRept_Positions_Long_All":"100",
        "NonRept_Positions_Short_All":"80"})
    w.writerow({"Market_and_Exchange_Names":"SILVER - COMMODITY EXCHANGE INC.",
        "FutOnly_or_Combined":"FutOnly", "Report_Date_as_YYYY-MM-DD":"2026-01-06",
        "Open_Interest_All":"200000", "Lev_Money_Positions_Long_All":"10",
        "Lev_Money_Positions_Short_All":"20"})
    return out.getvalue().encode()


class CftcCotTests(unittest.TestCase):
    def test_conservative_proxy_is_next_week_wednesday(self):
        got = cot.conservative_cot_available_at(date(2026, 1, 6))
        self.assertEqual(got.isoformat(), "2026-01-14T00:00:00-05:00")

    def test_tuesday_report_is_not_available_on_tuesday_or_friday(self):
        got = cot.conservative_cot_available_at(date(2026, 1, 6))
        self.assertGreater(got, datetime(2026, 1, 9, tzinfo=timezone.utc))

    def test_verified_release_timestamp_overrides_proxy_only_when_valid(self):
        verified = datetime(2026, 1, 9, 20, 30, tzinfo=timezone.utc)
        got, quality = cot.resolve_cot_available_at(date(2026, 1, 6), verified)
        self.assertEqual(got, verified)
        self.assertEqual(quality, "verified")
        got, quality = cot.resolve_cot_available_at(date(2026, 1, 6), None, ambiguous_calendar=True)
        self.assertIsNone(got)
        self.assertEqual(quality, "unknown")

    def test_gold_filter_excludes_other_contracts_and_options_combined(self):
        rows = cot.parse_cot_csv(payload(), manifest(payload()))
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0].value, 45000.0)
        self.assertEqual(rows[0].attributes["market"], "GOLD - COMMODITY EXCHANGE INC.")

    def test_invalid_positions_are_quarantined(self):
        raw = payload().decode().replace("120000", "NaN").encode()
        rows = cot.parse_cot_csv(raw, manifest(raw))
        self.assertEqual(rows, [])

    def test_truncated_or_oversized_zip_fails_closed(self):
        raw = b"PK\\x03\\x04broken"
        with self.assertRaises(ValueError):
            cot.parse_cot_csv(raw, manifest(raw))


if __name__ == "__main__":
    unittest.main()
