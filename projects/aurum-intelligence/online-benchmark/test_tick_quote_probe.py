import json
import unittest

import tick_quote_probe as probe


def tick(idx, stamp, bid=2000.0, ask=2000.25, bid_volume=0.1, ask_volume=0.2):
    return {
        "row_idx": idx,
        "row": {
            "timestamp": stamp,
            "bid_price": bid,
            "ask_price": ask,
            "bid_volume": bid_volume,
            "ask_volume": ask_volume,
        },
        "truncated_cells": [],
    }


class FakeResponse:
    def __init__(self, payload, final_url=None):
        self.payload = json.dumps(payload).encode("utf-8")
        self.headers = {"Content-Length": str(len(self.payload))}
        self.final_url = final_url or probe.build_rows_url(0, 1)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def geturl(self):
        return self.final_url

    def read(self, maximum):
        return self.payload[:maximum]


class TickQuoteProbeTests(unittest.TestCase):
    def test_build_rows_url_is_fixed_allowlisted_and_bounded(self):
        url = probe.build_rows_url(100, 100)
        self.assertIn("datasets-server.huggingface.co/rows?", url)
        self.assertIn("dataset=CarlosSilva1%2Fxauusd-ticks", url)
        self.assertIn("offset=100", url)
        with self.assertRaisesRegex(probe.ProbeError, "offset_must_be_nonnegative"):
            probe.build_rows_url(-1, 1)
        with self.assertRaisesRegex(probe.ProbeError, "page_length_must_be_between"):
            probe.build_rows_url(0, 101)

    def test_three_sample_windows_are_disjoint_and_inside_dataset(self):
        total = 286_823_542
        windows = probe.plan_window_offsets(total, 1000)
        self.assertEqual([w["window"] for w in windows], ["head", "middle", "tail"])
        self.assertEqual([w["start_offset"] for w in windows], [0, 143_411_700, 286_822_500])
        self.assertTrue(all(w["start_offset"] + 1000 <= total for w in windows))
        self.assertTrue(all(a["start_offset"] + 1000 <= b["start_offset"] for a, b in zip(windows, windows[1:])))
        with self.assertRaisesRegex(probe.ProbeError, "dataset_too_small_for_three_disjoint_windows"):
            probe.plan_window_offsets(4000, 2000)

    def test_summary_profiles_bid_ask_spread_in_price_units_and_bps(self):
        rows = [
            tick(10, "2024-01-01T00:00:00.001000", 2000.0, 2000.2),
            tick(11, "2024-01-01T00:00:00.002000", 2000.0, 2000.4),
            tick(12, "2024-01-01T00:00:00.003000", 2000.0, 2000.6),
        ]
        summary = probe.summarize_window(rows, window="fixture", start_offset=10, payload_sha256="a" * 64)
        self.assertEqual(summary["rows"], 3)
        self.assertTrue(summary["timestamp_ordered"])
        self.assertEqual(summary["positive_spread_rows"], 3)
        self.assertAlmostEqual(summary["spread_price_units"]["median"], 0.4)
        self.assertAlmostEqual(summary["spread_price_units"]["p95"], 0.58)
        self.assertAlmostEqual(summary["spread_basis_points"]["median"], 2.0, places=3)
        self.assertFalse(summary["raw_tick_rows_persisted"])

    def test_timestamp_timezone_claim_is_normalized_but_assumed(self):
        stamp = probe.parse_timestamp_utc("2024-07-10T13:23:26.657000")
        self.assertEqual(stamp.isoformat(), "2024-07-10T13:23:26.657000+00:00")
        offset = probe.parse_timestamp_utc("2024-07-10T15:23:26.657000+02:00")
        self.assertEqual(stamp, offset)

    def test_crossed_and_locked_quotes_are_reported_not_silently_dropped(self):
        rows = [
            tick(0, "2024-01-01T00:00:00.001000", 2000.2, 2000.1),
            tick(1, "2024-01-01T00:00:00.002000", 2000.0, 2000.0),
            tick(2, "2024-01-01T00:00:00.003000", 2000.0, 2000.3),
        ]
        summary = probe.summarize_window(rows, window="fixture", start_offset=0, payload_sha256="b" * 64)
        self.assertEqual(summary["crossed_bid_ask_rows"], 1)
        self.assertEqual(summary["locked_or_zero_spread_rows"], 1)
        self.assertIn("CROSSED_BID_ASK_QUOTES_PRESENT", summary["quality_flags"])
        self.assertEqual(summary["positive_spread_rows"], 1)
        self.assertAlmostEqual(summary["spread_price_units"]["median"], 0.3, places=6)

    def test_negative_volume_and_nonfinite_prices_fail_closed(self):
        with self.assertRaisesRegex(probe.ProbeError, "quote_volume_must_be_nonnegative"):
            probe.summarize_window([tick(0, "2024-01-01T00:00:00.001000", bid_volume=-0.1)],
                window="x", start_offset=0, payload_sha256="c" * 64)
        with self.assertRaisesRegex(probe.ProbeError, "ask_price_must_be_finite"):
            probe.summarize_window([tick(0, "2024-01-01T00:00:00.001000", ask=float("nan"))],
                window="x", start_offset=0, payload_sha256="d" * 64)

    def test_nonmonotonic_timestamps_are_flagged(self):
        rows = [
            tick(5, "2024-01-01T00:00:00.002000"),
            tick(6, "2024-01-01T00:00:00.001000"),
        ]
        summary = probe.summarize_window(rows, window="fixture", start_offset=5, payload_sha256="e" * 64)
        self.assertFalse(summary["timestamp_ordered"])
        self.assertGreater(summary["timestamp_order_violations"], 0)
        self.assertIn("NON_MONOTONIC_TIMESTAMP_WITHIN_WINDOW", summary["quality_flags"])

    def test_fetch_json_page_checks_host_schema_and_row_count(self):
        payload = {
            "features": [{"name": name} for name in probe.REQUIRED_FIELDS],
            "rows": [tick(0, "2021-05-24T00:00:00.256000")],
            "num_rows_total": 10,
            "partial": False,
        }
        opened = []
        def opener(request, timeout=30):
            opened.append((request.full_url, timeout))
            return FakeResponse(payload, probe.build_rows_url(0, 1))
        result, size = probe.fetch_json_page(0, 1, opener=opener, sleep=lambda _: None)
        self.assertEqual(result["num_rows_total"], 10)
        self.assertEqual(len(result["rows"]), 1)
        self.assertEqual(size, len(json.dumps(payload).encode("utf-8")))
        self.assertEqual(len(opened), 1)

    def test_fetch_json_page_rejects_redirect_to_other_host(self):
        payload = {
            "features": [{"name": name} for name in probe.REQUIRED_FIELDS],
            "rows": [tick(0, "2021-05-24T00:00:00.256000")],
            "num_rows_total": 10,
            "partial": False,
        }
        def opener(request, timeout=30):
            return FakeResponse(payload, "https://evil.example/rows")
        with self.assertRaisesRegex(probe.ProbeError, "redirect_outside_allowlist"):
            probe.fetch_json_page(0, 1, opener=opener, sleep=lambda _: None)

    def test_write_reports_roundtrips_json_and_emits_real_markdown_newlines(self):
        summary = probe.summarize_window(
            [tick(0, "2024-01-01T00:00:00.001000")],
            window="fixture", start_offset=0, payload_sha256="1" * 64,
        )
        report = {
            "source": {
                "dataset_id": probe.DATASET_ID,
                "dataset_url": "https://huggingface.co/datasets/" + probe.DATASET_ID,
                "dataset_card_claimed_license": "CC-BY-4.0",
                "attribution": "required",
            },
            "status": "PASS",
            "dataset_rows_reported_by_huggingface": 1,
            "sample_design": {
                "sampled_rows_total": 1, "window_count": 1,
                "request_count_including_retries": 1, "total_response_bytes": 123,
            },
            "windows": [summary],
        }
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            files = probe.write_reports(report, Path(folder))
            payload = files["json"].read_text(encoding="utf-8")
            self.assertTrue(payload.endswith("\n"))
            parsed = json.loads(payload)
            self.assertEqual(parsed["status"], "PASS")
            markdown = files["markdown"].read_text(encoding="utf-8")
            self.assertIn("\n## Interpretation boundary\n", markdown)
            self.assertNotIn("\\n", markdown)

    def test_report_never_claims_production_or_persists_raw_ticks(self):
        summary = probe.summarize_window([tick(0, "2024-01-01T00:00:00.001000")],
            window="fixture", start_offset=0, payload_sha256="f" * 64)
        self.assertFalse(summary["raw_tick_rows_persisted"])
        report = {
            "source": {
                "dataset_id": probe.DATASET_ID,
                "dataset_url": "https://huggingface.co/datasets/" + probe.DATASET_ID,
                "dataset_card_claimed_license": "CC-BY-4.0",
                "attribution": "required",
            },
            "status": "PASS",
            "dataset_rows_reported_by_huggingface": 1,
            "sample_design": {"sampled_rows_total": 1, "window_count": 1,
                "request_count_including_retries": 1, "total_response_bytes": 123},
            "windows": [summary],
        }
        markdown = probe.render_markdown(report)
        self.assertIn("not a strategy test", markdown)
        self.assertIn("Upstream broker lineage", markdown)
        self.assertIn("NO_TRADE", markdown)


if __name__ == "__main__":
    unittest.main(verbosity=2)
