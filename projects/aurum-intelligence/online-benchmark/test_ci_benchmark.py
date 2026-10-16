import csv
import io
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import ci_benchmark as b

def fixture(n=100):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["datetime", "open", "high", "low", "close", "volume"])
    start, price = datetime(2026, 1, 1), 3000.0
    for i in range(n):
        ts = start + timedelta(minutes=5*i)
        o, c = price, price + (0.4 if i % 3 else -0.25)
        writer.writerow([ts.isoformat(), o, max(o,c)+.2, min(o,c)-.2, c, 100+i])
        price = c
    return stream.getvalue().encode()

class OnlineBenchmarkTests(unittest.TestCase):
    def test_dataset_schema_and_spread_out_cases(self):
        rows = b.parse_csv(fixture(), 30)
        self.assertEqual(len(rows), 100)
        indices = b.choose_indices(35954, 50)
        self.assertEqual(len(set(indices)), 50)
        self.assertGreater(indices[-1], 35000)

    def test_invalid_timestamp_and_ohlc_fail_closed(self):
        with self.assertRaisesRegex(b.DatasetError, "strictly increasing"):
            b.parse_csv(b"timestamp,open,high,low,close,volume\n2026-01-01T00:05:00,1,2,1,1,1\n2026-01-01T00:00:00,1,2,1,1,1\n", 2)
        with self.assertRaisesRegex(b.DatasetError, "invalid OHLCV"):
            b.parse_csv(b"timestamp,open,high,low,close,volume\n2026-01-01T00:00:00,1,.5,.9,1,1\n", 1)

    def test_timezone_mixing_is_rejected(self):
        with self.assertRaisesRegex(b.DatasetError, "mixed timezone"):
            b.parse_csv(b"timestamp,open,high,low,close,volume\n2026-01-01T00:00:00Z,1,2,1,1,1\n2026-01-01T00:05:00,1,2,1,1,1\n", 2)

    def test_30_cases_produce_150_rule_evaluations(self):
        rows = b.parse_csv(fixture(), 30)
        result = b.case_replay(rows, 30)
        self.assertEqual(len(result), 150)
        self.assertEqual(len({x["decision_time"] for x in result}), 30)
        self.assertTrue(all(x["cost_is_assumption"] for x in result))

    def test_480_market_paths_and_240_software_stress_cases(self):
        rows = b.parse_csv(fixture(), 30)
        market = b.market_scenarios(rows, 2)
        software = b.software_scenarios(rows)
        self.assertEqual(len(market), 480)
        self.assertTrue(all(x["passed"] for x in market), [x for x in market if not x["passed"]][:4])
        self.assertEqual(len(software), 240)
        self.assertTrue(all(x["passed"] for x in software), [x for x in software if not x["passed"]][:4])

    def test_end_to_end_outputs_replayable_reports_and_no_trade(self):
        with tempfile.TemporaryDirectory() as td:
            result = b.run(fixture(), Path(td), 30, 30, 2, "test-fixture")
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["real_data_evaluations"], 150)
            self.assertEqual(result["market_scenarios"], 480)
            self.assertEqual(result["software_scenarios"], 240)
            self.assertEqual(result["decision_output"], "NO_TRADE")
            for name in ("online_dataset_benchmark.json", "online_dataset_benchmark.md", "real_data_case_results.csv", "market_scenario_results.csv", "software_scenario_results.csv"):
                self.assertTrue((Path(td)/name).exists(), name)

if __name__ == "__main__":
    unittest.main()
