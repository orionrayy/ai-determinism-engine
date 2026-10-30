from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import termux_doctor as doctor


class TermuxDoctorTests(unittest.TestCase):
    def test_pair_parser_accepts_canonical_indodax_sol_idr(self):
        result = doctor.parse_indodax_sol_idr_pairs([{
            "id": "solidr", "symbol": "SOL/IDR",
            "traded_currency_unit": "SOL", "base_currency": "IDR",
            "price_precision": "1", "quantity_increment": "0.00000001",
            "trade_min_base_currency": "10000", "trade_min_traded_currency": "0.005",
        }])
        self.assertEqual(result["pair_id"], "solidr")
        self.assertEqual(result["base_asset"], "SOL")
        self.assertEqual(result["quote_asset"], "IDR")

    def test_pair_parser_rejects_ambiguous_or_missing_markets(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            doctor.parse_indodax_sol_idr_pairs([])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            doctor.parse_indodax_sol_idr_pairs([
                {"id": "solidr", "symbol": "SOLIDR", "traded_currency_unit": "SOL", "base_currency": "IDR"},
                {"id": "SOLIDR", "symbol": "SOLIDR", "traded_currency_unit": "SOL", "base_currency": "IDR"},
            ])

    def test_sqlite_replay_is_idempotent_and_survives_reopen(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = doctor.sqlite_recovery_check(tmp)
        self.assertEqual(result["integrity"], "ok")
        self.assertEqual(result["rows_after_replay"], 1)

    def write_dataset(self, folder: Path, *, gap: bool = False):
        path = folder / "INDODAX_SOLIDR_1m.csv"
        rows = []
        for index in range(3):
            open_ms = 1_800_000_000_000 + index * (120_000 if gap and index == 2 else 60_000)
            rows.append({
                "open_time_ms": open_ms, "close_time_ms": open_ms + 59_999,
                "open": 100 + index, "high": 103 + index, "low": 99 + index,
                "close": 102 + index, "volume": 1.25,
            })
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                "open_time_ms", "close_time_ms", "open", "high", "low", "close", "volume",
            ])
            writer.writeheader()
            writer.writerows(rows)
        metadata = {
            "data_status": "OK", "venue": "INDODAX", "pair": "SOL/IDR",
            "input_price_currency": "IDR", "fx_conversion_applied": False,
            "count": 3, "dataset_sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
        }
        Path(str(path) + ".meta.json").write_text(json.dumps(metadata), encoding="utf-8")
        return path

    def test_native_dataset_stream_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_dataset(Path(tmp))
            result = doctor.verify_native_dataset(path)
        self.assertEqual(result["rows"], 3)
        self.assertTrue(result["sidecar_verified"])
        self.assertFalse(result["fx_conversion_applied"])

    def test_native_dataset_rejects_gaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_dataset(Path(tmp), gap=True)
            with self.assertRaisesRegex(ValueError, "gap or duplicate"):
                doctor.verify_native_dataset(path)

    def test_native_dataset_rejects_sidecar_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_dataset(Path(tmp))
            meta_path = Path(str(path) + ".meta.json")
            meta = json.loads(meta_path.read_text())
            meta["dataset_sha256"] = "0" * 64
            meta_path.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "does not match sidecar"):
                doctor.verify_native_dataset(path)


if __name__ == "__main__":
    unittest.main()
