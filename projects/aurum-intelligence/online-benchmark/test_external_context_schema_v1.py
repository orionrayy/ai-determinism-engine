import hashlib
import math
import unittest
from datetime import datetime, timezone

import external_context_schema_v1 as schema


def obs(**overrides):
    base = dict(
        source_id="fixture", source_record_id="record-1",
        dataset_or_series_id="fixture-series", schema_version="1.0",
        event_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        published_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        available_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        availability_quality="verified",
        retrieved_at=datetime(2026, 1, 3, tzinfo=timezone.utc),
        source_url="https://example.invalid/data", payload_sha256="a" * 64,
        source_revision=None, value=1.0, unit="unit", attributes={},
        rights_status="unknown", provenance_status="unverified", quality_flags=(),
    )
    base.update(overrides)
    return schema.ExternalObservation(**base)


class ExternalContextSchemaTests(unittest.TestCase):
    def test_rejects_naive_datetime(self):
        with self.assertRaises(ValueError):
            schema.parse_utc_timestamp("2026-01-01T00:00:00")

    def test_aware_timestamp_normalizes_to_utc(self):
        result = schema.parse_utc_timestamp("2026-01-01T01:00:00+01:00")
        self.assertEqual(result, datetime(2026, 1, 1, tzinfo=timezone.utc))

    def test_rejects_nonfinite_values(self):
        with self.assertRaises(ValueError):
            schema.validate_observations([obs(value=float("nan"))])

    def test_conflicting_duplicate_id_is_counted_not_silently_overwritten(self):
        audit = schema.validate_observations([obs(), obs(value=2.0)])
        self.assertEqual(audit.duplicate_id_count, 1)
        self.assertEqual(audit.conflicting_duplicate_count, 1)

    def test_manifest_hash_is_deterministic(self):
        payload = b"fixture bytes"
        self.assertEqual(schema.manifest_sha256(payload), hashlib.sha256(payload).hexdigest())

    def test_available_at_cannot_be_after_retrieval(self):
        with self.assertRaises(ValueError):
            schema.validate_observations([obs(
                available_at=datetime(2026, 1, 4, tzinfo=timezone.utc)
            )])

    def test_unknown_availability_is_flagged_in_audit(self):
        audit = schema.validate_observations([obs(
            available_at=None, availability_quality="unknown"
        )])
        self.assertEqual(audit.unknown_availability_count, 1)


if __name__ == "__main__":
    unittest.main()
