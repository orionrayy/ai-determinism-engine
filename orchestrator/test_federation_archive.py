import io
import json
import sys
import tempfile
import zipfile
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location(
    "_orchestrator_archive_under_test",
    ROOT / "orchestrator.py",
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("cannot load orchestrator module")
o = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = o
_SPEC.loader.exec_module(o)


class FederationArchiveTests(unittest.TestCase):
    def _zip_response(self, payload: bytes) -> bytes:
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("aggregate.json", payload)
        return stream.getvalue()

    def test_uncompressed_member_limit_is_checked_before_read(self):
        oversized = b"x" * (128 * 1024 + 1)
        archive_bytes = self._zip_response(oversized)

        class Response:
            status = 200

            def __init__(self):
                self.stream = io.BytesIO(archive_bytes)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self, size=-1):
                return self.stream.read(size)

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(
                o.os.environ,
                {"GITHUB_TOKEN": "test-token", "GITHUB_REPOSITORY": "owner/repo"},
                clear=False,
            ), patch.object(
                o,
                "github_repository",
                return_value="owner/repo",
            ), patch.object(
                o,
                "github_headers",
                return_value={},
            ), patch.object(
                o,
                "http_json",
                return_value={"data": {"expired": False, "digest": ""}},
            ), patch.object(
                o.urllib.request,
                "urlopen",
                return_value=Response(),
            ):
                with self.assertRaisesRegex(RuntimeError, "exceeds download limit"):
                    o.download_federation_aggregate(123)

    def test_valid_aggregate_still_loads(self):
        aggregate = {"federation_id": "fed", "workflow_id": "wf", "results": []}
        encoded = json.dumps(
            aggregate,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = o.hashlib.sha256(encoded).hexdigest()
        aggregate["aggregate_sha256"] = digest
        archive_bytes = self._zip_response(
            json.dumps(aggregate, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )

        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self, size=-1):
                return archive_bytes

        with patch.object(
            o,
            "github_repository",
            return_value="owner/repo",
        ), patch.object(
            o,
            "github_headers",
            return_value={},
        ), patch.object(
            o,
            "http_json",
            return_value={"data": {"expired": False, "digest": ""}},
        ), patch.object(
            o.urllib.request,
            "urlopen",
            return_value=Response(),
        ):
            value = o.download_federation_aggregate(123)

        self.assertEqual(value["federation_id"], "fed")


if __name__ == "__main__":
    unittest.main()
