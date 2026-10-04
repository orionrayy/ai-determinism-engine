import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


class ImportPathRegressionTests(unittest.TestCase):
    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, *args],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_package_import_from_repository_root(self):
        result = self._run(
            "-c",
            "from orchestrator.orchestrator import load_state; assert callable(load_state)",
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"package import failed: stdout={result.stdout!r} stderr={result.stderr!r}",
        )

    def test_direct_script_invocation_supports_help(self):
        result = self._run("orchestrator/orchestrator.py", "--help")
        self.assertEqual(
            result.returncode,
            0,
            msg=f"script invocation failed: stdout={result.stdout!r} stderr={result.stderr!r}",
        )
        self.assertIn("usage:", result.stdout.lower())


if __name__ == "__main__":
    unittest.main()
