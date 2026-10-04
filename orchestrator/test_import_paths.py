import subprocess
import sys
import unittest
from pathlib import Path


class ImportPathRegressionTests(unittest.TestCase):
    def test_package_import_from_repository_root(self):
        root = Path(__file__).resolve().parent.parent
        code = (
            "from orchestrator.orchestrator import load_state; "
            "assert callable(load_state)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"package import failed: stdout={result.stdout!r} stderr={result.stderr!r}",
        )


if __name__ == "__main__":
    unittest.main()
