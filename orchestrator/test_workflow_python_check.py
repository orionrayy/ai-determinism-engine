from __future__ import annotations

import unittest
from pathlib import Path

from workflow_python_check import extract_scripts, validate_workflows


class WorkflowPythonCheckTests(unittest.TestCase):
    def test_extracts_indented_heredoc_with_shell_suffix(self):
        text = """\
          run: |
            python - <<'PY' "$BODY"
            import json
            value = json.loads("{}")
            PY
        """
        scripts = extract_scripts(text)
        self.assertEqual(len(scripts), 1)
        self.assertIn("import json", scripts[0][1])

    def test_repository_workflows_compile(self):
        root = Path(__file__).resolve().parent.parent
        self.assertEqual(validate_workflows(root / ".github" / "workflows"), 0)


if __name__ == "__main__":
    unittest.main()
