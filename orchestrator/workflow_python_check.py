#!/usr/bin/env python3
"""Compile Python heredocs embedded in GitHub Actions workflow files."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import textwrap

HEREDOC_RE = re.compile(
    r"(?m)^\s*(?:python3?|python)\s+-\s+<<(['"]?)([A-Za-z_][A-Za-z0-9_]*)\1.*$"
)


def extract_scripts(text: str) -> list[tuple[int, str]]:
    lines = text.splitlines()
    scripts: list[tuple[int, str]] = []
    i = 0
    while i < len(lines):
        match = HEREDOC_RE.match(lines[i])
        if not match:
            i += 1
            continue
        marker = match.group(2)
        start_line = i + 1
        i += 1
        body: list[str] = []
        while i < len(lines) and lines[i].strip() != marker:
            body.append(lines[i])
            i += 1
        if i >= len(lines):
            raise ValueError(f"unterminated Python heredoc starting at line {start_line}")
        scripts.append((start_line, textwrap.dedent("\n".join(body))))
        i += 1
    return scripts


def validate_workflows(workflow_dir: Path) -> int:
    failures = 0
    for path in sorted(workflow_dir.glob("*.y*ml")):
        text = path.read_text(encoding="utf-8")
        try:
            scripts = extract_scripts(text)
        except ValueError as exc:
            print(f"{path}: {exc}")
            failures += 1
            continue
        for line, source in scripts:
            try:
                compile(source, f"{path}:{line}", "exec")
            except SyntaxError as exc:
                print(
                    f"{path}:{line}: Python syntax error: "
                    f"{exc.msg} (line {exc.lineno})"
                )
                failures += 1
        print(f"{path}: checked {len(scripts)} embedded Python block(s)")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workflow-dir",
        default=".github/workflows",
        type=Path,
    )
    args = parser.parse_args()
    failures = validate_workflows(args.workflow_dir)
    if failures:
        raise SystemExit(f"{failures} embedded Python block(s) failed validation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
