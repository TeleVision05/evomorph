"""Enforce the code-size rules: no file over 500 lines, no function over 75 lines.

Usage: .venv/bin/python scripts/check_sizes.py [paths...]   (default: evomorph tests scripts)

Each path may be a ``.py`` file (what pre-commit passes) or a directory, which is
searched recursively. Exits non-zero and lists offenders when a limit is exceeded.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

MAX_FILE, MAX_FUNC = 500, 75
DEFAULT_ROOTS = ("evomorph", "tests", "scripts")


def python_files(paths: list[str]) -> list[Path]:
    """Expand files and directories into the sorted list of .py files to check."""
    found: set[Path] = set()
    for raw in paths or DEFAULT_ROOTS:
        p = Path(raw)
        if p.is_file():
            if p.suffix == ".py":
                found.add(p)
        elif p.is_dir():
            found.update(p.rglob("*.py"))
        else:
            raise FileNotFoundError(f"check_sizes: no such file or directory: {p}")
    return sorted(found)


def offences(path: Path) -> list[str]:
    """Size-rule violations in one file."""
    text = path.read_text()
    bad = []
    n = text.count("\n") + (0 if text.endswith("\n") else 1)
    if n > MAX_FILE:
        bad.append(f"{path}: {n} lines (max {MAX_FILE})")
    for node in ast.walk(ast.parse(text, filename=str(path))):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            length = node.end_lineno - node.lineno + 1
            if length > MAX_FUNC:
                bad.append(f"{path}:{node.lineno} {node.name}() {length} lines (max {MAX_FUNC})")
    return bad


def main(paths: list[str]) -> int:
    bad = [msg for f in python_files(paths) for msg in offences(f)]
    print("\n".join(bad) or "sizes ok")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
