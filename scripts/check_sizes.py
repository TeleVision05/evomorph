"""Enforce the code-size rules: no file over 500 lines, no function over 75 lines.

Usage: .venv/bin/python scripts/check_sizes.py [paths...]   (default: evomorph tests scripts)
Exits non-zero and lists offenders when a limit is exceeded.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

MAX_FILE, MAX_FUNC = 500, 75


def main(paths: list[str]) -> int:
    bad = []
    for root in paths or ["evomorph", "tests", "scripts"]:
        for f in sorted(Path(root).rglob("*.py")):
            text = f.read_text()
            n = text.count("\n") + 1
            if n > MAX_FILE:
                bad.append(f"{f}: {n} lines")
            for node in ast.walk(ast.parse(text)):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    length = node.end_lineno - node.lineno + 1
                    if length > MAX_FUNC:
                        bad.append(f"{f}:{node.lineno} {node.name}() {length} lines")
    print("\n".join(bad) or "sizes ok")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
