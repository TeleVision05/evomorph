"""The code-size pre-commit hook must fail on oversized files passed by name."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_sizes.py"


def _run(*paths: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, paths)], capture_output=True, text=True
    )


def _write(path: Path, n_top: int, n_func: int) -> Path:
    body = ["x = 1"] * n_top + ["def f():"] + ["    y = 1"] * n_func + ["    return y"]
    path.write_text("\n".join(body) + "\n")
    return path


def test_oversized_file_passed_by_name_fails(tmp_path) -> None:
    # pre-commit passes individual file names, not directories
    out = _run(_write(tmp_path / "big.py", 600, 3))
    assert out.returncode == 1 and "big.py: 6" in out.stdout


def test_long_function_passed_by_name_fails(tmp_path) -> None:
    out = _run(_write(tmp_path / "longfn.py", 1, 100))
    assert out.returncode == 1 and "f() 102 lines" in out.stdout


def test_small_file_and_directory_pass(tmp_path) -> None:
    ok = _write(tmp_path / "ok.py", 10, 10)
    assert _run(ok).returncode == 0
    assert _run(tmp_path).returncode == 0
    _write(tmp_path / "sub_big.py", 600, 1)
    assert _run(tmp_path).returncode == 1  # directories are still searched


def test_hook_passes_filenames_to_the_checker() -> None:
    hooks = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())["repos"][0]["hooks"]
    size_hook = next(h for h in hooks if h["id"] == "code-size")
    assert "check_sizes.py" in size_hook["entry"]
    assert size_hook.get("pass_filenames", True)  # so the file-path branch above is what runs
