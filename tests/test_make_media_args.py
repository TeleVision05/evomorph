"""make_media.py refuses an unusable --out with a plain message, before simulating anything."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_media.py"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, cwd=ROOT
    )


def test_audit_out_directory_gives_plain_error(tmp_path: Path) -> None:
    r = _run("audit", str(tmp_path / "no_run"), "--out", str(tmp_path))
    assert r.returncode == 2
    assert "must be an image file name" in r.stderr
    assert "Traceback" not in r.stderr


def test_audit_out_without_extension_gives_plain_error(tmp_path: Path) -> None:
    r = _run("audit", str(tmp_path / "no_run"), "--out", str(tmp_path / "sheet"))
    assert r.returncode == 2
    assert "must be an image file name" in r.stderr


def test_evolution_out_file_gives_plain_error(tmp_path: Path) -> None:
    f = tmp_path / "x.mp4"
    f.write_text("x")
    r = _run("evolution", str(tmp_path / "no_run"), "--out", str(f))
    assert r.returncode == 2
    assert "is a file" in r.stderr
