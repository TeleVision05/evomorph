"""Shared helpers for the dashboard tests: fixture runs copied into a temp folder."""

from __future__ import annotations

import shutil
from pathlib import Path

FIXTURE_RUNS = Path(__file__).parent / "fixtures" / "dashboard" / "runs"
APP = str(Path(__file__).resolve().parents[1] / "evomorph" / "visualization" / "dashboard.py")
RUN_NAMES = ("ga_tiny", "me_tiny")


def copy_runs(dest: Path, names: tuple[str, ...] = RUN_NAMES) -> Path:
    """Copy the committed fixture runs to ``dest`` (tests may then damage the copies)."""
    dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copytree(FIXTURE_RUNS / name, dest / name)
    return dest


def make_in_progress(run: Path, keep_steps: int = 2) -> None:
    """Turn a finished copy into what a run looks like mid-way.

    Removes the files the engine writes only at the end, keeps the first
    ``keep_steps`` generation files, cuts metrics.csv to match plus a half
    written line, and leaves a half-written newest generation file.
    """
    for name in ("summary.md", "best_genome.json", "best_result.json", "best_heldout.json"):
        (run / name).unlink(missing_ok=True)
    shutil.rmtree(run / "archive")
    shutil.rmtree(run / "genomes" / "final", ignore_errors=True)
    gens = sorted((run / "generations").glob("gen_*.json"))
    for p in gens[keep_steps:]:
        p.unlink()
    for p in sorted((run / "genomes").glob("champion_gen_*.json"))[keep_steps:]:
        p.unlink()
    lines = (run / "metrics.csv").read_text().splitlines()
    partial = lines[keep_steps + 1][: len(lines[keep_steps + 1]) // 2]
    (run / "metrics.csv").write_text("\n".join([*lines[: keep_steps + 1], partial]) + "\n")
    newest = run / "generations" / f"gen_{keep_steps:04d}.json"
    newest.write_text('[{"id": "half-written", "par')
