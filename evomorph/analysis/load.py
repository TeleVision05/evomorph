"""Read run directories and experiment folders for analysis (no simulation).

* :func:`metrics_series`: a run's ``metrics.csv`` as numeric columns.
* :func:`plan_history`: body-plan counts of the current set after every step.
  For the GA that is the population, read from ``generations/gen_NNNN.json``
  (which holds the whole population). For MAP-Elites it is the archive, which
  is rebuilt by replaying every insertion recorded in those files (status
  ``new`` or ``improved`` puts the genome in its descriptor's cell), so the
  history is exact, not sampled.
* :func:`load_experiment_dir`: manifest, results table and run folders.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from evomorph.config import load_yaml
from evomorph.evolution.archive import ArchiveSpec
from evomorph.experiments.comparison import MANIFEST
from evomorph.experiments.table import read_table
from evomorph.utils.serialization import load_json

PLANS: tuple[str, ...] = tuple(f"{2 * p}x{s}" for p in (1, 2, 3, 4) for s in (1, 2))


def is_experiment(path: str | Path) -> bool:
    return (Path(path) / MANIFEST).is_file()


def is_run(path: str | Path) -> bool:
    return (Path(path) / "metrics.csv").is_file() and (Path(path) / "config.yaml").is_file()


def metrics_series(run_dir: str | Path) -> dict[str, np.ndarray]:
    """``metrics.csv`` columns; numeric ones as float arrays, others as string arrays."""
    with (Path(run_dir) / "metrics.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    out: dict[str, np.ndarray] = {}
    for key in rows[0] if rows else ():
        try:
            out[key] = np.array([float(r[key]) for r in rows])
        except ValueError:
            out[key] = np.array([r[key] for r in rows])
    return out


def _plan_of(record: dict[str, Any]) -> str:
    m = record["genome"]["morphology"]
    return f"{2 * int(m['n_pairs'])}x{int(m['n_segments'])}"


def _algorithm(run_dir: Path) -> str:
    return str(load_yaml(run_dir / "config.yaml")["evolution"]["algorithm"])


def _archive_spec(run_dir: Path) -> ArchiveSpec:
    me = load_yaml(run_dir / "config.yaml")["map_elites"]
    return ArchiveSpec(tuple(me["descriptors"]), tuple(me["bins"]), tuple(me["ranges"]))


def plan_history(run_dir: str | Path) -> tuple[np.ndarray, list[dict[str, int]]]:
    """(simulations after each step, body-plan counts of the current set at that step)."""
    run_dir = Path(run_dir)
    sims = metrics_series(run_dir)["sims"]
    files = sorted((run_dir / "generations").glob("gen_*.json"))
    counts: list[dict[str, int]] = []
    if _algorithm(run_dir) == "ga":
        for f in files:
            c: dict[str, int] = {}
            for rec in load_json(f):
                c[_plan_of(rec)] = c.get(_plan_of(rec), 0) + 1
            counts.append(c)
        return sims, counts
    spec = _archive_spec(run_dir)
    cells: dict[tuple[int, int], str] = {}
    for f in files:
        for rec in load_json(f):
            if rec["status"] in ("new", "improved"):
                cells[spec.cell(spec.descriptor(rec["metrics"]))] = _plan_of(rec)
        c = {}
        for plan in cells.values():
            c[plan] = c.get(plan, 0) + 1
        counts.append(c)
    return sims, counts


@dataclass
class ExperimentData:
    path: Path
    manifest: dict[str, Any]
    rows: list[dict[str, Any]]  # results.csv
    conditions: list[dict[str, Any]]  # manifest condition records, in design order

    def run_dir(self, condition: str, seed: int) -> Path:
        return self.path / "runs" / f"{condition}_s{seed}"

    def runs_of(self, condition: str) -> list[tuple[int, Path]]:
        """(seed, run folder) of the finished runs of ``condition``, by seed."""
        seeds = sorted(r["seed"] for r in self.rows if r["condition"] == condition)
        return [(s, self.run_dir(condition, s)) for s in seeds]

    def label(self, condition: str) -> str:
        for c in self.conditions:
            if c["key"] == condition:
                return str(c["label"])
        return condition


def load_experiment_dir(path: str | Path) -> ExperimentData:
    """An experiment folder made by ``evomorph experiment``."""
    p = Path(path)
    manifest = load_json(p / MANIFEST)
    table = p / "results.csv"
    rows = read_table(table) if table.is_file() else []
    return ExperimentData(p, manifest, rows, list(manifest["conditions"]))
