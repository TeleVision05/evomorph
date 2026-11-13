"""Archive and per-generation statistics derived from a run directory (no Streamlit).

Archive
-------
A finished run has ``archive/archive.json`` (written by ``Archive.save``). A
run that is still going has no archive file yet, so :func:`load_archive`
rebuilds it by replaying every evaluated record in order with the engine's
insertion rules (valid, never fell in any run, strictly better than the
incumbent; cell from :class:`ArchiveSpec`). Evaluation order is stored in the
generation files, so the replay gives the same archive the engine holds; a
test checks it against the saved file of the fixture runs. The view says which
source it used.

Statistics
----------
Per-generation numbers come from the individuals *evaluated in that step* (the
GA's new children, or a MAP-Elites batch), so both algorithms are summarised
the same way. ``metrics.csv`` columns (population or archive statistics as the
engine defines them) are plotted as they are.

A creature counts as **successful** when it is valid, never fell in any of its
runs, and its mean forward displacement is at least ``SUCCESS_DISTANCE``.
1 m is the threshold the core's random sweep reported against (7% of 300
random genomes moved a metre in 10 s), so it separates walkers from the
creatures evolution starts with. It is a reporting threshold only and plays
no part in selection.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from evomorph.evolution.archive import ArchiveSpec
from evomorph.utils.serialization import load_json
from evomorph.visualization.dashboard_data import (
    Index,
    Run,
    RunReadError,
    evaluated_records,
    load_generation,
)

SUCCESS_DISTANCE = 1.0  # m, see module doc


@dataclass
class CellView:
    """One occupied archive cell."""

    cell: tuple[int, int]
    genome_id: str
    fitness: float
    descriptor: tuple[float, float]
    metrics: dict[str, float]
    seed_fitness: tuple[float, ...] = ()
    genome: dict[str, Any] | None = None  # genome dict (reconstructed archives)
    genome_file: str = ""  # path of the saved elite genome (saved archives)


@dataclass
class ArchiveView:
    spec: ArchiveSpec
    cells: dict[tuple[int, int], CellView] = field(default_factory=dict)
    source: str = "saved"  # "saved" or "rebuilt from generation files"

    @property
    def coverage(self) -> float:
        return len(self.cells) / self.spec.n_cells

    def grid(self) -> np.ndarray:
        """(bins0, bins1) fitness grid, NaN where empty."""
        g = np.full(self.spec.bins, np.nan)
        for (i, j), c in self.cells.items():
            g[i, j] = c.fitness
        return g

    def edges(self, axis: int) -> np.ndarray:
        lo, hi = self.spec.ranges[axis]
        return np.linspace(lo, hi, self.spec.bins[axis] + 1)


def _saved_archive(run: Run) -> ArchiveView | None:
    meta_path = run.path / "archive" / "archive.json"
    if not meta_path.is_file():
        return None
    try:
        meta = load_json(meta_path)
        s = meta["spec"]
        spec = ArchiveSpec(tuple(s["descriptors"]), tuple(s["bins"]), tuple(s["ranges"]))
        view = ArchiveView(spec, source="saved")
        for row in meta["cells"]:
            key = (int(row["cell"][0]), int(row["cell"][1]))
            view.cells[key] = CellView(
                cell=key,
                genome_id=str(row["genome_id"]),
                fitness=float(row["fitness"]),
                descriptor=(float(row["descriptor"][0]), float(row["descriptor"][1])),
                metrics={k: float(v) for k, v in row["metrics"].items()},
                seed_fitness=tuple(row.get("seed_fitness") or ()),
                genome_file=str(run.path / "archive" / row["genome_file"]),
            )
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        raise RunReadError(
            f"The MAP-Elites archive file of '{run.name}' is damaged and cannot be read."
        ) from None
    return view


def _eligible(rec: dict[str, Any]) -> bool:
    m = rec.get("metrics") or {}
    return bool(rec.get("valid")) and bool(m) and float(m.get("fell", 0.0)) <= 0.0


def rebuild_archive(run: Run, index: Index) -> ArchiveView:
    """Replay the engine's insertion rules over every evaluated record (see module doc)."""
    spec = run.archive_spec
    view = ArchiveView(spec, source="rebuilt from generation files")
    for step in sorted(index.by_step):
        for gid in index.by_step[step]:
            rec = index.records[gid]
            if not _eligible(rec):
                continue
            desc = spec.descriptor(rec["metrics"])
            key = spec.cell(desc)
            old = view.cells.get(key)
            if old is not None and not float(rec["fitness"]) > old.fitness:
                continue
            view.cells[key] = CellView(
                cell=key,
                genome_id=gid,
                fitness=float(rec["fitness"]),
                descriptor=desc,
                metrics=dict(rec["metrics"]),
                seed_fitness=tuple(rec.get("seed_fitness") or ()),
                genome=rec.get("genome"),
            )
    return view


def load_archive(run: Run, index: Index) -> ArchiveView:
    """The saved archive of a finished run, or one rebuilt from the generation files."""
    saved = _saved_archive(run) if run.complete else None
    return saved if saved is not None else rebuild_archive(run, index)


# ---------------------------------------------------------------------------
# Per-generation statistics
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    vals = [v for v in values if isinstance(v, int | float) and math.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def is_success(rec: dict[str, Any]) -> bool:
    m = rec.get("metrics") or {}
    return _eligible(rec) and float(m.get("forward_displacement", 0.0)) >= SUCCESS_DISTANCE


def _step_row(step: int, recs: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in recs if r.get("valid") and r.get("metrics")]
    metric = lambda k: [float(r["metrics"].get(k, float("nan"))) for r in valid]  # noqa: E731
    return {
        "generation": step,
        "evaluated": len(recs),
        "valid": len(valid),
        "fell": sum(1 for r in valid if float(r["metrics"].get("fell", 0.0)) > 0),
        "successful": sum(1 for r in recs if is_success(r)),
        "mean_fitness_evaluated": _mean([float(r["fitness"]) for r in valid]),
        "mean_forward_displacement": _mean(metric("forward_displacement")),
        "mean_energy_efficiency": _mean(metric("energy_efficiency")),
        "mean_sustained_velocity": _mean(metric("sustained_velocity")),
    }


def generation_stats(run: Run, index: Index) -> pd.DataFrame:
    """One row per step over the individuals evaluated in it (see module doc)."""
    rows = []
    for step in sorted(index.by_step):
        recs = [index.records[g] for g in index.by_step[step]]
        rows.append(_step_row(step, recs))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if not run.metrics.empty and "sims" in run.metrics.columns:
        sims = run.metrics[["generation", "sims"]].drop_duplicates("generation")
        df = df.merge(sims, on="generation", how="left")
    return df


def fitness_samples(run: Run, index: Index) -> pd.DataFrame:
    """(generation, fitness, id) of every valid evaluated individual."""
    rows = []
    for step in sorted(index.by_step):
        for gid in index.by_step[step]:
            r = index.records[gid]
            if r.get("valid"):
                rows.append({"generation": step, "fitness": float(r["fitness"]), "id": gid})
    return pd.DataFrame(rows, columns=["generation", "fitness", "id"])


def body_plan_of(rec: dict[str, Any]) -> str:
    m = (rec.get("genome") or {}).get("morphology") or {}
    try:
        return f"{2 * int(m['n_pairs'])}x{int(m['n_segments'])}"
    except (KeyError, TypeError, ValueError):
        return "unknown"


def plan_counts(run: Run, index: Index) -> pd.DataFrame:
    """(generation, plan, count) over the whole recorded population of each step.

    Uses every record of the generation file (for the GA that includes the
    carried elites), i.e. who is alive in that step, which is what body-plan
    diversity is about.
    """
    rows = []
    for step in sorted(index.by_step):
        records = load_generation(run, step)
        for plan, n in sorted(Counter(body_plan_of(r) for r in records).items()):
            rows.append({"generation": step, "plan": plan, "count": n})
    return pd.DataFrame(rows, columns=["generation", "plan", "count"])


def step_records(run: Run, step: int) -> list[dict[str, Any]]:
    """Every record of one step, sorted best first (invalid last)."""
    recs = load_generation(run, step)
    return sorted(recs, key=lambda r: (not r.get("valid"), -float(r.get("fitness", -1e9))))


def step_best(run: Run, step: int) -> dict[str, Any] | None:
    """The best valid individual evaluated in ``step`` (not the all-time champion)."""
    recs = [r for r in evaluated_records(run, load_generation(run, step)) if r.get("valid")]
    return max(recs, key=lambda r: float(r["fitness"])) if recs else None
