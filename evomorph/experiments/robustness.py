"""Evaluate one genome across a terrain suite: per-environment rows and aggregates.

``evaluate_suite`` is a top-level function with picklable arguments and result,
so an experiment can map it over many genomes in its own process pool (call it
with ``workers=1`` there; pools must not nest). With ``workers > 1`` it runs its
own pool, capped at ``MAX_WORKERS``.

Scoring rules
-------------
* Each (environment, spawn seed) pair is one ``evaluate`` run.
* A run is a **failure** if the creature fell or the run was invalid (a physics
  glitch or a world that could not be built). ``failure_rate`` counts both;
  ``invalid_rate`` reports the invalid share separately so a terrain artefact
  cannot hide inside the fall count.
* An invalid run scores ``INVALID_FITNESS``, exactly as ``evaluate`` and
  ``evaluate_robust`` score it for selection, so a robustness number means the
  same thing as the fitness evolution saw. Locomotion metrics (velocity,
  efficiency, slip) are means over valid runs only (NaN when there are none).
* ``mean_fitness`` is the mean of the per-environment means (every environment
  weighs the same whatever its seed count); ``worst_fitness`` is the lowest
  per-environment mean, the score on the terrain the creature handles worst.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from evomorph.config import FitnessConfig, SimConfig
from evomorph.morphology.genome import Genome
from evomorph.simulation.environment import EnvSpec
from evomorph.simulation.evaluator import EvalResult, evaluate
from evomorph.simulation.suites import get_suite
from evomorph.simulation.terrain import TerrainParams, resolve

MAX_WORKERS = 6  # house rule: no process pool larger than 6 on the shared laptop

# Per-environment columns, in table and CSV order.
ROW_COLUMNS: tuple[str, ...] = (
    "env",
    "kind",
    "slope_deg",
    "roughness",
    "n_obstacles",
    "friction",
    "friction_spread",
    "terrain_seed",
    "runs",
    "fell",
    "invalid",
    "fitness",
    "fitness_min",
    "forward_velocity",
    "energy_efficiency",
    "slip_fraction",
    "foot_slip_speed",
)
AGGREGATE_KEYS: tuple[str, ...] = (
    "mean_fitness",
    "worst_fitness",
    "failure_rate",
    "invalid_rate",
    "mean_forward_velocity",
    "mean_energy_efficiency",
    "mean_slip_fraction",
)


@dataclass(frozen=True)
class EnvRow:
    """One environment's results over the spawn seeds."""

    env: EnvSpec
    terrain: TerrainParams  # what was actually built (random draws resolved)
    seeds: tuple[int, ...]
    fitness: tuple[float, ...]  # per seed
    fell: tuple[bool, ...]
    valid: tuple[bool, ...]
    reasons: tuple[str, ...]  # failure_reason per seed ("" when the run completed)
    metrics: dict[str, float]  # means over valid runs

    @property
    def mean_fitness(self) -> float:
        return float(np.mean(self.fitness))

    def as_dict(self) -> dict[str, object]:
        """The ``ROW_COLUMNS`` values."""
        t = self.terrain
        return {
            "env": self.env.label(),
            "kind": t.kind,
            "slope_deg": t.slope_deg,
            "roughness": t.roughness,
            "n_obstacles": t.n_obstacles,
            "friction": t.friction,
            "friction_spread": t.friction_spread,
            "terrain_seed": t.seed,
            "runs": len(self.seeds),
            "fell": sum(self.fell),
            "invalid": sum(not v for v in self.valid),
            "fitness": self.mean_fitness,
            "fitness_min": float(min(self.fitness)),
            "forward_velocity": self.metrics["avg_forward_velocity"],
            "energy_efficiency": self.metrics["energy_efficiency"],
            "slip_fraction": self.metrics["slip_fraction"],
            "foot_slip_speed": self.metrics["foot_slip_speed"],
        }


@dataclass(frozen=True)
class RobustnessReport:
    """A genome's results on a suite."""

    genome_id: str
    suite: str
    seeds: tuple[int, ...]
    rows: tuple[EnvRow, ...]
    aggregates: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "genome_id": self.genome_id,
            "suite": self.suite,
            "seeds": list(self.seeds),
            "aggregates": self.aggregates,
            "rows": [
                {**r.as_dict(), "env_spec": asdict(r.env), "per_seed": _per_seed(r)}
                for r in self.rows
            ],
        }


def _per_seed(row: EnvRow) -> list[dict[str, object]]:
    return [
        {"seed": s, "fitness": f, "fell": fl, "valid": v, "reason": why}
        for s, f, fl, v, why in zip(
            row.seeds, row.fitness, row.fell, row.valid, row.reasons, strict=True
        )
    ]


_ROW_METRICS = ("avg_forward_velocity", "energy_efficiency", "slip_fraction", "foot_slip_speed")


def _row(env: EnvSpec, runs: Sequence[EvalResult]) -> EnvRow:
    valid = [r for r in runs if r.valid]
    metrics = {
        k: float(np.mean([r.metrics[k] for r in valid])) if valid else math.nan
        for k in _ROW_METRICS
    }
    try:
        terrain = resolve(env)
    except ValueError:  # an unbuildable spec: every run is already invalid and says why
        e = env
        terrain = TerrainParams(
            e.kind, e.slope_deg, e.roughness, e.n_obstacles, e.friction, e.friction_spread, e.seed
        )
    return EnvRow(
        env=env,
        terrain=terrain,
        seeds=tuple(r.seed for r in runs),
        fitness=tuple(float(r.fitness) for r in runs),
        fell=tuple(bool(r.valid and r.fell) for r in runs),
        valid=tuple(bool(r.valid) for r in runs),
        reasons=tuple(r.failure_reason for r in runs),
        metrics=metrics,
    )


def aggregate(rows: Sequence[EnvRow]) -> dict[str, float]:
    """Suite-level numbers (see the module doc for the rules)."""
    n_runs = sum(len(r.seeds) for r in rows)
    failed = sum(sum(f or not v for f, v in zip(r.fell, r.valid, strict=True)) for r in rows)
    invalid = sum(sum(not v for v in r.valid) for r in rows)
    env_means = [r.mean_fitness for r in rows]

    def mean_metric(key: str) -> float:
        vals = [r.metrics[key] for r in rows if not math.isnan(r.metrics[key])]
        return float(np.mean(vals)) if vals else math.nan

    return {
        "mean_fitness": float(np.mean(env_means)),
        "worst_fitness": float(np.min(env_means)),
        "failure_rate": failed / n_runs,
        "invalid_rate": invalid / n_runs,
        "mean_forward_velocity": mean_metric("avg_forward_velocity"),
        "mean_energy_efficiency": mean_metric("energy_efficiency"),
        "mean_slip_fraction": mean_metric("slip_fraction"),
    }


def _run_one(task: tuple[Genome, EnvSpec, SimConfig, FitnessConfig, int]) -> EvalResult:
    genome, env, sim, fitness, seed = task
    return evaluate(genome, env=env, sim=sim, fitness=fitness, seed=seed)


def evaluate_suite(
    genome: Genome,
    suite: str | Sequence[EnvSpec],
    sim: SimConfig | None = None,
    fitness: FitnessConfig | None = None,
    seeds: Sequence[int] = (0, 1, 2),
    workers: int = 1,
) -> RobustnessReport:
    """Evaluate ``genome`` on every environment of ``suite`` for every spawn seed.

    ``suite`` is a name from ``suites.SUITES`` or a sequence of EnvSpec (reported
    as suite "custom"). Deterministic in all arguments; ``workers`` only changes
    speed (results are collected in task order).
    """
    if not seeds:
        raise ValueError("evaluate_suite needs at least one spawn seed")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    name, envs = (suite, get_suite(suite)) if isinstance(suite, str) else ("custom", tuple(suite))
    if not envs:
        raise ValueError("the terrain suite is empty")
    sim, fitness = sim or SimConfig(), fitness or FitnessConfig()
    tasks = [(genome, env, sim, fitness, int(s)) for env in envs for s in seeds]
    n = min(workers, MAX_WORKERS, len(tasks))
    if n == 1:
        results = [_run_one(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=n) as pool:
            results = list(pool.map(_run_one, tasks))
    k = len(seeds)
    rows = tuple(_row(env, results[i * k : (i + 1) * k]) for i, env in enumerate(envs))
    gid = getattr(genome, "id", "unknown")
    return RobustnessReport(gid, name, tuple(int(s) for s in seeds), rows, aggregate(rows))


# ----------------------------------------------------------------------------
# Output: a fixed-width table for the terminal, CSV and JSON for the record
# ----------------------------------------------------------------------------

_TABLE = (  # (column, header, width, format)
    ("env", "environment", 30, "s"),
    ("fitness", "fitness", 8, "+.3f"),
    ("fitness_min", "worst", 7, "+.3f"),
    ("forward_velocity", "fwd m/s", 8, "+.3f"),
    ("energy_efficiency", "eff", 6, ".3f"),
    ("fell", "fell", 5, "s"),
    ("slip_fraction", "slip", 6, ".3f"),
    ("invalid", "invalid", 7, "d"),
)


def table_lines(report: RobustnessReport) -> list[str]:
    """Per-environment table plus the aggregates, for printing."""
    head = " ".join(f"{h:>{w}}" if i else f"{h:<{w}}" for i, (_, h, w, _) in enumerate(_TABLE))
    title = f"{report.genome_id} on suite '{report.suite}', spawn seeds {list(report.seeds)}"
    lines = [title, head, "-" * len(head)]
    for row in report.rows:
        d = row.as_dict()
        d["fell"] = f"{d['fell']}/{d['runs']}"
        cells = []
        for i, (col, _, w, fmt) in enumerate(_TABLE):
            text = format(d[col], fmt)
            cells.append(f"{text:>{w}}" if i else f"{text:<{w}}")
        lines.append(" ".join(cells))
    a = report.aggregates
    lines += [
        "-" * len(head),
        f"mean fitness {a['mean_fitness']:+.3f}   worst environment {a['worst_fitness']:+.3f}   "
        f"failure rate {a['failure_rate']:.0%} (invalid {a['invalid_rate']:.0%})",
        f"mean forward velocity {a['mean_forward_velocity']:+.3f} m/s   mean efficiency "
        f"{a['mean_energy_efficiency']:.3f}   mean slip {a['mean_slip_fraction']:.3f}",
    ]
    return lines


def write_csv(report: RobustnessReport, path: str | Path) -> Path:
    """One row per environment, ``ROW_COLUMNS`` order."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ROW_COLUMNS))
        writer.writeheader()
        for row in report.rows:
            writer.writerow(row.as_dict())
    return out
