"""What both algorithms report after every generation (GA) or batch (MAP-Elites).

One :class:`GenerationReport` per step goes to an observer (the run-directory
writer, the progress printer, a test). ``report.row`` is one line of
``metrics.csv``; its columns are ``METRICS_COLUMNS``. Wall time is deliberately
*not* in the row: it lives in ``timing.csv`` so that ``metrics.csv`` is
byte-identical across worker counts (the determinism contract).

Column semantics
----------------
``generation``           step index (GA generation or MAP-Elites batch)
``sims``                 simulations used so far (genomes x runs), the budget unit
``evaluations``          genomes evaluated in this step
``best/mean/median/std/p10/p90_fitness``
                         over the *current set*: the GA population, or the
                         MAP-Elites archive elites; valid individuals only
``n_individuals``        size of that set; ``n_valid`` and ``n_fell`` (valid
                         ones that fell in at least one run) count within it
``mean_gene_distance``, ``n_body_plans``, ``behaviour_spread``
                         diversity of the current set (see selection.py)
``archive_cells``, ``archive_coverage``, ``qd_score``
                         the archive: MAP-Elites' own, or for the GA a passive
                         archive of every genome it has evaluated, so both
                         algorithms can be compared on quality-diversity
``champion_*``           best valid genome found so far (all-time)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from evomorph.evolution.archive import Archive
from evomorph.evolution.selection import (
    behaviour_spread,
    body_plan,
    gene_distance_matrix,
    mean_pairwise,
)
from evomorph.morphology.genome import Genome
from evomorph.simulation.evaluator import EvalResult

METRICS_COLUMNS: tuple[str, ...] = (
    "generation",
    "sims",
    "evaluations",
    "best_fitness",
    "mean_fitness",
    "median_fitness",
    "std_fitness",
    "p10_fitness",
    "p90_fitness",
    "n_individuals",
    "n_valid",
    "n_fell",
    "mean_gene_distance",
    "n_body_plans",
    "behaviour_spread",
    "archive_cells",
    "archive_coverage",
    "qd_score",
    "champion_id",
    "champion_fitness",
    "champion_plan",
    "champion_speed",
    "champion_efficiency",
)


@dataclass
class Individual:
    """A genome with its (aggregated) evaluation and what happened to it."""

    genome: Genome
    result: EvalResult
    status: str = ""  # GA: "new" or "elite"; MAP-Elites: the archive insert status

    @property
    def fitness(self) -> float:
        return float(self.result.fitness)


@dataclass
class GenerationReport:
    """Everything an observer may want after one generation or batch."""

    index: int
    sims_used: int
    evaluated: list[Individual]  # genomes evaluated in this step
    current: list[Individual]  # GA population or archive elites after this step
    champion: Individual | None
    archive: Archive
    row: dict[str, object] = field(default_factory=dict)
    wall_seconds: float = 0.0  # this step only
    algorithm: str = "ga"


class Observer(Protocol):
    def on_generation(self, report: GenerationReport) -> None: ...


def best_of(individuals: Sequence[Individual]) -> Individual | None:
    """Highest-fitness valid individual (first one on ties), or None."""
    best = None
    for ind in individuals:
        if ind.result.valid and (best is None or ind.fitness > best.fitness):
            best = ind
    return best


def _fitness_stats(valid: Sequence[Individual]) -> dict[str, float]:
    if not valid:
        return dict.fromkeys(
            ("best_fitness", "mean_fitness", "median_fitness", "std_fitness"), float("nan")
        ) | {"p10_fitness": float("nan"), "p90_fitness": float("nan")}
    f = np.array([i.fitness for i in valid])
    p10, p90 = np.percentile(f, [10, 90])
    return {
        "best_fitness": float(f.max()),
        "mean_fitness": float(f.mean()),
        "median_fitness": float(np.median(f)),
        "std_fitness": float(f.std()),
        "p10_fitness": float(p10),
        "p90_fitness": float(p90),
    }


def diversity_measures(
    current: Sequence[Individual], genes: Sequence[str], archive: Archive
) -> dict[str, float]:
    """Gene distance, body-plan count and behaviour spread of ``current``."""
    genomes = [i.genome for i in current]
    dist = gene_distance_matrix(genomes, genes)
    valid = [i for i in current if i.result.valid and i.result.metrics]
    pts = np.array(
        [archive.spec.normalise(archive.spec.descriptor(i.result.metrics)) for i in valid]
    )
    return {
        "mean_gene_distance": mean_pairwise(dist),
        "n_body_plans": len({body_plan(g) for g in genomes}),
        "behaviour_spread": behaviour_spread(pts) if len(pts) else 0.0,
    }


def make_row(
    index: int,
    sims_used: int,
    n_evaluated: int,
    current: Sequence[Individual],
    champion: Individual | None,
    archive: Archive,
    genes: Sequence[str],
) -> dict[str, object]:
    """One ``metrics.csv`` row (see the module doc for the columns)."""
    valid = [i for i in current if i.result.valid]
    row: dict[str, object] = {
        "generation": index,
        "sims": sims_used,
        "evaluations": n_evaluated,
        **_fitness_stats(valid),
        "n_individuals": len(current),
        "n_valid": len(valid),
        "n_fell": sum(1 for i in valid if i.result.metrics.get("fell", 0.0) > 0),
        **diversity_measures(current, genes, archive),
        "archive_cells": len(archive),
        "archive_coverage": archive.coverage,
        "qd_score": archive.qd_score,
    }
    if champion is None:
        row |= {"champion_id": "", "champion_fitness": float("nan"), "champion_plan": ""}
        row |= {"champion_speed": float("nan"), "champion_efficiency": float("nan")}
    else:
        speed, eff = archive.spec.descriptor(champion.result.metrics)
        row |= {
            "champion_id": champion.genome.id,
            "champion_fitness": champion.fitness,
            "champion_plan": body_plan(champion.genome),
            "champion_speed": speed,
            "champion_efficiency": eff,
        }
    return row
