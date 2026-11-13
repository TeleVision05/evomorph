"""Two-dimensional MAP-Elites archive over behaviour descriptors.

Default descriptors are ``sustained_velocity`` (m/s; the core's recommended
speed axis: the mean of the 3 slowest of 5 windows, so a lunge does not count)
and ``energy_efficiency`` (``1 / (1 + cost_of_transport)`` when moving forward,
0 otherwise). Both come from the multi-run aggregated result, so a cell is won
by a creature's typical behaviour, not by one lucky seed.

Default ranges, chosen from a measured probe (flat ground, friction 0.6 and 1.0,
seed 0, 202 genomes = 200 random + the two references, 404 simulations): valid
non-fallen random creatures reached sustained velocity p99 0.40 and max
0.64 m/s and efficiency max 0.55; the hand-designed references sit at
0.37-0.43 m/s and 0.44-0.46. Speed spans [0, 0.8] m/s so the references land
mid-range with room above for evolution; efficiency spans [0, 0.7], above the
best measured value. Values outside a range go to the edge cell (backward and
stationary creatures share the slowest column).

Insertion rules:

* only valid results that never fell (``metrics['fell'] == 0`` in every run)
  are eligible: a creature that falls in some runs is not a reliable elite;
* an empty cell takes the candidate (``"new"``);
* an occupied cell is replaced only by a strictly higher fitness
  (``"improved"``); ties keep the incumbent, so the result does not depend on
  anything but evaluation order, which is fixed.

QD-score is ``sum over filled cells of (fitness + qd_offset)``. The offset is
the magnitude of the lowest valid fitness (``penalty_cap + w_fall + w_still``,
3.2 with the default fitness config), so every filled cell adds a
non-negative amount and filling a cell never lowers the score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from evomorph.config import FitnessConfig
from evomorph.morphology.genome import Genome, load_genome, save_genome
from evomorph.simulation.evaluator import EvalResult
from evomorph.utils.serialization import dump_json, load_json

INSERT_STATUSES = ("new", "improved", "rejected", "fell", "invalid")


@dataclass(frozen=True)
class ArchiveSpec:
    """Descriptor names, bins per axis and value ranges (see module doc)."""

    descriptors: tuple[str, str] = ("sustained_velocity", "energy_efficiency")
    bins: tuple[int, int] = (10, 10)
    ranges: tuple[tuple[float, float], tuple[float, float]] = ((0.0, 0.8), (0.0, 0.7))

    def __post_init__(self) -> None:
        object.__setattr__(self, "descriptors", tuple(str(d) for d in self.descriptors))
        object.__setattr__(self, "bins", tuple(int(b) for b in self.bins))
        object.__setattr__(self, "ranges", tuple(tuple(map(float, r)) for r in self.ranges))
        if not len(self.descriptors) == len(self.bins) == len(self.ranges) == 2:
            raise ValueError("the archive is two-dimensional: give 2 descriptors, bins, ranges")
        if min(self.bins) < 1 or any(hi <= lo for lo, hi in self.ranges):
            raise ValueError("archive bins must be >= 1 and each range must have high > low")

    @property
    def n_cells(self) -> int:
        return self.bins[0] * self.bins[1]

    def descriptor(self, metrics: dict[str, float]) -> tuple[float, float]:
        return (float(metrics[self.descriptors[0]]), float(metrics[self.descriptors[1]]))

    def normalise(self, desc: tuple[float, float]) -> np.ndarray:
        """Descriptor scaled to [0, 1] per axis (clipped)."""
        out = [(v - lo) / (hi - lo) for v, (lo, hi) in zip(desc, self.ranges, strict=True)]
        return np.clip(np.asarray(out), 0.0, 1.0)

    def cell(self, desc: tuple[float, float]) -> tuple[int, int]:
        """Bin index per axis; out-of-range values go to the edge bins."""
        u = self.normalise(desc)
        idx = [min(int(x * b), b - 1) for x, b in zip(u, self.bins, strict=True)]
        return (idx[0], idx[1])


def qd_offset_for(fitness: FitnessConfig) -> float:
    """Magnitude of the lowest valid fitness (see module doc)."""
    return float(fitness.penalty_cap + fitness.w_fall + fitness.w_still)


@dataclass
class Elite:
    """One archive cell's occupant and the numbers it was admitted with."""

    genome: Genome
    fitness: float
    descriptor: tuple[float, float]
    metrics: dict[str, float]
    seed_fitness: tuple[float, ...] = ()
    inserted_at: int = 0  # simulations used when it entered the archive


@dataclass
class Archive:
    """Cells keyed by ``(i, j)``; see the module doc for the insertion rules."""

    spec: ArchiveSpec = field(default_factory=ArchiveSpec)
    qd_offset: float = 3.2
    cells: dict[tuple[int, int], Elite] = field(default_factory=dict)

    def add(self, genome: Genome, result: EvalResult, sims_used: int = 0) -> str:
        """Try to insert; returns one of ``INSERT_STATUSES``."""
        if not result.valid or not result.metrics:
            return "invalid"
        if result.metrics.get("fell", 0.0) > 0.0:
            return "fell"
        desc = self.spec.descriptor(result.metrics)
        key = self.spec.cell(desc)
        incumbent = self.cells.get(key)
        if incumbent is not None and not result.fitness > incumbent.fitness:
            return "rejected"
        self.cells[key] = Elite(
            genome=genome,
            fitness=float(result.fitness),
            descriptor=desc,
            metrics=dict(result.metrics),
            seed_fitness=tuple(float(x) for x in result.seed_fitness),
            inserted_at=int(sims_used),
        )
        return "new" if incumbent is None else "improved"

    def get(self, cell: tuple[int, int]) -> Elite | None:
        return self.cells.get(tuple(cell))

    def elites(self) -> list[Elite]:
        """Occupants in cell order (deterministic)."""
        return [self.cells[k] for k in sorted(self.cells)]

    def occupied(self) -> list[tuple[int, int]]:
        return sorted(self.cells)

    def __len__(self) -> int:
        return len(self.cells)

    @property
    def coverage(self) -> float:
        return len(self.cells) / self.spec.n_cells

    @property
    def qd_score(self) -> float:
        return float(sum(e.fitness + self.qd_offset for e in self.cells.values()))

    def best(self) -> Elite | None:
        """Highest-fitness occupant (lowest cell index on ties)."""
        es = self.elites()
        return max(es, key=lambda e: e.fitness) if es else None

    def fitness_grid(self) -> np.ndarray:
        """(bins0, bins1) array of occupant fitness, NaN where empty."""
        grid = np.full(self.spec.bins, np.nan)
        for (i, j), e in self.cells.items():
            grid[i, j] = e.fitness
        return grid

    # -- persistence ---------------------------------------------------------

    def save(self, folder: str | Path) -> Path:
        """Write ``archive.json`` plus one genome file per elite under ``elites/``."""
        out = Path(folder)
        rows = []
        for key in self.occupied():
            e = self.cells[key]
            name = f"cell_{key[0]:02d}_{key[1]:02d}.json"
            save_genome(e.genome, out / "elites" / name)
            rows.append(
                {
                    "cell": list(key),
                    "genome_file": f"elites/{name}",
                    "genome_id": e.genome.id,
                    "fitness": e.fitness,
                    "descriptor": list(e.descriptor),
                    "seed_fitness": list(e.seed_fitness),
                    "inserted_at": e.inserted_at,
                    "metrics": e.metrics,
                }
            )
        meta: dict[str, Any] = {
            "spec": asdict(self.spec),
            "qd_offset": self.qd_offset,
            "coverage": self.coverage,
            "qd_score": self.qd_score,
            "cells": rows,
        }
        return dump_json(meta, out / "archive.json")

    @classmethod
    def load(cls, folder: str | Path) -> Archive:
        """Inverse of :meth:`save`."""
        root = Path(folder)
        meta = load_json(root / "archive.json")
        s = meta["spec"]
        spec = ArchiveSpec(tuple(s["descriptors"]), tuple(s["bins"]), tuple(s["ranges"]))
        arch = cls(spec=spec, qd_offset=float(meta["qd_offset"]))
        for row in meta["cells"]:
            arch.cells[tuple(row["cell"])] = Elite(
                genome=load_genome(root / row["genome_file"]),
                fitness=float(row["fitness"]),
                descriptor=tuple(row["descriptor"]),
                metrics={k: float(v) for k, v in row["metrics"].items()},
                seed_fitness=tuple(row["seed_fitness"]),
                inserted_at=int(row["inserted_at"]),
            )
        return arch
