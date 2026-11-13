"""MAP-Elites (Mouret & Clune 2015) over a 2-D behaviour archive.

1. Initialisation: random genomes, in batches, until ``init_size`` have been
   evaluated (and for as long as the archive stays empty, since there is no
   elite to vary yet).
2. Then every batch: draw a parent uniformly from the archive's elites, with
   ``crossover_prob`` cross it with a second uniformly drawn elite, mutate, and
   evaluate; each child is offered to the archive (see archive.py for the
   rules). Uniform parent choice over *cells* (not individuals) is what gives
   rare behaviours as many offspring as common ones.

Descriptors and quality come from the injected evaluator's aggregated result
(several runs), never from a single run. Budget, determinism and reporting
follow the GA (see genetic_algorithm.py): the last batch is truncated to the
remaining budget, all random draws are made here, and one report per batch
goes to the observer with the archive's elites as the "current" set.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from evomorph.evolution.archive import Archive, ArchiveSpec, Elite
from evomorph.evolution.operators import Variation
from evomorph.evolution.pool import BatchRunner
from evomorph.evolution.progress import (
    GenerationReport,
    Individual,
    Observer,
    best_of,
    make_row,
)
from evomorph.simulation.environment import EnvSpec
from evomorph.simulation.evaluator import EvalResult
from evomorph.utils.random_state import make_rng


@dataclass(frozen=True)
class MapElitesConfig:
    """MAP-Elites settings; archive geometry is in ``archive`` (see archive.py).

    ``batch_size = 48`` matches the GA population so a "generation" row means
    the same number of evaluations in both. ``init_size = 192`` (4 batches)
    is spent on random genomes: with ~7% of random creatures moving a metre,
    that seeds the archive with a dozen or so movers before variation starts.
    ``crossover_prob = 0.2``: mostly mutation, the classic MAP-Elites setting,
    with some crossover to recombine elites from different cells.
    """

    batch_size: int = 48
    init_size: int = 192
    crossover_prob: float = 0.2
    archive: ArchiveSpec = field(default_factory=ArchiveSpec)

    def __post_init__(self) -> None:
        if self.batch_size < 1 or self.init_size < 1:
            raise ValueError("map_elites.batch_size and init_size must be at least 1")
        if not 0.0 <= self.crossover_prob <= 1.0:
            raise ValueError("map_elites.crossover_prob is a probability")


@dataclass
class MapElitesResult:
    archive: Archive
    champion: Individual | None
    sims_used: int
    batches: int


def batch_id(batch: int, k: int) -> str:
    return f"b{batch:04d}-{k:03d}"


class MapElites:
    """Runs MAP-Elites; see the module doc."""

    def __init__(
        self,
        cfg: MapElitesConfig,
        variation: Variation,
        runner: BatchRunner,
        sims_per_eval: int,
        budget: int,
        seed: int,
        qd_offset: float = 3.2,
        observer: Observer | None = None,
    ) -> None:
        self.cfg, self.variation, self.runner = cfg, variation, runner
        self.cost, self.budget, self.seed = int(sims_per_eval), int(budget), int(seed)
        self.archive = Archive(cfg.archive, qd_offset)
        self.observer = observer
        self.used, self.evaluated = 0, 0
        self.champion: Individual | None = None

    def _make_batch(self, rng, batch: int, n: int) -> list:
        if self.evaluated < self.cfg.init_size or len(self.archive) == 0:
            return [self.variation.random(rng, batch_id(batch, k), batch) for k in range(n)]
        elites = self.archive.elites()
        out = []
        for k in range(n):
            a = elites[int(rng.integers(len(elites)))].genome
            b = elites[int(rng.integers(len(elites)))].genome
            out.append(self.variation.child(rng, a, b, batch_id(batch, k), batch))
        return out

    def _step(self, rng, batch: int, n: int) -> list[Individual]:
        genomes = self._make_batch(rng, batch, n)
        results = self.runner.map(genomes)
        out = []
        for g, r in zip(genomes, results, strict=True):
            self.used += self.cost
            out.append(Individual(g, r, self.archive.add(g, r, self.used)))
        self.evaluated += len(genomes)
        best = best_of([i for i in out if i.status in ("new", "improved")])
        if best is not None and (self.champion is None or best.fitness > self.champion.fitness):
            self.champion = best
        return out

    def run(self) -> MapElitesResult:
        rng = make_rng(self.seed)
        if self.budget // self.cost < 1:
            raise ValueError(
                f"a budget of {self.budget} simulations cannot score one genome at "
                f"{self.cost} simulations each; raise the budget"
            )
        batch = -1
        while (n := min(self.cfg.batch_size, (self.budget - self.used) // self.cost)) >= 1:
            batch += 1
            t0 = time.perf_counter()
            new = self._step(rng, batch, n)
            self._report(batch, new, t0)
        return MapElitesResult(self.archive, self.champion, self.used, batch + 1)

    def _report(self, batch: int, new: list[Individual], t0: float) -> None:
        if self.observer is None:
            return
        current = [Individual(e.genome, elite_result(e), "elite") for e in self.archive.elites()]
        genes = self.variation.genes
        row = make_row(batch, self.used, len(new), current, self.champion, self.archive, genes)
        wall = time.perf_counter() - t0
        rep = GenerationReport(
            batch, self.used, new, current, self.champion, self.archive, row, wall, "map_elites"
        )
        self.observer.on_generation(rep)


def elite_result(elite: Elite) -> EvalResult:
    """A minimal valid EvalResult view of an archive elite (for statistics)."""
    return EvalResult(
        genome_id=elite.genome.id,
        valid=True,
        failure_reason="",
        fitness=elite.fitness,
        metrics=elite.metrics,
        fitness_terms={},
        env=EnvSpec(),
        seed=0,
        sim_seconds=0.0,
        wall_seconds=0.0,
        seed_fitness=elite.seed_fitness,
    )
