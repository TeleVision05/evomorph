"""Generational genetic algorithm with elitism, tournaments and diversity.

One generation (brief section 5):

1. score the population (random at generation 0) through the injected
   evaluator, in a batch over the process pool;
2. compute selection scores (rank, shared by niche count when diversity is on);
3. keep the elites unchanged: the global top ``elite_count`` plus, with
   diversity on, the best of every body plan (they are *not* re-evaluated:
   evaluation is deterministic, so their stored multi-run score is exact);
4. fill the rest with children (two tournament winners, crossover with
   ``crossover_prob``, then mutation) and a few random immigrants.

Budget: every evaluation costs ``sims_per_eval`` simulations. The run stops
when the next child would not fit. If a generation can only partly fit, the
children that fit are evaluated and the population is topped up with the best
already-scored non-elites of the previous generation, so the whole budget is
spent and no individual is ever unscored. ``max_generations`` optionally caps
the run earlier.

Determinism: every random draw happens here, in the parent process, from one
generator seeded with ``seed``; evaluations return in input order.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np

from evomorph.evolution.archive import Archive, ArchiveSpec
from evomorph.evolution.operators import MutationConfig, Variation
from evomorph.evolution.pool import BatchRunner
from evomorph.evolution.progress import (
    GenerationReport,
    Individual,
    Observer,
    best_of,
    make_row,
)
from evomorph.evolution.selection import (
    DiversityConfig,
    body_plan,
    choose_elites,
    gene_distance_matrix,
    rank_order,
    selection_scores,
    tournament,
)
from evomorph.utils.random_state import make_rng


@dataclass(frozen=True)
class GAConfig:
    """GA settings. Defaults are the quick laptop configuration's.

    48 individuals is 8 batches of 6 workers, and enough to hold several body
    plans at once; 2 global elites keep the champion without freezing the
    population; tournaments of 3 are mild pressure (the best of 48 wins ~6% of
    tournaments), which matters when most of generation 0 cannot walk.
    """

    population_size: int = 48
    elite_count: int = 2
    tournament_size: int = 3
    crossover_prob: float = 0.5
    max_generations: int | None = None  # None: run until the budget is spent
    mutation: MutationConfig = field(default_factory=MutationConfig)
    diversity: DiversityConfig = field(default_factory=DiversityConfig)

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if not 0 <= self.elite_count < self.population_size:
            raise ValueError("elite_count must be between 0 and population_size - 1")
        if self.tournament_size < 1:
            raise ValueError("tournament_size must be at least 1")
        if not 0.0 <= self.crossover_prob <= 1.0:
            raise ValueError("crossover_prob is a probability")


@dataclass
class GAResult:
    population: list[Individual]
    champion: Individual | None
    archive: Archive
    sims_used: int
    generations: int  # generations evaluated, including generation 0


def gene_id(generation: int, k: int) -> str:
    return f"g{generation:04d}-{k:03d}"


@dataclass
class _Breed:
    elites: list[Individual]
    children: list  # genomes to evaluate
    fill: list[Individual]


def _breed(
    pop: list[Individual],
    dist: np.ndarray,
    rng: np.random.Generator,
    cfg: GAConfig,
    variation: Variation,
    gen: int,
    max_new: int,
) -> _Breed:
    """Choose elites and make up to ``max_new`` children for generation ``gen``."""
    fitness = np.array([i.fitness for i in pop])
    valid = np.array([i.result.valid for i in pop])
    plans = [body_plan(i.genome) for i in pop]
    elite_idx = choose_elites(
        fitness, valid, plans, cfg.elite_count, cfg.diversity.effective_plan_elites
    )[: cfg.population_size // 2]  # never let protection crowd out reproduction
    scores = selection_scores(fitness, dist, cfg.diversity)
    n_new = cfg.population_size - len(elite_idx)
    n_make = min(n_new, max_new)
    n_imm = min(round(cfg.diversity.effective_immigrants * cfg.population_size), n_make)
    children = []
    for k in range(n_make - n_imm):
        a = tournament(scores, rng, cfg.tournament_size)
        b = tournament(scores, rng, cfg.tournament_size)
        children.append(variation.child(rng, pop[a].genome, pop[b].genome, gene_id(gen, k), gen))
    for k in range(n_make - n_imm, n_make):
        g = variation.random(rng, gene_id(gen, k), gen)
        children.append(replace(g, origin="immigrant"))
    rest = [int(i) for i in rank_order(fitness) if int(i) not in elite_idx]
    fill = [pop[i] for i in rest[: n_new - n_make]]
    return _Breed([pop[i] for i in elite_idx], children, fill)


class GeneticAlgorithm:
    """Runs the GA; see the module doc. ``run`` returns a :class:`GAResult`."""

    def __init__(
        self,
        cfg: GAConfig,
        variation: Variation,
        runner: BatchRunner,
        sims_per_eval: int,
        budget: int,
        seed: int,
        archive_spec: ArchiveSpec | None = None,
        qd_offset: float = 3.2,
        observer: Observer | None = None,
    ) -> None:
        self.cfg, self.variation, self.runner = cfg, variation, runner
        self.cost, self.budget, self.seed = int(sims_per_eval), int(budget), int(seed)
        self.archive = Archive(archive_spec or ArchiveSpec(), qd_offset)
        self.observer = observer
        self.used = 0
        self.champion: Individual | None = None

    def _evaluate(self, genomes: list) -> list[Individual]:
        results = self.runner.map(genomes)
        self.used += self.cost * len(genomes)
        out = [Individual(g, r, "new") for g, r in zip(genomes, results, strict=True)]
        for ind in out:
            self.archive.add(ind.genome, ind.result, self.used)
        best = best_of(out)
        if best is not None and (self.champion is None or best.fitness > self.champion.fitness):
            self.champion = best
        return out

    def _report(self, gen: int, new: list, pop: list, dist: np.ndarray, t0: float) -> None:
        if self.observer is None:
            return
        genes = self.variation.genes
        row = make_row(gen, self.used, len(new), pop, self.champion, self.archive, genes)
        rep = GenerationReport(
            gen, self.used, new, pop, self.champion, self.archive, row, time.perf_counter() - t0
        )
        self.observer.on_generation(rep)

    def run(self) -> GAResult:
        rng = make_rng(self.seed)
        n0 = min(self.cfg.population_size, self.budget // self.cost)
        if n0 < 2:
            raise ValueError(
                f"a budget of {self.budget} simulations cannot score even two genomes at "
                f"{self.cost} simulations each; raise the budget"
            )
        t0 = time.perf_counter()
        pop = self._evaluate([self.variation.random(rng, gene_id(0, k), 0) for k in range(n0)])
        dist = gene_distance_matrix([i.genome for i in pop], self.variation.genes)
        self._report(0, pop, pop, dist, t0)
        gen = 0
        while self.cfg.max_generations is None or gen + 1 < self.cfg.max_generations:
            max_new = (self.budget - self.used) // self.cost
            if max_new < 1:
                break
            gen += 1
            t0 = time.perf_counter()
            bred = _breed(pop, dist, rng, self.cfg, self.variation, gen, max_new)
            kids = self._evaluate(bred.children)
            elites = [replace(i, status="elite") for i in bred.elites]
            fill = [replace(i, status="carried") for i in bred.fill]
            pop = elites + kids + fill
            dist = gene_distance_matrix([i.genome for i in pop], self.variation.genes)
            self._report(gen, kids, pop, dist, t0)
        return GAResult(pop, self.champion, self.archive, self.used, gen + 1)
