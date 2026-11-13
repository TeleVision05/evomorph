"""One evolution run end to end: run directory, algorithm, final checks, summary.

:func:`run_evolution` is what ``python -m evomorph evolve`` calls, and what an
experiment calls with its own evaluator (any picklable ``Genome -> EvalResult``
callable plus its cost in simulations). After the search it

* saves the champion, the final population or archive, and the archive;
* re-scores the champion with the same evaluator and checks the numbers are
  identical (the reproducibility contract; costs one evaluation, outside the
  budget);
* scores the champion on held-out seeds (same environments, seeds + 1000) as
  a winner's-curse check, also outside the budget;
* writes ``summary.md``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from evomorph.evolution.archive import Archive, qd_offset_for
from evomorph.evolution.evalspec import EvalSpec, Evaluator, SpecEvaluator, sims_per_call
from evomorph.evolution.genetic_algorithm import GeneticAlgorithm
from evomorph.evolution.map_elites import MapElites, elite_result
from evomorph.evolution.modes import Freezer, load_template
from evomorph.evolution.operators import Variation
from evomorph.evolution.pool import BatchRunner, clamp_workers
from evomorph.evolution.progress import GenerationReport, Individual
from evomorph.evolution.rundir import RunWriter, create_run_dir
from evomorph.evolution.settings import RunConfig, to_mapping
from evomorph.evolution.summary import write_summary

HELDOUT_SEED_OFFSET = 1000


@dataclass
class RunOutcome:
    path: Path
    champion: Individual | None
    archive: Archive
    sims_used: int
    steps: int
    wall_seconds: float
    reproduced: bool | None
    heldout: Individual | None
    final: list[Individual]


class ProgressPrinter:
    """One line per generation / batch."""

    def __init__(self, budget: int, out: Callable[[str], None] = print) -> None:
        self.budget, self.out = budget, out

    def on_generation(self, rep: GenerationReport) -> None:
        r = rep.row
        label = "gen" if rep.algorithm == "ga" else "batch"
        self.out(
            f"{label} {rep.index:4d}  sims {rep.sims_used:6d}/{self.budget}  "
            f"best {r['best_fitness']:+.3f}  mean {r['mean_fitness']:+.3f}  "
            f"valid {r['n_valid']}/{r['n_individuals']}  fell {r['n_fell']}  "
            f"plans {r['n_body_plans']}  dist {r['mean_gene_distance']:.3f}  "
            f"cov {r['archive_coverage']:.0%}  qd {r['qd_score']:.1f}  "
            f"{rep.wall_seconds:.1f}s"
        )


class _Fanout:
    def __init__(self, observers: Sequence) -> None:
        self.observers = [o for o in observers if o is not None]

    def on_generation(self, rep: GenerationReport) -> None:
        for o in self.observers:
            o.on_generation(rep)


def heldout_spec(spec: EvalSpec) -> EvalSpec:
    """The same environments with fresh seeds (seed + HELDOUT_SEED_OFFSET)."""
    runs = tuple((env, seed + HELDOUT_SEED_OFFSET) for env, seed in spec.runs)
    return replace(spec, runs=runs)


def build_variation(cfg: RunConfig, crossover_prob: float) -> tuple[Variation, Freezer]:
    evo = cfg.evolution
    template = None
    if evo.mode == "controller":
        template = load_template(evo.frozen_body)
    elif evo.mode == "morphology":
        template = load_template(evo.frozen_controller)
    freezer = Freezer(evo.mode, template, evo.kind)
    return Variation(cfg.ga.mutation, freezer, crossover_prob), freezer


def _search(cfg: RunConfig, evaluator: Evaluator, cost: int, observer, workers: int):
    evo = cfg.evolution
    offset = qd_offset_for(cfg.fitness)
    with BatchRunner(evaluator, workers) as runner:
        if evo.algorithm == "ga":
            variation, freezer = build_variation(cfg, cfg.ga.crossover_prob)
            algo = GeneticAlgorithm(
                cfg.ga, variation, runner, cost, evo.budget, evo.seed,
                cfg.map_elites.archive, offset, observer,
            )  # fmt: skip
            res = algo.run()
            return (
                res.champion,
                res.archive,
                res.sims_used,
                res.generations,
                res.population,
                freezer,
            )
        variation, freezer = build_variation(cfg, cfg.map_elites.crossover_prob)
        algo = MapElites(
            cfg.map_elites, variation, runner, cost, evo.budget, evo.seed, offset, observer
        )
        res = algo.run()
        final = [Individual(e.genome, elite_result(e), "elite") for e in res.archive.elites()]
        return res.champion, res.archive, res.sims_used, res.batches, final, freezer


def _check_champion(champion: Individual, evaluator: Evaluator, cfg: RunConfig):
    again = evaluator(champion.genome)
    same = again.fitness == champion.result.fitness and again.metrics == champion.result.metrics
    heldout = None
    if isinstance(evaluator, SpecEvaluator):
        held = SpecEvaluator(heldout_spec(evaluator.spec))
        heldout = Individual(champion.genome, held(champion.genome), "heldout")
    return same, heldout


def run_evolution(
    cfg: RunConfig,
    results_root: str | Path = "results",
    name: str | None = None,
    evaluator: Evaluator | None = None,
    sims_per_eval: int | None = None,
    out: Callable[[str], None] | None = print,
    command: str = "",
) -> RunOutcome:
    """Run one evolution into a fresh run directory (see the module doc)."""
    evaluator = evaluator or SpecEvaluator(cfg.eval_spec)
    cost = sims_per_call(evaluator, sims_per_eval)
    workers = clamp_workers(cfg.evolution.workers)
    path = create_run_dir(results_root, name)
    writer = RunWriter(path)
    writer.write_config(to_mapping(cfg))
    printer = ProgressPrinter(cfg.evolution.budget, out) if out is not None else None
    if out is not None:
        out(f"run directory: {path}  ({workers} workers, {cost} simulations per genome)")
    t0 = time.perf_counter()
    champion, archive, used, steps, final, freezer = _search(
        cfg, evaluator, cost, _Fanout([writer, printer]), workers
    )
    wall = time.perf_counter() - t0
    if freezer.template is not None:
        writer.save_template(freezer.template)
    writer.save_final([i.genome for i in final])
    archive.save(path / "archive")
    reproduced, heldout = None, None
    if champion is not None:
        writer.save_best(champion)
        reproduced, heldout = _check_champion(champion, evaluator, cfg)
    outcome = RunOutcome(path, champion, archive, used, steps, wall, reproduced, heldout, final)
    write_summary(writer, cfg, outcome, workers, cost, command)
    if out is not None:
        out(f"done: {used} simulations in {wall:.1f}s; summary at {path / 'summary.md'}")
    return outcome
