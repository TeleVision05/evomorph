"""GA and MAP-Elites on a fast fake evaluator: budget, elitism, progress, modes."""

from __future__ import annotations

import numpy as np
import pytest
from evo_helpers import FakeEvaluator

from evomorph.evolution.archive import ArchiveSpec
from evomorph.evolution.genetic_algorithm import GAConfig, GeneticAlgorithm
from evomorph.evolution.map_elites import MapElites, MapElitesConfig
from evomorph.evolution.modes import Freezer
from evomorph.evolution.operators import MutationConfig, Variation
from evomorph.evolution.pool import BatchRunner
from evomorph.evolution.selection import DiversityConfig
from evomorph.morphology.genome import reference_quadruped


class Collect:
    def __init__(self) -> None:
        self.reports = []

    def on_generation(self, rep) -> None:
        self.reports.append(rep)


def _ga(budget: int, mode: str = "joint", cfg: GAConfig | None = None, seed: int = 0):
    ev = FakeEvaluator()
    template = None if mode == "joint" else reference_quadruped()
    var = Variation(MutationConfig(), Freezer(mode, template), 0.5)
    obs = Collect()
    algo = GeneticAlgorithm(
        cfg or GAConfig(population_size=12), var, BatchRunner(ev, 1), ev.n_sims, budget, seed,
        ArchiveSpec(), 3.2, obs,
    )  # fmt: skip
    return algo.run(), obs, ev


def test_ga_spends_exactly_the_budget_and_never_more() -> None:
    res, obs, ev = _ga(budget=12 * 3 * 5 + 7)  # 5 full generations + part of one + slack
    assert res.sims_used <= 12 * 3 * 5 + 7
    assert res.sims_used == 3 * ev.calls  # every evaluation was paid for
    assert (12 * 3 * 5 + 7) - res.sims_used < 3  # less than one genome left over
    assert [r.sims_used for r in obs.reports] == sorted(r.sims_used for r in obs.reports)
    assert all(len(r.current) == 12 for r in obs.reports)  # partial generation topped up


def test_ga_improves_and_keeps_its_elites() -> None:
    res, obs, _ = _ga(budget=12 * 3 * 25)
    best = [r.row["best_fitness"] for r in obs.reports]
    assert best == sorted(best)  # elitism: the best never gets worse
    assert best[-1] > best[0] + 0.3
    assert obs.reports[-1].row["mean_fitness"] > obs.reports[0].row["mean_fitness"]
    assert res.champion.fitness == max(best)
    for prev, cur in zip(obs.reports, obs.reports[1:], strict=False):
        prev_best = max(prev.current, key=lambda i: i.fitness).genome.id
        assert prev_best in {i.genome.id for i in cur.current}


def test_ga_children_record_lineage() -> None:
    _, obs, _ = _ga(budget=12 * 3 * 4)
    ids = {i.genome.id for r in obs.reports for i in r.evaluated}
    for rep in obs.reports[1:]:
        for ind in rep.evaluated:
            assert ind.genome.generation == rep.index
            if ind.genome.origin in ("mutation", "crossover"):
                assert ind.genome.parent_ids and set(ind.genome.parent_ids) <= ids


def test_max_generations_caps_the_run() -> None:
    res, obs, _ = _ga(budget=10_000, cfg=GAConfig(population_size=8, max_generations=3))
    assert res.generations == 3 and len(obs.reports) == 3


def test_budget_too_small_is_a_plain_error() -> None:
    with pytest.raises(ValueError, match="raise the budget"):
        _ga(budget=4)


def test_controller_mode_ga_never_changes_the_body() -> None:
    _, obs, _ = _ga(budget=12 * 3 * 6, mode="controller")
    body = reference_quadruped().morphology
    assert all(i.genome.morphology == body for r in obs.reports for i in r.evaluated)
    assert all(r.row["n_body_plans"] == 1 for r in obs.reports)


def test_diversity_mechanism_keeps_more_variety_than_none() -> None:
    on = GAConfig(population_size=16, diversity=DiversityConfig())
    off = GAConfig(population_size=16, diversity=DiversityConfig(method="none"))
    _, obs_on, _ = _ga(budget=16 * 3 * 30, cfg=on)
    _, obs_off, _ = _ga(budget=16 * 3 * 30, cfg=off)
    late_on = np.mean([r.row["mean_gene_distance"] for r in obs_on.reports[-10:]])
    late_off = np.mean([r.row["mean_gene_distance"] for r in obs_off.reports[-10:]])
    assert late_on > late_off
    assert any(i.genome.origin == "immigrant" for r in obs_on.reports for i in r.evaluated)
    assert not any(i.genome.origin == "immigrant" for r in obs_off.reports for i in r.evaluated)


def test_map_elites_fills_the_archive_within_budget() -> None:
    ev = FakeEvaluator()
    var = Variation(MutationConfig(), Freezer("joint"), 0.2)
    obs = Collect()
    cfg = MapElitesConfig(batch_size=10, init_size=20, archive=ArchiveSpec(bins=(6, 6)))
    me = MapElites(cfg, var, BatchRunner(ev, 1), ev.n_sims, 3 * 95, 1, 3.2, obs)
    res = me.run()
    assert res.sims_used == 3 * 95 == 3 * ev.calls
    assert [len(r.evaluated) for r in obs.reports] == [10] * 9 + [5]
    cov = [r.row["archive_coverage"] for r in obs.reports]
    qd = [r.row["qd_score"] for r in obs.reports]
    assert cov == sorted(cov) and qd == sorted(qd)  # an archive never loses cells or quality
    assert res.archive.coverage > 0.3
    # first two batches are random initialisation, later ones are children of elites
    assert all(i.genome.origin == "random" for r in obs.reports[:2] for i in r.evaluated)
    assert all(i.genome.origin != "random" for r in obs.reports[2:] for i in r.evaluated)
    assert res.champion.fitness == res.archive.best().fitness
