"""Tournament, elitism, fitness sharing and the diversity measures."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from evomorph.evolution.modes import mutable_genes
from evomorph.evolution.selection import (
    DiversityConfig,
    behaviour_spread,
    body_plan,
    choose_elites,
    gene_distance_matrix,
    mean_pairwise,
    rank_order,
    selection_scores,
    tournament,
)
from evomorph.morphology.genome import random_genome, reference_hexapod, reference_quadruped

GENES = mutable_genes("joint")


def test_rank_order_is_best_first_and_breaks_ties_by_index() -> None:
    assert list(rank_order(np.array([0.1, 0.5, 0.5, -2.0]))) == [1, 2, 0, 3]


def test_tournament_prefers_higher_scores() -> None:
    rng = np.random.default_rng(0)
    scores = np.arange(10, dtype=float)
    wins = np.bincount([tournament(scores, rng, 3) for _ in range(4000)], minlength=10)
    assert wins[9] > wins[5] > wins[1]
    assert wins[0] == wins.min()
    # size 1 is uniform random selection
    uniform = np.bincount([tournament(scores, rng, 1) for _ in range(4000)], minlength=10)
    assert uniform.min() > 300


def test_gene_distance_properties() -> None:
    rng = np.random.default_rng(1)
    gs = [random_genome(rng, f"r{i}") for i in range(30)]
    d = gene_distance_matrix(gs, GENES)
    assert d.shape == (30, 30)
    assert np.allclose(np.diag(d), 0)
    assert np.allclose(d, d.T)
    assert d.min() >= 0 and d.max() <= 1
    # random pairs sit near the documented ~0.32
    assert 0.25 < mean_pairwise(d) < 0.45
    clone = copy.deepcopy(gs[0])
    assert gene_distance_matrix([gs[0], clone], GENES)[0, 1] == 0.0


def test_gene_distance_counts_a_gait_change_as_one_gene() -> None:
    a = reference_quadruped()
    b = copy.deepcopy(a)
    b.controller.gait = "tripod"
    d = gene_distance_matrix([a, b], GENES)[0, 1]
    assert d == pytest.approx(1.0 / len(GENES))


def test_sharing_lowers_the_score_of_crowded_individuals() -> None:
    q = reference_quadruped()
    crowd = [copy.deepcopy(q) for _ in range(5)]
    loner = random_genome(np.random.default_rng(9), "loner")
    pop = [*crowd, loner]
    fitness = np.array([1.0, 0.99, 0.98, 0.97, 0.96, 0.5])  # loner is worst
    dist = gene_distance_matrix(pop, GENES)
    assert dist[0, 5] > DiversityConfig().sharing_radius  # truly outside the niche
    plain = selection_scores(fitness, dist, DiversityConfig(method="none"))
    shared = selection_scores(fitness, dist, DiversityConfig(method="sharing"))
    assert np.argmin(plain) == 5
    # clones split one niche five ways; the loner keeps its whole rank score
    assert shared[0] == pytest.approx(plain[0] / 5)
    assert shared[5] == pytest.approx(plain[5])
    assert shared[5] > shared[2]  # now outranks most of the crowd
    assert shared[5] / shared.sum() > 2 * plain[5] / plain.sum()


def test_sharing_works_with_negative_fitness() -> None:
    rng = np.random.default_rng(2)
    gs = [random_genome(rng, f"r{i}") for i in range(8)]
    fitness = -np.linspace(0.1, 3.0, 8)
    s = selection_scores(fitness, gene_distance_matrix(gs, GENES), DiversityConfig())
    assert np.all(s > 0)


def test_elites_include_global_best_and_best_of_each_plan() -> None:
    fitness = np.array([3.0, 2.0, 1.0, 0.5, -1.0, 5.0])
    valid = np.array([True, True, True, True, True, False])  # invalid never kept
    plans = ["4x2", "4x2", "6x1", "4x2", "2x2", "8x2"]
    assert choose_elites(fitness, valid, plans, 2, 0) == [0, 1]
    assert choose_elites(fitness, valid, plans, 2, 1) == [0, 1, 2, 4]
    assert choose_elites(fitness, valid, plans, 0, 1) == [0, 2, 4]


def test_diversity_off_disables_every_mechanism() -> None:
    off = DiversityConfig(method="none", plan_elites=3, immigrant_fraction=0.2)
    assert not off.enabled
    assert off.effective_plan_elites == 0 and off.effective_immigrants == 0.0
    with pytest.raises(ValueError):
        DiversityConfig(method="crowding-ish")


def test_behaviour_spread_and_body_plan() -> None:
    assert behaviour_spread(np.array([[0.0, 0.0], [1.0, 0.0]])) == pytest.approx(1.0)
    assert behaviour_spread(np.array([[0.2, 0.2]] * 4)) == 0.0
    assert body_plan(reference_hexapod()) == "6x2"
