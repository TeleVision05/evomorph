"""MAP-Elites archive: cells, insertion, replacement, persistence, reproduction."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from evo_helpers import short_spec

from evomorph.config import FitnessConfig
from evomorph.evolution.archive import Archive, ArchiveSpec, qd_offset_for
from evomorph.evolution.evalspec import SpecEvaluator
from evomorph.morphology.genome import random_genome, reference_hexapod, reference_quadruped
from evomorph.simulation.environment import EnvSpec
from evomorph.simulation.evaluator import EvalResult

SPEC = ArchiveSpec(("sustained_velocity", "energy_efficiency"), (10, 10), ((0, 0.8), (0, 0.7)))


def _result(fitness: float, speed: float, eff: float, fell: float = 0.0, valid: bool = True):
    return EvalResult(
        genome_id="x",
        valid=valid,
        failure_reason="",
        fitness=fitness,
        metrics={"sustained_velocity": speed, "energy_efficiency": eff, "fell": fell}
        if valid
        else {},
        fitness_terms={},
        env=EnvSpec(),
        seed=0,
        sim_seconds=0.0,
        wall_seconds=0.0,
        seed_fitness=(fitness, fitness),
    )


def test_cells_cover_the_ranges_and_clip_outside_values() -> None:
    assert SPEC.cell((0.0, 0.0)) == (0, 0)
    assert SPEC.cell((0.079, 0.069)) == (0, 0)
    assert SPEC.cell((0.081, 0.071)) == (1, 1)
    assert SPEC.cell((0.8, 0.7)) == (9, 9)  # the top edge belongs to the last bin
    assert SPEC.cell((-0.5, -1.0)) == (0, 0)  # backward walkers share the slowest column
    assert SPEC.cell((3.0, 0.35)) == (9, 5)
    assert SPEC.n_cells == 100


def test_insertion_and_replacement_rules() -> None:
    a = Archive(SPEC, qd_offset=3.2)
    g1, g2, g3 = (
        reference_quadruped(),
        reference_hexapod(),
        random_genome(np.random.default_rng(0)),
    )
    assert a.add(g1, _result(0.5, 0.4, 0.45), 10) == "new"
    assert a.add(g2, _result(0.4, 0.41, 0.46), 20) == "rejected"  # same cell, worse
    assert a.add(g2, _result(0.5, 0.41, 0.46), 20) == "rejected"  # tie keeps the incumbent
    assert a.get((5, 6)).genome is g1
    assert a.add(g2, _result(0.9, 0.42, 0.47), 30) == "improved"
    assert a.get((5, 6)).genome is g2 and a.get((5, 6)).inserted_at == 30
    assert a.add(g3, _result(2.0, 0.1, 0.1, fell=1 / 3), 40) == "fell"  # fell in one run
    assert a.add(g3, _result(-20.0, 0.0, 0.0, valid=False), 50) == "invalid"
    assert a.add(g3, _result(-0.1, 0.0, 0.0), 60) == "new"
    assert len(a) == 2
    assert a.coverage == pytest.approx(0.02)
    assert a.qd_score == pytest.approx((0.9 + 3.2) + (-0.1 + 3.2))
    assert a.best().genome is g2
    assert a.occupied() == [(0, 0), (5, 6)]
    grid = a.fitness_grid()
    assert grid[5, 6] == 0.9 and np.isnan(grid[1, 1])


def test_qd_offset_is_the_lowest_valid_fitness() -> None:
    assert qd_offset_for(FitnessConfig()) == pytest.approx(3.2)


def test_save_and_load_round_trip(tmp_path) -> None:
    a = Archive(SPEC, qd_offset=3.2)
    a.add(reference_quadruped(), _result(0.5, 0.4, 0.45), 10)
    a.add(reference_hexapod(), _result(0.123456789012345, 0.7, 0.2), 20)
    a.save(tmp_path / "archive")
    b = Archive.load(tmp_path / "archive")
    assert b.spec == a.spec and b.qd_offset == a.qd_offset and b.occupied() == a.occupied()
    for k in a.occupied():
        ea, eb = a.get(k), b.get(k)
        assert eb.genome.to_dict() == ea.genome.to_dict()
        assert (eb.fitness, eb.descriptor, eb.metrics) == (ea.fitness, ea.descriptor, ea.metrics)
        assert eb.seed_fitness == ea.seed_fitness and eb.inserted_at == ea.inserted_at
    assert b.qd_score == a.qd_score


def test_bad_archive_specs_are_rejected() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        ArchiveSpec(("a", "b", "c"), (2, 2, 2), ((0, 1), (0, 1), (0, 1)))
    with pytest.raises(ValueError):
        ArchiveSpec(bins=(0, 4))
    with pytest.raises(ValueError):
        ArchiveSpec(ranges=((1.0, 0.0), (0.0, 1.0)))


def test_saved_elites_reproduce_their_numbers_when_re_evaluated(tmp_path) -> None:
    """Real (short) simulations: every stored elite re-scores bit-identically."""
    evaluator = SpecEvaluator(short_spec(2))
    a = Archive(ArchiveSpec(ranges=((-0.2, 0.3), (0.0, 0.5))), qd_offset=3.2)
    rng = np.random.default_rng(3)
    genomes = [reference_quadruped(), reference_hexapod()]
    genomes += [random_genome(rng, f"r{i}") for i in range(6)]
    for g in genomes:
        a.add(g, evaluator(g))
    assert len(a) >= 2
    a.save(tmp_path / "arch")
    loaded = Archive.load(tmp_path / "arch")
    for key in loaded.occupied():
        e = loaded.get(key)
        again = evaluator(replace(e.genome))
        assert again.fitness == e.fitness
        assert again.metrics == e.metrics
        assert loaded.spec.cell(loaded.spec.descriptor(again.metrics)) == key
