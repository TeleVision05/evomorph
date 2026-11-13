"""EvalSpec aggregation, the evaluator contract and the batch runner."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from evo_helpers import SHORT_SIM, short_spec

import evomorph.evolution.evalspec as evalspec
from evomorph.config import FitnessConfig
from evomorph.evolution.evalspec import (
    EvalSpec,
    SpecEvaluator,
    evaluate_spec,
    sims_per_call,
    spec_from_mapping,
    spec_to_mapping,
)
from evomorph.evolution.pool import MAX_WORKERS, BatchRunner, clamp_workers
from evomorph.morphology.genome import reference_hexapod, reference_quadruped
from evomorph.simulation.environment import EnvSpec
from evomorph.simulation.evaluator import EvalResult, evaluate_robust
from evomorph.simulation.metrics import INVALID_FITNESS


def _fake_run(fitness: float, valid: bool = True, fell: float = 0.0, speed: float = 0.0):
    return EvalResult(
        genome_id="g",
        valid=valid,
        failure_reason="" if valid else "simulation glitch: boom",
        fitness=fitness if valid else INVALID_FITNESS,
        metrics={"fell": fell, "sustained_velocity": speed} if valid else {},
        fitness_terms={"progress": fitness} if valid else {},
        env=EnvSpec(),
        seed=0,
        sim_seconds=1.0,
        wall_seconds=0.0,
    )


@pytest.fixture
def three_runs():
    runs = ((EnvSpec(friction=0.6), 0), (EnvSpec(friction=0.8), 1), (EnvSpec(), 2))
    return EvalSpec(runs, SHORT_SIM, FitnessConfig())


def _patch(monkeypatch, results):
    it = iter(results)
    monkeypatch.setattr(evalspec, "evaluate", lambda *a, **k: next(it))


def test_mean_and_min_aggregation_keep_per_run_values(monkeypatch, three_runs) -> None:
    runs = [_fake_run(0.3, speed=0.1), _fake_run(0.6, fell=1.0, speed=0.4), _fake_run(0.9)]
    _patch(monkeypatch, runs)
    r = evaluate_spec(reference_quadruped(), three_runs)
    assert r.valid and r.fitness == pytest.approx(0.6)
    assert r.seed_fitness == (0.3, 0.6, 0.9)
    assert r.metrics["fell"] == pytest.approx(1 / 3)  # fraction of runs that fell
    assert r.metrics["sustained_velocity"] == pytest.approx(0.5 / 3)
    assert r.failure_reason == "fell in 1/3 runs"
    assert r.run_labels == ("flat_mu0.6/s0", "flat_mu0.8/s1", "flat/s2")
    assert [m["sustained_velocity"] for m in r.run_metrics] == [0.1, 0.4, 0.0]
    _patch(monkeypatch, runs)
    r_min = evaluate_spec(reference_quadruped(), EvalSpec(three_runs.runs, aggregate="min"))
    assert r_min.fitness == pytest.approx(0.3)


def test_any_invalid_run_invalidates_the_genome(monkeypatch, three_runs) -> None:
    _patch(monkeypatch, [_fake_run(0.3), _fake_run(0.0, valid=False), _fake_run(0.9)])
    r = evaluate_spec(reference_quadruped(), three_runs)
    assert not r.valid and r.fitness == INVALID_FITNESS and r.metrics == {}
    # the fake run reports env flat / seed 0; the reason names the run that failed
    assert r.failure_reason == "flat/s0: simulation glitch: boom"
    assert r.seed_fitness[1] == INVALID_FITNESS and len(r.seed_fitness) == 3


def test_matches_core_evaluate_robust_on_one_environment() -> None:
    """Same env, seeds 0 and 1: evaluate_spec and the core's evaluate_robust agree."""
    g = reference_hexapod()
    spec = EvalSpec(((EnvSpec(), 0), (EnvSpec(), 1)), SHORT_SIM, FitnessConfig())
    mine = evaluate_spec(g, spec)
    core = evaluate_robust(g, EnvSpec(), SHORT_SIM, FitnessConfig(), seeds=(0, 1))
    assert mine.fitness == core.fitness
    assert mine.metrics == core.metrics
    assert mine.seed_fitness == core.seed_fitness


def test_evaluator_is_picklable_and_reports_its_cost() -> None:
    ev = SpecEvaluator(short_spec(3))
    clone = pickle.loads(pickle.dumps(ev))
    assert clone.spec == ev.spec and ev.n_sims == 3
    assert sims_per_call(ev) == 3 and sims_per_call(lambda g: None) == 1
    assert sims_per_call(ev, 5) == 5
    with pytest.raises(ValueError):
        sims_per_call(ev, 0)


def test_spec_mapping_round_trip_and_unknown_keys() -> None:
    data = {"aggregate": "min", "runs": [{"env": {"friction": 0.5}, "seed": 4}]}
    spec = spec_from_mapping(data, SHORT_SIM, FitnessConfig())
    assert spec.runs == ((EnvSpec(friction=0.5), 4),) and spec.aggregate == "min"
    assert spec_from_mapping(spec_to_mapping(spec), SHORT_SIM, FitnessConfig()) == spec
    with pytest.raises(ValueError, match="frictoin"):
        spec_from_mapping({"runs": [{"env": {"frictoin": 1}}]}, SHORT_SIM, FitnessConfig())
    with pytest.raises(ValueError, match="aggregate"):
        EvalSpec(((EnvSpec(), 0),), aggregate="median")
    with pytest.raises(ValueError):
        EvalSpec(())


def test_batch_runner_caps_workers_and_keeps_order() -> None:
    assert clamp_workers(64) == MAX_WORKERS == 6
    assert clamp_workers(0) == 1
    ev = SpecEvaluator(short_spec(1))
    genomes = [reference_quadruped(), reference_hexapod()]
    with BatchRunner(ev, 1) as runner:
        res = runner.map(genomes)
    assert [r.genome_id for r in res] == ["ref-quadruped", "ref-hexapod"]
    assert runner.map([]) == []
    assert np.isfinite([r.fitness for r in res]).all()
