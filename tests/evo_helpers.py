"""Helpers for the evolution tests: a fast fake evaluator and short-sim configs."""

from __future__ import annotations

from evomorph.config import FitnessConfig, SimConfig
from evomorph.evolution.evalspec import EvalSpec
from evomorph.morphology.genome import GENE_BOUNDS, Genome
from evomorph.simulation.environment import EnvSpec
from evomorph.simulation.evaluator import EvalResult

# 1.2 s of simulated time instead of 11 s: ~10x cheaper, still real physics.
SHORT_SIM = SimConfig(duration=1.0, settle_time=0.2)


def short_spec(n_runs: int = 2) -> EvalSpec:
    runs = tuple((EnvSpec(friction=0.8 + 0.2 * i), i) for i in range(n_runs))
    return EvalSpec(runs, SHORT_SIM, FitnessConfig(), "mean")


def _norm(g: Genome, name: str) -> float:
    b = GENE_BOUNDS[name]
    group = g.morphology if b.group == "morphology" else g.controller
    return (float(getattr(group, name)) - b.low) / b.span


class FakeEvaluator:
    """Deterministic, simulation-free scorer: fitness rises with frequency,
    hip amplitude and torso length. Lets the algorithm tests run in milliseconds.
    """

    n_sims = 3

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, g: Genome) -> EvalResult:
        self.calls += 1
        f = _norm(g, "frequency") + _norm(g, "hip_amplitude") + _norm(g, "torso_length")
        metrics = {
            "sustained_velocity": 0.25 * f,
            "energy_efficiency": 0.7 * _norm(g, "torso_width"),
            "fell": 0.0,
        }
        return EvalResult(
            genome_id=g.id,
            valid=True,
            failure_reason="",
            fitness=float(f),
            metrics=metrics,
            fitness_terms={},
            env=EnvSpec(),
            seed=0,
            sim_seconds=0.0,
            wall_seconds=0.0,
            seed_fitness=(float(f),) * 3,
        )
