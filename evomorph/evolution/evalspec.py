"""Injected evaluation: what one genome is scored on, and how the scores combine.

The evolution engine never calls the simulator directly. It is handed a
callable ``Genome -> EvalResult`` plus the number of simulations that one call
costs, so a terrain suite or an experiment can swap in its own scoring without
touching the algorithms. :class:`SpecEvaluator` is the standard such callable:
it runs the core :func:`~evomorph.simulation.evaluator.evaluate` once per
``(EnvSpec, seed)`` pair of an :class:`EvalSpec` and aggregates.

Aggregation (the contract other stages code against)
---------------------------------------------------
* ``fitness``: mean or min of the per-run fitness values (``EvalSpec.aggregate``).
* ``metrics`` and ``fitness_terms``: per-key means over the runs. ``fell`` is
  therefore the *fraction* of runs that fell.
* ``valid``: False if any run is invalid; the result then carries that run's
  reason and ``INVALID_FITNESS`` (one glitch is enough to distrust a body).
* Per-run values are kept: ``seed_fitness`` (fitness of every run, in spec
  order), ``run_labels`` and ``run_metrics``.

Everything here is picklable, so evaluators can be mapped over a process pool.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from evomorph.config import FitnessConfig, SimConfig
from evomorph.morphology.genome import Genome
from evomorph.simulation.environment import EnvSpec
from evomorph.simulation.evaluator import EvalResult, evaluate
from evomorph.simulation.metrics import INVALID_FITNESS

AGGREGATES: tuple[str, ...] = ("mean", "min")

# A genome scorer: any picklable callable. ``n_sims`` (if present) is its cost.
Evaluator = Callable[[Genome], EvalResult]


@dataclass(frozen=True)
class EvalSpec:
    """The runs one genome is scored on and how their scores combine."""

    runs: tuple[tuple[EnvSpec, int], ...]
    sim: SimConfig = field(default_factory=SimConfig)
    fitness: FitnessConfig = field(default_factory=FitnessConfig)
    aggregate: str = "mean"

    def __post_init__(self) -> None:
        if not self.runs:
            raise ValueError("an evaluation needs at least one (environment, seed) run")
        if self.aggregate not in AGGREGATES:
            raise ValueError(f"aggregate must be one of {AGGREGATES}, got {self.aggregate!r}")
        object.__setattr__(self, "runs", tuple((env, int(seed)) for env, seed in self.runs))

    @property
    def n_sims(self) -> int:
        """Simulations one genome costs under this spec."""
        return len(self.runs)

    def labels(self) -> tuple[str, ...]:
        return tuple(f"{env.label()}/s{seed}" for env, seed in self.runs)


@dataclass
class SpecResult(EvalResult):
    """An aggregated :class:`EvalResult` that also keeps every run's values."""

    run_labels: tuple[str, ...] = ()
    run_metrics: tuple[dict[str, float], ...] = ()


def _invalid(genome_id: str, spec: EvalSpec, runs: Sequence[EvalResult], t0: float) -> SpecResult:
    bad = next(r for r in runs if not r.valid)
    return SpecResult(
        genome_id=genome_id,
        valid=False,
        failure_reason=f"{bad.env.label()}/s{bad.seed}: {bad.failure_reason}",
        fitness=INVALID_FITNESS,
        metrics={},
        fitness_terms={},
        env=spec.runs[0][0],
        seed=spec.runs[0][1],
        sim_seconds=float(sum(r.sim_seconds for r in runs)),
        wall_seconds=time.perf_counter() - t0,
        seed_fitness=tuple(float(r.fitness) for r in runs),
        run_labels=spec.labels(),
        run_metrics=tuple(dict(r.metrics) for r in runs),
    )


def evaluate_spec(genome: Genome, spec: EvalSpec) -> SpecResult:
    """Score ``genome`` on every run of ``spec`` and aggregate (see module doc).

    Deterministic in ``(genome genes, spec)``: the same inputs give bit-identical
    fitness and metrics in any process, which is what makes a saved genome
    reproducible and a run independent of worker count.
    """
    t0 = time.perf_counter()
    runs = [evaluate(genome, env, spec.sim, spec.fitness, seed=s) for env, s in spec.runs]
    gid = getattr(genome, "id", "unknown")
    if not all(r.valid for r in runs):
        return _invalid(gid, spec, runs, t0)
    scores = [float(r.fitness) for r in runs]
    n_fell = sum(r.fell for r in runs)
    agg = np.mean(scores) if spec.aggregate == "mean" else np.min(scores)
    return SpecResult(
        genome_id=gid,
        valid=True,
        failure_reason=f"fell in {n_fell}/{len(runs)} runs" if n_fell else "",
        fitness=float(agg),
        metrics={k: float(np.mean([r.metrics[k] for r in runs])) for k in runs[0].metrics},
        fitness_terms={
            k: float(np.mean([r.fitness_terms[k] for r in runs])) for k in runs[0].fitness_terms
        },
        env=spec.runs[0][0],
        seed=spec.runs[0][1],
        sim_seconds=float(sum(r.sim_seconds for r in runs)),
        wall_seconds=time.perf_counter() - t0,
        seed_fitness=tuple(scores),
        run_labels=spec.labels(),
        run_metrics=tuple(dict(r.metrics) for r in runs),
    )


@dataclass(frozen=True)
class SpecEvaluator:
    """Picklable ``Genome -> SpecResult`` callable bound to one :class:`EvalSpec`."""

    spec: EvalSpec

    @property
    def n_sims(self) -> int:
        return self.spec.n_sims

    def __call__(self, genome: Genome) -> SpecResult:
        return evaluate_spec(genome, self.spec)


def sims_per_call(evaluator: Evaluator, override: int | None = None) -> int:
    """How many simulations one call of ``evaluator`` costs (budget accounting)."""
    if override is not None:
        if override < 1:
            raise ValueError("sims_per_eval must be at least 1")
        return int(override)
    return int(getattr(evaluator, "n_sims", 1))


# ---------------------------------------------------------------------------
# Config <-> spec
# ---------------------------------------------------------------------------


def _env_from_mapping(data: dict[str, Any]) -> EnvSpec:
    from evomorph.config import from_mapping

    return from_mapping(EnvSpec, data)


def spec_from_mapping(
    data: dict[str, Any] | None, sim: SimConfig, fitness: FitnessConfig
) -> EvalSpec:
    """Build an :class:`EvalSpec` from the ``evolution.evaluation`` config mapping.

    Shape::

        aggregate: mean
        runs:
          - {env: {kind: flat, friction: 0.6}, seed: 0}
          - {env: {kind: flat}, seed: 1}
    """
    data = dict(data or {})
    unknown = sorted(set(data) - {"aggregate", "runs"})
    if unknown:
        raise ValueError(f"evolution.evaluation: unknown keys {unknown}")
    runs_in = data.get("runs") or [{"env": {}, "seed": 0}, {"env": {}, "seed": 1}]
    runs = []
    for i, run in enumerate(runs_in):
        extra = sorted(set(run) - {"env", "seed"})
        if extra:
            raise ValueError(f"evolution.evaluation.runs[{i}]: unknown keys {extra}")
        runs.append((_env_from_mapping(run.get("env") or {}), int(run.get("seed", 0))))
    return EvalSpec(tuple(runs), sim, fitness, str(data.get("aggregate", "mean")))


def spec_to_mapping(spec: EvalSpec) -> dict[str, Any]:
    """Inverse of :func:`spec_from_mapping` (sim and fitness are their own sections)."""
    from dataclasses import asdict

    return {
        "aggregate": spec.aggregate,
        "runs": [{"env": asdict(env), "seed": seed} for env, seed in spec.runs],
    }
