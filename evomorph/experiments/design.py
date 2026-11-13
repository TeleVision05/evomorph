"""The experiment's design: the ``experiment`` config section and per-run configs.

One :class:`ExperimentConfig` fixes everything that must be equal across
conditions: the training and test suites, the budget in simulations, the
search seeds, the population size, and how champions and final sets are
re-scored afterwards. :func:`run_config` then derives each condition's engine
config from the same base sections (``sim``, ``fitness``, ``evolution``,
``map_elites``); only ``algorithm`` and ``mode`` differ between conditions.

Equal budgets
-------------
Every genome is scored on every environment of the training suite, so one
genome costs ``len(train suite)`` simulations whatever the condition. The
engine stops when the next genome would not fit, so every run spends exactly
:attr:`ExperimentConfig.effective_budget` (the budget rounded down to whole
genomes). The experiment checks this after every run and refuses to tabulate
runs that spent different amounts.

MAP-Elites against the GA
-------------------------
MAP-Elites' batch size and its number of random initial genomes are both set
to the GA's population size, so D starts from as many random creatures as the
GA's generation 0 and then makes as many children per step as a GA
generation. With a small budget, the engine's default of four random batches
would otherwise spend a large share of D's budget on random search.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from evomorph.config import load_yaml
from evomorph.evolution.evalspec import EvalSpec
from evomorph.evolution.pool import MAX_WORKERS, clamp_workers
from evomorph.evolution.settings import RunConfig, from_sections, to_mapping
from evomorph.experiments.baselines import CONDITION_KEYS, REFERENCE, Condition, get_condition
from evomorph.simulation.suites import get_suite
from evomorph.utils.serialization import to_jsonable

# Fresh spawn seeds for re-scoring after the search. Training runs use spawn
# seeds train_seed + i (one per environment, i < 11), so seeds from 1000 up can
# never coincide with a seed a genome was selected on; the test suite gets its
# own block so train and test re-scores never share a seed either.
RESCORE_TRAIN_SEED = 1000
RESCORE_TEST_SEED = 2000


@dataclass(frozen=True)
class ExperimentConfig:
    """The ``experiment`` config section (see docs/EXPERIMENT.md for the numbers).

    Defaults are the full run's (configs/comparison.yaml). ``budget = 1441``
    is 131 genomes of 11 simulations, what fits 4 conditions x 5 seeds into
    about two and a half hours at the throughput measured in the quick run
    (3.89 simulations/s with 6 workers on the shared laptop; computed by
    scripts/plan_budget.py). ``n_seeds = 5`` because it is the fewest seeds
    with which a primary comparison can pass the Holm correction at all: the
    smallest two-sided Mann-Whitney p is 0.0079 for 5 vs 5 (x 6 tests = 0.048),
    0.029 for 4 vs 4 (x 6 = 0.17) and 0.10 for 3 vs 3.
    """

    conditions: tuple[str, ...] = CONDITION_KEYS
    n_seeds: int = 5  # search seeds seed_start .. seed_start + n_seeds - 1, for every condition
    seed_start: int = 0
    budget: int = 1441  # simulations per run, equal for every condition
    workers: int = MAX_WORKERS  # process pool, capped at 6; never changes a result
    train_suite: str = "train"  # selection happens here
    test_suite: str = "test"  # held out: never seen during the search
    train_seed: int = 0  # environment i of the training suite uses spawn seed train_seed + i
    population_size: int | None = None  # None: evolution.population_size from the config
    rescore_seeds: int = 3  # fresh spawn seeds per environment for each champion re-score
    set_size: int = 10  # best-K distinct genomes of the final population or archive
    set_seeds: int = 1  # fresh spawn seeds per environment for each set member

    def __post_init__(self) -> None:
        object.__setattr__(self, "conditions", tuple(self.conditions))
        for key in self.conditions:
            get_condition(key)
        if len(set(self.conditions)) != len(self.conditions):
            raise ValueError("experiment.conditions lists a condition twice")
        if self.n_seeds < 1 or self.budget < 1 or self.workers < 1:
            raise ValueError("experiment.n_seeds, budget and workers must be at least 1")
        if min(self.rescore_seeds, self.set_size, self.set_seeds) < 1:
            raise ValueError("experiment.rescore_seeds, set_size and set_seeds must be at least 1")
        if self.population_size is not None and self.population_size < 2:
            raise ValueError("experiment.population_size must be at least 2")
        get_suite(self.train_suite)
        get_suite(self.test_suite)
        if self.train_suite == self.test_suite:
            raise ValueError("the test suite must differ from the training suite")

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(range(self.seed_start, self.seed_start + self.n_seeds))

    @property
    def sims_per_genome(self) -> int:
        return len(get_suite(self.train_suite))

    @property
    def effective_budget(self) -> int:
        """The budget rounded down to whole genomes: what every run spends exactly."""
        return (self.budget // self.sims_per_genome) * self.sims_per_genome


_KEYS = {f.name for f in dataclasses.fields(ExperimentConfig)}


def experiment_from_mapping(
    data: dict[str, Any] | None, quick: bool = False, overrides: dict[str, Any] | None = None
) -> ExperimentConfig:
    """Parse the ``experiment`` section; ``quick`` applies its ``quick`` sub-section.

    Precedence: defaults < section < ``quick`` sub-section (if asked) <
    ``overrides`` (command-line options; None values are ignored). Unknown keys
    are errors at every level.
    """
    data = dict(data or {})
    preset = dict(data.pop("quick", None) or {})
    for where, block in (("experiment", data), ("experiment.quick", preset)):
        unknown = sorted(set(block) - _KEYS)
        if unknown:
            raise ValueError(f"{where}: unknown keys {unknown}")
    if quick and not preset:
        raise ValueError("this config has no experiment.quick section, so --quick has no size")
    merged = data | (preset if quick else {})
    merged |= {k: v for k, v in (overrides or {}).items() if v is not None}
    if "conditions" in merged:
        merged["conditions"] = tuple(merged["conditions"])
    return ExperimentConfig(**merged)


def load_experiment(
    path: str | Path, quick: bool = False, overrides: dict[str, Any] | None = None
) -> tuple[ExperimentConfig, RunConfig]:
    """The experiment section and the base engine config from one YAML file."""
    sections = load_yaml(path)
    exp = experiment_from_mapping(sections.get("experiment"), quick, overrides)
    base = from_sections({k: v for k, v in sections.items() if k != "experiment"})
    check_base(base)
    return exp, base


def check_base(base: RunConfig) -> None:
    """Refuse base settings that would make the budgets unequal."""
    if base.ga.max_generations is not None:
        raise ValueError(
            "evolution.max_generations must be null in an experiment: runs must stop on the "
            "simulation budget, or conditions with different costs would spend different amounts"
        )


def train_spec(exp: ExperimentConfig, base: RunConfig) -> EvalSpec:
    """Every training environment once, environment i with spawn seed train_seed + i."""
    envs = get_suite(exp.train_suite)
    runs = tuple((env, exp.train_seed + i) for i, env in enumerate(envs))
    return EvalSpec(runs, base.sim, base.fitness, "mean")


def run_config(exp: ExperimentConfig, base: RunConfig, cond: Condition, seed: int) -> RunConfig:
    """The engine config of one run: ``base`` with only algorithm, mode and seed set per run."""
    pop = exp.population_size or base.ga.population_size
    evolution = replace(
        base.evolution,
        algorithm=cond.algorithm,
        mode=cond.mode,
        seed=int(seed),
        budget=exp.effective_budget,
        workers=clamp_workers(exp.workers),
        frozen_body=REFERENCE,
        frozen_controller=REFERENCE,
    )
    return replace(
        base,
        evolution=evolution,
        ga=replace(base.ga, population_size=pop),
        map_elites=replace(base.map_elites, batch_size=pop, init_size=pop),
        spec=train_spec(exp, base),
    )


def run_name(cond: Condition, seed: int) -> str:
    """Run directory name of one (condition, seed)."""
    return f"{cond.key}_s{seed}"


def design_mapping(exp: ExperimentConfig, base: RunConfig) -> dict[str, Any]:
    """Everything that makes runs comparable.

    Excluded: the worker count (it never changes a result) and which seeds are
    run (more seeds of the same design add samples; they do not change the
    design).
    """
    runs = {}
    for key in exp.conditions:
        cfg = run_config(exp, base, get_condition(key), exp.seed_start)
        mapping = to_mapping(cfg)
        mapping["evolution"].pop("workers")
        mapping["evolution"].pop("seed")
        runs[key] = mapping
    exp_map = dataclasses.asdict(exp)
    for key in ("workers", "n_seeds", "seed_start"):
        exp_map.pop(key)
    return to_jsonable({"experiment": exp_map, "runs": runs})


def design_hash(exp: ExperimentConfig, base: RunConfig) -> str:
    """16-hex digest of :func:`design_mapping`; equal hashes mean comparable runs."""
    text = json.dumps(design_mapping(exp, base), sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]
