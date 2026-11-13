"""A tiny but real comparison experiment for the experiment and analysis tests.

Real physics, made cheap: 1.2 s of simulated time per run instead of 11 s, the
one-environment training suite "flat" and a two-environment held-out suite
registered here for the tests only (friction 0.5 and 1.5). Four genomes per
generation and 12 simulations per run give three GA generations, enough to
exercise selection, freezing and MAP-Elites insertion.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from evomorph.evolution.settings import RunConfig
from evomorph.experiments.comparison import ComparisonOutcome, run_comparison
from evomorph.experiments.design import ExperimentConfig, experiment_from_mapping, load_experiment
from evomorph.simulation.environment import EnvSpec
from evomorph.simulation.suites import SUITES

TEST_SUITE = "tiny-held-out"
SUITES.setdefault(
    TEST_SUITE,
    (EnvSpec(kind="friction", friction=0.5), EnvSpec(kind="friction", friction=1.5)),
)

TINY: dict[str, Any] = {
    "sim": {"duration": 1.0, "settle_time": 0.2},
    "evolution": {"population_size": 4, "workers": 1},
    "map_elites": {"batch_size": 4, "init_size": 4},
    "experiment": {
        "n_seeds": 1,
        "budget": 12,
        "workers": 1,
        "train_suite": "flat",
        "test_suite": TEST_SUITE,
        "population_size": 4,
        "rescore_seeds": 1,
        "set_size": 2,
        "set_seeds": 1,
        "quick": {"budget": 8, "population_size": 4},
    },
}


def tiny_sections(**experiment: Any) -> dict[str, Any]:
    sections = copy.deepcopy(TINY)
    sections["experiment"] |= experiment
    return sections


def write_config(path: Path, sections: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(sections))
    return path


def tiny_design(tmp: Path, **experiment: Any) -> tuple[ExperimentConfig, RunConfig]:
    cfg = write_config(tmp / "tiny_design.yaml", tiny_sections(**experiment))
    return load_experiment(cfg)


def run_tiny(root: Path, name: str = "exp", **experiment: Any) -> ComparisonOutcome:
    exp, base = tiny_design(root, **experiment)
    return run_comparison(exp, base, root, name, out=lambda line: None, command="test")


def tiny_experiment_config(**experiment: Any) -> ExperimentConfig:
    return experiment_from_mapping(tiny_sections(**experiment)["experiment"])
