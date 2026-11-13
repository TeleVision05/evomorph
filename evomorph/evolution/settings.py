"""The ``evolution`` and ``map_elites`` config sections, and the resolved run config.

``evolution`` holds what both algorithms share (algorithm, mode, seed, budget,
workers, frozen templates, the evaluation suite, mutation) plus the GA's own
settings; ``map_elites`` holds MAP-Elites' batch, initialisation and archive
geometry. Unknown keys are an error at every level, as in the core config.

:func:`resolve` turns a YAML file plus command-line overrides into a
:class:`RunConfig`; :func:`to_mapping` writes it back out fully resolved (every
default filled in), and loading that output gives the same RunConfig. That is
what ``config.yaml`` in a run directory is, and why a saved genome can be
re-scored exactly from it.
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from evomorph.config import FitnessConfig, SimConfig, from_mapping, load_yaml
from evomorph.evolution.archive import ArchiveSpec
from evomorph.evolution.evalspec import EvalSpec, spec_from_mapping, spec_to_mapping
from evomorph.evolution.genetic_algorithm import GAConfig
from evomorph.evolution.map_elites import MapElitesConfig
from evomorph.evolution.modes import MODES
from evomorph.evolution.operators import MutationConfig
from evomorph.evolution.pool import MAX_WORKERS
from evomorph.evolution.selection import DiversityConfig

ALGORITHMS: tuple[str, ...] = ("ga", "map_elites")


@dataclass(frozen=True)
class EvolutionConfig:
    """Shared run settings (the scalar keys of the ``evolution`` section)."""

    algorithm: str = "ga"
    mode: str = "joint"
    seed: int = 0
    budget: int = 6000  # simulations = genomes x runs per genome
    workers: int = MAX_WORKERS  # capped at MAX_WORKERS (6) whatever is asked
    frozen_body: str = "quadruped"  # controller mode: whose body is fixed
    frozen_controller: str = "quadruped"  # morphology mode: whose controller is fixed
    kind: str = "cpg"  # controller family under test (never mutated)

    def __post_init__(self) -> None:
        if self.algorithm not in ALGORITHMS:
            raise ValueError(f"evolution.algorithm must be one of {ALGORITHMS}")
        if self.mode not in MODES:
            raise ValueError(f"evolution.mode must be one of {MODES}")
        if self.budget < 1:
            raise ValueError("evolution.budget must be a positive number of simulations")
        if self.workers < 1:
            raise ValueError("evolution.workers must be at least 1")


@dataclass(frozen=True)
class RunConfig:
    """Everything a run needs, fully resolved."""

    sim: SimConfig = field(default_factory=SimConfig)
    fitness: FitnessConfig = field(default_factory=FitnessConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    ga: GAConfig = field(default_factory=GAConfig)
    map_elites: MapElitesConfig = field(default_factory=MapElitesConfig)
    spec: EvalSpec | None = None

    @property
    def eval_spec(self) -> EvalSpec:
        if self.spec is None:
            return spec_from_mapping(None, self.sim, self.fitness)
        return self.spec


_GA_KEYS = tuple(f.name for f in dataclasses.fields(GAConfig))
_EVO_KEYS = tuple(f.name for f in dataclasses.fields(EvolutionConfig))
_NESTED = ("mutation", "diversity", "evaluation")


def _split_evolution(data: dict[str, Any]) -> tuple[dict, dict, dict]:
    unknown = sorted(set(data) - set(_GA_KEYS) - set(_EVO_KEYS) - set(_NESTED))
    if unknown:
        raise ValueError(f"evolution: unknown keys {unknown}")
    evo = {k: data[k] for k in _EVO_KEYS if k in data}
    ga = {k: data[k] for k in _GA_KEYS if k in data and k not in ("mutation", "diversity")}
    ga["mutation"] = from_mapping(MutationConfig, data.get("mutation"))
    ga["diversity"] = from_mapping(DiversityConfig, data.get("diversity"))
    return evo, ga, dict(data.get("evaluation") or {})


def _map_elites(data: dict[str, Any] | None) -> MapElitesConfig:
    data = dict(data or {})
    arch_keys = ("descriptors", "bins", "ranges")
    arch = {k: data.pop(k) for k in arch_keys if k in data}
    defaults = ArchiveSpec()
    spec = ArchiveSpec(
        tuple(arch.get("descriptors", defaults.descriptors)),
        tuple(arch.get("bins", defaults.bins)),
        tuple(tuple(r) for r in arch.get("ranges", defaults.ranges)),
    )
    unknown = sorted(set(data) - {"batch_size", "init_size", "crossover_prob"})
    if unknown:
        raise ValueError(f"map_elites: unknown keys {unknown}")
    return MapElitesConfig(archive=spec, **data)


def from_sections(sections: dict[str, Any], overrides: dict[str, Any] | None = None) -> RunConfig:
    """Build a RunConfig from parsed YAML sections plus ``evolution`` overrides.

    ``overrides`` keys are ``evolution`` keys (e.g. ``budget``, ``mode``,
    ``diversity_method``); None values are ignored.
    """
    sim = from_mapping(SimConfig, sections.get("sim"))
    fit = from_mapping(FitnessConfig, sections.get("fitness"))
    evo, ga, evaluation = _split_evolution(dict(sections.get("evolution") or {}))
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        if key == "diversity_method":
            ga["diversity"] = dataclasses.replace(ga["diversity"], method=value)
        elif key in _EVO_KEYS:
            evo[key] = value
        elif key in _GA_KEYS:
            ga[key] = value
        else:
            raise ValueError(f"unknown override {key!r}")
    return RunConfig(
        sim=sim,
        fitness=fit,
        evolution=from_mapping(EvolutionConfig, evo),
        ga=GAConfig(**ga),
        map_elites=_map_elites(sections.get("map_elites")),
        spec=spec_from_mapping(evaluation, sim, fit),
    )


def resolve(path: str | Path | None, overrides: dict[str, Any] | None = None) -> RunConfig:
    """Load ``path`` (or all defaults when None) and apply overrides."""
    sections = load_yaml(path) if path is not None else {}
    return from_sections(sections, overrides)


def to_mapping(cfg: RunConfig) -> dict[str, Any]:
    """Fully resolved sections; ``from_sections(to_mapping(c)) == c``."""
    ga = asdict(cfg.ga)
    evolution = asdict(cfg.evolution) | ga | {"evaluation": spec_to_mapping(cfg.eval_spec)}
    me = asdict(cfg.map_elites)
    arch = me.pop("archive")
    return {
        "sim": asdict(cfg.sim),
        "fitness": asdict(cfg.fitness),
        "evolution": evolution,
        "map_elites": me | {k: _lists(v) for k, v in arch.items()},
    }


def _lists(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_lists(v) for v in value]
    return value
