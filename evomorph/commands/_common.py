"""Argument helpers shared by subcommands (private: not a command itself)."""

from __future__ import annotations

import argparse
from pathlib import Path

from evomorph.config import FitnessConfig, SimConfig, load_fitness_config, load_sim_config
from evomorph.morphology.genome import (
    Genome,
    load_genome,
    random_genome,
    reference_hexapod,
    reference_quadruped,
)
from evomorph.simulation.environment import ENV_KINDS, EnvSpec
from evomorph.utils.random_state import make_rng

REFERENCES = {"quadruped": reference_quadruped, "hexapod": reference_hexapod}


def add_genome_args(p: argparse.ArgumentParser) -> None:
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--genome", type=Path, help="genome JSON file")
    src.add_argument("--reference", choices=sorted(REFERENCES), help="built-in baseline")
    src.add_argument("--random", type=int, metavar="SEED", help="random genome from SEED")


def add_env_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=Path, help="YAML with sim/fitness sections")
    p.add_argument("--env", default="flat", choices=ENV_KINDS, help="terrain kind")
    p.add_argument("--friction", type=float, default=1.0, help="ground friction coefficient")
    p.add_argument("--env-seed", type=int, default=0, help="terrain seed")
    p.add_argument("--seed", type=int, default=0, help="evaluation seed (spawn noise)")


def genome_from_args(args: argparse.Namespace) -> Genome:
    if args.genome is not None:
        return load_genome(args.genome)
    if args.reference is not None:
        return REFERENCES[args.reference]()
    return random_genome(make_rng(args.random), genome_id=f"rnd-seed{args.random}")


def env_from_args(args: argparse.Namespace) -> EnvSpec:
    return EnvSpec(kind=args.env, friction=args.friction, seed=args.env_seed)


def configs_from_args(args: argparse.Namespace) -> tuple[SimConfig, FitnessConfig]:
    if getattr(args, "config", None):
        return load_sim_config(args.config), load_fitness_config(args.config)
    return SimConfig(), FitnessConfig()
