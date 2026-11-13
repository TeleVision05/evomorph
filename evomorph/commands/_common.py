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
    p.add_argument("--slope", type=float, default=0.0, help="slope in degrees (slope, random)")
    p.add_argument("--roughness", type=float, default=0.0, help="bump height in m (uneven, random)")
    p.add_argument("--obstacles", type=int, default=0, help="obstacle count (uneven, random)")
    p.add_argument(
        "--friction-spread", type=float, default=1.0, help="friction patch factor >= 1 (friction)"
    )
    p.add_argument("--env-seed", type=int, default=0, help="terrain seed")
    p.add_argument("--seed", type=int, default=0, help="evaluation seed (spawn noise)")


class GenomeLoadError(Exception):
    """A genome file could not be read; the message is meant for the person at the terminal."""


def genome_from_args(args: argparse.Namespace) -> Genome:
    """The genome named on the command line. Raises GenomeLoadError for an unreadable file."""
    if args.genome is not None:
        try:
            return load_genome(args.genome)
        except FileNotFoundError:
            raise GenomeLoadError(f"no genome file at {args.genome}") from None
        except (OSError, ValueError, TypeError) as exc:  # JSONDecodeError is a ValueError
            raise GenomeLoadError(f"could not read genome {args.genome}: {exc}") from None
    if args.reference is not None:
        return REFERENCES[args.reference]()
    return random_genome(make_rng(args.random), genome_id=f"rnd-seed{args.random}")


def env_from_args(args: argparse.Namespace) -> EnvSpec:
    return EnvSpec(
        kind=args.env,
        slope_deg=args.slope,
        friction=args.friction,
        roughness=args.roughness,
        n_obstacles=args.obstacles,
        seed=args.env_seed,
        friction_spread=args.friction_spread,
    )


def configs_from_args(args: argparse.Namespace) -> tuple[SimConfig, FitnessConfig]:
    if getattr(args, "config", None):
        return load_sim_config(args.config), load_fitness_config(args.config)
    return SimConfig(), FitnessConfig()
