"""``evolve``: run the GA or MAP-Elites into a fresh results directory.

Examples::

    python -m evomorph evolve --config configs/quick.yaml
    python -m evomorph evolve --config configs/quick.yaml --algorithm map_elites
    python -m evomorph evolve --config configs/quick.yaml --mode controller --budget 3000
    python -m evomorph evolve --config configs/quick.yaml --no-diversity --name ga_nodiv

Command-line options override the config's ``evolution`` section; the fully
resolved result is saved as ``config.yaml`` in the run directory.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from evomorph.evolution.modes import MODES
from evomorph.evolution.pool import MAX_WORKERS
from evomorph.evolution.rundir import RunDirExistsError
from evomorph.evolution.settings import ALGORITHMS, resolve


def _overrides(args: argparse.Namespace) -> dict[str, object]:
    return {
        "algorithm": args.algorithm,
        "mode": args.mode,
        "workers": args.workers,
        "budget": args.budget,
        "seed": args.seed,
        "diversity_method": "none" if args.no_diversity else None,
    }


def run(args: argparse.Namespace) -> int:
    from evomorph.evolution.runner import run_evolution

    if args.config is not None and not Path(args.config).is_file():
        print(f"error: no config file at {args.config}", file=sys.stderr)
        return 2
    try:
        cfg = resolve(args.config, _overrides(args))
    except (ValueError, TypeError) as exc:
        print(f"error: the configuration is not usable: {exc}", file=sys.stderr)
        return 2
    if args.workers is not None and args.workers > MAX_WORKERS:
        print(f"note: using {MAX_WORKERS} workers (the maximum), not {args.workers}")
    try:
        outcome = run_evolution(cfg, args.results, args.name, command=command_line(args))
    except RunDirExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0 if outcome.champion is not None else 1


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("evolve", help="evolve creatures (GA or MAP-Elites)")
    p.add_argument("--config", type=Path, help="YAML config (sim, fitness, evolution, map_elites)")
    p.add_argument("--algorithm", choices=ALGORITHMS, help="override evolution.algorithm")
    p.add_argument("--mode", choices=MODES, help="what evolves: joint, controller or morphology")
    p.add_argument("--workers", type=int, help=f"process-pool size (capped at {MAX_WORKERS})")
    p.add_argument("--budget", type=int, help="simulations to spend (genomes x runs)")
    p.add_argument("--seed", type=int, help="random seed of the search")
    p.add_argument("--name", help="run directory name (default: next free run_NNN)")
    p.add_argument("--results", type=Path, default=Path("results"), help="results root folder")
    p.add_argument(
        "--no-diversity", action="store_true", help="switch the GA's diversity mechanism off"
    )
    p.set_defaults(func=run)


def command_line(args: argparse.Namespace) -> str:
    """The equivalent command line, rebuilt from the parsed options (for summary.md)."""
    parts = ["python", "-m", "evomorph", "evolve"]
    for flag in ("config", "algorithm", "mode", "workers", "budget", "seed", "name"):
        value = getattr(args, flag)
        if value is not None:
            parts += [f"--{flag}", str(value)]
    if str(args.results) != "results":
        parts += ["--results", str(args.results)]
    if args.no_diversity:
        parts.append("--no-diversity")
    return " ".join(shlex.quote(p) for p in parts)
