"""``experiment``: run the comparison experiment (brief section 9), resumably.

Examples::

    python -m evomorph experiment --type comparison                # full design
    python -m evomorph experiment --type comparison --quick        # smoke-test size
    python -m evomorph experiment --type comparison --budget 2200 --seeds 3 --name small

The design comes from the config's ``experiment`` section (``--quick`` applies
its ``quick`` sub-section); command-line options override both. Re-running the
same command skips finished runs. Then: ``python -m evomorph analyze --run <dir>``.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from evomorph.evolution.pool import MAX_WORKERS

EXPERIMENT_TYPES: tuple[str, ...] = ("comparison",)
DEFAULT_CONFIG = Path("configs/comparison.yaml")


def _overrides(args: argparse.Namespace) -> dict[str, object]:
    return {
        "budget": args.budget,
        "n_seeds": args.seeds,
        "seed_start": args.seed_start,
        "workers": args.workers,
    }


def run(args: argparse.Namespace) -> int:
    from evomorph.experiments.comparison import (
        BudgetMismatchError,
        DesignMismatchError,
        run_comparison,
    )
    from evomorph.experiments.design import load_experiment

    if not Path(args.config).is_file():
        print(f"error: no config file at {args.config}", file=sys.stderr)
        return 2
    try:
        exp, base = load_experiment(args.config, args.quick, _overrides(args))
    except (ValueError, TypeError) as exc:
        print(f"error: the experiment settings are not usable: {exc}", file=sys.stderr)
        return 2
    if args.workers is not None and args.workers > MAX_WORKERS:
        print(f"note: using {MAX_WORKERS} workers (the maximum), not {args.workers}")
    name = args.name or ("comparison_quick" if args.quick else "comparison")
    print(
        f"experiment {name}: {len(exp.conditions)} conditions x {exp.n_seeds} seeds, "
        f"{exp.effective_budget} simulations per run ({exp.sims_per_genome} per genome), "
        f"train suite '{exp.train_suite}', test suite '{exp.test_suite}'",
        flush=True,
    )
    try:
        run_comparison(exp, base, args.results, name, command=command_line(args))
    except (DesignMismatchError, BudgetMismatchError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("experiment", help="run the comparison experiment (resumable)")
    p.add_argument("--type", required=True, choices=EXPERIMENT_TYPES, help="which experiment")
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML with an experiment section (default {DEFAULT_CONFIG})",
    )
    p.add_argument("--quick", action="store_true", help="use the config's quick (smoke-test) size")
    p.add_argument("--budget", type=int, help="simulations per run, equal for every condition")
    p.add_argument("--seeds", type=int, help="number of search seeds per condition")
    p.add_argument("--seed-start", type=int, help="first search seed (default 0)")
    p.add_argument("--workers", type=int, help=f"process-pool size (capped at {MAX_WORKERS})")
    p.add_argument("--name", help="experiment folder name (default comparison[_quick])")
    p.add_argument(
        "--results",
        type=Path,
        default=Path("results/experiments"),
        help="where experiment folders go (default results/experiments)",
    )
    p.set_defaults(func=run)


def command_line(args: argparse.Namespace) -> str:
    """The equivalent command line, rebuilt from the parsed options (for the manifest)."""
    parts = ["python", "-m", "evomorph", "experiment", "--type", args.type]
    parts += ["--config", str(args.config)]
    if args.quick:
        parts.append("--quick")
    for flag in ("budget", "seeds", "seed_start", "workers", "name"):
        value = getattr(args, flag)
        if value is not None:
            parts += [f"--{flag.replace('_', '-')}", str(value)]
    if str(args.results) != "results/experiments":
        parts += ["--results", str(args.results)]
    return " ".join(shlex.quote(p) for p in parts)
