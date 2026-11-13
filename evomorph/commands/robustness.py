"""``robustness``: evaluate one genome across a terrain suite and save the table.

    python -m evomorph robustness --genome best.json --suite test --seeds 3
    python -m evomorph robustness --list

Writes ``<out>/<genome id>_<suite>.csv`` (one row per environment) and a
``.json`` with the per-seed results, aggregates and the exact configs. An
existing file is never overwritten: the name gets ``-2``, ``-3`` ... instead.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import mujoco

from evomorph.commands._common import REFERENCES, configs_from_args, genome_from_args
from evomorph.experiments.robustness import (
    MAX_WORKERS,
    evaluate_suite,
    table_lines,
    write_csv,
)
from evomorph.morphology.genome import genome_hash
from evomorph.simulation.suites import SUITE_DOCS, SUITES, get_suite
from evomorph.simulation.terrain import resolve
from evomorph.utils.serialization import dump_json


def suite_listing() -> list[str]:
    """Every suite with its environments' exact parameters (markdown table rows)."""
    lines = []
    for name, envs in SUITES.items():
        lines += [f"### {name}: {SUITE_DOCS[name]}", ""]
        lines.append(
            "| environment | kind | slope deg | roughness m | obstacles | friction | spread "
            "| terrain seed |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for e in envs:
            lines.append(
                f"| {e.label()} | {e.kind} | {e.slope_deg:g} | {e.roughness:g} | "
                f"{e.n_obstacles} | {e.friction:g} | {e.friction_spread:g} | {e.seed} |"
            )
        lines.append("")
    return lines


def _free_path(path: Path) -> Path:
    """``path`` or, if it exists, the first of ``stem-2``, ``stem-3`` ... that does not."""
    k = 2
    out = path
    while out.exists() or out.with_suffix(".json").exists():
        out = path.with_name(f"{path.stem}-{k}{path.suffix}")
        k += 1
    return out


def run(args: argparse.Namespace) -> int:
    if args.list:
        print("\n".join(suite_listing()))
        return 0
    if args.genome is None and args.reference is None and args.random is None:
        print("error: name a genome with --genome, --reference or --random", file=sys.stderr)
        return 2
    try:
        envs = get_suite(args.suite)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.seeds < 1:
        print("error: --seeds must be at least 1", file=sys.stderr)
        return 2
    genome = genome_from_args(args)
    sim, fit = configs_from_args(args)
    seeds = tuple(range(args.seed_start, args.seed_start + args.seeds))
    workers = max(1, min(args.workers, MAX_WORKERS))
    report = evaluate_suite(genome, args.suite, sim, fit, seeds=seeds, workers=workers)
    print("\n".join(table_lines(report)))
    csv_path = _free_path(args.out / f"{report.genome_id}_{report.suite}.csv")
    write_csv(report, csv_path)
    record = {
        **report.to_dict(),
        "genome_hash": genome_hash(genome),
        "terrain_draws": [asdict(resolve(e)) for e in envs],
        "sim": asdict(sim),
        "fitness": asdict(fit),
        "mujoco_version": mujoco.__version__,
        "command": " ".join(["python -m evomorph", *sys.argv[1:]]),
    }
    json_path = dump_json(record, csv_path.with_suffix(".json"))
    print(f"wrote {csv_path} and {json_path}")
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("robustness", help="evaluate a genome across a terrain suite")
    # The same genome options as evaluate, but optional so that --list needs none.
    src = p.add_mutually_exclusive_group()
    src.add_argument("--genome", type=Path, help="genome JSON file")
    src.add_argument("--reference", choices=sorted(REFERENCES), help="built-in baseline")
    src.add_argument("--random", type=int, metavar="SEED", help="random genome from SEED")
    p.add_argument("--suite", default="test", help=f"terrain suite: {', '.join(SUITES)}")
    p.add_argument("--seeds", type=int, default=3, help="spawn seeds per environment")
    p.add_argument("--seed-start", type=int, default=0, help="first spawn seed")
    p.add_argument("--config", type=Path, help="YAML with sim/fitness sections")
    p.add_argument(
        "--workers", type=int, default=1, help=f"parallel processes (capped at {MAX_WORKERS})"
    )
    p.add_argument("--out", type=Path, default=Path("results/robustness"), help="output folder")
    p.add_argument("--list", action="store_true", help="print every suite's parameters and exit")
    p.set_defaults(func=run)
