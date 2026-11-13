"""``analyze``: plots and a summary for a run directory or an experiment folder.

Examples::

    python -m evomorph analyze --run results/experiments/comparison
    python -m evomorph analyze --run results/run_001
    python -m evomorph analyze --run results/run_001 --no-render   # skip the 3-D gallery

Plots are written with matplotlib's Agg backend into ``<dir>/plots/``; creature
images come from MuJoCo's offscreen renderer. Nothing opens a window.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def run(args: argparse.Namespace) -> int:
    from evomorph.analysis.analyze import NothingToAnalyseError, analyze

    if not Path(args.run).is_dir():
        print(f"error: there is no folder at {args.run}", file=sys.stderr)
        return 2
    try:
        summary, plots = analyze(args.run, render=not args.no_render)
    except NothingToAnalyseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {len(plots)} plots to {Path(args.run) / 'plots'} and {summary}")
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("analyze", help="plots and summary for a run or an experiment")
    p.add_argument("--run", type=Path, required=True, help="run directory or experiment folder")
    p.add_argument(
        "--no-render", action="store_true", help="skip the offscreen-rendered creature gallery"
    )
    p.set_defaults(func=run)
