"""``evaluate``: simulate a genome, print its metrics and fitness, optionally record video."""

from __future__ import annotations

import argparse
from pathlib import Path

from evomorph.commands._common import (
    add_env_args,
    add_genome_args,
    configs_from_args,
    env_from_args,
    genome_from_args,
)
from evomorph.simulation.evaluator import EvalResult, RecordSpec, evaluate
from evomorph.simulation.metrics import METRIC_KEYS
from evomorph.utils.serialization import dump_json


def result_lines(result: EvalResult) -> list[str]:
    status = "valid" if result.valid else "INVALID"
    lines = [
        f"{result.genome_id} on {result.env.label()} seed={result.seed}: {status}"
        + (f" ({result.failure_reason})" if result.failure_reason else ""),
        f"fitness {result.fitness:+.4f}   sim {result.sim_seconds:.2f} s in "
        f"{result.wall_seconds:.3f} s wall",
    ]
    if result.metrics:
        lines.append("metrics:")
        lines += [f"  {k:<24} {result.metrics[k]:+.4f}" for k in METRIC_KEYS]
        lines.append("fitness terms:")
        lines += [f"  {k:<24} {v:+.4f}" for k, v in result.fitness_terms.items()]
    return lines


def run(args: argparse.Namespace) -> int:
    genome = genome_from_args(args)
    env = env_from_args(args)
    sim, fit = configs_from_args(args)
    if args.video:
        from evomorph.visualization.renderer import render_video

        spec = RecordSpec(width=args.width, height=args.height, fps=args.fps, include_settle=True)
        result = render_video(genome, args.video, env, sim, fit, seed=args.seed, spec=spec)
        print(f"wrote {args.video}")
    else:
        result = evaluate(genome, env=env, sim=sim, fitness=fit, seed=args.seed)
    print("\n".join(result_lines(result)))
    if args.json:
        result.frames = None
        print(f"wrote {dump_json(result, args.json)}")
    return 0 if result.valid else 1


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("evaluate", help="simulate a genome and print its metrics")
    add_genome_args(p)
    add_env_args(p)
    p.add_argument("--video", type=Path, help="write an mp4 of the run")
    p.add_argument("--json", type=Path, help="write the EvalResult as JSON")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=float, default=30.0)
    p.set_defaults(func=run)
