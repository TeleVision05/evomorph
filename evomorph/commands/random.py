"""``random``: write N seeded random genomes to a folder."""

from __future__ import annotations

import argparse
from pathlib import Path

from evomorph.morphology.genome import random_genome, save_genome
from evomorph.utils.random_state import spawn_rngs


def run(args: argparse.Namespace) -> int:
    out: Path = args.out
    if out.exists() and any(out.iterdir()) and not args.force:
        print(f"{out} is not empty; pass --force to add files to it")
        return 1
    for i, rng in enumerate(spawn_rngs(args.seed, args.n)):
        g = random_genome(rng, genome_id=f"rnd{args.seed}-{i:04d}")
        save_genome(g, out / f"{g.id}.json")
    print(f"wrote {args.n} genomes (seed {args.seed}) to {out}")
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("random", help="write N seeded random genomes")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--force", action="store_true", help="allow writing into a non-empty folder")
    p.set_defaults(func=run)
