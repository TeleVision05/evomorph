"""Evaluate genomes across ground friction: does speed fall when feet skate?

Usage:
    .venv/bin/python scripts/friction_sweep.py GENOME.json [...] [--reference quadruped]
        [--frictions 0.2 0.3 0.45 0.6 1.0 1.5] [--seed 0]

One evaluation per (genome, friction) on flat ground with the default sim and
fitness config, printing fitness, forward displacement, slip fraction and
whether it fell. A walker whose feet grip should slow down as friction drops;
a skater keeps its speed while its slip fraction climbs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evomorph.evolution.modes import REFERENCE_TEMPLATES
from evomorph.morphology.genome import load_genome
from evomorph.simulation.environment import EnvSpec
from evomorph.simulation.evaluator import evaluate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("genomes", type=Path, nargs="*")
    ap.add_argument("--reference", action="append", choices=sorted(REFERENCE_TEMPLATES))
    ap.add_argument("--frictions", type=float, nargs="+", default=[0.2, 0.3, 0.45, 0.6, 1.0, 1.5])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    creatures = [load_genome(p) for p in args.genomes]
    creatures += [REFERENCE_TEMPLATES[name]() for name in args.reference or []]
    for g in creatures:
        for mu in args.frictions:
            r = evaluate(g, EnvSpec(friction=mu), seed=args.seed)
            m = r.metrics
            print(
                f"{g.id} friction {mu:g}: fitness {r.fitness:+.3f} "
                f"forward {m.get('forward_displacement', float('nan')):.2f} m "
                f"slip {m.get('slip_fraction', float('nan')):.2f} "
                f"fell {m.get('fell', float('nan')):.0f}"
            )


if __name__ == "__main__":
    main()
