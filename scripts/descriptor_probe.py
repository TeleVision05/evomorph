"""Measure throughput and the range of the MAP-Elites descriptors creatures reach.

Usage:
    .venv/bin/python scripts/descriptor_probe.py [--n 200] [--seed 7] [--workers 6]

Evaluates ``n`` random genomes plus the two references on flat ground at
friction 0.6 and 1.0 (spawn seed 0), and prints simulations per second and the
percentiles of ``sustained_velocity`` and ``energy_efficiency`` over valid,
non-fallen runs. The archive ranges in configs/*.yaml were chosen from its
output with ``--n 200 --seed 7 --workers 6``.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from evomorph.evolution.pool import clamp_workers
from evomorph.morphology.genome import random_genome, reference_hexapod, reference_quadruped
from evomorph.simulation.environment import EnvSpec
from evomorph.simulation.evaluator import evaluate
from evomorph.utils.random_state import spawn_rngs

FRICTIONS = (0.6, 1.0)


def _job(args):
    genome, mu = args
    return evaluate(genome, env=EnvSpec(friction=mu), seed=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    workers = clamp_workers(args.workers)
    gs = [random_genome(r, genome_id=f"p{i}") for i, r in enumerate(spawn_rngs(args.seed, args.n))]
    gs += [reference_quadruped(), reference_hexapod()]
    jobs = [(g, mu) for g in gs for mu in FRICTIONS]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        res = list(pool.map(_job, jobs, chunksize=1))
    wall = time.perf_counter() - t0
    print(f"{len(jobs)} simulations in {wall:.1f}s = {len(jobs) / wall:.1f}/s ({workers} workers)")
    ok = [r for r in res if r.valid and not r.fell]
    q = [0, 10, 50, 90, 99, 100]
    print(f"valid and not fallen: {len(ok)} of {len(res)}; percentiles {q}")
    for key in ("sustained_velocity", "energy_efficiency"):
        vals = np.array([r.metrics[key] for r in ok])
        print(f"  {key}: {np.round(np.percentile(vals, q), 3)}")
    for r in res[-2 * len(FRICTIONS) :]:
        m = r.metrics
        print(
            f"  {r.genome_id} friction {r.env.friction}: fitness {r.fitness:+.3f} "
            f"speed {m['sustained_velocity']:.3f} efficiency {m['energy_efficiency']:.3f}"
        )


if __name__ == "__main__":
    main()
