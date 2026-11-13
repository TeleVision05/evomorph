"""How condition B's fixed baseline controller drives bodies of every limb count.

Bodies: the reference quadruped re-shaped to 1-4 limb pairs and 1-2 segments
(nothing else changed, then repaired to valid), plus the reference hexapod's
own body. Each is driven by the baseline controller exactly as condition B
does (evomorph.experiments.baselines.baseline_controller_genome) and scored on
flat ground, friction 1, spawn seeds 0-2.

Usage:
    .venv/bin/python scripts/baseline_check.py [--workers 6]
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np

from evomorph.evolution.pool import clamp_workers
from evomorph.experiments.baselines import baseline_controller_genome
from evomorph.morphology.genome import Genome, reference_hexapod, reference_quadruped, repair
from evomorph.simulation.evaluator import evaluate

SEEDS = (0, 1, 2)


def bodies() -> list[Genome]:
    out = []
    base = reference_quadruped()
    for pairs in (1, 2, 3, 4):
        for segments in (1, 2):
            m = replace(base.morphology, n_pairs=pairs, n_segments=segments)
            out.append(repair(replace(base, id=f"quad-body-{2 * pairs}x{segments}", morphology=m)))
    out.append(replace(reference_hexapod(), id="hexapod-body-6x2"))
    return [baseline_controller_genome(g) for g in out]


def _run(task: tuple[Genome, int]) -> tuple[float, float, bool, bool]:
    genome, seed = task
    r = evaluate(genome, seed=seed)
    speed = r.metrics.get("avg_forward_velocity", float("nan")) if r.valid else float("nan")
    return float(r.fitness), float(speed), bool(r.valid and r.fell), bool(r.valid)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    genomes = bodies()
    tasks = [(g, s) for g in genomes for s in SEEDS]
    with ProcessPoolExecutor(max_workers=clamp_workers(args.workers)) as pool:
        results = list(pool.map(_run, tasks))
    print(f"{'body':<26} {'fitness':>8} {'speed m/s':>10} {'fell':>5} {'invalid':>8}")
    for i, g in enumerate(genomes):
        rs = results[i * len(SEEDS) : (i + 1) * len(SEEDS)]
        fit = np.mean([r[0] for r in rs])
        speed = np.nanmean([r[1] for r in rs]) if any(r[3] for r in rs) else float("nan")
        fell = sum(r[2] for r in rs)
        invalid = sum(not r[3] for r in rs)
        print(f"{g.id:<26} {fit:>+8.3f} {speed:>+10.3f} {fell:>3}/3 {invalid:>6}/3")


if __name__ == "__main__":
    main()
