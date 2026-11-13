"""Evaluate N seeded random genomes and summarise validity, displacement and fitness.

Usage:
    .venv/bin/python scripts/random_sweep.py --n 300 --seed 0 --workers 10 \
        --out results/sweep_random_300

Writes ``summary.json`` (per-genome results) and the genomes into ``--out`` and
prints validity, falls by body plan, distribution statistics, the extremes the
glitch guards watch, and the top creatures by fitness.
"""

from __future__ import annotations

import argparse
import collections
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from evomorph.morphology.genome import random_genome, save_genome
from evomorph.simulation.evaluator import EvalResult, evaluate
from evomorph.utils.random_state import spawn_rngs
from evomorph.utils.serialization import dump_json


def _row(g, r: EvalResult) -> dict:
    return {
        "id": g.id,
        "n_limbs": g.n_limbs,
        "n_segments": g.morphology.n_segments,
        "gait": g.controller.gait,
        "valid": r.valid,
        "failure_reason": r.failure_reason,
        "fitness": r.fitness,
        "metrics": r.metrics,
        "terms": r.fitness_terms,
        "wall_seconds": r.wall_seconds,
    }


def _pct(a: np.ndarray) -> str:
    q = np.percentile(a, [0, 10, 25, 50, 75, 90, 100])
    return " ".join(f"{v:+.3f}" for v in q)


def _print_population(genomes, results, wall: float, workers: int) -> None:
    n = len(results)
    valid = [r for r in results if r.valid]
    reasons = collections.Counter(
        re.sub(r"[-+]?\d+(\.\d+)?", "#", r.failure_reason) for r in results if r.failure_reason
    )
    nan_metrics = sum(any(not np.isfinite(v) for v in r.metrics.values()) for r in valid)
    print(f"evaluated {n} genomes in {wall:.1f}s with {workers} workers ({n / wall:.1f} eval/s)")
    print(f"valid {len(valid)}/{n}; fell {sum(r.fell for r in valid)}; "
          f"non-finite metrics in valid results: {nan_metrics}")  # fmt: skip
    print("failure reasons:", dict(reasons))
    plans: dict[tuple, list[int]] = collections.defaultdict(lambda: [0, 0])
    for g, r in zip(genomes, results, strict=True):
        key = (g.n_limbs, g.morphology.n_segments)
        plans[key][0] += int(r.fell)
        plans[key][1] += 1
    print("fell by plan (limbs x segments):",
          ", ".join(f"{k[0]}x{k[1]} {v[0]}/{v[1]}" for k, v in sorted(plans.items())))  # fmt: skip
    m = {k: np.array([r.metrics[k] for r in valid]) for k in valid[0].metrics} if valid else {}
    if not m:
        return
    print("percentiles               min    p10    p25    p50    p75    p90    max")
    print("forward_displacement (m):", _pct(m["forward_displacement"]))
    print("fitness:                 ", _pct(np.array([r.fitness for r in valid])))
    for thr in (0.25, 1.0, 2.0):
        print(
            f"  fraction moving >= {thr} m forward: {np.mean(m['forward_displacement'] >= thr):.1%}"
        )
    print(
        f"guard extremes: peak speed {m['peak_speed'].max():.2f} m/s, max height gain "
        f"{m['max_height_gain'].max():.3f} m, max penetration "
        f"{m['max_penetration'].max():.3f} m; bodies with rest-pose exclusions "
        f"{int(np.sum(m['rest_exclusions'] > 0))}"
    )


def _print_top(genomes, results, top: int) -> None:
    order = sorted(range(len(results)), key=lambda i: -results[i].fitness)
    print(f"top {top}:")
    for i in order[:top]:
        g, r = genomes[i], results[i]
        m = r.metrics
        print(
            f"  {g.id} fit={r.fitness:+.3f} limbs={g.n_limbs} seg={g.morphology.n_segments} "
            f"gait={g.controller.gait} f={g.controller.frequency:.2f} "
            f"fwd={m['forward_displacement']:+.2f} sus={m['sustained_velocity']:+.3f} "
            f"latmax={m['max_lateral_deviation']:.2f} yaw={m['yaw_drift']:.2f} "
            f"tilt={m['mean_tilt']:.2f} cot={m['cost_of_transport']:.2f} "
            f"support={m['foot_support_fraction']:.2f} slip={m['slip_fraction']:.2f} "
            f"air={m['airborne_fraction']:.2f} {r.failure_reason}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("results/sweep_random"))
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    genomes = [
        random_genome(rng, genome_id=f"rnd{args.seed}-{i:04d}")
        for i, rng in enumerate(spawn_rngs(args.seed, args.n))
    ]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(evaluate, genomes, chunksize=2))
    wall = time.perf_counter() - t0
    args.out.mkdir(parents=True, exist_ok=True)
    dump_json(
        [_row(g, r) for g, r in zip(genomes, results, strict=True)], args.out / "summary.json"
    )
    for g in genomes:
        save_genome(g, args.out / "genomes" / f"{g.id}.json")
    _print_population(genomes, results, wall, args.workers)
    _print_top(genomes, results, args.top)


if __name__ == "__main__":
    main()
