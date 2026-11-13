"""Evaluate N seeded random genomes and summarise validity, displacement and fitness.

Usage:
    .venv/bin/python scripts/random_sweep.py --n 300 --seed 0 --workers 10 \
        --out results/sweep_random_300

Writes ``summary.json`` (per-genome results) and the genomes into ``--out`` and
prints distribution statistics plus the top creatures by fitness.
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
    m = r.metrics
    return {
        "id": g.id,
        "n_limbs": g.n_limbs,
        "n_segments": g.morphology.n_segments,
        "gait": g.controller.gait,
        "valid": r.valid,
        "failure_reason": r.failure_reason,
        "fitness": r.fitness,
        "metrics": m,
        "terms": r.fitness_terms,
        "wall_seconds": r.wall_seconds,
    }


def _pct(a: np.ndarray) -> str:
    q = np.percentile(a, [0, 10, 25, 50, 75, 90, 100])
    return " ".join(f"{v:+.3f}" for v in q)


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
        results = list(pool.map(evaluate, genomes, chunksize=4))
    wall = time.perf_counter() - t0
    rows = [_row(g, r) for g, r in zip(genomes, results, strict=True)]
    args.out.mkdir(parents=True, exist_ok=True)
    dump_json(rows, args.out / "summary.json")
    for g in genomes:
        save_genome(g, args.out / "genomes" / f"{g.id}.json")

    valid = [r for r in results if r.valid]
    reasons = collections.Counter(
        re.sub(r"[-+]?\d+(\.\d+)?", "#", r.failure_reason) for r in results if r.failure_reason
    )
    fit = np.array([r.fitness for r in valid])
    fwd = np.array([r.metrics["forward_displacement"] for r in valid])
    nan_metrics = sum(any(not np.isfinite(v) for v in r.metrics.values()) for r in valid)
    print(
        f"evaluated {args.n} genomes in {wall:.1f}s with {args.workers} workers "
        f"({args.n / wall:.1f} eval/s); mean single-eval wall "
        f"{np.mean([r.wall_seconds for r in results]):.3f}s"
    )
    print(
        f"valid {len(valid)}/{args.n} ({len(valid) / args.n:.1%}); "
        f"fell {sum(r.fell for r in valid)}; non-finite metrics in valid results: {nan_metrics}"
    )
    print("failure reasons:", dict(reasons))
    print("percentiles               min    p10    p25    p50    p75    p90    max")
    print("forward_displacement (m):", _pct(fwd))
    print("fitness:                 ", _pct(fit))
    for thr in (0.25, 0.5, 1.0, 2.0):
        print(f"  fraction moving >= {thr} m forward: {np.mean(fwd >= thr):.1%}")
    order = sorted(range(len(results)), key=lambda i: -results[i].fitness)
    print(f"top {args.top}:")
    for i in order[: args.top]:
        g, r = genomes[i], results[i]
        m = r.metrics
        print(
            f"  {g.id} fit={r.fitness:+.3f} limbs={g.n_limbs} seg={g.morphology.n_segments} "
            f"gait={g.controller.gait} f={g.controller.frequency:.2f} "
            f"fwd={m['forward_displacement']:+.2f} sus={m['sustained_velocity']:+.3f} "
            f"lat={m['lateral_displacement']:.2f} yaw={m['yaw_drift']:.2f} "
            f"tilt={m['mean_tilt']:.2f} air={m['airborne_fraction']:.2f} "
            f"acc={m['accel_rms']:.1f} {r.failure_reason}"
        )


if __name__ == "__main__":
    main()
