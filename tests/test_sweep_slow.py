"""Full 300-genome random sweep (validation item 3). Run with ``pytest -m slow``."""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest

from evomorph.morphology.genome import random_genome
from evomorph.simulation.evaluator import evaluate
from evomorph.utils.random_state import spawn_rngs


@pytest.mark.slow
def test_random_sweep_300() -> None:
    genomes = [random_genome(r, genome_id=f"s{i}") for i, r in enumerate(spawn_rngs(0, 300))]
    with ProcessPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(evaluate, genomes, chunksize=4))
    assert all(r.valid for r in results)
    assert all(math.isfinite(v) for r in results for v in r.metrics.values())
    fwd = np.array([r.metrics["forward_displacement"] for r in results])
    assert np.median(np.abs(fwd)) < 0.3  # most move poorly
    assert np.sum(fwd > 1.0) >= 3  # a few move
    # No exploit among the best: they stay up, stand on their feet, keep a foot
    # down, do not launch and go roughly straight (checked visually 2026-11-13).
    for r in sorted(results, key=lambda r: -r.fitness)[:3]:
        m = r.metrics
        assert m["fell"] == 0.0 and m["foot_support_fraction"] > 0.9, (r.genome_id, m)
        assert m["airborne_fraction"] < 0.2 and m["max_height_gain"] < 0.15, (r.genome_id, m)
        assert m["max_lateral_deviation"] < 0.5 * m["forward_displacement"], (r.genome_id, m)
