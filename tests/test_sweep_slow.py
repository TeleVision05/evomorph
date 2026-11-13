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
