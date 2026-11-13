"""Shared fixtures and helpers."""

from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pytest

from evomorph.morphology.genome import Genome, reference_quadruped, repair

LIMB_PAIRS = (1, 2, 3, 4)
SEGMENTS = (1, 2)
ALL_CONFIGS = list(itertools.product(LIMB_PAIRS, SEGMENTS))


def configured_genome(n_pairs: int, n_segments: int, **morph) -> Genome:
    """Reference quadruped re-shaped to n_pairs x n_segments, repaired to valid."""
    base = reference_quadruped()
    m = replace(base.morphology, n_pairs=n_pairs, n_segments=n_segments, **morph)
    g = replace(base, id=f"cfg-{n_pairs}x{n_segments}", morphology=m)
    return repair(g)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)
