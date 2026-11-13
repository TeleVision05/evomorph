"""Seeded random-number helpers.

Every random draw in EVOMORPH goes through a ``numpy.random.Generator`` created
here, so experiments are reproducible from one integer seed.
"""

from __future__ import annotations

import numpy as np


def make_rng(seed: int | np.random.Generator | None) -> np.random.Generator:
    """Return a Generator. Passing a Generator returns it unchanged."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def spawn_rngs(seed: int, n: int) -> list[np.random.Generator]:
    """``n`` statistically independent Generators derived from one seed."""
    children = np.random.SeedSequence(seed).spawn(n)
    return [np.random.default_rng(c) for c in children]


def derive_seed(seed: int, *keys: int | str) -> int:
    """Deterministically derive a child seed from ``seed`` and any number of keys.

    Strings are hashed with a stable (non-salted) byte sum so the result does not
    change between Python processes.
    """
    entropy = [int(seed) & 0xFFFFFFFF]
    for key in keys:
        if isinstance(key, str):
            entropy.append(sum((i + 1) * b for i, b in enumerate(key.encode())) & 0xFFFFFFFF)
        else:
            entropy.append(int(key) & 0xFFFFFFFF)
    return int(np.random.SeedSequence(entropy).generate_state(1)[0])
