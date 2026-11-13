"""Genetic operators: mutation and crossover over ``GENE_BOUNDS``.

Every operator takes the tuple of *mutable* gene names (see
:func:`evomorph.evolution.modes.mutable_genes`) and never reads or writes any
other gene. :class:`Variation` then re-imposes the frozen group, repairs and
validates, so every child is a valid genome whatever the operators did.

Mutation, in the order the random draws are made (fixed, so runs reproduce):

1. Structural (discrete) changes, each with its own probability:
   limb-pair count +-1, segment count toggled, gait re-drawn.
   Changing the limb count resizes every per-limb gene: a new pair copies the
   current rear pair's values (gene duplication: the new legs move like their
   neighbours, rather than with a neutral zero that may break the gait), and a
   removed pair drops the rear pair's values.
2. Per gene: a random reset (uniform in bounds) with ``reset_rate``, otherwise
   a Gaussian step of ``sigma x span`` with ``gene_rate``. Per-limb and vector
   genes draw per element. Out-of-range values are reflected back into bounds
   rather than clipped, so children do not pile up exactly on a bound.
3. Per-limb phase mutation: with ``p_phase``, one limb's phase offset gets a
   larger step (``phase_sigma x span``), which is how a gait is reshaped one
   leg at a time.

Crossover is uniform per gene: each mutable gene comes from either parent with
probability 1/2. Per-limb arrays are inherited whole from one parent and then
resized to the child's limb count, so a leg's phase, hip bias and knee bias
stay a consistent set.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from evomorph.evolution.modes import Freezer, mutable_genes
from evomorph.morphology.genome import (
    GAITS,
    GENE_BOUNDS,
    PER_LIMB_GENES,
    Genome,
    random_genome,
)
from evomorph.morphology.validator import validate_genome


@dataclass(frozen=True)
class MutationConfig:
    """Probabilities and magnitude of mutation.

    The defaults change about three genes per child in joint mode (~23 mutable
    scalar genes x 0.12 plus a few per-limb elements): small enough that a
    walking parent usually has a walking child, large enough to cross a basin
    in a few generations. ``sigma = 0.1`` of a gene's span is the standard
    1/10-range step for real-coded GAs. Structural changes are rare (5%) because
    each one rebuilds the body or gait and usually costs fitness at first.
    """

    gene_rate: float = 0.12  # per-gene (per-element) probability of a Gaussian step
    sigma: float = 0.10  # Gaussian step std as a fraction of the gene's span (magnitude)
    reset_rate: float = 0.02  # per-gene probability of a uniform re-draw
    p_limb_pairs: float = 0.05  # add or remove one symmetric limb pair
    p_segments: float = 0.05  # toggle one- vs two-segment limbs
    p_gait: float = 0.05  # re-draw the gait pattern
    p_phase: float = 0.20  # per-limb phase mutation: one limb's phase offset ...
    phase_sigma: float = 0.25  # ... moves by this fraction of its span

    def __post_init__(self) -> None:
        for name in ("gene_rate", "reset_rate", "p_limb_pairs", "p_segments", "p_gait", "p_phase"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"mutation.{name} is a probability, got {v}")
        if self.sigma < 0 or self.phase_sigma < 0:
            raise ValueError("mutation magnitudes must be non-negative")


def reflect(x: np.ndarray | float, low: float, high: float) -> np.ndarray | float:
    """Fold values outside [low, high] back inside by mirror reflection."""
    span = high - low
    if span <= 0:
        return np.full_like(x, low) if isinstance(x, np.ndarray) else low
    y = np.mod(np.asarray(x, dtype=float) - low, 2.0 * span)
    y = np.where(y > span, 2.0 * span - y, y) + low
    return y if isinstance(x, np.ndarray) else float(y)


def resize_limbs(arr: np.ndarray, n_limbs: int) -> np.ndarray:
    """Resize a per-limb array to ``n_limbs`` (= 2 x pairs) by pairs.

    Growing appends copies of the rear pair (or zeros if the array is empty);
    shrinking drops pairs from the rear.
    """
    a = np.asarray(arr, dtype=float).ravel()
    if a.size >= n_limbs:
        return a[:n_limbs].copy()
    rear = a[-2:] if a.size >= 2 else np.zeros(2)
    reps = (n_limbs - a.size + 1) // 2
    return np.concatenate([a, np.tile(rear, reps)])[:n_limbs]


def _copy(genome: Genome) -> Genome:
    return replace(
        genome,
        morphology=copy.deepcopy(genome.morphology),
        controller=copy.deepcopy(genome.controller),
    )


def _get(g: Genome, name: str) -> Any:
    group = g.morphology if GENE_BOUNDS[name].group == "morphology" else g.controller
    return getattr(group, name)


def _set(g: Genome, name: str, value: Any) -> None:
    group = g.morphology if GENE_BOUNDS[name].group == "morphology" else g.controller
    setattr(group, name, value)


def _structural(g: Genome, rng: np.random.Generator, cfg: MutationConfig, genes: tuple) -> None:
    if "n_pairs" in genes and rng.random() < cfg.p_limb_pairs:
        values = GENE_BOUNDS["n_pairs"].values
        cur = values.index(g.morphology.n_pairs)
        step = -1 if rng.random() < 0.5 else 1
        if not 0 <= cur + step < len(values):
            step = -step  # at a bound the only move is inward
        g.morphology.n_pairs = values[cur + step]
        for name in PER_LIMB_GENES:
            _set(g, name, resize_limbs(_get(g, name), g.morphology.n_limbs))
    if "n_segments" in genes and rng.random() < cfg.p_segments:
        values = GENE_BOUNDS["n_segments"].values
        g.morphology.n_segments = next(v for v in values if v != g.morphology.n_segments)
    if "gait" in genes and rng.random() < cfg.p_gait:
        others = [x for x in GAITS if x != g.controller.gait]
        g.controller.gait = others[int(rng.integers(len(others)))]


_STRUCTURAL = ("n_pairs", "n_segments", "gait")


def _perturb(value: Any, bound, rng: np.random.Generator, cfg: MutationConfig) -> Any:
    """Reset or Gaussian-step one continuous gene (scalar or array, element-wise)."""
    arr = np.atleast_1d(np.asarray(value, dtype=float)).copy()
    reset = rng.random(arr.size) < cfg.reset_rate
    step = rng.random(arr.size) < cfg.gene_rate
    noise = rng.normal(0.0, cfg.sigma * bound.span, arr.size)
    fresh = rng.uniform(bound.low, bound.high, arr.size)
    arr = np.where(step, reflect(arr + noise, bound.low, bound.high), arr)
    arr = np.where(reset, fresh, arr)
    if bound.per_limb or bound.kind == "vector":
        return arr
    return float(arr[0])


def mutate(
    genome: Genome, rng: np.random.Generator, cfg: MutationConfig, genes: tuple[str, ...]
) -> Genome:
    """A mutated copy of ``genome``; only ``genes`` can change (see module doc)."""
    g = _copy(genome)
    _structural(g, rng, cfg, genes)
    for name in genes:
        bound = GENE_BOUNDS[name]
        if name in _STRUCTURAL:
            continue
        if bound.is_discrete:  # any other discrete gene: occasional re-draw
            if rng.random() < cfg.reset_rate:
                _set(g, name, bound.values[int(rng.integers(len(bound.values)))])
            continue
        _set(g, name, _perturb(_get(g, name), bound, rng, cfg))
    if "phase_offsets" in genes and rng.random() < cfg.p_phase:
        bound = GENE_BOUNDS["phase_offsets"]
        phases = np.asarray(g.controller.phase_offsets, dtype=float).copy()
        if phases.size:
            i = int(rng.integers(phases.size))
            phases[i] = reflect(
                phases[i] + rng.normal(0.0, cfg.phase_sigma * bound.span), bound.low, bound.high
            )
            g.controller.phase_offsets = phases
    return g


def crossover(a: Genome, b: Genome, rng: np.random.Generator, genes: tuple[str, ...]) -> Genome:
    """Uniform crossover of the ``genes`` of ``a`` and ``b``; other genes come from ``a``."""
    child = _copy(a)
    take_b = rng.random(len(genes)) < 0.5
    for name, from_b in zip(genes, take_b, strict=True):
        if from_b:
            _set(child, name, copy.deepcopy(_get(b, name)))
    for name in PER_LIMB_GENES:
        _set(child, name, resize_limbs(_get(child, name), child.morphology.n_limbs))
    return child


class InvalidChildError(RuntimeError):
    """Repair did not produce a valid genome (a bug: repair must always succeed)."""


@dataclass(frozen=True)
class Variation:
    """Makes valid children in one evolution mode: crossover, mutation, freeze, repair."""

    mutation: MutationConfig
    freezer: Freezer
    crossover_prob: float = 0.5

    @property
    def genes(self) -> tuple[str, ...]:
        return mutable_genes(self.freezer.mode)

    def finish(self, g: Genome, child_id: str, parents: tuple[str, ...], gen: int, origin: str):
        """Freeze, repair, validate and stamp identity on a raw child."""
        child = self.freezer.apply(g)
        child = replace(child, id=child_id, parent_ids=parents, generation=gen, origin=origin)
        check = validate_genome(child)
        if not check.ok:
            raise InvalidChildError(f"{child_id}: " + "; ".join(check.errors))
        return child

    def random(self, rng: np.random.Generator, child_id: str, gen: int) -> Genome:
        """A random individual of this mode (frozen genes from the template)."""
        g = random_genome(rng, genome_id=child_id, generation=gen, kind=self.freezer.kind)
        return self.finish(g, child_id, (), gen, "random")

    def child(
        self,
        rng: np.random.Generator,
        a: Genome,
        b: Genome | None,
        child_id: str,
        gen: int,
    ) -> Genome:
        """Crossover (with ``crossover_prob``, when ``b`` is given) then mutation."""
        genes = self.genes
        if b is not None and rng.random() < self.crossover_prob:
            raw, parents, origin = crossover(a, b, rng, genes), (a.id, b.id), "crossover"
        else:
            raw, parents, origin = a, (a.id,), "mutation"
        raw = mutate(raw, rng, self.mutation, genes)
        return self.finish(raw, child_id, parents, gen, origin)
