"""Parent selection, elitism and the diversity mechanism, plus diversity measures.

Selection is a tournament on a *selection score*. Without diversity the score
is the fitness rank. With ``method: sharing`` the rank score is divided by the
individual's niche count on normalised gene distance (fitness sharing,
Goldberg & Richardson 1987). Sharing works on ranks rather than raw fitness
because fitness here can be negative (falls, backward motion), and dividing a
negative number by a niche count would *reward* crowding.

Two further protections run alongside sharing when diversity is on:

* body-plan elites: the best valid individual of every body plan
  (limbs x segments) survives unchanged, so a new plan is not wiped out the
  generation after it appears, before its controller has been tuned;
* random immigrants: a small fraction of every generation is fresh random
  genomes, which keeps reintroducing plans the population has lost.

Diversity measures logged every generation:

* ``mean_gene_distance``: mean pairwise normalised gene distance (see
  :func:`gene_distance_matrix`), 0 = clones; about 0.35 for random genomes;
* ``n_body_plans``: distinct (limbs, segments) combinations present;
* ``behaviour_spread``: mean pairwise Euclidean distance between valid
  individuals in the normalised MAP-Elites descriptor space (speed, efficiency),
  0 = everyone moves the same way.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from evomorph.morphology.genome import GENE_BOUNDS, Genome

DIVERSITY_METHODS: tuple[str, ...] = ("sharing", "none")


@dataclass(frozen=True)
class DiversityConfig:
    """Diversity mechanism settings (``method: none`` switches all of it off).

    ``sharing_radius = 0.15`` is about 40% of the mean distance between two
    random genomes (~0.35): individuals closer than that are treated as one
    niche. ``plan_elites = 1`` protects the best of each of at most 8 body plans.
    ``immigrant_fraction = 0.05`` is 2-3 random genomes in a population of 48:
    enough to keep reintroducing lost plans, cheap enough not to stall progress.
    """

    method: str = "sharing"
    sharing_radius: float = 0.15
    sharing_alpha: float = 1.0  # sh(d) = 1 - (d / radius) ** alpha inside the radius
    plan_elites: int = 1
    immigrant_fraction: float = 0.05

    def __post_init__(self) -> None:
        if self.method not in DIVERSITY_METHODS:
            raise ValueError(f"diversity.method must be one of {DIVERSITY_METHODS}")
        if self.sharing_radius <= 0 or self.sharing_alpha <= 0:
            raise ValueError("diversity.sharing_radius and sharing_alpha must be > 0")
        if not 0.0 <= self.immigrant_fraction < 1.0:
            raise ValueError("diversity.immigrant_fraction must be in [0, 1)")

    @property
    def enabled(self) -> bool:
        return self.method != "none"

    @property
    def effective_plan_elites(self) -> int:
        return self.plan_elites if self.enabled else 0

    @property
    def effective_immigrants(self) -> float:
        return self.immigrant_fraction if self.enabled else 0.0


def body_plan(genome: Genome) -> str:
    """``"<limbs>x<segments>"``, e.g. ``"4x2"``."""
    return f"{genome.n_limbs}x{genome.morphology.n_segments}"


# ---------------------------------------------------------------------------
# Gene distance
# ---------------------------------------------------------------------------


def _scalar_features(genomes: Sequence[Genome], genes: Sequence[str]) -> np.ndarray:
    """(N, k) matrix; |difference| of a column is that gene's distance in [0, 1]."""
    cols = []
    for name in genes:
        b = GENE_BOUNDS[name]
        group = "morphology" if b.group == "morphology" else "controller"
        vals = [getattr(getattr(g, group), name) for g in genomes]
        if b.kind == "choice":  # one-hot x 1/2: two different choices differ by 1 in L1
            for v in b.values:
                cols.append([0.5 * (x == v) for x in vals])
        elif b.kind == "int":
            lo, hi = min(b.values), max(b.values)
            cols.append([(float(x) - lo) / max(hi - lo, 1) for x in vals])
        else:
            cols.append([(float(x) - b.low) / b.span for x in vals])
    return np.asarray(cols, dtype=float).T if cols else np.zeros((len(genomes), 0))


def _array_features(genomes: Sequence[Genome], name: str) -> np.ndarray:
    """(N, L) normalised elements padded with NaN to the longest array."""
    b = GENE_BOUNDS[name]
    arrays = [np.asarray(getattr(g.controller, name), dtype=float).ravel() for g in genomes]
    width = max((a.size for a in arrays), default=0)
    out = np.full((len(genomes), width), np.nan)
    for i, a in enumerate(arrays):
        out[i, : a.size] = (a - b.low) / b.span
    return out


def gene_distance_matrix(genomes: Sequence[Genome], genes: Sequence[str]) -> np.ndarray:
    """Pairwise normalised gene distance, (N, N), each entry in [0, 1].

    Every gene contributes a distance in [0, 1]: |delta| / span for continuous
    and integer genes, 0 or 1 for choice genes, and for per-limb or vector genes
    the mean |delta| / span over the limbs both genomes have (0 if they share
    none). The result is the mean over genes, so it does not depend on how many
    genes a mode evolves.
    """
    n = len(genomes)
    if n == 0:
        return np.zeros((0, 0))
    arrays = [g for g in genes if GENE_BOUNDS[g].per_limb or GENE_BOUNDS[g].kind == "vector"]
    scalars = [g for g in genes if g not in arrays]
    feats = _scalar_features(genomes, scalars)
    total = np.abs(feats[:, None, :] - feats[None, :, :]).sum(axis=2)
    for name in arrays:
        f = _array_features(genomes, name)
        if f.shape[1] == 0:
            continue
        diff = np.abs(f[:, None, :] - f[None, :, :])
        common = np.sum(np.isfinite(diff), axis=2)
        summed = np.nansum(diff, axis=2)
        total += np.divide(summed, common, out=np.zeros_like(summed), where=common > 0)
    n_genes = max(len(scalars) + len(arrays), 1)
    return total / n_genes


def mean_pairwise(dist: np.ndarray) -> float:
    """Mean of the off-diagonal entries (0 for fewer than two individuals)."""
    n = dist.shape[0]
    if n < 2:
        return 0.0
    return float(dist[np.triu_indices(n, k=1)].mean())


def behaviour_spread(points: np.ndarray) -> float:
    """Mean pairwise Euclidean distance between rows of ``points`` (normalised descriptors)."""
    if len(points) < 2:
        return 0.0
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    return mean_pairwise(d)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def rank_order(fitness: np.ndarray) -> np.ndarray:
    """Indices best first; ties broken by index so the order is deterministic."""
    return np.lexsort((np.arange(len(fitness)), -np.asarray(fitness, dtype=float)))


def selection_scores(fitness: np.ndarray, dist: np.ndarray | None, cfg: DiversityConfig):
    """Score used by the tournament: rank score, divided by niche count if sharing.

    Rank score is ``(N - rank) / N`` (best 1, worst 1/N). The niche count of i is
    ``sum_j sh(d_ij)`` with ``sh(d) = max(0, 1 - (d / radius) ** alpha)``, which
    counts i itself once, so an individual with no neighbours keeps its score.
    """
    n = len(fitness)
    order = rank_order(fitness)
    rank_score = np.empty(n)
    rank_score[order] = (n - np.arange(n)) / n
    if not cfg.enabled or dist is None:
        return rank_score
    sh = np.clip(1.0 - (dist / cfg.sharing_radius) ** cfg.sharing_alpha, 0.0, 1.0)
    return rank_score / sh.sum(axis=1)


def tournament(scores: np.ndarray, rng: np.random.Generator, size: int) -> int:
    """Index of the best-scoring of ``size`` individuals drawn with replacement."""
    picks = rng.integers(0, len(scores), size=max(1, size))
    best = picks[0]
    for i in picks[1:]:
        if scores[i] > scores[best] or (scores[i] == scores[best] and i < best):
            best = i
    return int(best)


def choose_elites(
    fitness: np.ndarray,
    valid: np.ndarray,
    plans: Sequence[str],
    elite_count: int,
    plan_elites: int,
) -> list[int]:
    """Indices that survive unchanged: the global top ``elite_count`` (valid only),
    then the best ``plan_elites`` valid individuals of every body plan.

    Returned in rank order without duplicates.
    """
    order = [int(i) for i in rank_order(fitness) if valid[i]]
    chosen = order[: max(elite_count, 0)]
    if plan_elites > 0:
        per_plan: dict[str, int] = {}
        for i in order:
            if per_plan.get(plans[i], 0) < plan_elites:
                per_plan[plans[i]] = per_plan.get(plans[i], 0) + 1
                if i not in chosen:
                    chosen.append(i)
    rank = {i: r for r, i in enumerate(order)}
    return sorted(chosen, key=lambda i: rank[i])
