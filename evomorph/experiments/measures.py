"""Pure measures for the comparison: suite summaries, diversity, the final set.

Nothing here runs a simulation; :mod:`evomorph.experiments.evaluation` feeds it
compact run records (:data:`RECORD_METRICS` of one ``evaluate`` call).

Suite summary rules (the same as :mod:`evomorph.experiments.robustness`)
-------------------------------------------------------------------------
* An invalid run scores ``INVALID_FITNESS`` (as ``evaluate`` already returns
  it), exactly what evolution would have seen.
* ``fitness`` is the mean of per-environment means (every environment weighs
  the same); ``worst_fitness`` is the lowest per-environment mean.
* ``failure_rate`` counts runs that fell *or* were invalid; ``invalid_rate``
  reports the invalid share alone so a physics artefact cannot hide in it.
* Locomotion numbers (speed, efficiency, tilt, slip) are means over the valid
  runs only; NaN if there are none.

Diversity
---------
* Morphological diversity: mean pairwise normalised gene distance over the
  *morphology* genes only (the engine's :func:`gene_distance_matrix`, values in
  [0, 1]), plus the number of distinct body plans. Using the same gene set for
  every condition keeps the numbers comparable; condition A has one body, so
  its morphological diversity is 0 by construction.
* Behavioural diversity: mean pairwise Euclidean distance between behaviour
  vectors, divided by sqrt(dimensions) so it lies in [0, 1]. A behaviour vector
  is :data:`BEHAVIOUR` scaled by fixed ranges that were chosen from the
  metrics' definitions before any experiment ran (not from the data, so a
  condition cannot change the scale it is measured on).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from evomorph.evolution.selection import body_plan, gene_distance_matrix, mean_pairwise
from evomorph.morphology.genome import MORPHOLOGY_GENES, Genome

# (metric, low, high). Speed and efficiency use the MAP-Elites archive ranges
# (measured probe, see evolution/archive.py). Tilt spans 0 to tilt_scale
# (60 deg), where the fitness's upright factor reaches 0. Slip and airborne
# are fractions. Yaw drift spans 0 to pi: a creature that turned round.
BEHAVIOUR: tuple[tuple[str, float, float], ...] = (
    ("sustained_velocity", 0.0, 0.8),
    ("energy_efficiency", 0.0, 0.7),
    ("mean_tilt", 0.0, math.pi / 3),
    ("slip_fraction", 0.0, 1.0),
    ("airborne_fraction", 0.0, 1.0),
    ("yaw_drift", 0.0, math.pi),
)

# Metrics kept from every post-hoc run: the behaviour vector plus speed and fall.
RECORD_METRICS: tuple[str, ...] = (
    "avg_forward_velocity",
    *(name for name, _, _ in BEHAVIOUR),
    "fell",
)

SUMMARY_KEYS: tuple[str, ...] = (
    "fitness",
    "worst_fitness",
    "failure_rate",
    "invalid_rate",
    "speed",
    "efficiency",
    "tilt",
    "slip",
)


def _nanmean(values: Sequence[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return float(np.mean(vals)) if vals else math.nan


def suite_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarise one genome's runs on one suite (rules in the module doc).

    Each record has ``env``, ``seed``, ``fitness``, ``valid``, ``fell`` and
    ``metrics`` (the :data:`RECORD_METRICS`, empty when invalid).
    """
    by_env: dict[str, list[float]] = {}
    for r in records:
        by_env.setdefault(r["env"], []).append(float(r["fitness"]))
    env_means = [float(np.mean(v)) for v in by_env.values()]
    valid = [r for r in records if r["valid"]]
    failed = sum(1 for r in records if r["fell"] or not r["valid"])

    def metric(key: str) -> float:
        return _nanmean([float(r["metrics"][key]) for r in valid])

    return {
        "fitness": float(np.mean(env_means)),
        "worst_fitness": float(np.min(env_means)),
        "failure_rate": failed / len(records),
        "invalid_rate": (len(records) - len(valid)) / len(records),
        "speed": metric("avg_forward_velocity"),
        "efficiency": metric("energy_efficiency"),
        "tilt": metric("mean_tilt"),
        "slip": metric("slip_fraction"),
        "behaviour": behaviour_mean(valid),
        "n_runs": len(records),
    }


def behaviour_vector(metrics: dict[str, float]) -> np.ndarray:
    """:data:`BEHAVIOUR` metrics scaled to [0, 1] by their fixed ranges (clipped)."""
    out = [(float(metrics[k]) - lo) / (hi - lo) for k, lo, hi in BEHAVIOUR]
    return np.clip(np.asarray(out, dtype=float), 0.0, 1.0)


def behaviour_mean(valid_records: Sequence[dict[str, Any]]) -> list[float] | None:
    """Mean behaviour vector over valid runs (None if there are none)."""
    if not valid_records:
        return None
    vecs = [behaviour_vector(r["metrics"]) for r in valid_records]
    return [float(x) for x in np.mean(vecs, axis=0)]


def behaviour_diversity(vectors: Sequence[Sequence[float] | None]) -> float:
    """Mean pairwise distance / sqrt(dims) over the non-None vectors, in [0, 1]."""
    pts = np.array([v for v in vectors if v is not None], dtype=float)
    if len(pts) < 2:
        return 0.0
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2) / math.sqrt(pts.shape[1])
    return mean_pairwise(d)


def morphology_diversity(genomes: Sequence[Genome]) -> float:
    """Mean pairwise morphology-gene distance, in [0, 1] (0 for one body)."""
    return mean_pairwise(gene_distance_matrix(list(genomes), MORPHOLOGY_GENES))


def plan_counts(genomes: Sequence[Genome]) -> dict[str, int]:
    """Body plan ("<limbs>x<segments>") -> how many genomes have it, sorted by plan."""
    counts: dict[str, int] = {}
    for g in genomes:
        counts[body_plan(g)] = counts.get(body_plan(g), 0) + 1
    return dict(sorted(counts.items()))


def final_set_measures(members: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Training-side measures of a whole final population or archive.

    ``members`` are dicts with ``genome``, ``fitness``, ``valid`` and ``metrics``
    (the stored training evaluation). Fitness statistics use valid members.
    """
    genomes = [m["genome"] for m in members]
    valid = [m for m in members if m["valid"] and m["metrics"]]
    fit = [float(m["fitness"]) for m in valid]
    return {
        "n": len(members),
        "n_valid": len(valid),
        "best_fitness": float(max(fit)) if fit else math.nan,
        "mean_fitness": float(np.mean(fit)) if fit else math.nan,
        "morph_diversity": morphology_diversity(genomes),
        "n_body_plans": len(plan_counts(genomes)),
        "plans": plan_counts(genomes),
        "behaviour_diversity": behaviour_diversity(
            [behaviour_vector(m["metrics"]).tolist() for m in valid]
        ),
    }


def top_k(members: Sequence[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    """The ``k`` best valid members by stored training fitness, one per genome id.

    Ties are broken by id so the choice is deterministic.
    """
    seen: set[str] = set()
    ranked = sorted(
        (m for m in members if m["valid"]), key=lambda m: (-float(m["fitness"]), m["genome"].id)
    )
    out = []
    for m in ranked:
        if m["genome"].id in seen:
            continue
        seen.add(m["genome"].id)
        out.append(m)
        if len(out) == k:
            break
    return out
