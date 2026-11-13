"""Statistics and pure measures: known values, edge cases, reproducibility."""

from __future__ import annotations

import math

import pytest

from evomorph.analysis.stats import (
    MEASURES,
    a12,
    all_comparisons,
    bootstrap_median_ci,
    effect_label,
    evidence_note,
    holm,
    mann_whitney_p,
    min_two_sided_p,
)
from evomorph.experiments.measures import (
    BEHAVIOUR,
    behaviour_diversity,
    morphology_diversity,
    suite_summary,
    top_k,
)
from evomorph.morphology.genome import reference_hexapod, reference_quadruped
from evomorph.simulation.metrics import INVALID_FITNESS


def test_a12_known_values_and_direction() -> None:
    assert a12([3, 4], [1, 2]) == 1.0
    assert a12([1, 2], [3, 4]) == 0.0
    assert a12([1, 2], [1, 2]) == 0.5
    assert a12([3, 4], [1, 2], higher_is_better=False) == 0.0  # lower is better: 3, 4 lose
    assert a12([1, float("nan")], [0]) == 1.0  # nan dropped
    assert math.isnan(a12([], [1]))
    assert effect_label(0.5) == "negligible" and effect_label(0.6) == "small"
    assert effect_label(0.33) == "medium" and effect_label(0.9) == "large"


def test_mann_whitney_exact_floor_and_degenerate_cases() -> None:
    assert mann_whitney_p([1, 2, 3], [4, 5, 6]) == pytest.approx(0.1)
    assert min_two_sided_p(3, 3) == pytest.approx(0.1)
    assert min_two_sided_p(4, 4) == pytest.approx(2 / 70)
    assert min_two_sided_p(5, 5) == pytest.approx(2 / 252)
    assert mann_whitney_p([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]) == pytest.approx(2 / 252)
    assert math.isnan(mann_whitney_p([1], [2, 3]))
    assert mann_whitney_p([0, 0], [0, 0]) == 1.0


def test_holm_adjustment() -> None:
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    adj = holm([0.5, float("nan"), 0.01])
    assert adj[0] == pytest.approx(0.5) and math.isnan(adj[1]) and adj[2] == pytest.approx(0.02)


def test_bootstrap_interval_is_reproducible_and_needs_three_values() -> None:
    vals = [0.1, 0.4, 0.2, 0.9, 0.3]
    lo, hi = bootstrap_median_ci(vals)
    assert lo <= 0.3 <= hi and (lo, hi) == bootstrap_median_ci(vals)
    assert all(math.isnan(x) for x in bootstrap_median_ci([1.0, 2.0]))


def test_evidence_note_is_plain_about_few_seeds() -> None:
    assert "WEAK" in evidence_note([3, 3, 3, 3]) and "0.1" in evidence_note([3, 3])
    assert "WEAK" in evidence_note([4, 4])  # 6 x 0.029 > 0.05 after Holm
    assert "WEAK" not in evidence_note([5, 5]) and "complete separation" in evidence_note([5, 6])
    assert "no evidence" in evidence_note([])


def test_holm_covers_only_the_primary_family() -> None:
    rows = []
    for cond, base in (("controller", 0.0), ("morphology", 0.5), ("joint", 1.0), ("joint_me", 1.5)):
        for seed in range(5):
            rows.append({"condition": cond} | {m.column: base + 0.01 * seed for m in MEASURES})
    comps = all_comparisons(rows, ["controller", "morphology", "joint", "joint_me"])
    assert len(comps) == 3 * len(MEASURES)
    primary = [c for c in comps if c.measure.primary]
    assert len(primary) == 6 and all(c.p_holm >= c.p for c in primary)
    assert all(math.isnan(c.p_holm) for c in comps if not c.measure.primary)
    c_vs_a = next(c for c in primary if (c.first, c.second) == ("joint", "controller"))
    assert c_vs_a.a12 == 1.0 and c_vs_a.p == pytest.approx(2 / 252)
    tilt = next(c for c in comps if c.measure.column == "champ_test_tilt" and c.first == "joint")
    assert tilt.a12 == 0.0  # higher tilt is worse: C loses every pairing


def _rec(env: str, fitness: float, valid: bool = True, fell: bool = False, **metrics) -> dict:
    base = {k: 0.0 for k, _, _ in BEHAVIOUR} | {"avg_forward_velocity": 0.0, "fell": float(fell)}
    return {
        "env": env,
        "seed": 0,
        "fitness": fitness,
        "valid": valid,
        "fell": fell,
        "metrics": (base | metrics) if valid else {},
    }


def test_suite_summary_counts_invalid_runs_as_failures_and_scores_them() -> None:
    recs = [
        _rec("a", 1.0, avg_forward_velocity=0.2),
        _rec("a", 0.0, fell=True, avg_forward_velocity=0.4),
        _rec("b", INVALID_FITNESS, valid=False),
    ]
    s = suite_summary(recs)
    assert s["fitness"] == pytest.approx((0.5 + INVALID_FITNESS) / 2)  # mean of env means
    assert s["worst_fitness"] == INVALID_FITNESS
    assert s["failure_rate"] == pytest.approx(2 / 3) and s["invalid_rate"] == pytest.approx(1 / 3)
    assert s["speed"] == pytest.approx(0.3)  # valid runs only


def test_diversity_measures_have_known_extremes() -> None:
    assert behaviour_diversity([[0.2] * 6, [0.2] * 6, None]) == 0.0
    assert behaviour_diversity([[0.0] * 6, [1.0] * 6]) == pytest.approx(1.0)
    assert morphology_diversity([reference_quadruped(), reference_quadruped()]) == 0.0
    assert morphology_diversity([reference_quadruped(), reference_hexapod()]) > 0.0


def test_top_k_is_deterministic_and_distinct() -> None:
    q = reference_quadruped()
    members = [
        {"genome": q, "fitness": 1.0, "valid": True, "metrics": {}},
        {"genome": q, "fitness": 1.0, "valid": True, "metrics": {}},
        {"genome": reference_hexapod(), "fitness": 1.0, "valid": True, "metrics": {}},
        {"genome": reference_hexapod(), "fitness": 5.0, "valid": False, "metrics": {}},
    ]
    chosen = top_k(members, 3)
    assert [m["genome"].id for m in chosen] == ["ref-hexapod", "ref-quadruped"]
