"""One targeted test per fitness term, plus the exploits the review found (pure, no MuJoCo).

Each quality-factor test perturbs exactly one raw metric of a good walker and
checks that (a) the score falls, (b) the matching term carries the loss, (c) a
worse flaw always costs more (no saturation), and (d) the factor's weight is
what does it: with its exponent at 0 the perturbation is free. Deleting any
term from ``compute_fitness`` fails its test.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from evomorph.config import FitnessConfig
from evomorph.simulation.fitness import (
    FITNESS_TERMS,
    INVALID_FITNESS,
    QUALITY_FACTORS,
    compute_fitness,
)
from evomorph.simulation.metrics import METRIC_KEYS

CFG = FitnessConfig()

# A clean straight walker: 0.4 m/s sustained, 4 m in 10 s, every flaw near zero.
GOOD = dict.fromkeys(METRIC_KEYS, 0.0) | {
    "forward_displacement": 4.0,
    "sustained_velocity": 0.4,
    "avg_forward_velocity": 0.4,
    "path_length": 4.0,
    "cost_of_transport": 0.5,
    "foot_support_fraction": 1.0,
    "accel_rms": 2.0,
    "actuator_effort": 0.05,
    "airborne_fraction": 0.02,
    "slip_fraction": 0.1,
    "survival_time": 10.0,
    "total_mass": 5.0,
}

# term -> (metric, three increasingly bad values)
FLAWS = {
    "upright": ("mean_tilt", (0.2, 0.4, 0.8)),
    "lateral": ("max_lateral_deviation", (0.5, 1.5, 4.0)),
    "rotation": ("yaw_drift", (0.3, 1.0, 3.0)),
    "energy": ("cost_of_transport", (1.5, 4.0, 12.0)),
    "slip": ("slip_fraction", (0.4, 0.7, 1.0)),
    "support": ("foot_support_fraction", (0.8, 0.5, 0.2)),
    "airborne": ("airborne_fraction", (0.15, 0.3, 0.45)),
    "effort": ("actuator_effort", (0.2, 0.5, 0.9)),
    "smoothness": ("accel_rms", (15.0, 40.0, 120.0)),
}


def score(metrics: dict, cfg: FitnessConfig = CFG) -> tuple[float, dict]:
    return compute_fitness(metrics, cfg)


def test_every_quality_factor_has_a_flaw_case() -> None:
    assert set(FLAWS) == set(QUALITY_FACTORS)
    assert set(FITNESS_TERMS) == {"progress", *QUALITY_FACTORS, "fall", "stationary"}


@pytest.mark.parametrize("term", sorted(FLAWS))
def test_each_quality_factor_costs_and_never_saturates(term: str) -> None:
    metric, values = FLAWS[term]
    clean, clean_terms = score(GOOD)
    previous = clean
    for v in values:
        f, terms = score(GOOD | {metric: v})
        assert f < previous, (term, v)  # strictly worse at every step: no cap
        assert terms[term] < clean_terms[term] <= 0.0, (term, v)
        previous = f
    off = dataclasses.replace(CFG, **{f"w_{term}": 0.0})
    assert score(GOOD | {metric: values[-1]}, off)[0] == pytest.approx(score(GOOD, off)[0])


@pytest.mark.parametrize("term", sorted(set(FLAWS) - {"lateral"}))
def test_quality_factors_scale_progress_at_any_speed(term: str) -> None:
    """A flaw costs the same *fraction* of the score whether walking or sprinting.

    (Lateral deviation is judged per metre forward, so the crab test covers it.)
    """
    metric, values = FLAWS[term]

    def ratio(speed: float) -> float:
        fast = GOOD | {"sustained_velocity": speed, "forward_displacement": 10 * speed}
        return score(fast | {metric: values[1]})[0] / score(fast)[0]

    slow, fast = ratio(0.4), ratio(2.0)
    assert slow == pytest.approx(fast, rel=1e-9)
    assert slow < 1.0


def test_fall_and_stationary_are_additive_penalties() -> None:
    clean, _ = score(GOOD)
    fell, terms = score(GOOD | {"fell": 1.0})
    assert fell == pytest.approx(clean - CFG.w_fall) and terms["fall"] == -CFG.w_fall
    still, terms = score(GOOD | {"forward_displacement": 0.0, "sustained_velocity": 0.0})
    assert terms["stationary"] == pytest.approx(-CFG.w_still) and still < 0
    _, terms = score(GOOD | {"forward_displacement": 0.05, "sustained_velocity": 0.0})
    assert terms["stationary"] == pytest.approx(-0.5 * CFG.w_still)


def test_backward_progress_is_not_softened_by_poor_quality() -> None:
    back = GOOD | {"forward_displacement": -2.0, "sustained_velocity": -0.2}
    clean, _ = score(back)
    sloppy, _ = score(back | {"cost_of_transport": 50.0, "yaw_drift": 3.0})
    assert sloppy == pytest.approx(clean) and clean < 0
    assert score(back | {"sustained_velocity": -50.0})[0] >= -CFG.penalty_cap - 1.0


def test_terms_sum_to_fitness_for_mixed_flaws() -> None:
    m = GOOD | {"mean_tilt": 0.3, "yaw_drift": 0.8, "cost_of_transport": 3.0, "fell": 1.0}
    f, terms = score(m)
    assert set(terms) == set(FITNESS_TERMS)
    assert f == pytest.approx(sum(terms.values()))
    zero_factor = GOOD | {"foot_support_fraction": 0.0}  # wipes out progress entirely
    f, terms = score(zero_factor)
    assert f == pytest.approx(0.0) and terms["support"] == pytest.approx(-terms["progress"])


# --- the reviewer's exploit cases ---------------------------------------------------


def test_speed_cannot_buy_back_energy() -> None:
    """Twice as fast at four times the cost of transport must lose (was: energy capped)."""
    efficient = GOOD | {"cost_of_transport": 1.2}
    brute = GOOD | {"sustained_velocity": 0.8, "forward_displacement": 8.0,
                    "cost_of_transport": 4.75}  # fmt: skip
    assert score(brute)[0] < score(efficient)[0]


def test_diagonal_crab_cannot_outrun_a_straight_walker() -> None:
    """r991-1266 in review: 8.14 m forward, 8.45 m sideways, yaw 1.64 rad, ranked 9/1500."""
    crab = GOOD | {"sustained_velocity": 0.8, "forward_displacement": 8.14,
                   "max_lateral_deviation": 8.45, "lateral_displacement": 8.45,
                   "yaw_drift": 1.64}  # fmt: skip
    walker = GOOD | {"sustained_velocity": 0.3, "forward_displacement": 3.0,
                     "max_lateral_deviation": 0.3}  # fmt: skip
    assert score(crab)[0] < score(walker)[0]


def test_wandering_and_returning_is_penalised() -> None:
    """Lateral uses the max deviation, not the end point."""
    back_on_line = GOOD | {"max_lateral_deviation": 3.0, "lateral_displacement": 0.0}
    assert score(back_on_line)[0] < 0.5 * score(GOOD)[0]


def test_shin_dragger_and_skater_lose_to_a_gripping_walker() -> None:
    """tv424242-0054 in review: fastest creature, 52% of its load on its shins."""
    dragger = GOOD | {"sustained_velocity": 1.05, "forward_displacement": 10.5,
                      "foot_support_fraction": 0.48, "slip_fraction": 0.89,
                      "cost_of_transport": 4.0}  # fmt: skip
    walker = GOOD | {"sustained_velocity": 0.45, "forward_displacement": 4.5,
                     "cost_of_transport": 1.2, "slip_fraction": 0.4}  # fmt: skip
    assert score(dragger)[0] < score(walker)[0]


def test_hopper_at_walking_speed_is_penalised() -> None:
    """Seed-1 GA winners in review: 29-41% airborne at Froude < 0.5 cost only -0.26 of 4.3."""
    hopper = GOOD | {"airborne_fraction": 0.41}
    assert score(hopper)[0] < 0.4 * score(GOOD)[0]


def test_missing_or_non_finite_metric_is_invalid() -> None:
    assert score(GOOD | {"energy": math.nan})[0] == INVALID_FITNESS
    broken = dict(GOOD)
    del broken["slip_fraction"]
    assert score(broken)[0] == INVALID_FITNESS
