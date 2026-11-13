"""Combined locomotion fitness from raw metrics (pure python, no MuJoCo).

Structure
---------
::

    base     = w_progress * sustained_velocity / velocity_scale
    quality  = prod_i  factor_i ** w_i            each factor_i in [0, 1]
    fitness  = base * quality   (base > 0)        quality only ever scales progress down
             = max(base, -penalty_cap)  (base <= 0, i.e. backward or stalled)
             - w_fall * fell
             - w_still * stationary_shortfall

Quality factors (1 = flawless, falling toward 0 as the flaw grows; no cap, so a
twice-as-bad gait always loses more):

=========== ================================================ ==========================
factor      formula                                          defeats
=========== ================================================ ==========================
upright     1 - mean_tilt / tilt_scale, floored at 0         leaning, rolling crawlers
lateral     1 / (1 + (off_axis / lateral_ratio)^2),          sideways or diagonal crabs,
            off_axis = max_lateral_deviation / forward       wandering and returning
rotation    1 / (1 + (yaw_drift / yaw_scale)^2)              spinners, circlers
energy      1 / (1 + cost_of_transport / cot_scale)          thrashing, brute force
slip        1 - slip_penalty * slip_fraction                 skating on sliding feet
support     foot_support_fraction                            shin/knee/belly draggers
airborne    1 - excess_airborne / airborne_scale, >= 0       hopping, bounding, launching
effort      1 / (1 + actuator_effort / effort_scale)         saturated, straining servos
smoothness  1 / (1 + accel_rms / (smooth_scale * g))         vibration, impacts
=========== ================================================ ==========================

Because the factors multiply progress, speed cannot buy its way out of a bad
gait: halving a factor halves the score at any speed. That is the fix for the
earlier additive design, where every penalty saturated at a small cap and a
creature faster than ~0.7 m/s won however it moved. Backward motion is scored
unscaled (a negative progress must not be *softened* by poor quality) and is
capped at ``-penalty_cap`` so valid scores stay bounded above ``INVALID_FITNESS``.

Reported terms: ``progress`` is ``base`` (or the capped negative progress); each
factor's term is its share of the loss ``base * (1 - quality)``, split in
proportion to ``-w_i * log(factor_i)``; ``fall`` and ``stationary`` are additive.
The terms sum exactly to the fitness.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from evomorph.config import FitnessConfig

GRAVITY = 9.81
INVALID_FITNESS = -20.0  # below any valid score (valid scores are >= -(cap + w_fall + w_still))

QUALITY_FACTORS: tuple[str, ...] = (
    "upright",
    "lateral",
    "rotation",
    "energy",
    "slip",
    "support",
    "airborne",
    "effort",
    "smoothness",
)
FITNESS_TERMS: tuple[str, ...] = ("progress", *QUALITY_FACTORS, "fall", "stationary")


def _clip01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


def quality_factors(m: Mapping[str, float], cfg: FitnessConfig) -> dict[str, float]:
    """Each quality factor in [0, 1] (before its exponent weight)."""
    forward = max(m["forward_displacement"], cfg.still_distance)
    off_axis = m["max_lateral_deviation"] / forward
    excess_air = max(m["airborne_fraction"] - cfg.airborne_allowance, 0.0)
    return {
        "upright": _clip01(1.0 - m["mean_tilt"] / cfg.tilt_scale),
        "lateral": 1.0 / (1.0 + (off_axis / cfg.lateral_ratio) ** 2),
        "rotation": 1.0 / (1.0 + (m["yaw_drift"] / cfg.yaw_scale) ** 2),
        "energy": 1.0 / (1.0 + m["cost_of_transport"] / cfg.cot_scale),
        "slip": _clip01(1.0 - cfg.slip_penalty * m["slip_fraction"]),
        "support": _clip01(m["foot_support_fraction"]),
        "airborne": _clip01(1.0 - excess_air / cfg.airborne_scale),
        "effort": 1.0 / (1.0 + m["actuator_effort"] / cfg.effort_scale),
        "smoothness": 1.0 / (1.0 + m["accel_rms"] / (cfg.smooth_scale * GRAVITY)),
    }


def _loss_shares(factors: dict[str, float], cfg: FitnessConfig) -> tuple[float, dict[str, float]]:
    """(quality, share of the loss per factor); shares sum to 1 when quality < 1."""
    weights = {k: float(getattr(cfg, f"w_{k}")) for k in factors}
    zero = [k for k, f in factors.items() if f <= 0.0 and weights[k] > 0]
    if zero:  # a zero factor wipes out progress: it takes the whole loss
        return 0.0, {k: (1.0 / len(zero) if k in zero else 0.0) for k in factors}
    logs = {k: -weights[k] * math.log(f) for k, f in factors.items()}
    total = sum(logs.values())
    quality = math.exp(-total)
    if total <= 0.0:
        return quality, dict.fromkeys(factors, 0.0)
    return quality, {k: v / total for k, v in logs.items()}


def compute_fitness(
    metrics: Mapping[str, float], cfg: FitnessConfig
) -> tuple[float, dict[str, float]]:
    """Combine raw metrics into one score; returns (fitness, signed per-term contributions).

    See the module docstring for the formula. A non-finite or missing metric
    yields ``INVALID_FITNESS`` with all terms 0. ``metrics`` is never modified.
    """
    try:
        if not all(math.isfinite(float(v)) for v in metrics.values()):
            return INVALID_FITNESS, dict.fromkeys(FITNESS_TERMS, 0.0)
        factors = quality_factors(metrics, cfg)
        base = cfg.w_progress * metrics["sustained_velocity"] / cfg.velocity_scale
        fell, forward = metrics["fell"], metrics["forward_displacement"]
    except (KeyError, TypeError, ValueError):
        return INVALID_FITNESS, dict.fromkeys(FITNESS_TERMS, 0.0)
    terms = dict.fromkeys(FITNESS_TERMS, 0.0)
    if base > 0:
        quality, shares = _loss_shares(factors, cfg)
        loss = base * (1.0 - quality)
        terms["progress"] = base
        for k, share in shares.items():
            terms[k] = -loss * share
    else:
        terms["progress"] = max(base, -cfg.penalty_cap)
    terms["fall"] = -cfg.w_fall * fell
    terms["stationary"] = -cfg.w_still * _clip01(1.0 - max(forward, 0.0) / cfg.still_distance)
    return float(sum(terms.values())), {k: float(v) for k, v in terms.items()}
