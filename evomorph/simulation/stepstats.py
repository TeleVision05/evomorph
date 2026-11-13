"""Physics-step-resolution sums recorded per control interval (pure numpy).

The run loop advances the simulation one physics step at a time and adds every
step's contribution into the row of the control interval it belongs to. Row
``k`` covers the interval that ends at ``Trajectory.time[k]``; row 0 (the
starting sample) is all zeros. Keeping sums rather than samples means energy,
acceleration, airborne time, contact loads and foot slip are integrated at the
physics rate (500 Hz by default) instead of being read off once per control
step, which under-counted energy by ~1.1-1.7x on real creatures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# A stance contact sliding faster than this (m/s) counts as slipping, not gripping.
SLIP_SPEED = 0.05

STEP_FIELDS: dict[str, str] = {
    "n_steps": "physics steps in the interval",
    "energy": "J, sum over steps of sum_j |torque_j * joint_velocity_j| * dt",
    "accel_sq": "m^2/s^3, sum over steps of |torso linear acceleration|^2 * dt",
    "airborne": "steps with no creature geom touching the ground",
    "torso_touch": "steps with the torso touching the ground",
    "nonfoot_touch": "steps with a limb segment or the torso (not a foot) touching the ground",
    "foot_impulse": "N s, ground normal impulse carried by feet",
    "limb_impulse": "N s, ground normal impulse carried by limb segments (shins, thighs)",
    "torso_impulse": "N s, ground normal impulse carried by the torso",
    "slip_impulse": "N m, sum over foot contacts of normal force * tangential slip speed * dt",
    "sliding_impulse": f"N s, foot normal impulse delivered while sliding > {SLIP_SPEED} m/s",
    "max_penetration": "m, deepest creature-ground contact penetration in the interval",
}


@dataclass
class StepStats:
    """Per-interval sums; every array has one row per Trajectory sample."""

    dt: float  # physics timestep (s)
    n_steps: np.ndarray
    energy: np.ndarray
    accel_sq: np.ndarray
    airborne: np.ndarray
    torso_touch: np.ndarray
    nonfoot_touch: np.ndarray
    foot_impulse: np.ndarray
    limb_impulse: np.ndarray
    torso_impulse: np.ndarray
    slip_impulse: np.ndarray
    sliding_impulse: np.ndarray
    max_penetration: np.ndarray

    @classmethod
    def zeros(cls, n_rows: int, dt: float) -> StepStats:
        return cls(dt=dt, **{k: np.zeros(n_rows) for k in STEP_FIELDS})

    def truncated(self, n_rows: int) -> StepStats:
        return StepStats(dt=self.dt, **{k: getattr(self, k)[:n_rows] for k in STEP_FIELDS})

    def total(self, name: str) -> float:
        return float(np.sum(getattr(self, name)))

    @property
    def total_steps(self) -> float:
        return self.total("n_steps")

    def step_fraction(self, name: str) -> float:
        """Fraction of physics steps for which a per-step counter was set."""
        n = self.total_steps
        return self.total(name) / n if n > 0 else 0.0
