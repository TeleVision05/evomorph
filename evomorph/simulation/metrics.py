"""Locomotion metrics and the combined fitness.

Pure numpy: nothing here imports MuJoCo, so every metric and the fitness are
unit-testable on synthetic trajectories. Raw metrics (``compute_metrics``) are
always kept separate from the combined score (``compute_fitness``).

Forward is +x in the world frame. A ``Trajectory`` is sampled at the control
rate from the start of measurement (after settling) until the run ends or the
creature falls.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from evomorph.config import FitnessConfig

GRAVITY = 9.81
# Sustained velocity: the run is cut into N_WINDOWS equal windows and the score is
# the mean velocity of the N_SLOWEST slowest. A lunge then collapse fills one
# window; walking the whole time fills all of them.
N_WINDOWS = 5
N_SLOWEST = 3
MIN_COT_DISTANCE = 0.01  # m; COT denominator floor so a stationary body has finite COT
INVALID_FITNESS = -20.0  # below any valid score (valid scores are bounded below by ~-9)

METRIC_DOCS: dict[str, str] = {
    "forward_displacement": "m, torso x at end minus x at start (+ is forward)",
    "avg_forward_velocity": "m/s, forward_displacement / survival_time",
    "sustained_velocity": "m/s, mean velocity of the 3 slowest of 5 equal windows of the "
    "intended duration; windows after a fall count as 0",
    "path_length": "m, horizontal distance travelled by the torso along its path",
    "lateral_displacement": "m, |torso y at end - y at start|",
    "max_lateral_deviation": "m, max |y(t) - y(0)| over the run",
    "survival_time": "s, time until a fall (= duration if it never fell)",
    "survival_fraction": "survival_time / intended duration, in [0, 1]",
    "fell": "1 if the run ended in a fall (flipped or torso resting on ground), else 0",
    "mean_tilt": "rad, mean angle between torso up-axis and world up",
    "max_tilt": "rad, max angle between torso up-axis and world up",
    "upright_score": "1 - mean_tilt / (pi/2), in [0, 1]; 1 = perfectly level",
    "yaw_drift": "rad, max |heading(t) - heading(0)| with heading unwrapped (spinning grows it)",
    "net_yaw": "rad, unwrapped heading change start to end (signed, + = turned left)",
    "energy": "J, integral over time of sum_j |torque_j * joint_velocity_j|",
    "mean_power": "W, energy / survival_time",
    "cost_of_transport": "energy / (mass * g * max(forward_displacement, 0.01 m)), dimensionless",
    "energy_efficiency": "d / (d + energy/(m g)) with d = max(forward_displacement, 0); "
    "= 1/(1+COT) when moving forward, 0 otherwise. In [0, 1)",
    "actuator_effort": "mean over time and joints of (torque / torque_limit)^2, in [0, 1]",
    "accel_rms": "m/s^2, RMS torso linear acceleration (finite-differenced at control rate)",
    "smoothness": "1 / (1 + accel_rms / g), in (0, 1]; vibration drives it down",
    "foot_contact_fraction": "fraction of samples with at least one foot on the ground",
    "airborne_fraction": "fraction of samples with no creature geom touching the ground",
    "torso_contact_fraction": "fraction of samples with the torso touching the ground",
    "mean_height": "m, mean torso z (world frame)",
    "max_height_gain": "m, max torso z minus torso z at measurement start",
    "peak_speed": "m/s, max torso linear speed",
    "total_mass": "kg, total creature mass",
    "n_joints": "number of actuated joints",
    "n_limbs": "number of limbs",
}
METRIC_KEYS: tuple[str, ...] = tuple(METRIC_DOCS)


@dataclass
class Trajectory:
    """Recorded state of one run, one row per control step.

    ``torso_quat`` is (w, x, y, z). ``torso_linvel`` is world frame. Joint arrays
    follow actuator order. ``duration`` is the intended run length (s), which can
    exceed ``time[-1]`` when the creature fell.
    """

    time: np.ndarray  # (T,)
    torso_pos: np.ndarray  # (T, 3)
    torso_quat: np.ndarray  # (T, 4)
    torso_linvel: np.ndarray  # (T, 3)
    joint_vel: np.ndarray  # (T, J)
    actuator_force: np.ndarray  # (T, J)
    foot_contacts: np.ndarray  # (T, L) bool
    torso_contact: np.ndarray  # (T,) bool
    any_contact: np.ndarray  # (T,) bool
    duration: float
    total_mass: float
    torque_limit: float
    n_limbs: int
    fell: bool = False
    fall_time: float | None = None
    extra: dict = field(default_factory=dict)

    @property
    def n_joints(self) -> int:
        return int(self.joint_vel.shape[1]) if self.joint_vel.ndim == 2 else 0

    @property
    def survival_time(self) -> float:
        if self.fell and self.fall_time is not None:
            return float(self.fall_time)
        return float(self.time[-1]) if self.time.size else 0.0


# ----------------------------------------------------------------------------
# Orientation helpers
# ----------------------------------------------------------------------------


def quat_tilt(quat: np.ndarray) -> np.ndarray:
    """Angle (rad) between the body z-axis and world z for (w, x, y, z) quats."""
    q = np.atleast_2d(quat)
    up_z = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
    return np.arccos(np.clip(up_z, -1.0, 1.0))


def quat_yaw(quat: np.ndarray) -> np.ndarray:
    """Heading (rad) of the body x-axis projected on the ground plane."""
    q = np.atleast_2d(quat)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))


# ----------------------------------------------------------------------------
# Individual metrics
# ----------------------------------------------------------------------------


def forward_displacement(traj: Trajectory) -> float:
    return float(traj.torso_pos[-1, 0] - traj.torso_pos[0, 0])


def path_length(traj: Trajectory) -> float:
    steps = np.diff(traj.torso_pos[:, :2], axis=0)
    return float(np.sum(np.linalg.norm(steps, axis=1)))


def sustained_velocity(traj: Trajectory) -> float:
    """Mean velocity of the N_SLOWEST slowest of N_WINDOWS windows over ``duration``."""
    edges = np.linspace(0.0, traj.duration, N_WINDOWS + 1)
    x, t = traj.torso_pos[:, 0], traj.time
    width = traj.duration / N_WINDOWS
    vels = []
    for a, b in itertools.pairwise(edges):
        # After the last sample (a fall), position is frozen: windows add nothing.
        xa = float(np.interp(a, t, x))
        xb = float(np.interp(min(b, traj.survival_time), t, x)) if a < traj.survival_time else xa
        vels.append((xb - xa) / width)
    return float(np.mean(np.sort(vels)[:N_SLOWEST]))


def heading_metrics(traj: Trajectory) -> tuple[float, float]:
    """(yaw_drift, net_yaw) from the unwrapped heading."""
    yaw = np.unwrap(quat_yaw(traj.torso_quat))
    rel = yaw - yaw[0]
    return float(np.max(np.abs(rel))), float(rel[-1])


def energy(traj: Trajectory) -> float:
    """Trapezoidal integral of total absolute mechanical power (J)."""
    if traj.time.size < 2 or traj.n_joints == 0:
        return 0.0
    power = np.sum(np.abs(traj.actuator_force * traj.joint_vel), axis=1)
    return float(np.sum(0.5 * (power[1:] + power[:-1]) * np.diff(traj.time)))


def accel_rms(traj: Trajectory) -> float:
    if traj.time.size < 3:
        return 0.0
    acc = np.diff(traj.torso_linvel, axis=0) / np.diff(traj.time)[:, None]
    return float(np.sqrt(np.mean(np.sum(acc**2, axis=1))))


def actuator_effort(traj: Trajectory) -> float:
    if traj.n_joints == 0 or traj.torque_limit <= 0:
        return 0.0
    return float(np.mean((traj.actuator_force / traj.torque_limit) ** 2))


def _frac(flags: np.ndarray) -> float:
    return float(np.mean(flags)) if flags.size else 0.0


def compute_metrics(traj: Trajectory) -> dict[str, float]:
    """Every key of ``METRIC_KEYS``, as plain floats."""
    m, g = traj.total_mass, GRAVITY
    surv = traj.survival_time
    fwd = forward_displacement(traj)
    e = energy(traj)
    tilt = quat_tilt(traj.torso_quat)
    yaw_drift, net_yaw = heading_metrics(traj)
    rel_y = traj.torso_pos[:, 1] - traj.torso_pos[0, 1]
    z0 = traj.torso_pos[0, 2]
    a_rms = accel_rms(traj)
    d_pos = max(fwd, 0.0)
    out = {
        "forward_displacement": fwd,
        "avg_forward_velocity": fwd / surv if surv > 0 else 0.0,
        "sustained_velocity": sustained_velocity(traj),
        "path_length": path_length(traj),
        "lateral_displacement": float(abs(rel_y[-1])),
        "max_lateral_deviation": float(np.max(np.abs(rel_y))),
        "survival_time": surv,
        "survival_fraction": min(1.0, surv / traj.duration) if traj.duration > 0 else 0.0,
        "fell": 1.0 if traj.fell else 0.0,
        "mean_tilt": float(np.mean(tilt)),
        "max_tilt": float(np.max(tilt)),
        "upright_score": float(max(0.0, 1.0 - np.mean(tilt) / (math.pi / 2))),
        "yaw_drift": yaw_drift,
        "net_yaw": net_yaw,
        "energy": e,
        "mean_power": e / surv if surv > 0 else 0.0,
        "cost_of_transport": e / (m * g * max(fwd, MIN_COT_DISTANCE)),
        "energy_efficiency": d_pos / (d_pos + e / (m * g)) if d_pos > 0 else 0.0,
        "actuator_effort": actuator_effort(traj),
        "accel_rms": a_rms,
        "smoothness": 1.0 / (1.0 + a_rms / g),
        "foot_contact_fraction": _frac(np.any(traj.foot_contacts, axis=1)),
        "airborne_fraction": _frac(~traj.any_contact),
        "torso_contact_fraction": _frac(traj.torso_contact),
        "mean_height": float(np.mean(traj.torso_pos[:, 2])),
        "max_height_gain": float(np.max(traj.torso_pos[:, 2]) - z0),
        "peak_speed": float(np.max(np.linalg.norm(traj.torso_linvel, axis=1))),
        "total_mass": float(m),
        "n_joints": float(traj.n_joints),
        "n_limbs": float(traj.n_limbs),
    }
    return {k: float(out[k]) for k in METRIC_KEYS}


# ----------------------------------------------------------------------------
# Fitness
# ----------------------------------------------------------------------------

FITNESS_TERMS: tuple[str, ...] = (
    "progress",
    "lateral",
    "rotation",
    "energy",
    "fall",
    "stationary",
    "smoothness",
    "airborne",
)


def compute_fitness(
    metrics: dict[str, float], cfg: FitnessConfig
) -> tuple[float, dict[str, float]]:
    """Combine raw metrics into one score; returns (fitness, signed per-term contributions).

    ``progress`` = w * (sustained_velocity / velocity_scale) * upright gate, where the
    gate falls linearly from 1 (level) to 0 at ``tilt_scale`` mean tilt. Every other
    term is a penalty in units of its scale, clipped to ``penalty_cap`` and weighted:

    * lateral: lateral_displacement / lateral_scale            (sideways crab)
    * rotation: yaw_drift / yaw_scale                          (spinner, circler)
    * energy: mean_power / (m g velocity_scale)                (thrashing, vibrating)
    * fall: 1 if fell                                          (lunge-and-collapse)
    * stationary: 1 - forward/still_distance, floored at 0     (standing still)
    * smoothness: accel_rms / g                                (vibration, impacts)
    * airborne: excess of airborne_fraction over the allowance (launching, hopping)

    Using *sustained* rather than total velocity means progress must be held for
    most of the run; a single lunge scores close to nothing. A non-finite metric
    yields ``INVALID_FITNESS``.
    """
    vals = [metrics.get(k, math.nan) for k in METRIC_KEYS]
    if not all(math.isfinite(v) for v in vals):
        return INVALID_FITNESS, dict.fromkeys(FITNESS_TERMS, 0.0)
    cap = cfg.penalty_cap

    def pen(x: float) -> float:
        return float(min(max(x, 0.0), cap))

    gate = min(max(1.0 - metrics["mean_tilt"] / cfg.tilt_scale, 0.0), 1.0)
    weight = metrics["total_mass"] * GRAVITY
    forward = metrics["forward_displacement"]
    air = (metrics["airborne_fraction"] - cfg.airborne_allowance) / (1.0 - cfg.airborne_allowance)
    terms = {
        # Backward motion is penalised, but only down to -cap so the score stays bounded.
        "progress": cfg.w_progress
        * max(metrics["sustained_velocity"] / cfg.velocity_scale * gate, -cap),
        "lateral": -cfg.w_lateral * pen(metrics["lateral_displacement"] / cfg.lateral_scale),
        "rotation": -cfg.w_rotation * pen(metrics["yaw_drift"] / cfg.yaw_scale),
        "energy": -cfg.w_energy * pen(metrics["mean_power"] / (weight * cfg.velocity_scale)),
        "fall": -cfg.w_fall * metrics["fell"],
        "stationary": -cfg.w_still * pen(1.0 - max(forward, 0.0) / cfg.still_distance),
        "smoothness": -cfg.w_smooth * pen(metrics["accel_rms"] / GRAVITY),
        "airborne": -cfg.w_airborne * pen(air),
    }
    return float(sum(terms.values())), {k: float(v) for k, v in terms.items()}
