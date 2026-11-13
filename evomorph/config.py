"""Simulation and fitness configuration, loadable from YAML.

A config YAML is a mapping of named sections. This module owns ``sim`` and
``fitness``; other stages append their own sections (``evolution``,
``map_elites``, ``experiment`` ...) and read them with :func:`load_section`.
Unknown keys inside a section are an error, so a typo never silently falls back
to a default.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


@dataclass(frozen=True)
class SimConfig:
    """How one creature is simulated."""

    duration: float = 10.0  # s of measured locomotion after settling
    # 2 ms is MuJoCo's default; with implicitfast integration it keeps stiff
    # position servos on light limbs stable while a 10 s run stays cheap.
    timestep: float = 0.002
    settle_time: float = 1.0  # s the creature stands still at its rest pose before measuring
    # 50 Hz control is ~17x the fastest allowed CPG (3 Hz), plenty to track a sine,
    # and amortises Python overhead over 10 physics steps.
    control_rate: float = 50.0
    init_noise: float = 0.01  # rad std of seeded joint-angle noise at spawn (seed matters)
    spawn_clearance: float = 0.005  # m gap between lowest geom and the ground at spawn
    # Fall detection: the torso is flipped past this tilt, or it has touched the
    # ground for fall_contact_time in total within the last fall_contact_window
    # (accumulated at physics resolution, so brief lifts do not reset the clock).
    fall_tilt_deg: float = 75.0
    fall_contact_time: float = 0.25  # s of torso-ground contact ...
    fall_contact_window: float = 1.0  # s ... within this sliding window counts as a fall
    # Glitch guards, checked on the peak over every physics step. Speed and launch
    # height are capped both absolutely and relative to body size (torso length
    # + 2 x leg reach): 10 sizes/s is well above any legged gait at these scales
    # (a cheetah peaks near 20 body lengths/s, small robots near 5), and one body
    # size of torso rise is a jump, not a step.
    max_linear_speed: float = 5.0  # m/s absolute
    max_speed_per_size: float = 10.0  # body sizes per second
    max_angular_speed: float = 40.0  # rad/s
    max_height_gain: float = 1.0  # m above the start clearance: higher is a launch
    max_height_gain_per_size: float = 1.0  # body sizes above the start clearance
    max_ground_penetration: float = 0.05  # m of torso centre below the local ground: clipping
    # m any geom may sink into the ground. Soft contact lets heavy creatures sink
    # a few cm (measured up to ~75 mm on a 56 kg body); deeper means tunnelling.
    max_contact_penetration: float = 0.12

    def __post_init__(self) -> None:
        for name in ("duration", "timestep", "control_rate", "fall_contact_window"):
            if not getattr(self, name) > 0:
                raise ValueError(f"SimConfig.{name} must be > 0, got {getattr(self, name)}")
        if self.settle_time < 0:
            raise ValueError("SimConfig.settle_time must be >= 0")
        ratio = 1.0 / (self.control_rate * self.timestep)
        if ratio < 1.0 - 1e-9 or abs(ratio - round(ratio)) > 1e-6:
            raise ValueError(
                f"control period 1/control_rate = {1 / self.control_rate:g} s must be a whole "
                f"number of timesteps ({self.timestep:g} s); got {ratio:.4g}"
            )

    @property
    def n_substeps(self) -> int:
        """Physics steps per control step (exact: validated in ``__post_init__``)."""
        return round(1.0 / (self.control_rate * self.timestep))


@dataclass(frozen=True)
class FitnessConfig:
    """Weights and normalisation scales of the combined locomotion fitness.

    Each raw metric is divided by a scale that a competent walker of this size
    reaches, so every term is O(1); weights then set relative importance. See
    :func:`evomorph.simulation.metrics.compute_fitness` for the exact formula.
    """

    # --- progress -------------------------------------------------------------
    velocity_scale: float = 0.2  # m/s: a brisk walk for a 0.3-1 m creature (~0.5 body/s)
    # --- quality factor scales (each factor is 1 when flawless, see fitness.py) --
    tilt_scale: float = 1.0472  # rad (60 deg) mean tilt at which the upright factor hits 0
    # Max sideways deviation per metre forward that halves the lateral factor:
    # 0.5 is a ~27 deg off-axis path; a 45 deg diagonal crab keeps 20%.
    lateral_ratio: float = 0.5
    yaw_scale: float = 0.7  # rad (40 deg) heading excursion that halves the rotation factor
    # Cost of transport that halves the energy factor. The reference walkers sit
    # near 1.2-1.3 at physics-rate energy; brute-force movers at 3-6.
    cot_scale: float = 2.0
    slip_penalty: float = 0.5  # share of progress lost if every stance skates (slip_fraction 1)
    # Flight: walking keeps a foot down; 5% covers foot-switch transients.
    airborne_allowance: float = 0.05
    airborne_scale: float = 0.5  # excess airborne fraction at which the airborne factor hits 0
    effort_scale: float = 0.5  # mean (torque/limit)^2 that halves the effort factor
    # RMS torso acceleration (in g) that halves the smoothness factor. Physics-rate
    # values include footfall impacts: ~0.8-1.1 g for the reference walkers.
    smooth_scale: float = 3.0
    still_distance: float = 0.1  # m of forward progress below which "stationary" is penalised
    # --- weights: w_progress scales progress, factor weights are exponents (0 = off) --
    w_progress: float = 1.0
    w_upright: float = 1.0
    w_lateral: float = 1.0
    w_rotation: float = 1.0
    w_energy: float = 1.0
    w_slip: float = 1.0
    w_support: float = 1.0
    w_airborne: float = 1.0
    w_effort: float = 1.0
    w_smoothness: float = 1.0
    w_fall: float = 1.0  # additive penalty for a fall
    w_still: float = 0.2  # additive penalty for not moving at all
    penalty_cap: float = 2.0  # backward progress is penalised down to -penalty_cap only


def from_mapping(cls: type[T], data: dict[str, Any] | None) -> T:
    """Build dataclass ``cls`` from a mapping, rejecting unknown keys."""
    data = dict(data or {})
    names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(data) - names)
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown keys {unknown}")
    return cls(**data)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a config YAML into a dict of sections."""
    with open(path) as fh:
        content = yaml.safe_load(fh) or {}
    if not isinstance(content, dict):
        raise ValueError(f"{path}: top level must be a mapping of sections")
    return content


def load_section(path: str | Path, section: str, cls: type[T]) -> T:
    """Load one named section of a YAML config into dataclass ``cls``.

    A missing section yields the dataclass defaults.
    """
    return from_mapping(cls, load_yaml(path).get(section))


def load_sim_config(path: str | Path) -> SimConfig:
    return load_section(path, "sim", SimConfig)


def load_fitness_config(path: str | Path) -> FitnessConfig:
    return load_section(path, "fitness", FitnessConfig)
