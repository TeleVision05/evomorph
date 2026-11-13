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
    # Fall detection: the torso is flipped past this tilt, or it rests on the ground.
    fall_tilt_deg: float = 75.0
    fall_contact_time: float = 0.25  # s of continuous torso-ground contact counted as a fall
    # Glitch guards. A walking creature under ~1.5 m long never needs 5 m/s or
    # 40 rad/s; exceeding either means the integrator blew up or found an exploit.
    max_linear_speed: float = 5.0
    max_angular_speed: float = 40.0
    max_height_gain: float = 1.0  # m above spawn height: higher is a launch, not a gait
    max_ground_penetration: float = 0.05  # m of torso below the ground: clipping

    @property
    def n_substeps(self) -> int:
        """Physics steps per control step."""
        return max(1, round(1.0 / (self.control_rate * self.timestep)))


@dataclass(frozen=True)
class FitnessConfig:
    """Weights and normalisation scales of the combined locomotion fitness.

    Each raw metric is divided by a scale that a competent walker of this size
    reaches, so every term is O(1); weights then set relative importance. See
    :func:`evomorph.simulation.metrics.compute_fitness` for the exact formula.
    """

    velocity_scale: float = 0.2  # m/s: a brisk walk for a 0.3-1 m creature (~0.5 body/s)
    lateral_scale: float = 1.0  # m of sideways drift per penalty unit over one run
    yaw_scale: float = 1.5708  # rad (90 deg) of heading excursion per penalty unit
    tilt_scale: float = 1.0472  # rad (60 deg) mean tilt at which the progress gate hits zero
    still_distance: float = 0.1  # m of forward progress below which "stationary" is penalised
    airborne_allowance: float = 0.2  # fraction of time with no ground contact allowed for free
    w_progress: float = 1.0
    w_lateral: float = 0.5
    w_rotation: float = 0.5
    w_energy: float = 0.1
    w_fall: float = 1.0
    w_still: float = 0.2
    w_smooth: float = 0.1
    w_airborne: float = 1.0
    penalty_cap: float = 2.0  # each penalty is clipped to this many units before weighting


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
