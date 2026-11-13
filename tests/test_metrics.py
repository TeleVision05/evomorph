"""Metrics and fitness on synthetic trajectories, no MuJoCo involved (spec test 9)."""

from __future__ import annotations

import math
import subprocess
import sys

import numpy as np
import pytest

from evomorph.config import FitnessConfig
from evomorph.simulation.metrics import (
    INVALID_FITNESS,
    METRIC_KEYS,
    Trajectory,
    compute_fitness,
    compute_metrics,
)

DT, T = 0.02, 10.0
MASS = 5.0
CFG = FitnessConfig()


def make_traj(
    x=lambda t: 0 * t,
    y=lambda t: 0 * t,
    yaw=lambda t: 0 * t,
    roll=lambda t: 0 * t,
    fall_at: float | None = None,
    power: float = 2.0,
    z=lambda t: 0.15 + 0 * t,
    airborne=lambda t: np.zeros_like(t, dtype=bool),
) -> Trajectory:
    """Synthetic run: torso path x(t), y(t), heading yaw(t), roll(t), constant joint power."""
    end = fall_at if fall_at is not None else T
    t = np.arange(0.0, end + 1e-9, DT)
    n, j = t.size, 8
    pos = np.stack([x(t), y(t), z(t)], axis=1)
    hy, hr = yaw(t) / 2, roll(t) / 2
    quat = np.stack(
        [np.cos(hy) * np.cos(hr), np.sin(hr) * np.cos(hy), np.sin(hr) * np.sin(hy),
         np.sin(hy) * np.cos(hr)], axis=1)  # fmt: skip
    vel = np.gradient(pos, t, axis=0) if n > 1 else np.zeros((n, 3))
    # constant power split over joints: |f * qdot| sums to `power`
    jv = np.full((n, j), 1.0)
    force = np.full((n, j), power / j)
    air = airborne(t)
    return Trajectory(
        time=t, torso_pos=pos, torso_quat=quat, torso_linvel=vel, joint_vel=jv,
        actuator_force=force, foot_contacts=np.tile(~air[:, None], (1, 4)),
        torso_contact=np.zeros(n, dtype=bool), any_contact=~air, duration=T,
        total_mass=MASS, torque_limit=5.0, n_limbs=4,
        fell=fall_at is not None, fall_time=fall_at,
    )  # fmt: skip


def wobble(t):
    return 0.01 * np.sin(2 * math.pi * 1.5 * t)


SCENARIOS = {
    "steady": make_traj(x=lambda t: 0.25 * t, y=wobble, yaw=lambda t: 0.05 * np.sin(9 * t)),
    "slow": make_traj(x=lambda t: 0.05 * t, y=wobble, power=1.0),
    "stationary": make_traj(power=0.5),
    # 1.2 m surge in the first second, then collapse at 1.5 s
    "lunge": make_traj(x=lambda t: 1.2 * np.minimum(t, 1.0), fall_at=1.5, power=30.0),
    "spinner": make_traj(x=lambda t: 0.02 * np.sin(t), yaw=lambda t: 3.0 * t, power=4.0),
    "crab": make_traj(y=lambda t: 0.25 * t, power=2.0),
    "vibrator": make_traj(x=lambda t: 0.003 * np.sin(2 * math.pi * 20 * t), power=40.0),
    "hopper": make_traj(
        x=lambda t: 0.25 * t,
        z=lambda t: 0.15 + 0.3 * np.abs(np.sin(2 * math.pi * 1.0 * t)),
        airborne=lambda t: np.abs(np.sin(2 * math.pi * 1.0 * t)) > 0.2,
        power=20.0,
    ),
}


def fitness(name: str) -> float:
    return compute_fitness(compute_metrics(SCENARIOS[name]), CFG)[0]


def test_all_metric_keys_present_and_finite() -> None:
    for name, traj in SCENARIOS.items():
        m = compute_metrics(traj)
        assert set(m) == set(METRIC_KEYS), name
        assert all(math.isfinite(v) for v in m.values()), name


def test_walker_ordering() -> None:
    assert fitness("steady") > fitness("slow") > fitness("stationary")


@pytest.mark.parametrize("cheat", ["lunge", "spinner", "crab", "vibrator", "hopper"])
def test_cheats_score_below_slow_straight_walker(cheat) -> None:
    assert fitness(cheat) < fitness("slow"), compute_fitness(compute_metrics(SCENARIOS[cheat]), CFG)


def test_basic_metric_values() -> None:
    m = compute_metrics(SCENARIOS["steady"])
    assert m["forward_displacement"] == pytest.approx(2.5, abs=1e-6)
    assert m["avg_forward_velocity"] == pytest.approx(0.25, abs=1e-6)
    assert m["sustained_velocity"] == pytest.approx(0.25, abs=1e-6)
    assert m["energy"] == pytest.approx(20.0, rel=1e-6)  # 2 W for 10 s
    assert m["cost_of_transport"] == pytest.approx(20.0 / (MASS * 9.81 * 2.5), rel=1e-6)
    assert m["survival_time"] == pytest.approx(T) and m["fell"] == 0.0
    assert m["lateral_displacement"] < 0.02 and m["mean_tilt"] < 1e-9


def test_lunge_has_no_sustained_progress() -> None:
    m = compute_metrics(SCENARIOS["lunge"])
    assert m["forward_displacement"] == pytest.approx(1.2)
    assert m["sustained_velocity"] == pytest.approx(0.0, abs=1e-9)
    assert m["fell"] == 1.0 and m["survival_time"] == pytest.approx(1.5)


def test_spinner_accumulates_yaw_through_wraparound() -> None:
    m = compute_metrics(SCENARIOS["spinner"])
    assert m["yaw_drift"] == pytest.approx(30.0, rel=1e-3)
    assert m["net_yaw"] == pytest.approx(30.0, rel=1e-3)


def test_tilt_and_upright_score() -> None:
    m = compute_metrics(make_traj(roll=lambda t: 0.5 + 0 * t))
    assert m["mean_tilt"] == pytest.approx(0.5, abs=1e-9)
    assert m["upright_score"] == pytest.approx(1 - 0.5 / (math.pi / 2))


def test_tilted_walker_progress_is_gated() -> None:
    upright = compute_fitness(compute_metrics(SCENARIOS["steady"]), CFG)[1]["progress"]
    tilted = make_traj(x=lambda t: 0.25 * t, roll=lambda t: 0.6 + 0 * t)
    leaning = compute_fitness(compute_metrics(tilted), CFG)[1]["progress"]
    assert 0 < leaning < upright


def test_terms_sum_to_fitness_and_raw_metrics_untouched() -> None:
    m = compute_metrics(SCENARIOS["steady"])
    before = dict(m)
    f, terms = compute_fitness(m, CFG)
    assert f == pytest.approx(sum(terms.values()))
    assert m == before


def test_non_finite_metrics_are_invalid() -> None:
    m = compute_metrics(SCENARIOS["steady"])
    m["energy"] = float("nan")
    assert compute_fitness(m, CFG)[0] == INVALID_FITNESS


def test_valid_fitness_is_bounded_below_invalid_sentinel() -> None:
    worst = make_traj(
        x=lambda t: -4.0 * t,
        y=lambda t: 4.0 * t,
        yaw=lambda t: 9.0 * t,
        fall_at=9.0,
        power=1e4,
        airborne=lambda t: np.ones_like(t, dtype=bool),
    )
    assert compute_fitness(compute_metrics(worst), CFG)[0] > INVALID_FITNESS


def test_metrics_module_does_not_import_mujoco() -> None:
    code = "import sys, evomorph.simulation.metrics; print('mujoco' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
