"""Glitch guards, fall detection, terrain-relative guards and physics-rate instrumentation.

Each guard is triggered on purpose and must produce ``valid=False`` with its own
reason; disabling any guard fails its test.
"""

from __future__ import annotations

import math
from dataclasses import replace

import mujoco
import numpy as np
import pytest
from conftest import configured_genome

import evomorph.simulation.environment as environment
import evomorph.simulation.evaluator as evaluator
from evomorph.config import SimConfig
from evomorph.controllers.base import Controller, make_controller
from evomorph.morphology.genome import reference_quadruped
from evomorph.simulation.environment import EnvSpec, WorldFragments
from evomorph.simulation.evaluator import (
    _FallDetector,
    build_body,
    evaluate,
    observe,
    settle,
    spawn,
)
from evomorph.simulation.stepping import (
    IntervalStepper,
    contact_force_and_slip,
    speed_limit,
)
from evomorph.utils.random_state import make_rng

SHORT = SimConfig(duration=1.0)
# Guards are checked in order (state, speed, spin, penetration, launch, clip). To
# isolate one, the ones that a violent kick would trip first are relaxed.
NO_SPEED = replace(SHORT, max_linear_speed=math.inf, max_speed_per_size=math.inf)
NO_SPEED_SPIN = replace(NO_SPEED, max_angular_speed=math.inf)
NO_PENETRATION = replace(NO_SPEED_SPIN, max_contact_penetration=math.inf)


def _kick_during_run(monkeypatch, kick, at: float = 0.2) -> None:
    """Apply ``kick(data)`` at measured time ``at``, between two control steps."""
    holder = {}
    real_build, real_make = evaluator.build_body, evaluator.make_controller

    def build(*args, **kwargs):
        holder["body"] = real_build(*args, **kwargs)
        return holder["body"]

    def make(genome):
        inner = real_make(genome)

        class Kicker(Controller):
            def reset(self, layout):
                super().reset(layout)
                inner.reset(layout)

            def act(self, t, obs):
                if abs(t - at) < 1e-9:
                    kick(holder["body"].data)
                return inner.act(t, obs)

        return Kicker()

    monkeypatch.setattr(evaluator, "build_body", build)
    monkeypatch.setattr(evaluator, "make_controller", make)


def _set(attr: str, index, value=None, add=None):
    def kick(d):
        arr = getattr(d, attr)
        arr[index] = arr[index] + add if add is not None else value

    return kick


@pytest.mark.parametrize(
    ("kick", "sim", "reason"),
    [
        (_set("qvel", 0, 40.0), SHORT, "runaway linear speed"),
        (_set("qvel", slice(3, 6), 80.0), NO_SPEED, "runaway angular speed"),
        (_set("qvel", 0, math.nan), SHORT, "integrator instability"),
        (_set("qpos", 2, add=-0.3), NO_SPEED_SPIN, "sank"),
        (_set("qpos", 2, add=1.5), NO_SPEED_SPIN, "launched"),
        (_set("qpos", 2, -0.3), NO_PENETRATION, "torso clipped"),
    ],
)
def test_each_glitch_guard_marks_the_run_invalid(monkeypatch, kick, sim, reason) -> None:
    r = evaluate(reference_quadruped(), sim=sim)
    assert r.valid, r.failure_reason  # the unkicked run is fine, even with guards relaxed
    _kick_during_run(monkeypatch, kick)
    r = evaluate(reference_quadruped(), sim=sim)
    assert not r.valid and reason in r.failure_reason, r.failure_reason
    assert r.fitness == evaluator.INVALID_FITNESS and r.metrics == {}


def test_speed_limit_scales_with_body_size() -> None:
    sim = SimConfig()
    small = configured_genome(2, 1, torso_length=0.15, upper_limb_length=0.05)
    big = configured_genome(2, 2, torso_length=0.8, upper_limb_length=0.35, lower_limb_length=0.35)
    lim_small = speed_limit(build_body(small, EnvSpec(), sim), sim)
    lim_big = speed_limit(build_body(big, EnvSpec(), sim), sim)
    assert lim_small == pytest.approx(10.0 * 0.25)  # 0.15 m torso + 2 x 0.05 m legs
    assert lim_big == sim.max_linear_speed


def test_point_foot_biped_cannot_stand_and_falls_with_a_reason() -> None:
    """Two point feet share one x: the torso pitches onto the ground (a real fall)."""
    g = configured_genome(1, 2, torso_length=0.6, foot_length=0.0)
    g = replace(g, controller=replace(g.controller, hip_amplitude=0.0, knee_amplitude=0.0))
    r = evaluate(g, sim=SimConfig(duration=4.0))
    assert r.valid and r.fell
    assert r.failure_reason.startswith("fell at") and "torso resting" in r.failure_reason
    assert r.fitness_terms["fall"] == -1.0 and r.fitness < 0


def test_sole_lets_the_same_biped_stand() -> None:
    g = configured_genome(1, 2, torso_length=0.6, foot_length=0.12)
    g = replace(g, controller=replace(g.controller, hip_amplitude=0.0, knee_amplitude=0.0))
    r = evaluate(g, sim=SimConfig(duration=4.0))
    assert r.valid and not r.fell, r.failure_reason


def test_flip_is_detected() -> None:
    body = build_body(reference_quadruped(), EnvSpec(), SimConfig())
    spawn(body, SimConfig(), make_rng(0))
    body.data.qpos[3:7] = [0.0, 1.0, 0.0, 0.0]  # upside down
    assert _FallDetector(SimConfig()).update(body, 0.0) == "flipped over"


def test_intermittent_torso_contact_accumulates_into_a_fall() -> None:
    """Belly contact every other interval never lasted 0.25 s in a row, but it is a fall."""
    sim = SimConfig()
    body = build_body(reference_quadruped(), EnvSpec(), sim)
    spawn(body, sim, make_rng(0))
    det = _FallDetector(sim)
    dt = 1.0 / sim.control_rate
    reasons = [det.update(body, dt if k % 2 == 0 else 0.0) for k in range(50)]
    assert "torso resting on the ground" in reasons
    light = _FallDetector(sim)  # 20% of the time: under the 0.25 s per 1 s budget
    assert all(light.update(body, dt if k % 5 == 0 else 0.0) is None for k in range(200))


def _tilted_world(slope_deg: float, with_height: bool):
    a = math.radians(slope_deg)

    def build(env: EnvSpec) -> WorldFragments:
        flat = environment._flat(EnvSpec(friction=env.friction))
        tilt = f'euler="0 {-a:.6g} 0" material="ground"'
        floor = flat.worldbody.replace('material="ground"', tilt)
        height = (lambda x, y: x * math.tan(a)) if with_height else None
        return WorldFragments(flat.assets, floor, spawn_height=0.0, height_at=height)

    return build


@pytest.mark.parametrize("slope", [-10.0, -5.0, 10.0])
def test_guards_follow_the_terrain_height(monkeypatch, slope) -> None:
    monkeypatch.setattr(environment, "build_terrain_extension", _tilted_world(slope, True))
    r = evaluate(reference_quadruped(), env=EnvSpec(kind="slope", slope_deg=slope),
                 sim=SimConfig(duration=6.0))  # fmt: skip
    assert r.valid, r.failure_reason


def test_without_height_at_a_downhill_walker_is_flagged(monkeypatch) -> None:
    """Documents the seam contract: a slope must supply height_at."""
    monkeypatch.setattr(environment, "build_terrain_extension", _tilted_world(-10.0, False))
    r = evaluate(reference_quadruped(), env=EnvSpec(kind="slope"), sim=SimConfig(duration=6.0))
    assert not r.valid and "clipped" in r.failure_reason


def _settled_reference(sim: SimConfig):
    g = reference_quadruped()
    body = build_body(g, EnvSpec(), sim)
    spawn(body, sim, make_rng(0))
    settle(body, sim, None)
    ctrl = make_controller(g)
    ctrl.reset(body.layout)
    return body, ctrl


def test_energy_is_integrated_at_physics_rate() -> None:
    """Matches an independent step-by-step integral (control-rate sampling read ~0.67x)."""
    sim = SimConfig(duration=2.0)
    body, ctrl = _settled_reference(sim)
    m, d = body.model, body.data
    feet, energy = np.zeros(4, bool), 0.0
    for k in range(round(sim.duration * sim.control_rate)):
        t = k / sim.control_rate
        d.ctrl[:] = ctrl.act(t, observe(t, body, feet))
        for _ in range(sim.n_substeps):
            mujoco.mj_step(m, d)
            energy += np.sum(np.abs(d.actuator_force * d.qvel[body.dof_adr])) * m.opt.timestep
        feet = evaluator.ground_contacts(body)[0]
    assert evaluate(reference_quadruped(), sim=sim).metrics["energy"] == pytest.approx(energy)


def test_fast_contact_path_matches_mujoco_contact_force_and_point_velocity() -> None:
    sim = SimConfig()
    body, ctrl = _settled_reference(sim)
    checked = 0
    for k in range(100):
        body.data.ctrl[:] = ctrl.act(k * 0.02, observe(k * 0.02, body, np.zeros(4, bool)))
        mujoco.mj_step(body.model, body.data, nstep=7)
        if body.data.ncon == 0:
            continue
        idx = np.arange(body.data.ncon)
        body.fast_contacts = True
        fast = contact_force_and_slip(body, idx)
        body.fast_contacts = False
        slow = contact_force_and_slip(body, idx)
        np.testing.assert_allclose(fast[0], slow[0], rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(fast[1], slow[1], rtol=1e-5, atol=1e-7)
        checked += 1
    assert checked > 50


def test_ground_reaction_impulse_balances_weight() -> None:
    """On flat ground the mean normal force over a run equals m g (momentum balance)."""
    sim = SimConfig()
    body, ctrl = _settled_reference(sim)
    stepper = IntervalStepper(body, sim, 501)
    for k in range(500):
        body.data.ctrl[:] = ctrl.act(k * 0.02, observe(k * 0.02, body, stepper.feet))
        stepper.advance(k + 1)
    s = stepper.stats
    impulse = s.total("foot_impulse") + s.total("limb_impulse") + s.total("torso_impulse")
    assert impulse / (10.0 * body.total_mass * 9.81) == pytest.approx(1.0, abs=0.03)
    assert s.total("foot_impulse") > 0.999 * impulse  # the reference walks on its feet
