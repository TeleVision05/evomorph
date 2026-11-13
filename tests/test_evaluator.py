"""Physics simulation end to end (spec tests 8, 10 and validation items 1, 2, 3, 5, 6)."""

from __future__ import annotations

import math
import pickle
import time
from dataclasses import replace

import mujoco
import numpy as np
import pytest
from conftest import ALL_CONFIGS, configured_genome

from evomorph.config import SimConfig
from evomorph.morphology.genome import (
    load_genome,
    random_genome,
    reference_hexapod,
    reference_quadruped,
    save_genome,
)
from evomorph.simulation.environment import EnvSpec
from evomorph.simulation.evaluator import (
    EvalResult,
    build_body,
    evaluate,
    evaluate_robust,
    spawn,
)
from evomorph.simulation.metrics import INVALID_FITNESS
from evomorph.utils.random_state import make_rng, spawn_rngs


def _walks_straight(r: EvalResult) -> None:
    m = r.metrics
    assert r.valid and not r.failure_reason, r.failure_reason
    assert m["forward_displacement"] >= 1.0
    assert m["survival_time"] == pytest.approx(10.0)
    assert m["lateral_displacement"] < 0.25 * m["forward_displacement"]
    assert m["yaw_drift"] < 0.5  # rad, under ~30 degrees of heading excursion
    assert m["mean_tilt"] < 0.35


@pytest.mark.parametrize("make", [reference_quadruped, reference_hexapod])
def test_reference_creatures_walk_forward(make) -> None:
    _walks_straight(evaluate(make()))


def test_reference_walk_is_robust_to_seed() -> None:
    for seed in (1, 2):
        _walks_straight(evaluate(reference_quadruped(), seed=seed))


@pytest.mark.parametrize(("n_pairs", "n_segments"), ALL_CONFIGS)
def test_every_configuration_simulates(n_pairs, n_segments) -> None:
    g = configured_genome(n_pairs, n_segments)
    r = evaluate(g, sim=SimConfig(duration=2.0))
    assert r.valid, r.failure_reason
    assert r.metrics["n_joints"] == g.n_joints == 2 * n_pairs * n_segments
    assert r.metrics["n_limbs"] == 2 * n_pairs
    assert all(math.isfinite(v) for v in r.metrics.values())


def test_one_segment_quadruped_can_walk() -> None:
    """A hip-only quadruped walks with the four-beat ("tripod") gait even with zero tilt.

    Measured 2026-11-13: hip_splay 65, upper 0.25 m, tilt 0 -> 4.14 m, lateral 0.04 m.
    The builder's tilted diagonal variant (splay 30, tilt +20) goes 1.79 m but
    turns 1.5 rad; with the hexapod recipe (splay 50) it manages only 0.34 m.
    """
    base = reference_quadruped()
    m = replace(base.morphology, n_segments=1, hip_splay=65.0, upper_limb_length=0.25,
                hip_axis_tilt=0.0)  # fmt: skip
    walker = replace(base, morphology=m, controller=replace(base.controller, gait="tripod"))
    _walks_straight(evaluate(walker))
    assert evaluate(walker).metrics["foot_support_fraction"] > 0.99


def test_one_segment_hexapod_walks_through_hip_tilt() -> None:
    """For the hexapod tripod the tilted hip axis is what makes progress: 0 tilt shuffles."""
    base = reference_hexapod()

    def with_tilt(tilt: float):
        m = replace(base.morphology, n_segments=1, hip_splay=50.0, upper_limb_length=0.25,
                    hip_axis_tilt=tilt)  # fmt: skip
        return replace(base, morphology=m)

    walker = evaluate(with_tilt(20.0)).metrics
    shuffler = evaluate(with_tilt(0.0)).metrics
    assert walker["forward_displacement"] > 0.5 and walker["fell"] == 0.0
    assert abs(shuffler["forward_displacement"]) < 0.3


def test_determinism_and_json_reload(tmp_path) -> None:
    g = random_genome(np.random.default_rng(42))
    a, b = evaluate(g, seed=5), evaluate(g, seed=5)
    assert a.metrics == b.metrics and a.fitness == b.fitness
    reloaded = load_genome(save_genome(g, tmp_path / "g.json"))
    c = evaluate(reloaded, seed=5)
    assert c.metrics == a.metrics and c.fitness == a.fitness
    assert evaluate(g, seed=6).metrics != a.metrics  # seed drives spawn noise


def test_result_is_picklable_for_process_pools() -> None:
    r = evaluate(reference_quadruped(), sim=SimConfig(duration=1.0))
    back = pickle.loads(pickle.dumps(r))
    assert back.metrics == r.metrics and back.env == r.env
    pickle.dumps((evaluate, reference_quadruped(), EnvSpec(), SimConfig()))


def test_invalid_genome_yields_invalid_result_not_crash() -> None:
    g = reference_quadruped()
    g.morphology.torso_width = -1.0
    g.controller.phase_offsets = np.zeros(3)
    r = evaluate(g)
    assert not r.valid and "invalid genome" in r.failure_reason
    assert r.fitness == INVALID_FITNESS and r.metrics == {}
    # An unbuildable world (slope out of range) is an invalid result, not a crash.
    r = evaluate(reference_quadruped(), env=EnvSpec(kind="slope", slope_deg=60.0))
    assert not r.valid and "build failed" in r.failure_reason and "slope" in r.failure_reason


def test_glitch_guard_marks_launch_invalid() -> None:
    sim = SimConfig(max_height_gain=-1.0)  # any state "launches": exercise the guard path
    r = evaluate(reference_quadruped(), sim=replace(sim, duration=1.0))
    assert not r.valid and "glitch" in r.failure_reason


def test_robust_evaluation_aggregates_seeds() -> None:
    g, sim = reference_quadruped(), SimConfig(duration=3.0)
    singles = [evaluate(g, sim=sim, seed=s) for s in (0, 1, 2)]
    mean = evaluate_robust(g, sim=sim, seeds=(0, 1, 2))
    worst = evaluate_robust(g, sim=sim, seeds=(0, 1, 2), aggregate="min")
    assert mean.valid and mean.seed_fitness == tuple(r.fitness for r in singles)
    assert mean.fitness == pytest.approx(np.mean(mean.seed_fitness))
    assert worst.fitness == min(mean.seed_fitness) <= mean.fitness
    fwd = np.mean([r.metrics["forward_displacement"] for r in singles])
    assert mean.metrics["forward_displacement"] == pytest.approx(fwd)
    assert mean.fitness == pytest.approx(sum(mean.fitness_terms.values()))
    half_dt = evaluate_robust(g, sim=sim, seeds=(0,), sims=(replace(sim, timestep=0.001),))
    assert len(half_dt.seed_fitness) == 2
    pickle.dumps((evaluate_robust, g, mean))


def test_robust_evaluation_is_invalid_if_any_run_is() -> None:
    sim = SimConfig(duration=1.0)
    bad = replace(sim, max_height_gain=-1.0)  # every state "launches"
    r = evaluate_robust(reference_quadruped(), sim=sim, seeds=(0,), sims=(bad,))
    assert not r.valid and "launched" in r.failure_reason and len(r.seed_fitness) == 2


def test_friction_changes_outcome() -> None:
    g = reference_quadruped()
    slippery = evaluate(g, env=EnvSpec(friction=0.1)).metrics["forward_displacement"]
    normal = evaluate(g).metrics["forward_displacement"]
    assert slippery < normal


def test_small_random_sweep_has_no_crash_or_nan() -> None:
    results = [evaluate(random_genome(r), sim=SimConfig(duration=4.0)) for r in spawn_rngs(3, 24)]
    assert all(r.valid for r in results), [r.failure_reason for r in results if not r.valid]
    for r in results:
        assert all(math.isfinite(v) for v in r.metrics.values())
        assert math.isfinite(r.fitness)
    fwd = np.array([r.metrics["forward_displacement"] for r in results])
    assert np.median(np.abs(fwd)) < 0.5  # most random creatures barely move


def test_single_process_throughput() -> None:
    g = reference_hexapod()
    evaluate(g, sim=SimConfig(duration=1.0))  # warm-up
    cpu = time.process_time()
    evaluate(g)
    assert time.process_time() - cpu < 0.5  # CPU seconds for a 10 s evaluation


def test_viewer_control_callback_reproduces_walking() -> None:
    from evomorph.visualization.viewer import make_control_callback

    g, sim = reference_quadruped(), SimConfig()
    body = build_body(g, EnvSpec(), sim)
    spawn(body, sim, make_rng(0))
    mujoco.set_mjcb_control(make_control_callback(g, body, sim))
    try:
        x0 = float(body.data.qpos[0])
        mujoco.mj_step(body.model, body.data, nstep=round(6.0 / sim.timestep))
    finally:
        mujoco.set_mjcb_control(None)
    assert float(body.data.qpos[0]) - x0 > 1.0


def test_viewer_reset_respawns_on_the_ground() -> None:
    from evomorph.visualization.viewer import prepare_body

    sim = SimConfig()
    body = prepare_body(reference_quadruped(), None, sim, seed=0)
    spawned = body.data.qpos.copy()
    mujoco.mj_step(body.model, body.data, nstep=500)
    mujoco.mj_resetData(body.model, body.data)  # what Backspace does
    np.testing.assert_array_equal(body.data.qpos, spawned)
    assert spawned[2] < 0.3  # standing height, not the 0.5 m+ build height


@pytest.mark.gl
def test_recording_captures_frames() -> None:
    from evomorph.simulation.evaluator import RecordSpec

    r = evaluate(reference_quadruped(), sim=SimConfig(duration=1.0),
                 record=RecordSpec(width=160, height=120, fps=10))  # fmt: skip
    assert r.frames is not None and len(r.frames) == 10
    assert r.frames[0].shape == (120, 160, 3) and r.frames[-1].std() > 0
