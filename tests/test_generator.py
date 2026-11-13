"""MJCF generation, compilation and body-plan physics conventions (spec tests 4, 5, 6)."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
from conftest import ALL_CONFIGS, configured_genome

from evomorph.config import SimConfig
from evomorph.morphology.generator import (
    build_mjcf,
    describe_layout,
    estimate_mass,
    limb_tag,
)
from evomorph.morphology.genome import random_genome, reference_quadruped
from evomorph.morphology.validator import ModelCompileError, compile_model
from evomorph.simulation.environment import EnvSpec, build_world
from evomorph.simulation.evaluator import build_body, lowest_point, settle, spawn
from evomorph.utils.random_state import make_rng


def _foot(model, data, tag: str) -> np.ndarray:
    return data.geom_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{tag}")].copy()


def _posed(genome, joint: str, angle: float):
    model = compile_model(build_mjcf(genome))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    data.qpos[model.joint(joint).qposadr[0]] = angle
    mujoco.mj_forward(model, data)
    return model, data


@pytest.mark.parametrize(("n_pairs", "n_segments"), ALL_CONFIGS)
def test_every_configuration_compiles_with_matching_counts(n_pairs, n_segments) -> None:
    g = configured_genome(n_pairs, n_segments)
    model = compile_model(build_mjcf(g))
    layout = describe_layout(g)
    n_joints = 2 * n_pairs * n_segments
    assert model.nu == layout.n_actuators == n_joints == g.n_joints
    assert model.njnt == n_joints + 1  # + free root joint
    assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert model.nq == 7 + n_joints
    assert all((limb.knee_index is not None) == (n_segments == 2) for limb in layout.limbs)
    for kind in (mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_ACTUATOR):
        count = {"body": model.nbody, "joint": model.njnt, "actuator": model.nu}
        n = count[kind.name.split("_")[-1].lower()]
        names = [mujoco.mj_id2name(model, kind, i) for i in range(n)]
        named = [x for x in names if x]
        assert len(named) == len(set(named))
    # actuator i drives the joint the layout says it does
    for limb in layout.limbs:
        assert model.actuator(limb.hip_index).name == f"{limb.hip_joint}_act"
        if limb.knee_index is not None:
            assert model.actuator(limb.knee_index).name == f"{limb.knee_joint}_act"


def test_compile_error_is_wrapped() -> None:
    with pytest.raises(ModelCompileError):
        compile_model("<mujoco><worldbody><geom type='nonsense'/></worldbody></mujoco>")


def test_limbs_are_mirrored_across_sagittal_plane() -> None:
    g = configured_genome(3, 2, hip_axis_tilt=20.0)
    model = compile_model(build_mjcf(g))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    torso = data.xpos[1]
    for p in range(3):
        left = _foot(model, data, limb_tag(p, 0)) - torso
        right = _foot(model, data, limb_tag(p, 1)) - torso
        np.testing.assert_allclose(left * [1, -1, 1], right, atol=1e-9)
    xs = [(_foot(model, data, limb_tag(p, 0)) - torso)[0] for p in range(3)]
    assert xs[0] > xs[1] > xs[2]  # pair 0 is the front pair


@pytest.mark.parametrize("n_segments", [1, 2])
def test_positive_hip_swings_foot_forward_on_both_sides(n_segments) -> None:
    g = configured_genome(2, n_segments)
    for side in (0, 1):
        tag = limb_tag(0, side)
        m0, d0 = _posed(g, f"hip_{tag}", 0.0)
        m1, d1 = _posed(g, f"hip_{tag}", 0.2)
        assert _foot(m1, d1, tag)[0] > _foot(m0, d0, tag)[0] + 0.01


def test_positive_knee_lifts_foot_on_both_sides() -> None:
    g = configured_genome(2, 2)
    for side in (0, 1):
        tag = limb_tag(1, side)
        m0, d0 = _posed(g, f"knee_{tag}", 0.0)
        m1, d1 = _posed(g, f"knee_{tag}", 0.3)
        assert _foot(m1, d1, tag)[2] > _foot(m0, d0, tag)[2] + 0.01


def test_hip_axis_tilt_makes_forward_swing_lift_the_foot() -> None:
    flat = configured_genome(2, 1, hip_splay=30.0, hip_axis_tilt=0.0)
    tilted = configured_genome(2, 1, hip_splay=30.0, hip_axis_tilt=25.0)
    for side in (0, 1):
        tag = limb_tag(0, side)
        lifts = []
        for g in (flat, tilted):
            m0, d0 = _posed(g, f"hip_{tag}", 0.0)
            m1, d1 = _posed(g, f"hip_{tag}", 0.3)
            lifts.append(_foot(m1, d1, tag)[2] - _foot(m0, d0, tag)[2])
        assert lifts[1] > lifts[0] + 0.01


def test_mass_comes_from_density_and_genes() -> None:
    g = reference_quadruped()
    model = compile_model(build_mjcf(g))
    assert float(model.body_subtreemass[1]) == pytest.approx(estimate_mass(g), rel=1e-3)
    g.morphology.torso_mass_scale = 2.0
    heavier = compile_model(build_mjcf(g))
    assert float(heavier.body_subtreemass[1]) > float(model.body_subtreemass[1])
    assert np.all(model.body_inertia[1:] > 0)


def test_spawn_puts_lowest_point_just_above_ground_then_feet_take_load() -> None:
    sim = SimConfig()
    body = build_body(reference_quadruped(), EnvSpec(), sim)
    spawn(body, sim, make_rng(0))
    assert lowest_point(body.model, body.data) == pytest.approx(sim.spawn_clearance, abs=1e-9)
    settle(body, sim, None)
    from evomorph.simulation.evaluator import ground_contacts

    feet, torso, _ = ground_contacts(body)
    assert feet.all() and not torso


def test_rest_pose_overlaps_are_excluded_not_exploded() -> None:
    # Four thick-legged pairs crammed on a short torso overlap at rest.
    g = configured_genome(4, 2, torso_length=0.15, limb_thickness=0.035, hip_splay=60.0)
    xml = build_mjcf(g)
    assert xml.count("<exclude") > 0
    sim = SimConfig()
    body = build_body(g, EnvSpec(), sim)
    spawn(body, sim, make_rng(0))
    settle(body, sim, None)  # raises SimulationGlitchError if overlaps exploded


def test_random_genomes_always_compile(rng) -> None:
    for _ in range(40):
        compile_model(build_mjcf(random_genome(rng)))


def test_flat_world_honours_friction_and_other_kinds_are_a_seam() -> None:
    assert 'friction="0.3 ' in build_world(EnvSpec(friction=0.3)).worldbody
    for kind in ("slope", "uneven", "friction", "random"):
        with pytest.raises(NotImplementedError):
            build_world(EnvSpec(kind=kind))
    with pytest.raises(ValueError):
        build_world(EnvSpec(kind="lava"))
