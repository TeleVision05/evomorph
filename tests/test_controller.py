"""Controller interface, CPG output dimensions and gait patterns (spec test 7)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import ALL_CONFIGS, configured_genome

from evomorph.controllers.base import (
    Controller,
    Observation,
    make_controller,
    register_controller,
    registered_kinds,
)
from evomorph.controllers.cpg import CPGController, gait_phase, gait_phases, ramp
from evomorph.morphology.generator import describe_layout
from evomorph.morphology.genome import GAITS


def _obs(n: int, n_limbs: int, t: float = 0.0) -> Observation:
    return Observation(t, np.zeros(n), np.zeros(n), np.array([1.0, 0, 0, 0]), np.zeros(n_limbs))


@pytest.mark.parametrize(("n_pairs", "n_segments"), ALL_CONFIGS)
@pytest.mark.parametrize("gait", GAITS)
def test_output_shape_and_limits(n_pairs, n_segments, gait) -> None:
    g = configured_genome(n_pairs, n_segments)
    g.controller.gait = gait
    g.controller.hip_amplitude = 0.9  # above hip_range of 45 deg -> must be clipped
    layout = describe_layout(g)
    ctrl = make_controller(g)
    ctrl.reset(layout)
    for t in np.linspace(0, 5, 101):
        out = ctrl.act(float(t), _obs(layout.n_actuators, layout.n_limbs, float(t)))
        assert out.shape == (layout.n_actuators,) == (g.n_joints,)
        assert np.all(np.isfinite(out))
        assert np.all(np.abs(out) <= max(layout.hip_range, layout.knee_range) + 1e-12)


def test_ramp_starts_at_zero_and_is_smooth() -> None:
    g = configured_genome(2, 2)
    ctrl = make_controller(g)
    ctrl.reset(describe_layout(g))
    assert np.all(ctrl.act(0.0, None) == 0.0)
    assert ramp(0.0) == 0.0 and ramp(1.0) == 1.0 and ramp(5.0) == 1.0
    ts = np.linspace(0, 1, 1001)
    steps = np.abs(np.diff([ramp(t) for t in ts]))
    assert steps.max() < 0.002  # no jump anywhere in the ramp


def test_cpg_formula_matches_spec() -> None:
    g = configured_genome(2, 2)
    c = g.controller
    c.hip_amplitude, c.knee_amplitude, c.frequency, c.knee_phase_offset = 0.3, 0.2, 1.3, 0.7
    c.phase_offsets = np.array([0.1, -0.2, 0.05, 0.0])
    layout = describe_layout(g)
    ctrl = make_controller(g)
    ctrl.reset(layout)
    t = 2.37
    out = ctrl.act(t, None)
    for limb in layout.limbs:
        phi = gait_phase(c.gait, limb.pair, 0 if limb.side == "left" else 1, 2)
        phi += c.phase_offsets[limb.limb_index]
        w = 2 * math.pi * c.frequency * t
        assert out[limb.hip_index] == pytest.approx(0.3 * math.sin(w + phi))
        assert out[limb.knee_index] == pytest.approx(0.2 * math.sin(w + phi + 0.7))


def test_one_segment_controller_ignores_knee_genes() -> None:
    g = configured_genome(3, 1)
    ctrl = make_controller(g)
    ctrl.reset(describe_layout(g))
    assert ctrl.act(3.0, None).shape == (6,)


def _pi_groups(gait: str, n_pairs: int) -> np.ndarray:
    return np.round(gait_phases(gait, n_pairs) / (math.pi / 2)).astype(int) % 4


def test_gait_patterns() -> None:
    # quadruped (limb order LF, RF, LH, RH): trot pairs diagonals, pace pairs sides
    assert list(_pi_groups("diagonal", 2)) == [0, 2, 2, 0]
    assert list(_pi_groups("alternating", 2)) == [0, 2, 0, 2]
    assert sorted(_pi_groups("tripod", 2)) == [0, 1, 2, 3]  # four-beat walk
    # hexapod tripod: L0, R1, L2 together vs R0, L1, R2
    assert list(_pi_groups("tripod", 3)) == [0, 2, 2, 0, 0, 2]
    # metachronal: phase changes monotonically front to back on each side
    left = gait_phases("metachronal", 4)[0::2]
    assert np.all(np.diff(left) < 0)


def test_documented_coincidences() -> None:
    for gait in GAITS:  # 2 limbs: everything collapses to plain alternation
        np.testing.assert_allclose(gait_phases(gait, 1), [0.0, math.pi])
    np.testing.assert_allclose(gait_phases("tripod", 3), gait_phases("diagonal", 3))
    shifted = (gait_phases("diagonal", 2) + math.pi) % (2 * math.pi)
    np.testing.assert_allclose(gait_phases("metachronal", 2), shifted)
    p8 = [tuple(np.round(gait_phases(g, 4), 6)) for g in GAITS]
    assert len(set(p8)) == 4


def test_registry_dispatch_and_extension() -> None:
    assert "cpg" in registered_kinds()
    assert isinstance(make_controller(configured_genome(2, 2)), CPGController)

    class Zero(Controller):
        def act(self, t, obs):
            return np.zeros(self.layout.n_actuators)

    register_controller("zero-test")(lambda genome: Zero())
    g = configured_genome(2, 1)
    g.controller.kind = "zero-test"
    z = make_controller(g)
    z.reset(describe_layout(g))
    assert z.act(1.0, None).shape == (4,)
    g.controller.kind = "missing"
    with pytest.raises(KeyError):
        make_controller(g)


def test_mismatched_limb_arrays_are_rejected() -> None:
    g = configured_genome(2, 2)
    g.controller.phase_offsets = np.zeros(6)
    with pytest.raises(ValueError):
        make_controller(g).reset(describe_layout(g))
