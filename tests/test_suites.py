"""Terrain suites: the train/test split is disjoint on every axis, and every suite builds."""

from __future__ import annotations

import pytest

from evomorph.simulation.environment import ENV_KINDS, build_world
from evomorph.simulation.suites import SUITE_DOCS, SUITES, TEST, TRAIN, get_suite

SEEDED = ("uneven", "random")  # kinds whose geometry depends on the terrain seed


def _axis(envs, kind, field):
    return {getattr(e, field) for e in envs if e.kind == kind}


def uniform(envs):
    """Friction values of the uniform-friction environments."""
    return {e.friction for e in envs if e.kind == "friction" and e.friction_spread == 1.0}


def seeded(envs):
    """Terrain seeds of environments whose geometry depends on the seed."""
    return {e.seed for e in envs if e.kind in SEEDED or e.friction_spread > 1}


def test_train_and_test_share_no_environment() -> None:
    assert not set(TRAIN) & set(TEST)
    assert not {e.label() for e in TRAIN} & {e.label() for e in TEST}


def test_test_suite_shifts_every_axis_of_variation() -> None:
    assert not _axis(TRAIN, "slope", "slope_deg") & _axis(TEST, "slope", "slope_deg")
    assert not uniform(TRAIN) & uniform(TEST)
    assert min(uniform(TEST)) < min(uniform(TRAIN)) and max(uniform(TEST)) > max(uniform(TRAIN))
    assert max(_axis(TEST, "slope", "slope_deg")) > max(_axis(TRAIN, "slope", "slope_deg"))
    assert min(_axis(TEST, "slope", "slope_deg")) < min(_axis(TRAIN, "slope", "slope_deg"))
    for field in ("roughness", "n_obstacles"):
        assert not _axis(TRAIN, "uneven", field) & _axis(TEST, "uneven", field)
        assert max(_axis(TEST, "uneven", field)) > max(_axis(TRAIN, "uneven", field))
    for field in ("slope_deg", "roughness", "n_obstacles", "friction_spread"):
        assert max(_axis(TEST, "random", field)) > max(_axis(TRAIN, "random", field))


def test_test_suite_uses_unseen_terrain_seeds() -> None:
    assert not seeded(TRAIN) & seeded(TEST)


def test_training_includes_every_terrain_kind_and_low_friction() -> None:
    assert {e.kind for e in TRAIN} == set(ENV_KINDS)
    assert min(e.friction for e in TRAIN if e.kind == "friction") < 0.5  # skaters must pay


@pytest.mark.parametrize("name", sorted(SUITES))
def test_every_suite_builds_with_unique_labels(name) -> None:
    envs = get_suite(name)
    assert envs and name in SUITE_DOCS
    labels = [e.label() for e in envs]
    assert len(set(labels)) == len(labels)
    for env in envs:
        assert build_world(env).ground_height(0.0, 0.0) == pytest.approx(0.0, abs=1e-5)


def test_unknown_suite_is_a_plain_error() -> None:
    with pytest.raises(ValueError, match=r"no terrain suite called 'mars'.*train"):
        get_suite("mars")
