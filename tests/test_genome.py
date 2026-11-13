"""Genome generation, bounds, repair, validation and serialization (spec tests 1 and 10)."""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pytest

from evomorph.morphology.genome import (
    CONTROLLER_GENES,
    GENE_BOUNDS,
    MORPHOLOGY_GENES,
    ControllerGenes,
    Genome,
    MorphologyGenes,
    from_dict,
    genome_hash,
    load_genome,
    random_genome,
    reference_hexapod,
    reference_quadruped,
    repair,
    save_genome,
    to_dict,
)
from evomorph.morphology.validator import validate_genome


def _assert_within_bounds(g: Genome) -> None:
    for name, b in GENE_BOUNDS.items():
        part = g.morphology if b.group == "morphology" else g.controller
        val = getattr(part, name)
        if b.per_limb or b.kind == "vector":
            arr = np.asarray(val)
            assert arr.shape == ((g.n_limbs,) if b.per_limb else arr.shape), name
            assert np.all((arr >= b.low) & (arr <= b.high)), name
        elif b.is_discrete:
            assert val in b.values, name
        else:
            assert b.low <= val <= b.high, name


def test_gene_bounds_cover_every_dataclass_field() -> None:
    morph_fields = {f.name for f in dataclasses.fields(MorphologyGenes)}
    ctrl_fields = {f.name for f in dataclasses.fields(ControllerGenes)}
    assert set(MORPHOLOGY_GENES) == morph_fields
    assert set(CONTROLLER_GENES) == ctrl_fields
    for b in GENE_BOUNDS.values():
        assert b.is_discrete or b.low < b.high


def test_random_genomes_are_valid_and_diverse(rng) -> None:
    genomes = [random_genome(rng) for _ in range(200)]
    for g in genomes:
        _assert_within_bounds(g)
        assert validate_genome(g).ok
    assert {g.n_limbs for g in genomes} == {2, 4, 6, 8}
    assert {g.morphology.n_segments for g in genomes} == {1, 2}
    assert len({g.controller.gait for g in genomes}) == 4
    assert len({g.id for g in genomes}) == 200


def test_random_genome_is_seed_deterministic() -> None:
    a = random_genome(np.random.default_rng(7))
    b = random_genome(np.random.default_rng(7))
    assert to_dict(a) == to_dict(b)


def test_joint_count_follows_segments() -> None:
    g = reference_quadruped()
    assert (g.n_limbs, g.n_joints) == (4, 8)
    g.morphology.n_segments = 1
    assert g.n_joints == 4
    assert reference_hexapod().n_joints == 12


def test_repair_clips_snaps_and_resizes() -> None:
    g = reference_quadruped()
    g.morphology.torso_length = 99.0
    g.morphology.foot_radius = float("nan")
    g.morphology.n_pairs = 3.4  # type: ignore[assignment]
    g.morphology.n_segments = 7
    g.controller.gait = "moonwalk"
    g.controller.frequency = -5.0
    g.controller.phase_offsets = np.array([9.0, -9.0])  # too short and out of range
    g.controller.hip_bias = np.zeros(20)  # too long
    assert not validate_genome(g).ok
    r = repair(g)
    assert validate_genome(r).ok, validate_genome(r).errors
    assert r.morphology.n_pairs == 3 and r.morphology.n_segments == 2
    assert r.morphology.torso_length == GENE_BOUNDS["torso_length"].high
    assert math.isfinite(r.morphology.foot_radius)
    assert r.controller.gait in GENE_BOUNDS["gait"].values
    assert r.controller.frequency == GENE_BOUNDS["frequency"].low
    assert r.controller.phase_offsets.shape == (6,)
    assert r.controller.phase_offsets[0] == GENE_BOUNDS["phase_offsets"].high
    assert np.all(r.controller.phase_offsets[2:] == 0.0)  # new limbs follow the gait exactly
    assert r.controller.hip_bias.shape == (6,)


def test_repair_is_identity_on_valid_genomes(rng) -> None:
    for _ in range(20):
        g = random_genome(rng)
        assert to_dict(repair(g)) == to_dict(g)


def test_validator_reports_every_problem() -> None:
    g = reference_quadruped()
    g.morphology.hip_range = 500.0
    g.controller.knee_bias = np.zeros(3)
    g.controller.kind = "transformer"
    errors = validate_genome(g).errors
    assert any("hip_range" in e for e in errors)
    assert any("knee_bias" in e for e in errors)
    assert any("kind" in e for e in errors)
    assert not validate_genome("not a genome").ok  # type: ignore[arg-type]


def test_json_round_trip_is_exact(tmp_path, rng) -> None:
    for i in range(10):
        g = random_genome(rng)
        g.parent_ids = ("p1", "p2")
        g.generation = i
        path = save_genome(g, tmp_path / f"g{i}.json")
        back = load_genome(path)
        assert to_dict(back) == to_dict(g)
        for name in ("phase_offsets", "hip_bias", "knee_bias"):
            assert np.array_equal(getattr(back.controller, name), getattr(g.controller, name))
        assert back.morphology.torso_length == g.morphology.torso_length  # bit-exact
        assert back.parent_ids == ("p1", "p2") and back.generation == i


def test_to_dict_is_plain_json() -> None:
    d = to_dict(reference_hexapod())
    assert from_dict(json.loads(json.dumps(d))).id == "ref-hexapod"


def test_genomes_compare_by_value_and_hash_by_genes() -> None:
    a, b = reference_quadruped(), reference_quadruped()
    assert a == b  # used to raise "truth value of an array is ambiguous"
    b.controller.phase_offsets[0] = 0.1
    assert a != b and genome_hash(a) != genome_hash(b)
    renamed = from_dict(to_dict(a) | {"id": "other", "parent_ids": ["x"]})
    assert genome_hash(renamed) == genome_hash(a) and renamed != a  # same genes, other id
    with pytest.raises(TypeError):
        hash(a)


def test_from_dict_rejects_typos_and_sizes_missing_arrays() -> None:
    d = to_dict(reference_hexapod())
    bad = json.loads(json.dumps(d))
    bad["morphology"]["torso_lenght"] = 0.3
    with pytest.raises(ValueError, match="torso_lenght"):
        from_dict(bad)
    with pytest.raises(ValueError, match="colour"):
        from_dict(d | {"colour": "red"})
    old = json.loads(json.dumps(d))
    del old["controller"]["phase_offsets"], old["morphology"]["foot_length"]
    del old["controller"]["params"]
    g = from_dict(old)  # a file written before these genes existed
    assert g.controller.phase_offsets.shape == (6,) and validate_genome(g).ok
    assert g.morphology.foot_length == 0.0 and g.controller.params.size == 0


def test_params_vector_gene_round_trips_and_is_bounded() -> None:
    g = reference_quadruped()
    g.controller.params = np.array([0.5, -2.0, 99.0, np.nan])
    assert not validate_genome(g).ok
    r = repair(g)
    assert np.array_equal(r.controller.params, [0.5, -2.0, 10.0, 0.0])  # length kept
    assert validate_genome(r).ok
    assert np.array_equal(from_dict(to_dict(r)).controller.params, r.controller.params)


def test_registering_a_controller_makes_its_kind_a_valid_gene(monkeypatch) -> None:
    from evomorph.controllers import base

    monkeypatch.setitem(GENE_BOUNDS, "kind", GENE_BOUNDS["kind"])
    monkeypatch.setattr(base, "_REGISTRY", dict(base._REGISTRY))

    @base.register_controller("mlp")
    def _make(genome):  # pragma: no cover - only registration is under test
        raise NotImplementedError

    g = random_genome(np.random.default_rng(0), kind="mlp")
    g.controller.params = np.zeros(12)
    assert validate_genome(g).ok and repair(g).controller.kind == "mlp"
    assert random_genome(np.random.default_rng(0)).controller.kind == "cpg"  # not sampled
