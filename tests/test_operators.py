"""Mutation and crossover: validity, bounds, frozen groups, limb-count resizing."""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest

from evomorph.evolution.modes import Freezer, load_template, mutable_genes
from evomorph.evolution.operators import (
    MutationConfig,
    Variation,
    crossover,
    mutate,
    reflect,
    resize_limbs,
)
from evomorph.morphology.genome import (
    CONTROLLER_GENES,
    GENE_BOUNDS,
    MORPHOLOGY_GENES,
    PER_LIMB_GENES,
    random_genome,
    reference_hexapod,
    reference_quadruped,
    repair,
)
from evomorph.morphology.validator import validate_genome

# Exaggerated rates so a few hundred children exercise every code path.
WILD = MutationConfig(
    gene_rate=0.6, sigma=0.5, reset_rate=0.2, p_limb_pairs=0.5, p_segments=0.5, p_gait=0.5
)


def _variation(mode: str, cfg: MutationConfig = WILD) -> Variation:
    template = None if mode == "joint" else reference_quadruped()
    return Variation(cfg, Freezer(mode, template), crossover_prob=0.5)


def _population(var: Variation, rng: np.random.Generator, n: int = 12):
    return [var.random(rng, f"p{i}", 0) for i in range(n)]


def _children(mode: str, n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    var = _variation(mode)
    pop = _population(var, rng)
    kids = []
    for k in range(n):
        a, b = pop[int(rng.integers(len(pop)))], pop[int(rng.integers(len(pop)))]
        kids.append(var.child(rng, a, b, f"c{k}", 3))
        if k % 10 == 0:  # breed from children too, so drift compounds
            pop[int(rng.integers(len(pop)))] = kids[-1]
    return var, kids


def _gene(g, name):
    group = g.morphology if GENE_BOUNDS[name].group == "morphology" else g.controller
    return getattr(group, name)


@pytest.mark.parametrize("mode", ["joint", "controller", "morphology"])
def test_every_child_is_valid_and_in_bounds(mode: str) -> None:
    _, kids = _children(mode)
    for g in kids:
        assert validate_genome(g).ok, validate_genome(g).errors
        for name in PER_LIMB_GENES:
            assert len(_gene(g, name)) == g.n_limbs


def test_joint_mode_changes_both_groups_and_every_discrete_gene() -> None:
    _, kids = _children("joint")
    ref = _children("joint")[1][0]
    for names in (MORPHOLOGY_GENES, CONTROLLER_GENES):
        assert any(
            not np.array_equal(np.asarray(_gene(k, n)), np.asarray(_gene(ref, n)))
            for k in kids
            for n in names
            if n != "kind"
        )
    assert {k.n_limbs for k in kids} == {2, 4, 6, 8}
    assert {k.morphology.n_segments for k in kids} == {1, 2}
    assert len({k.controller.gait for k in kids}) == 4


def test_controller_mode_never_touches_the_body() -> None:
    _, kids = _children("controller")
    body = reference_quadruped().morphology
    assert all(k.morphology == body for k in kids)
    # ... while the controller genes do evolve
    assert len({round(k.controller.frequency, 9) for k in kids}) > 50


def test_morphology_mode_keeps_the_baseline_controller_adapted_to_limb_count() -> None:
    _, kids = _children("morphology")
    base = reference_quadruped()
    assert {k.n_limbs for k in kids} == {2, 4, 6, 8}  # bodies do evolve
    for k in kids:
        expected = repair(replace(k, controller=copy.deepcopy(base.controller))).controller
        assert k.controller == expected
        for name in ("gait", "frequency", "hip_amplitude", "knee_amplitude", "knee_phase_offset"):
            assert getattr(k.controller, name) == getattr(base.controller, name)


def test_kind_is_never_mutated() -> None:
    _, kids = _children("joint")
    assert {k.controller.kind for k in kids} == {"cpg"}
    assert "kind" not in mutable_genes("joint")


def test_operators_respect_the_gene_list_directly() -> None:
    """mutate/crossover themselves (before the freezer) only change listed genes."""
    rng = np.random.default_rng(5)
    a, b = random_genome(rng, "a"), random_genome(rng, "b")
    b = repair(replace(b, morphology=copy.deepcopy(a.morphology)))
    genes = tuple(n for n in CONTROLLER_GENES if n not in ("kind", *PER_LIMB_GENES))
    for _ in range(50):
        m = mutate(a, rng, WILD, genes)
        x = crossover(a, b, rng, genes)
        assert m.morphology == a.morphology and x.morphology == a.morphology
        for name in PER_LIMB_GENES:
            assert np.array_equal(_gene(m, name), _gene(a, name))


def test_limb_pair_mutation_resizes_per_limb_genes_by_copying_the_rear_pair() -> None:
    rng = np.random.default_rng(1)
    g = reference_quadruped()
    g.controller.phase_offsets = np.array([0.1, 0.2, 0.3, 0.4])
    cfg = MutationConfig(
        gene_rate=0, reset_rate=0, p_limb_pairs=1, p_segments=0, p_gait=0, p_phase=0
    )
    sizes = set()
    for _ in range(40):
        child = mutate(g, rng, cfg, mutable_genes("joint"))
        sizes.add(child.n_limbs)
        ph = child.controller.phase_offsets
        assert len(ph) == child.n_limbs
        if child.n_limbs == 6:
            assert np.allclose(ph, [0.1, 0.2, 0.3, 0.4, 0.3, 0.4])
        else:
            assert np.allclose(ph, [0.1, 0.2])
    assert sizes == {2, 6}
    assert np.allclose(resize_limbs(np.array([1.0, 2.0]), 8), [1, 2, 1, 2, 1, 2, 1, 2])


def test_crossover_with_different_limb_counts_is_valid() -> None:
    rng = np.random.default_rng(2)
    var = _variation("joint")
    a, b = reference_quadruped(), reference_hexapod()
    plans = set()
    for k in range(60):
        child = var.child(rng, a, b, f"x{k}", 1)
        assert validate_genome(child).ok
        plans.add(child.n_limbs)
    assert {4, 6} <= plans


def test_children_record_parents_generation_and_origin() -> None:
    rng = np.random.default_rng(3)
    var = Variation(MutationConfig(), Freezer("joint"), crossover_prob=1.0)
    a, b = reference_quadruped(), reference_hexapod()
    child = var.child(rng, a, b, "kid", 7)
    assert child.parent_ids == ("ref-quadruped", "ref-hexapod")
    assert (child.id, child.generation, child.origin) == ("kid", 7, "crossover")
    solo = Variation(MutationConfig(), Freezer("joint"), crossover_prob=0.0)
    m = solo.child(rng, a, b, "kid2", 8)
    assert m.parent_ids == ("ref-quadruped",) and m.origin == "mutation"
    r = solo.random(rng, "r0", 0)
    assert r.parent_ids == () and r.origin == "random"


def test_variation_is_deterministic_in_the_seed() -> None:
    a = _children("joint", n=40, seed=11)[1]
    b = _children("joint", n=40, seed=11)[1]
    assert [x.to_dict() for x in a] == [y.to_dict() for y in b]


def test_reflect_folds_back_into_bounds() -> None:
    assert reflect(1.2, 0.0, 1.0) == pytest.approx(0.8)
    assert reflect(-0.3, 0.0, 1.0) == pytest.approx(0.3)
    assert reflect(2.5, 0.0, 1.0) == pytest.approx(0.5)
    arr = reflect(np.array([-5.0, 0.5, 7.3]), -1.0, 1.0)
    assert np.all((arr >= -1) & (arr <= 1))


def test_zero_rates_leave_genes_alone() -> None:
    rng = np.random.default_rng(4)
    g = reference_hexapod()
    off = MutationConfig(0, 0.1, 0, 0, 0, 0, 0, 0.25)
    assert mutate(g, rng, off, mutable_genes("joint")).to_dict() == g.to_dict()


def test_bad_mode_and_missing_template_are_plain_errors() -> None:
    with pytest.raises(ValueError, match="mode"):
        mutable_genes("everything")
    with pytest.raises(ValueError, match="template"):
        Freezer("controller", None)
    with pytest.raises(ValueError, match="neither"):
        load_template("no-such-creature")
    with pytest.raises(ValueError, match="probability"):
        MutationConfig(gene_rate=1.5)
