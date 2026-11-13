"""The comparison experiment: reproducibility, frozen genes, equal budgets, resumption."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import pytest
from conftest import ALL_CONFIGS, configured_genome
from exp_helpers import (  # noqa: F401  (tiny_suite is an autouse fixture)
    run_tiny,
    tiny_design,
    tiny_experiment_config,
    tiny_sections,
    tiny_suite,
    write_config,
)

from evomorph.cli import main
from evomorph.experiments.baselines import (
    CONDITION_KEYS,
    baseline_controller_genome,
    get_condition,
)
from evomorph.experiments.comparison import (
    BudgetMismatchError,
    DesignMismatchError,
    check_run,
    run_comparison,
)
from evomorph.experiments.design import (
    experiment_from_mapping,
    load_experiment,
    run_config,
)
from evomorph.morphology.genome import from_dict, reference_quadruped, to_dict
from evomorph.utils.serialization import load_json

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """One tiny experiment: all four conditions, seed 0."""
    root = tmp_path_factory.mktemp("exp")
    return run_tiny(root, n_seeds=1)


def _all_genomes(run_dir: Path) -> list:
    out = []
    for f in sorted((run_dir / "generations").glob("gen_*.json")):
        out += [from_dict(r["genome"]) for r in load_json(f)]
    return out


def test_same_seed_same_table(tiny, tmp_path) -> None:
    again = run_tiny(tmp_path, n_seeds=1)
    assert again.table.read_bytes() == tiny.table.read_bytes()
    rows = list(csv.DictReader(tiny.table.open()))
    assert [(r["condition"], r["seed"]) for r in rows] == [(k, "0") for k in CONDITION_KEYS]
    other = run_tiny(tmp_path, name="other", n_seeds=1, seed_start=5, conditions=["controller"])
    (first,) = list(csv.DictReader(other.table.open()))
    assert first["champ_test_fitness"] != rows[0]["champ_test_fitness"]


def test_controller_condition_never_changes_the_body(tiny) -> None:
    ref = to_dict(reference_quadruped())["morphology"]
    genomes = _all_genomes(tiny.path / "runs" / "controller_s0")
    assert len(genomes) >= 8
    assert all(to_dict(g)["morphology"] == ref for g in genomes)
    # ... while the controller really evolves (the check is not vacuous)
    assert len({str(to_dict(g)["controller"]) for g in genomes}) > 1


def test_morphology_condition_never_changes_the_controller(tiny) -> None:
    genomes = _all_genomes(tiny.path / "runs" / "morphology_s0")
    for g in genomes:
        expected = baseline_controller_genome(g)
        assert to_dict(g)["controller"] == to_dict(expected)["controller"]
        c = g.controller
        assert (c.gait, c.frequency, c.hip_amplitude, c.knee_amplitude) == (
            "diagonal",
            1.5,
            0.45,
            0.40,
        )
        assert not any(c.phase_offsets) and not any(c.hip_bias) and not any(c.knee_bias)
    assert len({str(to_dict(g)["morphology"]) for g in genomes}) > 1


def test_baseline_controller_adapts_to_every_limb_count() -> None:
    ref = reference_quadruped().controller
    for pairs, segments in ALL_CONFIGS:
        g = baseline_controller_genome(configured_genome(pairs, segments))
        c = g.controller
        assert len(c.phase_offsets) == len(c.hip_bias) == len(c.knee_bias) == 2 * pairs
        assert not any(c.phase_offsets) and not any(c.hip_bias) and not any(c.knee_bias)
        assert (c.gait, c.frequency, c.hip_amplitude) == (ref.gait, ref.frequency, 0.45)
        assert c.knee_amplitude == ref.knee_amplitude
        assert c.knee_phase_offset == ref.knee_phase_offset
        assert g.morphology.n_pairs == pairs and g.morphology.n_segments == segments


def test_every_run_spent_exactly_the_same_budget(tiny) -> None:
    rows = list(csv.DictReader(tiny.table.open()))
    assert {int(r["sims"]) for r in rows} == {12}
    manifest = load_json(tiny.path / "manifest.json")
    assert manifest["budget"] == 12 and manifest["sims_per_genome"] == 1


def test_budget_is_rounded_down_to_whole_genomes_for_everyone() -> None:
    exp = experiment_from_mapping({"budget": 4405})  # 11 simulations per genome
    assert exp.sims_per_genome == 11 and exp.effective_budget == 4400
    assert experiment_from_mapping({}).effective_budget == 1441  # the default is whole genomes


def test_unequal_budget_or_config_is_refused(tiny, tmp_path) -> None:
    exp, base = tiny_design(tmp_path, n_seeds=2)
    src = tiny.path / "runs" / "joint_s0"
    cfg = run_config(exp, base, get_condition("joint"), 0)
    assert check_run(src, cfg) == 12
    tampered = tmp_path / "joint_s0"
    shutil.copytree(src, tampered)
    lines = (tampered / "metrics.csv").read_text().splitlines()
    lines[-1] = lines[-1].replace(",12,", ",11,", 1)
    (tampered / "metrics.csv").write_text("\n".join(lines) + "\n")
    with pytest.raises(BudgetMismatchError, match="spent 11"):
        check_run(tampered, cfg)
    other = run_config(exp, base, get_condition("joint"), 1)  # different seed = different config
    with pytest.raises(BudgetMismatchError, match="different configuration"):
        check_run(src, other)


def test_max_generations_cap_is_refused(tmp_path) -> None:
    sections = tiny_sections()
    sections["evolution"]["max_generations"] = 2
    with pytest.raises(ValueError, match="max_generations must be null"):
        load_experiment(write_config(tmp_path / "c.yaml", sections))


def test_rerun_skips_finished_runs_and_redoes_interrupted_ones(tiny, tmp_path) -> None:
    copy = tmp_path / "exp"
    shutil.copytree(tiny.path, copy)
    original = (copy / "results.csv").read_bytes()
    exp, base = tiny_design(tmp_path, n_seeds=1)
    lines: list[str] = []
    again = run_comparison(exp, base, tmp_path, "exp", out=lines.append)
    assert again.ran == [] and len(again.skipped) == 4
    # Evaluation lost: only the post-hoc evaluation is redone, not the search.
    (copy / "runs" / "joint_s0" / "posthoc.json").unlink()
    gen_file = copy / "runs" / "joint_s0" / "generations" / "gen_0000.json"
    stamp = gen_file.stat().st_mtime_ns
    # Search interrupted (no summary.md, so no evaluation either): moved aside and redone.
    (copy / "runs" / "controller_s0" / "posthoc.json").unlink()
    (copy / "runs" / "controller_s0" / "summary.md").unlink()
    lines.clear()
    again = run_comparison(exp, base, tmp_path, "exp", out=lines.append)
    assert sorted(again.ran) == ["controller_s0", "joint_s0"]
    assert gen_file.stat().st_mtime_ns == stamp
    assert (copy / "incomplete" / "controller_s0-1").is_dir()
    assert (copy / "results.csv").read_bytes() == original


def test_a_different_design_is_refused_in_the_same_folder(tiny, tmp_path) -> None:
    shutil.copytree(tiny.path, tmp_path / "exp")
    exp, base = tiny_design(tmp_path, budget=16)
    with pytest.raises(DesignMismatchError, match="different settings"):
        run_comparison(exp, base, tmp_path, "exp", out=lambda line: None)
    more_seeds, base = tiny_design(tmp_path, n_seeds=3)  # more seeds: same design, allowed
    assert tiny_experiment_config(n_seeds=3).seeds == (0, 1, 2)
    assert more_seeds.effective_budget == 12


def test_unknown_experiment_keys_and_bad_suites_are_plain_errors() -> None:
    with pytest.raises(ValueError, match="budgte"):
        experiment_from_mapping({"budgte": 3})
    with pytest.raises(ValueError, match="no terrain suite"):
        experiment_from_mapping({"test_suite": "moon"})
    with pytest.raises(ValueError, match="no condition"):
        experiment_from_mapping({"conditions": ["joint", "brain"]})


def test_cli_quick_experiment_and_errors(tmp_path, capsys) -> None:
    cfg = write_config(tmp_path / "tiny.yaml", tiny_sections(conditions=["controller"]))
    args = ["experiment", "--type", "comparison", "--config", str(cfg)]
    args += ["--results", str(tmp_path / "res"), "--quick", "--workers", "9"]
    assert main(args) == 0
    text = capsys.readouterr().out
    assert "using 6 workers" in text and "8 simulations per run" in text
    assert (tmp_path / "res" / "comparison_quick" / "results.csv").is_file()
    assert main(args) == 0 and "already finished" in capsys.readouterr().out
    assert main(["experiment", "--type", "comparison", "--config", str(tmp_path / "no")]) == 2
    bad = write_config(tmp_path / "bad.yaml", tiny_sections(budget=0))
    assert main(["experiment", "--type", "comparison", "--config", str(bad)]) == 2
    assert "not usable" in capsys.readouterr().err


def test_shipped_config_resolves_with_capped_workers() -> None:
    for quick in (False, True):
        exp, base = load_experiment(ROOT / "configs" / "comparison.yaml", quick)
        assert exp.workers <= 6 and base.evolution.workers <= 6
        assert exp.train_suite == "train" and exp.test_suite == "test"
        assert exp.effective_budget == exp.budget  # a whole number of genomes
