"""Run directories, config round trip, exact reproduction, worker-count determinism, CLI."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from evomorph.cli import main
from evomorph.config import load_yaml
from evomorph.evolution.archive import Archive
from evomorph.evolution.evalspec import SpecEvaluator
from evomorph.evolution.progress import METRICS_COLUMNS
from evomorph.evolution.rundir import RunDirExistsError, create_run_dir
from evomorph.evolution.runner import heldout_spec, run_evolution
from evomorph.evolution.settings import from_sections, resolve, to_mapping
from evomorph.morphology.genome import load_genome
from evomorph.utils.serialization import load_json

ROOT = Path(__file__).resolve().parents[1]

TINY = {
    "sim": {"duration": 1.0, "settle_time": 0.2},
    "evolution": {
        "budget": 36,
        "population_size": 6,
        "workers": 1,
        "evaluation": {"runs": [{"env": {"friction": 0.7}, "seed": 0}, {"env": {}, "seed": 1}]},
    },
    "map_elites": {"batch_size": 6, "init_size": 6},
}


def _tiny(**evolution) -> dict:
    sections = {k: dict(v) for k, v in TINY.items()}
    sections["evolution"] |= evolution
    return sections


def test_run_dirs_are_fresh_and_never_overwritten(tmp_path) -> None:
    assert create_run_dir(tmp_path).name == "run_001"
    assert create_run_dir(tmp_path).name == "run_002"
    (tmp_path / "run_010").mkdir()
    assert create_run_dir(tmp_path).name == "run_011"
    assert create_run_dir(tmp_path, "mine").name == "mine"
    with pytest.raises(RunDirExistsError, match="never overwritten"):
        create_run_dir(tmp_path, "mine")
    with pytest.raises(ValueError, match="letters"):
        create_run_dir(tmp_path, "../escape")


def test_quick_and_research_configs_resolve_and_round_trip() -> None:
    for name in ("quick", "research"):
        cfg = resolve(ROOT / "configs" / f"{name}.yaml")
        assert from_sections(to_mapping(cfg)) == cfg
        assert cfg.evolution.workers <= 6
        assert len({s for _, s in cfg.eval_spec.runs}) >= 2  # multi-seed scoring
        assert len({e.friction for e, _ in cfg.eval_spec.runs}) >= 2  # friction spread


def test_heldout_spec_uses_fresh_seeds_on_the_same_environments() -> None:
    spec = resolve(ROOT / "configs" / "quick.yaml").eval_spec
    held = heldout_spec(spec)
    assert [e for e, _ in held.runs] == [e for e, _ in spec.runs]
    assert [s for _, s in held.runs] == [s + 1000 for _, s in spec.runs]
    assert held.sim == spec.sim and held.fitness == spec.fitness


def test_unknown_evolution_keys_are_errors() -> None:
    with pytest.raises(ValueError, match="populaton_size"):
        from_sections({"evolution": {"populaton_size": 3}})
    with pytest.raises(ValueError, match="bins_x"):
        from_sections({"map_elites": {"bins_x": 3}})


@pytest.mark.parametrize("algorithm", ["ga", "map_elites"])
def test_run_directory_contents_and_exact_reproduction(tmp_path, algorithm: str) -> None:
    cfg = from_sections(_tiny(algorithm=algorithm))
    out = run_evolution(cfg, tmp_path, "r", out=None)
    p = out.path
    expected = ["config.yaml", "metrics.csv", "timing.csv", "best_genome.json"]
    expected += ["best_result.json", "best_heldout.json", "summary.md", "archive/archive.json"]
    for name in expected:
        assert (p / name).is_file(), name
    assert out.sims_used == 36 and out.reproduced is True
    rows = list(csv.DictReader((p / "metrics.csv").open()))
    assert tuple(rows[0]) == METRICS_COLUMNS and len(rows) == out.steps
    assert sorted(x.name for x in (p / "generations").iterdir())[0] == "gen_0000.json"
    gen0 = load_json(p / "generations" / "gen_0000.json")
    assert {"id", "parent_ids", "fitness", "metrics", "seed_fitness", "genome"} <= set(gen0[0])
    assert len(list((p / "genomes").glob("champion_gen_*.json"))) == out.steps
    assert any((p / "genomes" / "final").iterdir())
    # Re-score the saved champion from the saved config alone: bit-identical.
    saved = from_sections(load_yaml(p / "config.yaml"))
    assert saved == cfg
    best = load_json(p / "best_result.json")
    again = SpecEvaluator(saved.eval_spec)(load_genome(p / "best_genome.json"))
    assert again.fitness == best["fitness"] and again.metrics == best["metrics"]
    assert list(again.seed_fitness) == best["seed_fitness"]
    assert len(Archive.load(p / "archive")) == len(out.archive)


def test_same_seed_same_metrics_with_one_or_six_workers(tmp_path) -> None:
    one = run_evolution(from_sections(_tiny(workers=1)), tmp_path, "w1", out=None)
    six = run_evolution(from_sections(_tiny(workers=6)), tmp_path, "w6", out=None)
    assert (one.path / "metrics.csv").read_bytes() == (six.path / "metrics.csv").read_bytes()
    for f in sorted((one.path / "generations").iterdir()):
        assert f.read_bytes() == (six.path / "generations" / f.name).read_bytes()
    other = run_evolution(from_sections(_tiny(seed=1)), tmp_path, "s1", out=None)
    assert (other.path / "metrics.csv").read_bytes() != (one.path / "metrics.csv").read_bytes()


def test_cli_evolve_runs_and_refuses_to_overwrite(tmp_path, capsys) -> None:
    import yaml

    cfg = tmp_path / "tiny.yaml"
    cfg.write_text(yaml.safe_dump(_tiny()))
    args = ["evolve", "--config", str(cfg), "--results", str(tmp_path / "res"), "--name", "a"]
    assert main([*args, "--budget", "24", "--mode", "controller", "--no-diversity"]) == 0
    text = capsys.readouterr().out
    assert "gen    0" in text and "done: 24 simulations" in text
    saved = load_yaml(tmp_path / "res" / "a" / "config.yaml")
    assert saved["evolution"]["mode"] == "controller"
    assert saved["evolution"]["diversity"]["method"] == "none"
    assert (tmp_path / "res" / "a" / "frozen_template.json").is_file()
    assert main(args) == 2
    assert "never overwritten" in capsys.readouterr().err
    assert main(["evolve", "--config", str(tmp_path / "missing.yaml")]) == 2
    bad = _tiny(mode="controller", frozen_body=str(tmp_path / "nope.json"))
    cfg.write_text(yaml.safe_dump(bad))
    assert main(["evolve", "--config", str(cfg), "--results", str(tmp_path / "r2")]) == 2
    assert "neither a built-in reference" in capsys.readouterr().err
    assert not any((tmp_path / "r2").glob("run_*"))  # no half-made run directory
