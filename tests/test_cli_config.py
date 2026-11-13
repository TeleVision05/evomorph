"""CLI subcommands and YAML config loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from evomorph.cli import build_parser, main
from evomorph.config import (
    FitnessConfig,
    SimConfig,
    load_fitness_config,
    load_section,
    load_sim_config,
)
from evomorph.morphology.genome import load_genome

ROOT = Path(__file__).resolve().parents[1]


def test_quick_config_loads_and_matches_defaults() -> None:
    path = ROOT / "configs" / "quick.yaml"
    assert load_sim_config(path) == SimConfig()
    assert load_fitness_config(path) == FitnessConfig()


def test_unknown_config_key_is_an_error(tmp_path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("sim:\n  duraton: 3\n")
    with pytest.raises(ValueError, match="duraton"):
        load_sim_config(p)
    assert load_section(p, "fitness", FitnessConfig) == FitnessConfig()


def test_commands_are_discovered() -> None:
    text = build_parser().format_help()
    for cmd in ("inspect", "evaluate", "random"):
        assert cmd in text


def test_random_then_inspect_then_evaluate(tmp_path, capsys) -> None:
    out = tmp_path / "genomes"
    assert main(["random", "--n", "3", "--seed", "9", "--out", str(out)]) == 0
    files = sorted(out.glob("*.json"))
    assert len(files) == 3
    assert main(["random", "--n", "3", "--seed", "9", "--out", str(out)]) == 1  # no overwrite
    assert load_genome(files[0]).id == "rnd9-0000"
    assert main(["inspect", "--genome", str(files[0])]) == 0
    assert "VALIDATION: ok" in capsys.readouterr().out
    cfg = tmp_path / "short.yaml"
    cfg.write_text("sim:\n  duration: 1.0\n")
    js = tmp_path / "res.json"
    code = main(["evaluate", "--reference", "hexapod", "--config", str(cfg), "--json", str(js)])
    text = capsys.readouterr().out
    assert code == 0 and "forward_displacement" in text and js.exists()


def test_inspect_png(tmp_path) -> None:
    png = tmp_path / "q.png"
    assert main(["inspect", "--reference", "quadruped", "--png", str(png)]) == 0
    assert png.stat().st_size > 10_000
