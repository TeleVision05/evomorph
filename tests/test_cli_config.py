"""CLI subcommands and YAML config loading."""

from __future__ import annotations

import json
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


def test_inspect_reports_an_invalid_genome_without_a_traceback(tmp_path, capsys) -> None:
    ref = tmp_path / "ref.json"
    assert main(["inspect", "--reference", "hexapod", "--save", str(ref)]) == 0
    data = json.loads(ref.read_text())
    data["controller"]["gait"] = "bogus"
    bad = tmp_path / "bad_gait.json"
    bad.write_text(json.dumps(data))
    capsys.readouterr()
    assert main(["inspect", "--genome", str(bad)]) == 1
    out = capsys.readouterr().out
    assert "VALIDATION: invalid" in out and "gait='bogus'" in out
    assert main(["inspect", "--genome", str(tmp_path / "missing.json")]) == 2
    assert "no genome file" in capsys.readouterr().err
    (tmp_path / "junk.json").write_text("{not json")
    assert main(["evaluate", "--genome", str(tmp_path / "junk.json")]) == 2
    assert "could not read genome" in capsys.readouterr().err


def test_evaluate_does_not_claim_a_video_it_did_not_write(tmp_path, capsys) -> None:
    video = tmp_path / "slope.mp4"
    # Slope terrain exists now; a 60 degree slope is still out of range and cannot be built.
    argv = ["evaluate", "--reference", "quadruped", "--env", "slope", "--slope", "60"]
    assert main([*argv, "--video", str(video)]) == 1
    out = capsys.readouterr().out
    assert "wrote" not in out and "no video written" in out and not video.exists()


def test_sim_config_rejects_drifting_control_period_and_zero_duration() -> None:
    with pytest.raises(ValueError, match="whole number of timesteps"):
        SimConfig(timestep=0.003, control_rate=50.0)
    with pytest.raises(ValueError, match="duration"):
        SimConfig(duration=0.0)
    assert SimConfig(timestep=0.001).n_substeps == 20


@pytest.mark.gl
def test_inspect_png(tmp_path) -> None:
    png = tmp_path / "q.png"
    assert main(["inspect", "--reference", "quadruped", "--png", str(png)]) == 0
    assert png.stat().st_size > 10_000
