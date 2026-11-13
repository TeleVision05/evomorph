"""``analyze`` on a fixture experiment folder and a fixture run directory."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from exp_helpers import run_tiny, tiny_suite  # noqa: F401  (tiny_suite is an autouse fixture)

from evomorph.analysis.analyze import NothingToAnalyseError, analyze
from evomorph.analysis.load import load_experiment_dir, plan_history
from evomorph.analysis.stats import MEASURES
from evomorph.cli import main
from evomorph.evolution.archive import Archive
from evomorph.evolution.selection import body_plan

PNG = b"\x89PNG"


@pytest.fixture(scope="module")
def experiment(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("analysis")
    return run_tiny(root, n_seeds=2).path


def test_experiment_analysis_writes_every_plot_stats_and_summary(experiment) -> None:
    summary, plots = analyze(experiment, render=False)
    names = {p.name for p in plots}
    assert names == {
        "fitness_vs_simulations.png",
        "train_vs_test.png",
        "locomotion.png",
        "diversity.png",
        "body_plans.png",
        "archive_heatmap.png",
    }
    for p in plots:
        assert p.parent == experiment / "plots" and p.read_bytes()[:4] == PNG
    text = summary.read_text()
    for heading in ("How strong is the evidence?", "Verdict", "Key comparisons", "Throughput"):
        assert heading in text
    assert "WEAK" in text and "NOT SHOWN" in text  # 2 seeds can never show anything
    with (experiment / "stats.csv").open() as fh:
        stats = list(csv.DictReader(fh))
    assert len(stats) == 3 * len(MEASURES)


def test_gallery_renders_offscreen(experiment) -> None:
    _, plots = analyze(experiment, render=True)
    gallery = next(p for p in plots if p.name == "gallery.png")
    assert gallery.read_bytes()[:4] == PNG and gallery.stat().st_size > 20_000


def test_map_elites_plan_history_replays_to_the_saved_archive(experiment) -> None:
    run = experiment / "runs" / "joint_me_s0"
    _, counts = plan_history(run)
    archive = Archive.load(run / "archive")
    expected: dict[str, int] = {}
    for e in archive.elites():
        expected[body_plan(e.genome)] = expected.get(body_plan(e.genome), 0) + 1
    assert counts[-1] == expected


def test_ga_plan_history_counts_the_whole_population(experiment) -> None:
    sims, counts = plan_history(experiment / "runs" / "joint_s0")
    assert all(sum(c.values()) == 4 for c in counts) and sims[-1] == 12


def test_single_run_analysis_keeps_the_engine_summary(experiment) -> None:
    run = experiment / "runs" / "controller_s1"
    before = (run / "summary.md").read_text()
    summary, plots = analyze(run, render=True)
    assert {p.name for p in plots} >= {"fitness_vs_simulations.png", "body_plans.png"}
    assert {"archive_heatmap.png", "gallery.png", "diversity.png"} <= {p.name for p in plots}
    text = summary.read_text()
    assert text.startswith(before.rstrip("\n")) and text.count("## Analysis plots") == 1
    analyze(run, render=False)  # again: the section is replaced, not added twice
    assert summary.read_text().count("## Analysis plots") == 1


def test_loaded_experiment_matches_the_table(experiment) -> None:
    data = load_experiment_dir(experiment)
    assert len(data.rows) == 8 and [s for s, _ in data.runs_of("joint")] == [0, 1]
    assert data.label("joint_me") == "D joint (MAP-Elites)"


def test_cli_analyze_and_its_errors(experiment, tmp_path, capsys) -> None:
    assert main(["analyze", "--run", str(experiment), "--no-render"]) == 0
    assert "wrote 6 plots" in capsys.readouterr().out
    assert main(["analyze", "--run", str(tmp_path / "missing")]) == 2
    assert main(["analyze", "--run", str(tmp_path)]) == 2
    assert "neither an experiment" in capsys.readouterr().err
    with pytest.raises(NothingToAnalyseError):
        analyze(tmp_path)
