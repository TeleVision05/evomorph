"""Robustness evaluation: table columns, aggregates, determinism, pool use and the CLI."""

from __future__ import annotations

import csv
import json
import math
import pickle

import pytest

from evomorph.cli import main
from evomorph.config import SimConfig
from evomorph.experiments.robustness import (
    AGGREGATE_KEYS,
    MAX_WORKERS,
    ROW_COLUMNS,
    evaluate_suite,
    table_lines,
)
from evomorph.morphology.genome import reference_quadruped
from evomorph.simulation.environment import EnvSpec
from evomorph.simulation.evaluator import INVALID_FITNESS, evaluate

SHORT = SimConfig(duration=1.5, settle_time=0.3)
ENVS = (EnvSpec(kind="uneven", roughness=0.03, n_obstacles=10, seed=1), EnvSpec(kind="slope"))


def test_report_has_the_stated_columns_and_aggregates() -> None:
    rep = evaluate_suite(reference_quadruped(), ENVS, SHORT, seeds=(0, 1))
    assert rep.suite == "custom" and rep.seeds == (0, 1) and len(rep.rows) == 2
    for row in rep.rows:
        d = row.as_dict()
        assert tuple(d) == ROW_COLUMNS and d["runs"] == 2
        assert all(math.isfinite(d[k]) for k in ("fitness", "forward_velocity", "slip_fraction"))
    assert set(rep.aggregates) == set(AGGREGATE_KEYS)
    assert rep.aggregates["worst_fitness"] == min(r.mean_fitness for r in rep.rows)
    lines = table_lines(rep)
    assert "fitness" in lines[1] and "slip" in lines[1] and "fell" in lines[1]
    assert any("failure rate" in line for line in lines)


def test_rows_match_plain_evaluate_and_are_deterministic() -> None:
    g = reference_quadruped()
    a = evaluate_suite(g, ENVS[:1], SHORT, seeds=(3,))
    b = evaluate_suite(g, ENVS[:1], SHORT, seeds=(3,))
    assert a == b
    direct = evaluate(g, ENVS[0], SHORT, seed=3)
    assert a.rows[0].fitness == (direct.fitness,)


def test_a_process_pool_gives_the_same_report() -> None:
    g = reference_quadruped()
    serial = evaluate_suite(g, ENVS, SHORT, seeds=(0,))
    pooled = evaluate_suite(g, ENVS, SHORT, seeds=(0,), workers=2)
    assert serial == pooled
    assert MAX_WORKERS == 6
    pickle.loads(pickle.dumps((evaluate_suite, g, ENVS, SHORT, serial)))


def test_terrain_seed_changes_results_and_spawn_seed_is_repeatable() -> None:
    g = reference_quadruped()
    fwd = {
        s: evaluate(
            g, EnvSpec(kind="uneven", roughness=0.05, n_obstacles=20, seed=s), SHORT
        ).metrics["forward_displacement"]
        for s in (0, 1)
    }
    again = evaluate(g, EnvSpec(kind="uneven", roughness=0.05, n_obstacles=20, seed=0), SHORT)
    assert again.metrics["forward_displacement"] == fwd[0]
    assert fwd[0] != fwd[1]


def test_invalid_runs_count_as_failures_at_the_invalid_score() -> None:
    bad = EnvSpec(kind="slope", slope_deg=60.0)  # out of range: cannot be built
    rep = evaluate_suite(reference_quadruped(), (EnvSpec(), bad), SHORT, seeds=(0, 1))
    row = rep.rows[1].as_dict()
    assert row["invalid"] == 2 and row["fitness"] == INVALID_FITNESS
    assert math.isnan(row["forward_velocity"])
    assert rep.aggregates["invalid_rate"] == pytest.approx(0.5)
    assert rep.aggregates["failure_rate"] >= 0.5
    assert rep.aggregates["worst_fitness"] == INVALID_FITNESS
    with pytest.raises(ValueError):
        evaluate_suite(reference_quadruped(), (), SHORT)


def test_cli_prints_saves_and_never_overwrites(tmp_path, capsys) -> None:
    cfg = tmp_path / "short.yaml"
    cfg.write_text("sim:\n  duration: 1.0\n  settle_time: 0.2\n")
    argv = ["robustness", "--reference", "quadruped", "--suite", "flat", "--seeds", "2",
            "--config", str(cfg), "--out", str(tmp_path)]  # fmt: skip
    assert main(argv) == 0
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "mean fitness" in out and "flat" in out
    first, second = tmp_path / "ref-quadruped_flat.csv", tmp_path / "ref-quadruped_flat-2.csv"
    assert first.exists() and second.exists()
    with open(first) as fh:
        rows = list(csv.DictReader(fh))
    assert tuple(rows[0]) == ROW_COLUMNS and rows[0]["runs"] == "2"
    record = json.loads(first.with_suffix(".json").read_text())
    assert record["seeds"] == [0, 1] and record["sim"]["duration"] == 1.0
    assert set(record["aggregates"]) == set(AGGREGATE_KEYS)


def test_cli_lists_suites_and_rejects_unknown_ones(capsys) -> None:
    assert main(["robustness", "--list"]) == 0
    listing = capsys.readouterr().out
    assert "### train" in listing and "### test" in listing and "| slope+15 |" in listing
    assert main(["robustness", "--reference", "quadruped", "--suite", "mars"]) == 2
    assert "no terrain suite called 'mars'" in capsys.readouterr().err
