"""Dashboard data layer: discovery, loading, archive replay, lineage, media keys."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace

import pytest
from dashboard_helpers import FIXTURE_RUNS, RUN_NAMES, copy_runs, make_in_progress

from evomorph.simulation.environment import EnvSpec
from evomorph.visualization import dashboard_analysis as an
from evomorph.visualization import dashboard_data as dd
from evomorph.visualization import dashboard_describe as desc
from evomorph.visualization import dashboard_media as media


def _load(name: str, root=FIXTURE_RUNS) -> dd.Run:
    entry = next(e for e in dd.discover_runs(root) if e.name == name)
    return dd.load_run(entry)


def test_discovers_runs_and_experiment_folders(tmp_path):
    copy_runs(tmp_path / "exp" / "cond_a", ("ga_tiny",))
    copy_runs(tmp_path, ("me_tiny",))
    (tmp_path / ".dashboard_cache" / "thumb").mkdir(parents=True)
    names = [e.name for e in dd.discover_runs(tmp_path)]
    assert names == ["exp/cond_a/ga_tiny", "me_tiny"]
    assert dd.discover_runs(tmp_path / "missing") == []


@pytest.mark.parametrize("name", RUN_NAMES)
def test_fixture_runs_load_complete(name):
    run = _load(name)
    assert run.status == "complete"
    assert not run.notes
    assert run.metrics["sims"].iloc[-1] == run.budget == 64
    assert len(run.eval_runs) == 2  # every stored fitness is a two-run score


@pytest.mark.parametrize("name", RUN_NAMES)
def test_rebuilt_archive_matches_saved_archive(name):
    run = _load(name)
    idx = dd.build_index(run)
    saved = an.load_archive(run, idx)
    rebuilt = an.rebuild_archive(run, idx)
    assert saved.source == "read from archive/archive.json"
    assert {k: (c.genome_id, c.fitness) for k, c in saved.cells.items()} == {
        k: (c.genome_id, c.fitness) for k, c in rebuilt.cells.items()
    }
    assert saved.cells  # the fixture archive is not empty


def test_champion_history_and_lineage():
    run = _load("ga_tiny")
    idx = dd.build_index(run)
    hist = dd.champion_history(run)
    assert bool(hist["changed"].iloc[0])
    final = str(hist.iloc[-1]["champion_id"])
    assert final == json.loads((run.path / "best_genome.json").read_text())["id"]
    rows = dd.lineage(idx, final)
    assert rows[0]["id"] == final and rows[0]["depth"] == 0
    assert all(r["recorded"] for r in rows)  # GA parents are all in the generation files
    assert min(r["generation"] for r in rows) == 0  # traced back to the random start


def test_in_progress_run_is_readable(tmp_path):
    copy_runs(tmp_path, ("me_tiny",))
    make_in_progress(tmp_path / "me_tiny", keep_steps=2)
    run = _load("me_tiny", tmp_path)
    assert run.status == "in progress"
    assert any("could not be read" in n for n in run.notes)  # the half-written csv line
    idx = dd.build_index(run)
    assert idx.skipped == [2]  # the half-written newest generation file
    view = an.load_archive(run, idx)
    assert view.source.startswith("rebuilt")


def test_damaged_files_raise_friendly_problems(tmp_path):
    copy_runs(tmp_path, ("ga_tiny",))
    run_dir = tmp_path / "ga_tiny"
    (run_dir / "generations" / "gen_0001.json").write_text("{not json")
    run = _load("ga_tiny", tmp_path)
    with pytest.raises(dd.RunReadError, match=r"generation 1 .* damaged"):
        dd.build_index(run)
    (run_dir / "config.yaml").write_text("sim: [unclosed")
    with pytest.raises(dd.RunReadError, match=r"config\.yaml"):
        _load("ga_tiny", tmp_path)


def test_metrics_reader_skips_malformed_lines(tmp_path):
    shutil.copy(FIXTURE_RUNS / "ga_tiny" / "metrics.csv", tmp_path / "metrics.csv")
    with (tmp_path / "metrics.csv").open("a") as fh:
        fh.write("7,72,4\n")
    df, bad = dd.read_metrics(tmp_path)
    assert bad == 1 and len(df) == 7


def test_generation_stats_count_evaluated_individuals():
    run = _load("ga_tiny")
    idx = dd.build_index(run)
    steps = an.generation_stats(run, idx)
    assert list(steps["generation"]) == sorted(idx.by_step)
    assert steps["evaluated"].sum() * 2 == run.metrics["sims"].iloc[-1]  # 2 runs per genome
    assert (steps["successful"] <= steps["valid"]).all()


def test_media_key_tracks_genes_not_ids():
    run = _load("ga_tiny")
    genome = json.loads((run.path / "best_genome.json").read_text())
    t = media.Target(genome, EnvSpec(friction=0.6), 0, run.config["sim"], run.config["fitness"])
    renamed = replace(t, genome={**genome, "id": "other-id"})
    assert media.media_key(t, "thumb") == media.media_key(renamed, "thumb")
    assert media.media_key(t, "thumb") != media.media_key(replace(t, env=EnvSpec()), "thumb")
    assert media.media_key(t, "clip") != media.media_key(replace(t, seed=3), "clip")


def test_clip_is_a_real_resimulation(tmp_path):
    """The rendered clip re-simulates the stored genome and reproduces its stored fitness."""
    run = _load("ga_tiny")
    rec = json.loads((run.path / "best_result.json").read_text())
    env, seed = run.eval_runs[0]
    t = media.Target(rec["genome"], env, seed, run.config["sim"], run.config["fitness"])
    report = media.render(tmp_path, [t], "clip")
    assert report.errors == []
    path = media.cached(tmp_path, t, "clip")
    assert path is not None and path.stat().st_size > 10_000
    assert media.clip_meta(path)["fitness"] == rec["seed_fitness"][0]
    # The clip plays in real time: its length is the simulated settle + run time.
    import imageio.v2 as imageio

    reader = imageio.get_reader(path)
    seconds = reader.count_frames() / reader.get_meta_data()["fps"]
    sim = run.sim
    assert seconds == pytest.approx(sim.settle_time + sim.duration, abs=0.1)


def test_viewer_command_is_built_but_never_launched(monkeypatch, tmp_path):
    started = []
    monkeypatch.setattr(media.subprocess, "Popen", lambda cmd, **kw: started.append((cmd, kw)))
    env = EnvSpec(kind="slope", slope_deg=5.0, friction=0.8)
    cmd = media.viewer_command(tmp_path / "g.json", env, 2, tmp_path / "config.yaml")
    assert cmd[1:5] == ["-m", "evomorph", "inspect", "--genome"]
    assert "--view" in cmd and cmd[cmd.index("--slope") + 1] == "5.0"
    media.launch_viewer(cmd)
    assert started[0][0] == cmd and started[0][1]["start_new_session"] is True


def test_describe_tables_follow_the_body():
    run = _load("me_tiny")
    idx = dd.build_index(run)
    for rec in list(idx.records.values())[:6]:
        genome = dd.record_genome(rec)
        joints = desc.joint_table(genome)
        assert len(joints) == genome.n_limbs
        one_segment = genome.morphology.n_segments == 1
        assert joints["knee joint"].str.startswith("none").all() == one_segment
        assert len(desc.metric_table(rec)) == len(rec["metrics"])
        assert len(desc.run_table(rec)) == 2
