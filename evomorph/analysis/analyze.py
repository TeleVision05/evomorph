"""``analyze`` for an experiment folder or a single run directory.

Experiment folder (has ``manifest.json``): comparison plots into ``plots/``,
``stats.csv`` and ``summary.md``. Run directory (has ``metrics.csv``): the
run's own plots into ``plots/`` and an analysis section added to (or replaced
in) its ``summary.md``; the engine's summary above it is kept.
"""

from __future__ import annotations

from pathlib import Path

from evomorph.analysis import plots_experiment as px
from evomorph.analysis.gallery import Tile, gallery
from evomorph.analysis.load import ExperimentData, is_experiment, is_run, load_experiment_dir
from evomorph.analysis.plots_run import plot_run
from evomorph.analysis.report import LETTER, add_run_section, experiment_summary
from evomorph.analysis.style import setup
from evomorph.morphology.genome import load_genome


class NothingToAnalyseError(ValueError):
    """The folder is neither an experiment nor a run, or holds no finished run."""


def champion_gallery(data: ExperimentData, out: Path) -> Path:
    """Rows = conditions, columns = seeds; each tile titled with plan and held-out fitness."""
    rows, labels = [], []
    for c in data.conditions:
        cond = c["key"]
        table = {r["seed"]: r for r in data.rows if r["condition"] == cond}
        if not table:
            continue
        tiles = []
        for seed, run in data.runs_of(cond):
            r = table[seed]
            title = (
                f"seed {seed}: {r['champion_plan']}\n"
                f"train {r['champ_train_fitness']:+.2f} / held-out {r['champ_test_fitness']:+.2f}"
            )
            tiles.append(Tile(load_genome(run / "best_genome.json"), title))
        rows.append(tiles)
        labels.append(f"{LETTER.get(cond, cond)}: {c['label'].split(' ', 1)[-1]}")
    return gallery(rows, labels, "Champions of every run (rendered at rest)", out / "gallery.png")


def analyze_experiment(path: Path, render: bool = True) -> tuple[Path, list[Path]]:
    data = load_experiment_dir(path)
    if not data.rows:
        raise NothingToAnalyseError(
            f"{path} has no finished runs yet; run the experiment first (it is resumable)"
        )
    out = path / "plots"
    plots = [
        px.fitness_over_time(data, out),
        px.train_vs_test(data, out),
        px.locomotion(data, out),
        px.diversity(data, out),
        px.body_plans(data, out),
    ]
    heat = px.archive_heatmaps(data, out)
    if heat is not None:
        plots.append(heat)
    if render:
        plots.append(champion_gallery(data, out))
    return experiment_summary(data, plots, path / "summary.md"), plots


def analyze(path: str | Path, render: bool = True) -> tuple[Path, list[Path]]:
    """Analyse ``path``; returns (summary file, plot files)."""
    p = Path(path)
    setup()
    if is_experiment(p):
        return analyze_experiment(p, render)
    if is_run(p):
        plots = plot_run(p, p / "plots", render)
        return add_run_section(p, plots), plots
    raise NothingToAnalyseError(
        f"{p} is neither an experiment folder (manifest.json) nor a run directory (metrics.csv)"
    )
