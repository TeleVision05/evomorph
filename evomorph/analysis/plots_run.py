"""Plots for one run directory (files only, Agg backend).

The run's own records only: ``metrics.csv`` for the time series, the
generation files for body plans, ``archive/`` for the heat map (MAP-Elites'
archive, or the GA's passive archive of everything it evaluated) and the saved
genomes for the gallery.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evomorph.analysis.gallery import Tile, gallery
from evomorph.analysis.load import PLANS, metrics_series, plan_history
from evomorph.analysis.style import INK, PLAN_COLORS, SEQUENTIAL, save
from evomorph.config import load_yaml
from evomorph.evolution.archive import Archive
from evomorph.evolution.selection import body_plan
from evomorph.experiments.evaluation import champion_member, final_members
from evomorph.experiments.measures import top_k

BLUE, GREEN = "#2a78d6", "#008300"


def fitness(run: Path, out: Path) -> Path:
    m = metrics_series(run)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(m["sims"], m["p10_fitness"], m["p90_fitness"], color=BLUE, alpha=0.15, lw=0,
                    label="10th-90th percentile of current set")  # fmt: skip
    ax.plot(m["sims"], m["mean_fitness"], color=BLUE, label="mean of current set")
    ax.plot(m["sims"], m["champion_fitness"], color=GREEN, label="best genome so far")
    ax.set_xlabel("simulations used")
    ax.set_ylabel("training fitness (dimensionless)")
    ax.set_title(f"{run.name}: fitness against simulations")
    ax.legend(loc="lower right")
    return save(fig, out / "fitness_vs_simulations.png")


def diversity(run: Path, out: Path) -> Path:
    m = metrics_series(run)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    panels = (
        ("mean_gene_distance", "Mean gene distance (evolving genes)", "0-1"),
        ("behaviour_spread", "Behaviour spread (speed x efficiency)", "mean distance (0-1.41)"),
        ("archive_coverage", "Archive coverage", "share of cells"),
    )
    for ax, (col, title, unit) in zip(axes, panels, strict=True):
        ax.plot(m["sims"], m[col], color=BLUE)
        ax.set_title(title)
        ax.set_xlabel("simulations used")
        ax.set_ylabel(unit)
    fig.suptitle(f"{run.name}: diversity of the current set")
    return save(fig, out / "diversity.png")


def body_plans(run: Path, out: Path) -> Path:
    sims, counts = plan_history(run)
    mat = np.array([[c.get(p, 0) for p in PLANS] for c in counts], dtype=float).T
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.stackplot(sims, mat, colors=PLAN_COLORS, labels=PLANS, edgecolor="white", lw=0.5)
    ax.set_xlabel("simulations used")
    ax.set_ylabel("genomes in population / archive")
    ax.set_title(f"{run.name}: body plans (limbs x segments) over time")
    ax.grid(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return save(fig, out / "body_plans.png")


def archive_heatmap(run: Path, out: Path) -> Path | None:
    if not (run / "archive" / "archive.json").is_file():
        return None
    arch = Archive.load(run / "archive")
    (x0, x1), (y0, y1) = arch.spec.ranges
    fig, ax = plt.subplots(figsize=(6, 4.6))
    im = ax.imshow(arch.fitness_grid().T, origin="lower", extent=(x0, x1, y0, y1), aspect="auto",
                   cmap=SEQUENTIAL)  # fmt: skip
    fig.colorbar(im, ax=ax, label="training fitness")
    ax.set_xlabel("sustained velocity (m/s)")
    ax.set_ylabel("energy efficiency (0-1)")
    ax.grid(False)
    algorithm = load_yaml(run / "config.yaml")["evolution"]["algorithm"]
    kind = "MAP-Elites archive" if algorithm == "map_elites" else "GA passive archive"
    ax.set_title(f"{run.name}: {kind}, {len(arch)}/{arch.spec.n_cells} cells (empty = blank)",
                 color=INK)  # fmt: skip
    return save(fig, out / "archive_heatmap.png")


def run_gallery(run: Path, out: Path, k: int = 5) -> Path:
    """The champion and the next best distinct genomes of the final set."""
    champ = champion_member(run)
    others = [m for m in top_k(final_members(run), k + 1) if m["genome"].id != champ["genome"].id]
    members = [champ, *others[:k]]
    tiles = [
        Tile(m["genome"], f"{m['genome'].id} {body_plan(m['genome'])}\nfitness {m['fitness']:+.3f}")
        for m in members
    ]
    return gallery([tiles], ["final"], f"{run.name}: champion (left) and best of final set",
                   out / "gallery.png")  # fmt: skip


def plot_run(run: Path, out: Path, render: bool = True) -> list[Path]:
    """Every single-run plot into ``out``; returns the files written."""
    made = [fitness(run, out), diversity(run, out), body_plans(run, out)]
    heat = archive_heatmap(run, out)
    if heat is not None:
        made.append(heat)
    if render:
        made.append(run_gallery(run, out))
    return made
