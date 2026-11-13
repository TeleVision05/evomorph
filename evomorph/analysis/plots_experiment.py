"""Comparison plots for an experiment folder (files only, Agg backend).

Every plot shows each seed as its own point or line, so a reader sees the
spread rather than only a summary, plus the median per condition. Fitness is
dimensionless (see evomorph/simulation/fitness.py: ~1 = 0.2 m/s of clean
walking; a fall costs 1). The x axis of every time plot is simulations, the
unit budgets are counted in.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evomorph.analysis.load import PLANS, ExperimentData, metrics_series, plan_history
from evomorph.analysis.style import (
    INK,
    MUTED,
    PLAN_COLORS,
    SEQUENTIAL,
    color,
    marker,
    save,
)
from evomorph.evolution.archive import Archive


def _step_on_grid(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Value of a step series (last value at or before each grid point; nan before)."""
    idx = np.searchsorted(x, grid, side="right") - 1
    out = np.where(idx >= 0, y[np.clip(idx, 0, len(y) - 1)], np.nan)
    return out.astype(float)


def _conditions(data: ExperimentData) -> list[str]:
    present = {r["condition"] for r in data.rows}
    return [c["key"] for c in data.conditions if c["key"] in present]


def _time_grid(data: ExperimentData, n: int) -> np.ndarray:
    """Simulation counts from the latest first step of any run to the budget.

    Starting where every run has reported at least once keeps medians and bands
    from being computed over runs that have not started.
    """
    starts = [
        float(metrics_series(run)["sims"][0])
        for cond in _conditions(data)
        for _, run in data.runs_of(cond)
    ]
    return np.linspace(max(starts), int(data.manifest["budget"]), n)


def _caption(fig, data: ExperimentData) -> None:
    """Spell out the condition letters used on category axes."""
    names = "   ".join(data.label(c) for c in _conditions(data))
    fig.supxlabel(names, fontsize=8, color=MUTED)


def fitness_over_time(data: ExperimentData, out: Path) -> Path:
    """Best-so-far and current-set mean training fitness against simulations."""
    grid = _time_grid(data, 200)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    panels = (
        ("champion_fitness", "Best genome so far (training fitness)"),
        ("mean_fitness", "Mean of current population / archive"),
    )
    for ax, (col, title) in zip(axes, panels, strict=True):
        for cond in _conditions(data):
            series = []
            for _, run in data.runs_of(cond):
                m = metrics_series(run)
                series.append(_step_on_grid(m["sims"], m[col], grid))
            arr = np.array(series)
            c = color(cond)
            ax.fill_between(grid, np.nanmin(arr, 0), np.nanmax(arr, 0), color=c, alpha=0.15, lw=0)
            n = len(series)
            ax.plot(grid, np.nanmedian(arr, 0), color=c, label=f"{data.label(cond)} (n={n})")
        ax.set_title(title)
        ax.set_xlabel("simulations used")
        ax.set_ylabel("fitness (dimensionless)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels))
    fig.suptitle("Training fitness against simulations: median line, band = range over seeds")
    return save(fig, out / "fitness_vs_simulations.png")


def train_vs_test(data: ExperimentData, out: Path) -> Path:
    """Training against held-out fitness, per seed, for the champion and the best-K set."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    panels = (
        ("champ", "Champion (fresh seeds)"),
        ("set", "Best-K final set (mean over members)"),
    )
    for ax, (prefix, title) in zip(axes, panels, strict=True):
        xs, ys = [], []
        for cond in _conditions(data):
            rows = [r for r in data.rows if r["condition"] == cond]
            x = [r[f"{prefix}_train_fitness"] for r in rows]
            y = [r[f"{prefix}_test_fitness"] for r in rows]
            xs += x
            ys += y
            ax.scatter(x, y, s=46, color=color(cond), marker=marker(cond), label=data.label(cond),
                       edgecolor="white", linewidth=0.8, zorder=3)  # fmt: skip
        lo, hi = min(xs + ys), max(xs + ys)
        pad = 0.05 * (hi - lo or 1.0)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUTED, lw=1, ls="--", zorder=1)
        ax.annotate("test = train", (hi + pad, hi + pad), color=MUTED, fontsize=7, ha="right",
                    va="bottom")  # fmt: skip
        ax.set_title(title)
        ax.set_xlabel("training-terrain fitness")
        ax.set_ylabel("held-out-terrain fitness")
        ax.grid(axis="x", color="#e4e4e1", lw=0.8)
    axes[0].legend(loc="upper left")
    fig.suptitle("Training vs held-out performance, one point per seed (below the line = worse)")
    return save(fig, out / "train_vs_test.png")


def _strip(ax, data: ExperimentData, column: str, scale: float = 1.0, both: bool = True) -> None:
    """Per-seed points per condition (train hollow, test filled) with median ticks."""
    conds = _conditions(data)
    for i, cond in enumerate(conds):
        rows = [r for r in data.rows if r["condition"] == cond]
        suites = (
            (("train", -0.17, "none"), ("test", 0.17, color(cond))) if both else (("", 0, None),)
        )
        for suite, dx, face in suites:
            col = column.format(suite=suite)
            vals = np.array([r[col] for r in rows], dtype=float) * scale
            jitter = np.linspace(-0.05, 0.05, len(vals)) if len(vals) > 1 else np.zeros(1)
            c = color(cond)
            ax.scatter(i + dx + jitter, vals, s=34, marker=marker(cond), zorder=3,
                       facecolor=face or c, edgecolor=c, linewidth=1.3)  # fmt: skip
            if np.isfinite(vals).any():
                ax.hlines(np.nanmedian(vals), i + dx - 0.12, i + dx + 0.12, color=INK, lw=2)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([data.label(c).split(" ", 1)[0] for c in conds])
    ax.set_xlim(-0.6, len(conds) - 0.4)


def locomotion(data: ExperimentData, out: Path) -> Path:
    """Champion speed, efficiency, tilt and failure rate on training and held-out terrain."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    panels = (
        ("champ_{suite}_speed", 1.0, "Forward speed", "m/s"),
        ("champ_{suite}_efficiency", 1.0, "Energy efficiency", "1/(1 + cost of transport)"),
        ("champ_{suite}_tilt", 180 / np.pi, "Mean torso tilt (lower = more stable)", "degrees"),
        ("champ_{suite}_failure_rate", 100.0, "Failure rate (fell or invalid)", "% of runs"),
    )
    for ax, (col, scale, title, unit) in zip(axes.flat, panels, strict=True):
        _strip(ax, data, col, scale)
        ax.set_title(title)
        ax.set_ylabel(unit)
    axes[1][1].set_ylim(-3, 103)  # a rate: always show the whole 0-100% scale
    _caption(fig, data)
    fig.suptitle(
        "Champions re-scored with fresh seeds: hollow = training terrain, filled = held-out; "
        "bar = median"
    )
    return save(fig, out / "locomotion.png")


def diversity(data: ExperimentData, out: Path) -> Path:
    """Morphological and behavioural diversity per condition."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    panels = (
        ("final_morph_diversity", "Morphological diversity\n(final population / archive)",
         "mean gene distance (0-1)"),
        ("set_test_behaviour_diversity", "Behavioural diversity\n(best-K set, held-out terrain)",
         "mean behaviour distance (0-1)"),
        ("final_body_plans", "Body plans\n(final population / archive)", "distinct plans (of 8)"),
    )  # fmt: skip
    for ax, (col, title, unit) in zip(axes, panels, strict=True):
        _strip(ax, data, col, both=False)
        ax.set_title(title)
        ax.set_ylabel(unit)
    _caption(fig, data)
    fig.suptitle("Diversity, one point per seed, bar = median (condition A has one body by design)")
    return save(fig, out / "diversity.png")


def _mean_plan_shares(runs: Sequence[Path], grid: np.ndarray) -> np.ndarray:
    """(plans, grid) mean share of each body plan across runs."""
    shares = []
    for run in runs:
        sims, counts = plan_history(run)
        mat = np.array([[c.get(p, 0) for p in PLANS] for c in counts], dtype=float)
        frac = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1)
        shares.append(np.array([_step_on_grid(sims, frac[:, k], grid) for k in range(len(PLANS))]))
    return np.nanmean(np.array(shares), axis=0)


def body_plans(data: ExperimentData, out: Path) -> Path:
    """Share of each body plan in the current set over time, averaged over seeds."""
    conds = _conditions(data)
    grid = _time_grid(data, 120)
    fig, axes = plt.subplots(1, len(conds), figsize=(3.6 * len(conds), 3.8), sharey=True)
    for ax, cond in zip(np.atleast_1d(axes), conds, strict=True):
        runs = [p for _, p in data.runs_of(cond)]
        shares = np.nan_to_num(_mean_plan_shares(runs, grid))
        ax.stackplot(grid, shares, colors=PLAN_COLORS, labels=PLANS, edgecolor="white", lw=0.5)
        ax.set_title(f"{data.label(cond)} (n={len(runs)})")
        ax.set_xlabel("simulations used")
        ax.set_ylim(0, 1)
        ax.grid(False)
    np.atleast_1d(axes)[0].set_ylabel("share of population / archive")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, [f"{p} (limbs x segments)" if i == 0 else p for i, p in enumerate(labels)],
               loc="outside lower center", ncol=8)  # fmt: skip
    fig.suptitle("Body plans over time, mean share over seeds")
    return save(fig, out / "body_plans.png")


def archive_heatmaps(data: ExperimentData, out: Path) -> Path | None:
    """MAP-Elites (condition D) archive fitness per seed; None without condition D."""
    runs = data.runs_of("joint_me")
    if not runs:
        return None
    archives = [(s, Archive.load(p / "archive")) for s, p in runs]
    vmin = min(np.nanmin(a.fitness_grid()) for _, a in archives if len(a))
    vmax = max(np.nanmax(a.fitness_grid()) for _, a in archives if len(a))
    fig, axes = plt.subplots(
        1, len(archives), figsize=(3.8 * len(archives) + 1, 3.8), squeeze=False
    )
    im = None
    for ax, (seed, arch) in zip(axes[0], archives, strict=True):
        (x0, x1), (y0, y1) = arch.spec.ranges
        im = ax.imshow(arch.fitness_grid().T, origin="lower", extent=(x0, x1, y0, y1),
                       aspect="auto", cmap=SEQUENTIAL, vmin=vmin, vmax=vmax)  # fmt: skip
        ax.set_title(f"seed {seed}: {len(arch)}/{arch.spec.n_cells} cells")
        ax.set_xlabel("sustained velocity (m/s)")
        ax.grid(False)
    axes[0][0].set_ylabel("energy efficiency (0-1)")
    fig.colorbar(im, ax=axes[0].tolist(), label="training fitness", shrink=0.9)
    fig.suptitle(f"{data.label('joint_me')}: final archive, empty cells blank")
    return save(fig, out / "archive_heatmap.png")
