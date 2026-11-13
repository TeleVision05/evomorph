"""Plot evolution runs side by side from their metrics.csv (files only, Agg backend).

Usage:
    .venv/bin/python scripts/plot_runs.py --out DIR results/runA results/runB ...
        [--labels "with diversity" "without"]

Writes into ``--out``:

* ``fitness.png``: best and mean fitness of the current set against simulations;
* ``diversity.png``: mean gene distance, body plans and behaviour spread;
* ``qd.png``: archive coverage and QD-score against simulations;
* ``archive_<label>.png`` for every run with an archive: the fitness heat map.

The x axis is always simulations used, the unit budgets are counted in, so
runs with different population or batch sizes line up fairly.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # never open a window
import matplotlib.pyplot as plt
import numpy as np

from evomorph.evolution.archive import Archive


def read_metrics(run: Path) -> dict[str, np.ndarray]:
    with (run / "metrics.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for key in rows[0]:
        try:
            out[key] = np.array([float(r[key]) for r in rows])
        except ValueError:
            out[key] = np.array([r[key] for r in rows])
    return out


def _panel(ax, runs, labels, key: str, title: str) -> None:
    for m, label in zip(runs, labels, strict=True):
        ax.plot(m["sims"], m[key], label=label, linewidth=1.6)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("simulations")
    ax.grid(alpha=0.3)


def plot_all(runs: list[Path], labels: list[str], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    data = [read_metrics(r) for r in runs]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    _panel(axes[0], data, labels, "best_fitness", "best fitness")
    _panel(axes[1], data, labels, "mean_fitness", "mean fitness (valid individuals)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fitness.png", dpi=120)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    _panel(axes[0], data, labels, "mean_gene_distance", "mean pairwise gene distance")
    _panel(axes[1], data, labels, "n_body_plans", "distinct body plans (limbs x segments)")
    _panel(axes[2], data, labels, "behaviour_spread", "behaviour spread (speed x efficiency)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "diversity.png", dpi=120)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    _panel(axes[0], data, labels, "archive_coverage", "archive coverage")
    _panel(axes[1], data, labels, "qd_score", "QD-score")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "qd.png", dpi=120)
    plt.close(fig)
    for run, label in zip(runs, labels, strict=True):
        if (run / "archive" / "archive.json").exists():
            _heatmap(Archive.load(run / "archive"), label, out)


def _heatmap(archive: Archive, label: str, out: Path) -> None:
    grid = archive.fitness_grid()
    (x0, x1), (y0, y1) = archive.spec.ranges
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(grid.T, origin="lower", extent=(x0, x1, y0, y1), aspect="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, label="fitness")
    ax.set_xlabel(archive.spec.descriptors[0])
    ax.set_ylabel(archive.spec.descriptors[1])
    ax.set_title(f"{label}: {len(archive)} cells, QD {archive.qd_score:.1f}", fontsize=10)
    fig.tight_layout()
    safe = "".join(c if c.isalnum() else "_" for c in label)
    fig.savefig(out / f"archive_{safe}.png", dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--labels", nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    labels = args.labels or [r.name for r in args.runs]
    if len(labels) != len(args.runs):
        ap.error("give one label per run")
    plot_all(args.runs, labels, args.out)
    print(f"wrote fitness.png, diversity.png, qd.png to {args.out}")


if __name__ == "__main__":
    main()
