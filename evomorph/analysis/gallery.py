"""A gallery of creatures rendered offscreen (MuJoCo's offscreen renderer, no window).

Each tile is the creature's real body, built from its saved genome and
settled under gravity on flat ground, seen from the renderer's ``iso`` view.
Titles carry the numbers that matter for the comparison, so a picture is never
shown without its result.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evomorph.analysis.style import INK, save
from evomorph.morphology.genome import Genome
from evomorph.visualization.renderer import render_views

TILE = (360, 270)  # px per view; four views are rendered, the first (iso) is kept


@dataclass(frozen=True)
class Tile:
    genome: Genome | None  # None leaves the slot empty
    title: str


def iso_image(genome: Genome) -> np.ndarray:
    """The iso view of ``genome`` at rest (H x W x 3 uint8)."""
    views = render_views(genome, size=TILE)
    return views[:, : TILE[0]]


def gallery(rows: Sequence[Sequence[Tile]], row_labels: Sequence[str], title: str, path: Path):
    """Grid of rendered creatures; rows are e.g. conditions and columns seeds."""
    n_rows, n_cols = len(rows), max(len(r) for r in rows)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.9 * n_cols + 0.6, 2.5 * n_rows + 0.4), squeeze=False
    )
    for i, row in enumerate(rows):
        for j in range(n_cols):
            ax = axes[i][j]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if j >= len(row) or row[j].genome is None:
                continue
            ax.imshow(iso_image(row[j].genome))
            ax.set_title(row[j].title, fontsize=7.5, color=INK)
        axes[i][0].set_ylabel(row_labels[i], fontsize=9, color=INK)
    fig.suptitle(title)
    return save(fig, path)
