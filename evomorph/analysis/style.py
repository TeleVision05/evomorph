"""Shared plot style: files only (Agg), colour-blind-safe colours, quiet axes.

Colours come from a validated categorical palette and were checked with a
colour-vision-deficiency validator on a light surface:

* conditions (all four appear together, so every pair was checked): blue,
  yellow, green, magenta; worst simulated colour-blind separation dE 13.0,
  normal-vision floor passed. Yellow and magenta are below 3:1 contrast on
  white, so every condition also has its own marker shape and a legend:
  identity never rests on colour alone.
* body plans (stacked areas, adjacent pairs checked): the palette's eight
  hues in fixed order; worst adjacent colour-blind separation dE 9.1.
* heat maps: a single-hue sequential ramp (light = low, dark = high); empty
  cells stay the background colour.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # never open a window
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

CONDITION_COLORS: dict[str, str] = {
    "controller": "#2a78d6",
    "morphology": "#eda100",
    "joint": "#008300",
    "joint_me": "#e87ba4",
}
CONDITION_MARKERS: dict[str, str] = {
    "controller": "o",
    "morphology": "s",
    "joint": "^",
    "joint_me": "D",
}
FALLBACK_COLOR = "#6b6b6b"
PLAN_COLORS: tuple[str, ...] = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
INK = "#2b2b2b"
MUTED = "#6b6b6b"
GRID = "#e4e4e1"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "evomorph_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)


def color(condition: str) -> str:
    return CONDITION_COLORS.get(condition, FALLBACK_COLOR)


def marker(condition: str) -> str:
    return CONDITION_MARKERS.get(condition, "o")


def setup() -> None:
    """Quiet defaults: no top/right spines, light horizontal grid, readable text."""
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "lines.linewidth": 2.0,
            "figure.constrained_layout.use": True,  # titles, ticks and legends never collide
            "figure.dpi": 110,
            "savefig.dpi": 130,
            "font.size": 9,
        }
    )


def save(fig, path: str | Path) -> Path:
    """Write ``fig`` to ``path`` (png) and close it."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out
