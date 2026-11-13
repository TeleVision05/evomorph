"""Plotly figures for the research dashboard (no Streamlit here).

Colours: the categorical hues are the dataviz reference palette's dark-mode
steps, always assigned in the same fixed order (never cycled), so a series
keeps its colour whatever else is on the chart. Body plans have a fixed slot
each, so "6x2" is the same colour on every page. Magnitude (archive fitness)
uses one blue ramp, dark (low) to light (high) on the dark surface.

Every figure plots numbers read from a run directory; nothing is smoothed or
rescaled for appearance.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from evomorph.visualization.dashboard_analysis import ArchiveView

SURFACE = "#0e1316"
PANEL = "#161d21"
INK = "#e4ebe8"
MUTED = "#9aa7a3"
GRID = "#26323a"
# Dark-mode categorical steps, fixed order (blue, orange, aqua, yellow,
# magenta, green, violet, red).
SERIES = ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767")
PLANS = ("2x1", "2x2", "4x1", "4x2", "6x1", "6x2", "8x1", "8x2")
PLAN_COLOURS = dict(zip(PLANS, SERIES, strict=True))
# Blue ramp from near-surface to light: low fitness recedes, high fitness glows.
FITNESS_SCALE = [
    [0.0, "#0d366b"],
    [0.25, "#1c5cab"],
    [0.5, "#3987e5"],
    [0.75, "#86b6ef"],
    [1.0, "#cde2fb"],
]


def _layout(fig: go.Figure, title: str, x: str, y: str, height: int = 340) -> go.Figure:
    fig.update_layout(
        title={"text": title, "font": {"size": 15, "color": INK}},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"color": INK, "size": 12},
        height=height,
        margin={"l": 60, "r": 20, "t": 48, "b": 48},
        legend={"orientation": "h", "y": -0.22, "font": {"color": MUTED}},
        hovermode="x unified",
    )
    fig.update_xaxes(title=x, gridcolor=GRID, zerolinecolor=GRID, color=MUTED)
    fig.update_yaxes(title=y, gridcolor=GRID, zerolinecolor=GRID, color=MUTED)
    return fig


def fitness_progress(metrics: pd.DataFrame, x: str) -> go.Figure:
    """Best / mean / median fitness per step with the p10-p90 band (metrics.csv)."""
    fig = go.Figure()
    xs = metrics[x]
    if {"p10_fitness", "p90_fitness"} <= set(metrics.columns):
        fig.add_trace(go.Scatter(x=xs, y=metrics["p90_fitness"], line={"width": 0}, name="p90"))
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=metrics["p10_fitness"],
                fill="tonexty",
                fillcolor="rgba(57,135,229,0.15)",
                line={"width": 0},
                name="p10-p90 band",
            )
        )
    for i, col in enumerate(("best_fitness", "mean_fitness", "median_fitness")):
        if col in metrics.columns:
            name = col.replace("_fitness", "")
            fig.add_trace(
                go.Scatter(x=xs, y=metrics[col], name=name, line={"width": 2, "color": SERIES[i]})
            )
    return _layout(fig, "Fitness of the current set", _x_title(x), "fitness")


def _x_title(x: str) -> str:
    return "simulations used" if x == "sims" else "generation / batch"


def lines(df: pd.DataFrame, x: str, cols: Sequence[str], title: str, y_title: str) -> go.Figure:
    """One line per column, colours in fixed slot order."""
    fig = go.Figure()
    for i, col in enumerate(cols):
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df[x],
                    y=df[col],
                    name=col.replace("_", " "),
                    mode="lines+markers",
                    marker={"size": 8},
                    line={"width": 2, "color": SERIES[i % len(SERIES)]},
                )
            )
    if len(cols) == 1:
        fig.update_layout(showlegend=False)
    return _layout(fig, title, _x_title(x), y_title)


def fitness_distribution(samples: pd.DataFrame, max_boxes: int = 24) -> go.Figure:
    """Box plot of evaluated fitness per step (at most ``max_boxes`` evenly spaced steps)."""
    steps = sorted(samples["generation"].unique()) if not samples.empty else []
    if len(steps) > max_boxes:
        pick = np.linspace(0, len(steps) - 1, max_boxes).round().astype(int)
        steps = [steps[i] for i in pick]
    fig = go.Figure()
    for s in steps:
        vals = samples.loc[samples["generation"] == s, "fitness"]
        fig.add_trace(
            go.Box(
                y=vals,
                name=str(s),
                marker={"color": SERIES[0], "size": 4},
                line={"width": 1},
                boxpoints="outliers",
                showlegend=False,
            )
        )
    fig = _layout(fig, "Fitness of the individuals evaluated in each step", "step", "fitness")
    fig.update_layout(hovermode="closest")
    return fig


def plan_area(counts: pd.DataFrame) -> go.Figure:
    """Stacked body-plan counts per step (fixed colour per plan)."""
    fig = go.Figure()
    known = [p for p in PLANS if p in set(counts["plan"])] if not counts.empty else []
    other = sorted(set(counts["plan"]) - set(PLANS)) if not counts.empty else []
    steps = sorted(counts["generation"].unique()) if not counts.empty else []
    for plan in [*known, *other]:
        sub = counts[counts["plan"] == plan].set_index("generation")["count"]
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=[int(sub.get(s, 0)) for s in steps],
                name=plan,
                stackgroup="plans",
                line={"width": 0.5, "color": PLAN_COLOURS.get(plan, MUTED)},
            )
        )
    return _layout(fig, "Body plans alive in each step (limbs x segments)", "step", "creatures")


def archive_heatmap(view: ArchiveView, selected: tuple[int, int] | None = None) -> go.Figure:
    """Archive fitness grid plus a clickable marker on every occupied cell."""
    spec = view.spec
    xe, ye = view.edges(0), view.edges(1)
    xc, yc = (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2
    grid = view.grid()
    fig = go.Figure(
        go.Heatmap(
            x=xc,
            y=yc,
            z=grid.T,  # rows = second descriptor
            colorscale=FITNESS_SCALE,
            colorbar={"title": "fitness", "tickfont": {"color": MUTED}},
            hoverinfo="skip",
            xgap=2,
            ygap=2,
        )
    )
    cells = sorted(view.cells)
    custom = [[i, j, view.cells[(i, j)].genome_id] for i, j in cells]
    fig.add_trace(
        go.Scatter(
            x=[xc[i] for i, _ in cells],
            y=[yc[j] for _, j in cells],
            mode="markers",
            marker={"size": 14, "color": "rgba(0,0,0,0)", "line": {"width": 1, "color": MUTED}},
            customdata=custom,
            text=[f"{view.cells[c].fitness:+.3f}" for c in cells],
            hovertemplate="cell (%{customdata[0]}, %{customdata[1]})<br>%{customdata[2]}"
            "<br>fitness %{text}<extra></extra>",
            showlegend=False,
        )
    )
    if selected in view.cells:
        i, j = selected
        fig.add_shape(
            type="rect",
            x0=xe[i], x1=xe[i + 1], y0=ye[j], y1=ye[j + 1],
            line={"color": "#3ecf9a", "width": 3},
        )  # fmt: skip
    fig = _layout(fig, "MAP-Elites archive", spec.descriptors[0], spec.descriptors[1], 480)
    fig.update_layout(hovermode="closest", clickmode="event+select", dragmode=False)
    fig.update_xaxes(range=[xe[0], xe[-1]], showgrid=False)
    fig.update_yaxes(range=[ye[0], ye[-1]], showgrid=False)
    return fig


def lineage_figure(rows: Sequence[dict[str, Any]]) -> go.Figure:
    """Ancestors as fitness over generation of origin, with parent-child edges."""
    known = {r["id"]: r for r in rows if r.get("recorded")}
    fig = go.Figure()
    ex, ey = [], []
    for r in known.values():
        for p in r.get("parent_ids") or []:
            if p in known:
                ex += [known[p]["generation"], r["generation"], None]
                ey += [known[p]["fitness"], r["fitness"], None]
    fig.add_trace(
        go.Scatter(x=ex, y=ey, mode="lines", line={"color": GRID, "width": 1.5}, hoverinfo="skip")
    )
    pts = list(known.values())
    fig.add_trace(
        go.Scatter(
            x=[r["generation"] for r in pts],
            y=[r["fitness"] for r in pts],
            mode="markers",
            marker={
                "size": [16 if r["depth"] == 0 else 10 for r in pts],
                "color": [SERIES[1] if r["depth"] == 0 else SERIES[0] for r in pts],
                "line": {"width": 2, "color": SURFACE},
            },
            text=[f"{r['id']} ({r['origin']})" for r in pts],
            hovertemplate="%{text}<br>generation %{x}<br>fitness %{y:+.3f}<extra></extra>",
        )
    )
    fig = _layout(fig, "Lineage (orange: this creature)", "generation of origin", "fitness")
    fig.update_layout(showlegend=False, hovermode="closest")
    return fig


def comparison_bars(names: Sequence[str], values: Sequence[float], title: str) -> go.Figure:
    """One small bar chart per metric; bars keep the creature's slot colour."""
    fig = go.Figure(
        go.Bar(
            x=list(names),
            y=list(values),
            marker={"color": [SERIES[i % len(SERIES)] for i in range(len(names))]},
            text=[f"{v:+.3g}" if np.isfinite(v) else "n/a" for v in values],
            textposition="outside",
        )
    )
    fig = _layout(fig, title, "", "", 260)
    fig.update_layout(showlegend=False, hovermode="closest", margin={"t": 40, "b": 30})
    return fig
