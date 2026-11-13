"""Section C, the MAP-Elites archive as an interactive heat map."""

from __future__ import annotations

from collections import Counter

import pandas as pd
import streamlit as st

from evomorph.evolution.archive import qd_offset_for
from evomorph.visualization import dashboard_analysis as an
from evomorph.visualization.dashboard_common import Ctx, chart, images, select_creature, table
from evomorph.visualization.dashboard_plots import archive_heatmap

PICK = "archive_cell"  # session key of the cell picker
CHART = "archive_chart"  # session key of the chart's selection state
APPLIED = "archive_applied"  # last chart selection already copied into the picker
ELITE_METRICS = (
    "forward_displacement",
    "sustained_velocity",
    "energy_efficiency",
    "cost_of_transport",
    "slip_fraction",
    "foot_support_fraction",
    "mean_tilt",
)


def _cell_label(view: an.ArchiveView, cell: tuple[int, int]) -> str:
    c = view.cells[cell]
    d0, d1 = view.spec.descriptors
    return (
        f"cell ({cell[0]}, {cell[1]}) · {c.genome_id} · fitness {c.fitness:+.3f} · "
        f"{d0} {c.descriptor[0]:.3f} · {d1} {c.descriptor[1]:.3f}"
    )


def _sync_from_chart(view: an.ArchiveView, labels: dict[tuple[int, int], str]) -> None:
    """Copy a new click on the chart into the picker (before the picker is drawn)."""
    state = st.session_state.get(CHART)
    points = []
    try:
        points = list(state["selection"]["points"]) if state else []
    except (KeyError, TypeError):
        points = []
    for p in points:
        data = p.get("customdata")
        if not data or len(data) < 2:
            continue
        cell = (int(data[0]), int(data[1]))
        token = f"{cell}"
        if cell in labels and st.session_state.get(APPLIED) != token:
            st.session_state[PICK] = labels[cell]
            st.session_state[APPLIED] = token
        break


def _header(ctx: Ctx, view: an.ArchiveView) -> None:
    offset = qd_offset_for(ctx.run.fitness_config)
    qd = sum(c.fitness + offset for c in view.cells.values())
    a, b, c = st.columns(3)
    a.metric("Occupied cells", f"{len(view.cells)} / {view.spec.n_cells}")
    b.metric("Coverage", f"{view.coverage:.0%}")
    c.metric("QD-score", f"{qd:.1f}", help=f"sum of (fitness + {offset:g}) over filled cells")
    what = (
        "the MAP-Elites archive"
        if ctx.run.algorithm == "map_elites"
        else "the GA's passive archive (every genome the GA evaluated, binned the same way)"
    )
    st.caption(
        f"This is {what}, {view.source}. A cell holds the best creature that never fell in any "
        "evaluation run and whose mean behaviour falls in that cell. Click a cell, or pick one "
        "below."
    )


def _elite(ctx: Ctx, view: an.ArchiveView, cell: tuple[int, int]) -> None:
    c = view.cells[cell]
    rec = ctx.record(c.genome_id)
    if rec is None:
        st.warning(f"Creature {c.genome_id} is in the archive but not in the generation files.")
        return
    left, right = st.columns([1, 1])
    with left:
        images(ctx, [rec], "thumb", [_cell_label(view, cell)])
    with right:
        rows = [{"metric": k, "value": c.metrics.get(k)} for k in ELITE_METRICS]
        rows.insert(0, {"metric": "fitness", "value": c.fitness})
        table(pd.DataFrame(rows), hide_index=True)
        st.button(
            "Open this elite in creature inspection",
            on_click=select_creature,
            args=(ctx.entry.name, c.genome_id, "Creature inspection"),
            key="archive_open",
        )


def render(ctx: Ctx) -> None:
    st.header("MAP-Elites archive")
    view = ctx.archive
    if not view.cells:
        st.info(
            "The archive is empty so far: no creature has yet been valid in every run "
            "without falling. It fills in as the run proceeds."
        )
        return
    _header(ctx, view)
    cells = sorted(view.cells, key=lambda k: -view.cells[k].fitness)
    labels = {k: _cell_label(view, k) for k in cells}
    _sync_from_chart(view, labels)
    if st.session_state.get(PICK) not in labels.values():
        st.session_state[PICK] = labels[cells[0]]
    chosen = next(k for k, v in labels.items() if v == st.session_state[PICK])
    fig = archive_heatmap(view, chosen)
    chart(fig, CHART, on_select="rerun", selection_mode="points")
    st.selectbox("Occupied cell (best first)", list(labels.values()), key=PICK)
    _elite(ctx, view, chosen)
    plans = Counter(an.body_plan_of(ctx.record(c.genome_id) or {}) for c in view.cells.values())
    st.caption(
        "Body plans in the archive: "
        + ", ".join(f"{p} x{n}" for p, n in sorted(plans.items(), key=lambda kv: -kv[1]))
    )
