"""Section F, generation history: browse every step's champion and population."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from evomorph.visualization import dashboard_analysis as an
from evomorph.visualization import dashboard_data as dd
from evomorph.visualization.dashboard_common import Ctx, images, select_creature
from evomorph.visualization.dashboard_describe import short_label


def _population_table(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        m = r.get("metrics") or {}
        rows.append(
            {
                "id": r.get("id"),
                "status": r.get("status", ""),
                "origin": r.get("origin", ""),
                "parents": ", ".join(r.get("parent_ids") or []) or "-",
                "plan": an.body_plan_of(r),
                "fitness": r.get("fitness") if r.get("valid") else None,
                "forward (m)": m.get("forward_displacement"),
                "fell (share of runs)": m.get("fell"),
                "note": "" if r.get("valid") else (r.get("failure_reason") or "invalid"),
            }
        )
    return pd.DataFrame(rows)


def render(ctx: Ctx) -> None:
    st.header("Generation history")
    hist = dd.champion_history(ctx.run)
    steps = sorted(ctx.index.by_step)
    if not steps:
        st.info("No generation has finished yet; champions appear here as the run proceeds.")
        return
    label = "batch" if ctx.run.algorithm == "map_elites" else "generation"
    step = (
        steps[0]
        if len(steps) == 1
        else st.select_slider(f"Pick a {label}", options=steps, value=steps[-1], key="history_step")
    )
    champ_row = hist[hist["generation"] == step]
    champ = ctx.record(str(champ_row.iloc[0]["champion_id"])) if len(champ_row) else None
    best_here = an.step_best(ctx.run, step)
    st.caption(
        "Left: the run's champion after this step (best found so far, kept in "
        f"genomes/champion_gen_{step:04d}.json). Right: the best creature evaluated in this "
        "step itself, which shows what the search was producing at that moment."
    )
    pair = [r for r in (champ, best_here) if r is not None]
    caps = []
    if champ is not None:
        caps.append(f"champion after step {step}: {short_label(champ)}")
    if best_here is not None:
        caps.append(f"best evaluated in step {step}: {short_label(best_here)}")
    images(ctx, pair, "thumb", caps)
    records = an.step_records(ctx.run, step)
    st.subheader(f"Everyone recorded in {label} {step} ({len(records)})")
    st.dataframe(_population_table(records), hide_index=True, width="stretch")
    ids = [r["id"] for r in records]
    if ids:
        pick = st.selectbox("Inspect one of them", ids, key="history_pick")
        st.button(
            "Open in creature inspection",
            on_click=select_creature,
            args=(ctx.entry.name, pick, "Creature inspection"),
            key="history_open",
        )
