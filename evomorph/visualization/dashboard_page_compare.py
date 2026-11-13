"""Section E, creature comparison: bodies, statistics and clips side by side."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from evomorph.visualization import dashboard_data as dd
from evomorph.visualization import dashboard_media as media
from evomorph.visualization.dashboard_analysis import body_plan_of
from evomorph.visualization.dashboard_common import Ctx, images, other_ctx, select_creature
from evomorph.visualization.dashboard_plots import comparison_bars

# Bars keep one colour per creature in fixed palette order; four creatures stay
# within the palette's validated adjacent-pair range and fit one row of columns.
MAX_CREATURES = 4
COMPARE_METRICS = (
    "forward_displacement",
    "sustained_velocity",
    "energy_efficiency",
    "cost_of_transport",
    "slip_fraction",
    "mean_tilt",
    "max_lateral_deviation",
    "foot_support_fraction",
    "fell",
)
SEP = "::"


def _candidates(ctx: Ctx) -> dict[str, str]:
    """key 'run::id' -> label. Every run's champion, plus this run's champions and elites."""
    out: dict[str, str] = {}
    for entry in ctx.runs:
        try:
            other = other_ctx(ctx, entry.name)
            hist = dd.champion_history(other.run)
        except dd.RunReadError:
            continue
        if not hist.empty:
            gid = str(hist.iloc[-1]["champion_id"])
            out[f"{entry.name}{SEP}{gid}"] = f"{entry.name}: champion {gid}"
    hist = dd.champion_history(ctx.run)
    for _, row in hist[hist["changed"]].iterrows():
        key = f"{ctx.entry.name}{SEP}{row['champion_id']}"
        out.setdefault(key, f"{ctx.entry.name}: champion of step {int(row['generation'])}")
    for c in sorted(ctx.archive.cells.values(), key=lambda c: -c.fitness):
        key = f"{ctx.entry.name}{SEP}{c.genome_id}"
        out.setdefault(key, f"{ctx.entry.name}: archive cell {c.cell}")
    return out


def _resolve(ctx: Ctx, key: str) -> tuple[Ctx, dict[str, Any]] | None:
    run_name, gid = key.split(SEP, 1)
    other = other_ctx(ctx, run_name)
    rec = other.record(gid)
    return (other, rec) if rec is not None else None


def _metrics_table(picked: list[tuple[str, Ctx, dict[str, Any]]]) -> pd.DataFrame:
    rows = {"fitness": [r.get("fitness") for _, _, r in picked]}
    rows["body plan"] = [body_plan_of(r) for _, _, r in picked]
    rows["gait"] = [r["genome"]["controller"]["gait"] for _, _, r in picked]
    rows["frequency (Hz)"] = [r["genome"]["controller"]["frequency"] for _, _, r in picked]
    for k in COMPARE_METRICS:
        rows[k] = [(r.get("metrics") or {}).get(k) for _, _, r in picked]
    cols = [f"{i + 1}. {r['id']}" for i, (_, _, r) in enumerate(picked)]
    return pd.DataFrame(rows, index=cols).T.astype(str)


def _value(rec: dict[str, Any], metric: str) -> float:
    v = rec.get("fitness") if metric == "fitness" else (rec.get("metrics") or {}).get(metric)
    return float(v) if isinstance(v, int | float) else float("nan")


def _clips(picked: list[tuple[str, Ctx, dict[str, Any]]]) -> None:
    targets = [(c, c.target(r)) for _, c, r in picked]
    missing = [(c, t) for c, t in targets if media.cached(c.cache, t, "clip") is None]
    if missing and st.button(f"Render {len(missing)} missing clip(s)", key="compare_clips"):
        with st.spinner("Re-simulating and recording offscreen..."):
            for c, t in missing:
                media.render(c.cache, [t], "clip")
    cols = st.columns(len(targets))
    for i, (c, t) in enumerate(targets):
        path = media.cached(c.cache, t, "clip")
        with cols[i]:
            if path is None:
                st.caption("no clip rendered yet")
            else:
                st.video(str(path))
                st.caption(f"{t.genome_id} on {t.env.label()}, seed {t.seed}")


def render(ctx: Ctx) -> None:
    st.header("Creature comparison")
    options = _candidates(ctx)
    if not options:
        st.info("Nothing to compare yet: no run has a champion so far.")
        return
    keys = list(options)
    default = keys[:2]
    chosen = st.multiselect(
        f"Creatures to compare (up to {MAX_CREATURES})",
        keys,
        default=default,
        format_func=lambda k: options[k],
        max_selections=MAX_CREATURES,
        key="compare_pick",
    )
    picked = []
    for k in chosen:
        hit = _resolve(ctx, k)
        if hit is not None:
            picked.append((k, *hit))
    if not picked:
        st.info("Pick one or more creatures above.")
        return
    cols = st.columns(len(picked))
    for i, (k, c, r) in enumerate(picked):
        with cols[i]:
            images(c, [r], "thumb", [f"{i + 1}. {options[k]} ({r['id']})"])
            st.button(
                "Inspect", key=f"cmp_open_{i}", on_click=select_creature,
                args=(c.entry.name, r["id"], "Creature inspection"),
            )  # fmt: skip
    st.dataframe(_metrics_table(picked), width="stretch")
    names = [f"{i + 1}. {r['id']}" for i, (_, _, r) in enumerate(picked)]
    grid = st.columns(3)
    for j, metric in enumerate(("fitness", *COMPARE_METRICS[:5])):
        vals = [_value(r, metric) for _, _, r in picked]
        with grid[j % 3]:
            st.plotly_chart(comparison_bars(names, vals, metric), width="stretch", key=f"cb{j}")
    st.subheader("Clips side by side")
    _clips(picked)
