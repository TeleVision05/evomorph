"""Section A, the evolution viewer: where the run stands and how the champion changed."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import streamlit as st

from evomorph.visualization import dashboard_data as dd
from evomorph.visualization.dashboard_common import Ctx, images, select_creature

# The strip shows every change of champion; a long run can change it dozens of
# times, so at most this many evenly spaced changes (always including the
# first and the latest) are drawn to keep the page quick. 16 fits 4 rows of 4.
MAX_STRIP = 16


def _fmt(x: object, spec: str = "+.3f") -> str:
    return format(x, spec) if isinstance(x, int | float) and math.isfinite(x) else "n/a"


def _tiles(ctx: Ctx) -> None:
    run, m = ctx.run, ctx.run.metrics
    last = m.iloc[-1] if not m.empty else None
    budget = run.budget
    c = st.columns(6)
    c[0].metric("Generation / batch", int(last["generation"]) if last is not None else "none yet")
    used = int(last["sims"]) if last is not None else 0
    c[1].metric(f"Simulations used (of {budget:,})" if budget else "Simulations used", f"{used:,}")
    evaluated = int(m["evaluations"].sum()) if "evaluations" in m.columns else 0
    c[2].metric("Individuals evaluated", f"{evaluated:,}")
    c[3].metric(
        "Best fitness",
        _fmt(last["champion_fitness"]) if last is not None else "n/a",
        help="The champion: the best valid creature found so far.",
    )
    what = "population" if run.algorithm == "ga" else "archive elites"
    c[4].metric(
        "Mean fitness",
        _fmt(last["mean_fitness"]) if last is not None else "n/a",
        help=f"Mean over the valid {what} after the latest step.",
    )
    if last is not None and "archive_cells" in m.columns:
        n = ctx.run.archive_spec.n_cells
        c[5].metric("Archive occupancy", f"{int(last['archive_cells'])} / {n}")
    else:
        c[5].metric("Archive occupancy", "n/a")


def _settings(ctx: Ctx) -> None:
    run, evo = ctx.run, ctx.run.evolution
    mut = evo.get("mutation") or {}
    runs = ", ".join(f"{e.label()}/s{s}" for e, s in run.eval_runs)
    rows = {
        "algorithm": run.algorithm,
        "mode (what evolves)": evo.get("mode", "joint"),
        "seed": evo.get("seed"),
        "budget (simulations)": evo.get("budget"),
        "evaluation runs per genome": runs,
        "aggregate over runs": (evo.get("evaluation") or {}).get("aggregate", "mean"),
    }
    if run.algorithm == "ga":
        rows["population"] = evo.get("population_size")
        rows["diversity mechanism"] = (evo.get("diversity") or {}).get("method", "n/a")
    else:  # MAP-Elites ignores the GA's population and diversity settings
        me = run.config.get("map_elites") or {}
        spec = run.archive_spec
        rows["batch / initial random genomes"] = f"{me.get('batch_size')} / {me.get('init_size')}"
        rows["archive axes"] = " x ".join(
            f"{d} ({b} bins, {lo:g}-{hi:g})"
            for d, b, (lo, hi) in zip(spec.descriptors, spec.bins, spec.ranges, strict=True)
        )
    left, right = st.columns(2)
    with left:
        st.markdown("**Run settings**")
        st.table(_kv(rows).set_index("setting"))
    with right:
        st.markdown("**Mutation settings** (fixed for the whole run)")
        st.table(_kv(mut).set_index("setting"))


def _kv(d: dict) -> pd.DataFrame:
    return pd.DataFrame({"setting": list(d), "value": [str(v) for v in d.values()]})


def _strip(ctx: Ctx) -> None:
    hist = dd.champion_history(ctx.run)
    st.subheader("How the champion changed")
    if hist.empty:
        st.info("No champion yet: the first generation has not finished.")
        return
    changes = hist[hist["changed"]]
    st.caption(
        f"The champion (best creature found so far) changed {len(changes)} time(s) in "
        f"{len(hist)} steps. Each picture is that genome rendered at rest after settling on "
        f"{ctx.run.eval_runs[0][0].label()}; open any of them in the inspector below."
    )
    if len(changes) > MAX_STRIP:
        pick = np.unique(np.linspace(0, len(changes) - 1, MAX_STRIP).round().astype(int))
        changes = changes.iloc[pick]
        st.caption(f"Showing {len(changes)} evenly spaced changes, including the latest.")
    records, caps = [], []
    for _, row in changes.iterrows():
        rec = ctx.record(str(row["champion_id"]))
        if rec is not None:
            records.append(rec)
            caps.append(f"step {int(row['generation'])} · {rec['id']} · {rec['fitness']:+.3f}")
    images(ctx, records, "thumb", caps)
    if records:
        pick = st.selectbox(
            "Inspect a champion", [r["id"] for r in records], index=len(records) - 1
        )
        st.button(
            "Open in creature inspection",
            on_click=select_creature,
            args=(ctx.entry.name, pick, "Creature inspection"),
            key="strip_open",
        )


def render(ctx: Ctx) -> None:
    run = ctx.run
    st.header(f"Evolution viewer: {run.name}")
    status = run.status
    (st.success if status == "complete" else st.info)(f"Run status: {status}.")
    for note in run.notes:
        st.warning(note)
    _tiles(ctx)
    _settings(ctx)
    _strip(ctx)
    summary = run.path / "summary.md"
    if summary.is_file():
        with st.expander("summary.md written by the run"):
            st.markdown(summary.read_text())
