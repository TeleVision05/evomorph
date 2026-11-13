"""Section B, creature inspection: body, controller, metrics, lineage, clip and viewer."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from evomorph.visualization import dashboard_data as dd
from evomorph.visualization import dashboard_describe as desc
from evomorph.visualization import dashboard_media as media
from evomorph.visualization.dashboard_common import SELECTED, Ctx, images
from evomorph.visualization.dashboard_plots import lineage_figure

LINEAGE_DEPTH = 12  # generations back; enough to see a quick run's whole ancestry


def _default_id(ctx: Ctx, ids: list[str]) -> str | None:
    chosen = st.session_state.get(SELECTED) or {}
    if chosen.get("run") == ctx.entry.name and chosen.get("id") in ctx.index.records:
        return chosen["id"]
    hist = dd.champion_history(ctx.run)
    if not hist.empty and str(hist.iloc[-1]["champion_id"]) in ctx.index.records:
        return str(hist.iloc[-1]["champion_id"])
    return ids[0] if ids else None


def _pick(ctx: Ctx) -> dict[str, Any] | None:
    recs = ctx.index.records
    ids = sorted(recs, key=lambda g: (not recs[g].get("valid"), -float(recs[g]["fitness"])))
    if not ids:
        st.info("No creature has been recorded in this run yet.")
        return None
    default = _default_id(ctx, ids)
    gid = st.selectbox(
        f"Creature ({len(ids)} recorded, best first)",
        ids,
        index=ids.index(default) if default in ids else 0,
        format_func=lambda g: desc.short_label(recs[g]),
        key="inspect_pick",
    )
    st.session_state[SELECTED] = {"run": ctx.entry.name, "id": gid}
    return recs[gid]


def _identity(rec: dict[str, Any]) -> None:
    parents = ", ".join(rec.get("parent_ids") or []) or "none (made at random or a reference)"
    status = "valid" if rec.get("valid") else f"invalid: {rec.get('failure_reason') or '?'}"
    st.markdown(
        f"**{rec['id']}** · generation of origin **{rec.get('generation')}** · origin "
        f"**{rec.get('origin', '?')}** · parents: {parents} · {status}"
    )


def _body(ctx: Ctx, rec: dict[str, Any]) -> None:
    genome = dd.record_genome(rec)
    env = ctx.run.eval_runs[0][0]
    images(ctx, [rec], "still", [f"{rec['id']} at rest on {env.label()}: iso, side, front, top"])
    facts = desc.body_facts(genome)
    cols = st.columns(4)
    for i, (k, v) in enumerate(facts.items()):
        cols[i % 4].metric(k, v)
    left, right = st.columns(2)
    with left:
        st.markdown("**Body proportions (morphology genes)**")
        st.dataframe(desc.morphology_table(genome), hide_index=True, width="stretch")
    with right:
        st.markdown("**Controller parameters**")
        st.dataframe(desc.controller_table(genome), hide_index=True, width="stretch")
        st.markdown("**Joints and per-limb controller genes**")
        st.dataframe(desc.joint_table(genome), hide_index=True, width="stretch")


def _scores(rec: dict[str, Any]) -> None:
    fit = rec.get("fitness")
    st.markdown(f"**Fitness {fit:+.4f}**" if rec.get("valid") else "**Invalid evaluation**")
    a, b = st.columns([1, 2])
    with a:
        st.markdown("Per evaluation run")
        st.dataframe(desc.run_table(rec), hide_index=True, width="stretch")
        st.markdown("Fitness terms (they sum to the fitness)")
        st.dataframe(desc.terms_table(rec), hide_index=True, width="stretch")
    with b:
        st.markdown("Every metric (mean over the runs)")
        st.dataframe(desc.metric_table(rec), hide_index=True, width="stretch", height=420)


def _lineage(ctx: Ctx, rec: dict[str, Any]) -> None:
    rows = dd.lineage(ctx.index, rec["id"], LINEAGE_DEPTH)
    known = [r for r in rows if r.get("recorded")]
    if len(known) > 1:
        st.plotly_chart(lineage_figure(rows), width="stretch", key="lineage_chart")
    table = pd.DataFrame(
        [
            {
                "generations back": r["depth"],
                "id": r["id"],
                "generation": r.get("generation"),
                "origin": r.get("origin", "not recorded in this run"),
                "fitness": r.get("fitness"),
                "parents": ", ".join(r.get("parent_ids") or []),
            }
            for r in rows
        ]
    )
    st.dataframe(table, hide_index=True, width="stretch")


def _clip(ctx: Ctx, rec: dict[str, Any]) -> None:
    target = ctx.target(rec)
    path = media.cached(ctx.cache, target, "clip")
    label = f"{target.env.label()}, spawn seed {target.seed}"
    if path is None and st.button(f"Render a clip ({label})", key="clip_button"):
        with st.spinner("Re-simulating and recording offscreen (a few seconds)..."):
            report = media.render(ctx.cache, [target], "clip")
        path = media.cached(ctx.cache, target, "clip")
        for err in report.errors:
            st.warning(f"The clip could not be made ({err}).")
    if path is None:
        st.caption("No clip rendered yet. Clips are cached, so each is made only once.")
        return
    st.video(str(path))
    meta = media.clip_meta(path) or {}
    stored = (rec.get("seed_fitness") or [None])[0]
    again = meta.get("fitness")
    same = isinstance(stored, float) and again == stored
    st.caption(
        f"Clip of the first evaluation run ({label}), re-simulated from the genome: "
        f"fitness {again:+.4f} vs stored {stored:+.4f}"
        + (" (identical)." if same else " (different: see docs/DASHBOARD.md).")
        if isinstance(again, float) and isinstance(stored, float)
        else "Clip re-simulated from the genome."
    )


def _viewer(ctx: Ctx, rec: dict[str, Any]) -> None:
    env, seed = ctx.run.eval_runs[0]
    gfile = media.genome_file(ctx.cache, rec["genome"])
    cmd = media.viewer_command(gfile, env, seed, ctx.run.path / "config.yaml")
    if st.button("Open the interactive MuJoCo viewer (new window)", key="viewer_button"):
        media.launch_viewer(cmd)
        st.success("The viewer is starting in its own window. Close that window when done.")
    with st.expander("Command the button runs"):
        st.code(" ".join(cmd), language="bash")


def render(ctx: Ctx) -> None:
    st.header("Creature inspection")
    rec = _pick(ctx)
    if rec is None:
        return
    _identity(rec)
    try:
        _body(ctx, rec)
    except dd.RunReadError as exc:
        st.error(str(exc))
        return
    st.subheader("Fitness and metrics")
    _scores(rec)
    st.subheader("Lineage")
    _lineage(ctx, rec)
    st.subheader("Watch it move")
    _clip(ctx, rec)
    _viewer(ctx, rec)
