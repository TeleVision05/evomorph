"""Section D, evolution statistics: fitness, diversity, success, archive, efficiency."""

from __future__ import annotations

import streamlit as st

from evomorph.visualization import dashboard_plots as plots
from evomorph.visualization.dashboard_analysis import SUCCESS_DISTANCE
from evomorph.visualization.dashboard_common import Ctx, chart


def _chart(fig, key: str) -> None:
    chart(fig, key)


def _fitness(ctx: Ctx, x: str) -> None:
    m = ctx.run.metrics
    what = "population" if ctx.run.algorithm == "ga" else "archive elites"
    a, b = st.columns(2)
    with a:
        _chart(plots.fitness_progress(m, x), "st_fitness")
        st.caption(f"From metrics.csv: the current set is the {what} after each step.")
    with b:
        _chart(plots.fitness_distribution(ctx.stats["samples"]), "st_dist")
        st.caption("Every valid creature evaluated in a step (from the generation files).")


def _diversity(ctx: Ctx, x: str) -> None:
    m = ctx.run.metrics
    a, b = st.columns(2)
    with a:
        fig = plots.lines(
            m, x, ["mean_gene_distance"], "Morphological and controller diversity",
            "mean pairwise gene distance (0-1)",
        )  # fmt: skip
        _chart(fig, "st_genedist")
        st.caption("Random genomes average about 0.32 apart (engine measurement, seed 0).")
    with b:
        _chart(plots.plan_area(ctx.stats["plans"]), "st_plans")


def _success(ctx: Ctx, x: str) -> None:
    steps = ctx.stats["steps"]
    sx = x if x in steps.columns else "generation"
    a, b = st.columns(2)
    with a:
        fig = plots.lines(
            steps, sx, ["successful", "fell"], "Successful and fallen creatures per step",
            "creatures evaluated in the step",
        )  # fmt: skip
        _chart(fig, "st_success")
        st.caption(
            f"Successful = valid, never fell, and moved at least {SUCCESS_DISTANCE:g} m forward "
            "on average (a reporting threshold; selection never sees it)."
        )
    with b:
        m = ctx.run.metrics
        _chart(
            plots.lines(m, x, ["archive_coverage"], "Archive occupancy", "share of cells filled"),
            "st_cov",
        )


def _efficiency(ctx: Ctx, x: str) -> None:
    steps = ctx.stats["steps"]
    sx = x if x in steps.columns else "generation"
    a, b = st.columns(2)
    with a:
        fig = plots.lines(
            steps, sx, ["mean_energy_efficiency"], "Energy efficiency",
            "mean 1 / (1 + cost of transport)",
        )  # fmt: skip
        _chart(fig, "st_eff")
    with b:
        fig = plots.lines(
            steps, sx, ["mean_forward_displacement"], "Average displacement",
            "mean forward displacement (m)",
        )  # fmt: skip
        _chart(fig, "st_disp")
    st.caption("Both are means over the creatures evaluated in each step.")


def render(ctx: Ctx) -> None:
    st.header("Evolution statistics")
    if ctx.run.metrics.empty:
        st.info("No statistics yet: the first generation has not finished.")
        return
    choice = st.radio(
        "Horizontal axis",
        ["simulations used", "generation / batch"],
        horizontal=True,
        key="stats_axis",
        help="Simulations are the budget unit, so runs are compared at equal cost.",
    )
    x = "sims" if choice == "simulations used" else "generation"
    _fitness(ctx, x)
    _diversity(ctx, x)
    _success(ctx, x)
    _efficiency(ctx, x)
    a, b = st.columns(2)
    with a:
        m = ctx.run.metrics
        _chart(plots.lines(m, x, ["qd_score"], "QD-score", "sum of (fitness + offset)"), "st_qd")
    with b:
        _chart(plots.lines(m, x, ["n_body_plans"], "Distinct body plans", "plans"), "st_np")
