"""Streamlit glue shared by the dashboard pages: cached loading, selection, images.

Caching: every loader takes the run's :func:`signature` as an argument, so a
run that writes a new generation gets fresh data on the next rerun while a
finished run is read from disk once. The individual index is a
``cache_resource`` (shared, not copied on every rerun) because a 6000-
simulation run holds a few thousand records; pages treat it as read-only.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import streamlit as st

from evomorph.visualization import dashboard_analysis as an
from evomorph.visualization import dashboard_data as dd
from evomorph.visualization import dashboard_media as media
from evomorph.visualization.dashboard_describe import short_label

DEFAULT_RESULTS = "results"
SELECTED = "creature"  # session key: {"run": run name, "id": genome id}
SECTION = "section"  # session key of the section radio


def results_root() -> Path:
    """The folder searched for runs: the sidebar field, else $EVOMORPH_RESULTS, else results/."""
    chosen = st.session_state.get("results_root")
    return Path(chosen or os.environ.get("EVOMORPH_RESULTS") or DEFAULT_RESULTS)


@st.cache_data(max_entries=32, show_spinner=False)
def get_run(path: str, name: str, sig: tuple) -> dd.Run:
    return dd.load_run(dd.RunEntry(Path(path), name))


@st.cache_resource(max_entries=4, show_spinner=False)
def get_index(path: str, name: str, sig: tuple) -> dd.Index:
    return dd.build_index(get_run(path, name, sig))


@st.cache_data(max_entries=8, show_spinner=False)
def get_archive(path: str, name: str, sig: tuple) -> an.ArchiveView:
    return an.load_archive(get_run(path, name, sig), get_index(path, name, sig))


@st.cache_data(max_entries=8, show_spinner=False)
def get_stats(path: str, name: str, sig: tuple) -> dict[str, Any]:
    run, idx = get_run(path, name, sig), get_index(path, name, sig)
    return {
        "steps": an.generation_stats(run, idx),
        "samples": an.fitness_samples(run, idx),
        "plans": an.plan_counts(run, idx),
    }


class Ctx:
    """The run a page works on, with lazy access to its index and archive."""

    def __init__(self, entry: dd.RunEntry, runs: list[dd.RunEntry]) -> None:
        self.entry, self.runs = entry, runs
        self.key = (str(entry.path), entry.name, dd.signature(entry.path))
        self.run = get_run(*self.key)

    @property
    def index(self) -> dd.Index:
        return get_index(*self.key)

    @property
    def archive(self) -> an.ArchiveView:
        return get_archive(*self.key)

    @property
    def stats(self) -> dict[str, Any]:
        return get_stats(*self.key)

    @property
    def cache(self) -> Path:
        return media.cache_dir(results_root())

    def record(self, genome_id: str) -> dict[str, Any] | None:
        return self.index.records.get(genome_id)

    def target(self, record: dict[str, Any]) -> media.Target:
        """Render target: the creature in the run's first evaluation environment."""
        env, seed = self.run.eval_runs[0]
        return media.Target(
            genome=record["genome"],
            env=env,
            seed=seed,
            sim=dict(self.run.config.get("sim") or {}),
            fitness=dict(self.run.config.get("fitness") or {}),
        )


def other_ctx(ctx: Ctx, name: str) -> Ctx:
    """Context of another discovered run (for cross-run comparisons)."""
    entry = next(e for e in ctx.runs if e.name == name)
    return ctx if entry.name == ctx.entry.name else Ctx(entry, ctx.runs)


def select_creature(run: str, genome_id: str, go_to: str | None = None) -> None:
    """Button callback: remember a creature and optionally switch section."""
    st.session_state[SELECTED] = {"run": run, "id": genome_id}
    st.session_state["run_pick"] = run
    st.session_state.pop("inspect_pick", None)
    if go_to:
        st.session_state[SECTION] = go_to


def images(ctx: Ctx, records: list[dict[str, Any]], kind: str, captions: list[str]) -> None:
    """Render (cached) and show ``kind`` images of ``records`` in a grid."""
    if not records:
        return
    targets = [ctx.target(r) for r in records]
    missing = [t for t in targets if media.cached(ctx.cache, t, kind) is None]
    if missing:
        with st.spinner(f"Rendering {len(missing)} creature image(s) offscreen..."):
            report = media.render(ctx.cache, targets, kind)
        for err in report.errors[:3]:
            st.warning(f"A picture could not be made ({err}).")
    per_row = 1 if kind == "still" else min(4, len(records))
    cols = st.columns(per_row)
    for i, (t, cap) in enumerate(zip(targets, captions, strict=True)):
        path = media.cached(ctx.cache, t, kind)
        with cols[i % per_row]:
            if path is None:
                st.caption(f"{cap}\n\n(no picture: the body could not be rendered)")
            else:
                st.image(str(path), caption=cap, width="stretch")


ROW_PX = 35  # Streamlit's dataframe row height
MAX_TABLE_PX = 1300  # about 36 rows; longer tables scroll inside themselves


def table(df, **kwargs) -> None:
    """A dataframe tall enough to show every row (up to MAX_TABLE_PX), so nothing hides."""
    height = min(ROW_PX * (len(df) + 1) + 3, MAX_TABLE_PX)
    st.dataframe(df, width="stretch", height=height, **kwargs)


def chart(fig, key: str, **kwargs) -> Any:
    """Plotly chart with our own palette (theme=None keeps Streamlit from recolouring it)."""
    return st.plotly_chart(fig, width="stretch", key=key, theme=None, **kwargs)


def friendly(problem: Exception) -> None:
    st.error(str(problem))
    st.caption("Other runs can still be opened from the sidebar.")


def caption_for(record: dict[str, Any]) -> str:
    return short_label(record)
