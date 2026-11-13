"""EVOMORPH research dashboard (brief section 10).

Launch from the repository root::

    streamlit run evomorph/visualization/dashboard.py
    streamlit run evomorph/visualization/dashboard.py -- --results path/to/results

It reads run directories written by ``python -m evomorph evolve`` (and folders
of them, such as experiment directories) under ``results/``. Every number and
picture on it comes from a saved genome and its saved or re-simulated results;
rendering is offscreen and cached (see ``dashboard_media``). Sections:

A. Evolution viewer  B. Creature inspection  C. MAP-Elites archive
D. Statistics        E. Comparison           F. Generation history

The page can be opened on a given run and section with query parameters,
e.g. ``?run=quick_ga_joint_s0&section=archive``. Guide: docs/DASHBOARD.md.
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")  # never open a plotting window, whatever a helper imports

import streamlit as st

from evomorph.visualization import dashboard_data as dd
from evomorph.visualization import (
    dashboard_page_archive,
    dashboard_page_compare,
    dashboard_page_creature,
    dashboard_page_evolution,
    dashboard_page_history,
    dashboard_page_stats,
)
from evomorph.visualization.dashboard_common import (
    DEFAULT_RESULTS,
    SECTION,
    Ctx,
    friendly,
    results_root,
)

PAGES = {
    "Evolution viewer": dashboard_page_evolution.render,
    "Creature inspection": dashboard_page_creature.render,
    "MAP-Elites archive": dashboard_page_archive.render,
    "Statistics": dashboard_page_stats.render,
    "Comparison": dashboard_page_compare.render,
    "Generation history": dashboard_page_history.render,
}
# Short names accepted in the ?section= query parameter.
SLUGS = {
    "evolution": "Evolution viewer",
    "creature": "Creature inspection",
    "archive": "MAP-Elites archive",
    "stats": "Statistics",
    "compare": "Comparison",
    "history": "Generation history",
}


def _cli_results() -> str | None:
    """``--results DIR`` passed after ``--`` on the streamlit command line."""
    args = sys.argv[1:]
    if "--results" in args and args.index("--results") + 1 < len(args):
        return args[args.index("--results") + 1]
    return None


def _first_visit() -> None:
    """Apply the command line and query parameters once per browser session."""
    if st.session_state.get("_started"):
        return
    st.session_state["_started"] = True
    qp = st.query_params
    root = qp.get("results") or _cli_results() or os.environ.get("EVOMORPH_RESULTS")
    st.session_state["results_root"] = root or DEFAULT_RESULTS
    if qp.get("section") in SLUGS:
        st.session_state[SECTION] = SLUGS[qp["section"]]
    if qp.get("run"):
        st.session_state["run_pick"] = qp["run"]


def _empty(root: str) -> None:
    where = f"`{root}`"
    if not os.path.isdir(root):
        st.info(
            f"There is no results folder at {where} yet. Run an evolution first, for example\n\n"
            "`python -m evomorph evolve --config configs/quick.yaml`\n\n"
            "and its run folder will appear here. You can also type another folder in the "
            "sidebar."
        )
    else:
        st.info(
            f"No runs found in {where}. Each run that `python -m evomorph evolve` starts "
            "gets its own folder here; start one and it appears in the sidebar, even while "
            "it is still running."
        )


def _sidebar(runs: list[dd.RunEntry]) -> dd.RunEntry | None:
    st.sidebar.title("EVOMORPH")
    st.sidebar.caption("Evolutionary robotics lab: bodies and gaits that evolved, not designed.")
    st.sidebar.text_input("Results folder", key="results_root")
    if not runs:
        return None
    names = [r.name for r in runs]
    if st.session_state.get("run_pick") not in names:
        st.session_state["run_pick"] = names[-1]
    pick = st.sidebar.selectbox("Run", names, key="run_pick")
    st.sidebar.radio("Section", list(PAGES), key=SECTION)
    st.sidebar.caption(
        "Everything shown is read from the run folder or re-simulated from a saved genome. "
        "Pictures and clips are rendered offscreen and cached."
    )
    if st.sidebar.button("Reload from disk"):
        st.cache_data.clear()
        st.cache_resource.clear()
    return next(r for r in runs if r.name == pick)


def main() -> None:
    st.set_page_config(page_title="EVOMORPH lab", page_icon=":material/biotech:", layout="wide")
    _first_visit()
    root = str(results_root())
    runs = dd.discover_runs(root)
    entry = _sidebar(runs)
    if entry is None:
        st.title("EVOMORPH research dashboard")
        _empty(root)
        return
    section = st.session_state.get(SECTION) or "Evolution viewer"
    try:
        ctx = Ctx(entry, runs)
        PAGES[section](ctx)
    except dd.RunReadError as exc:
        friendly(exc)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(
            f"Part of the run '{entry.name}' could not be read, so this section cannot be "
            "shown. The folder may be damaged or still being written; try Reload in a moment."
        )
        with st.expander("Technical details"):
            st.code(f"{type(exc).__name__}: {exc}")


main()
