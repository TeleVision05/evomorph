"""The Streamlit dashboard, run headless with AppTest on the fixture runs.

Rendering is real (offscreen, in a child process) and cached in one folder
for the whole session, so only the first test pays for the pictures. The
interactive-viewer button is never pressed here (it would open a window);
its command is tested in test_dashboard_data.py with Popen replaced.
"""

from __future__ import annotations

import json

import pytest
from dashboard_helpers import APP, RUN_NAMES, copy_runs, make_in_progress
from streamlit.testing.v1 import AppTest

SECTIONS = (
    "Evolution viewer",
    "Creature inspection",
    "MAP-Elites archive",
    "Statistics",
    "Comparison",
    "Generation history",
)
TIMEOUT = 240  # s; the first run renders every picture it shows


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("dashboard_cache")


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    return copy_runs(tmp_path_factory.mktemp("results"))


def _app(monkeypatch, root, cache) -> AppTest:
    monkeypatch.setenv("EVOMORPH_RESULTS", str(root))
    monkeypatch.setenv("EVOMORPH_DASHBOARD_CACHE", str(cache))
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    return at


def _open(at: AppTest, run: str, section: str) -> AppTest:
    at.selectbox(key="run_pick").set_value(run).run()
    at.radio(key="section").set_value(section).run()
    return at


def _clean(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


@pytest.mark.parametrize("run", RUN_NAMES)
@pytest.mark.parametrize("section", SECTIONS)
def test_every_section_renders(monkeypatch, results, cache_dir, run, section):
    at = _open(_app(monkeypatch, results, cache_dir), run, section)
    _clean(at)
    assert any(section.split()[0].lower() in h.value.lower() for h in at.header)


def test_inspector_shows_the_selected_creature(monkeypatch, results, cache_dir):
    at = _open(_app(monkeypatch, results, cache_dir), "ga_tiny", "Creature inspection")
    pick = at.selectbox(key="inspect_pick")
    other = pick.options[-1].split(" · ")[0]  # the worst creature, not the default champion
    assert other != pick.value
    pick.set_value(other).run()
    _clean(at)
    shown = " ".join(m.value for m in at.markdown)
    assert f"**{other}**" in shown
    assert at.session_state["creature"] == {"run": "ga_tiny", "id": other}
    assert any(df.value.shape[0] > 0 for df in at.dataframe)


@pytest.mark.gl
def test_clip_button_renders_a_real_clip(monkeypatch, results, tmp_path):
    at = _open(_app(monkeypatch, results, tmp_path), "ga_tiny", "Creature inspection")
    assert not at.get("video")
    at.button(key="clip_button").click().run()
    _clean(at)
    assert len(at.get("video")) == 1
    assert any("(identical)" in c.value for c in at.caption)  # re-simulation matches


def test_selecting_an_archive_cell_opens_that_elite(monkeypatch, results, cache_dir):
    at = _open(_app(monkeypatch, results, cache_dir), "me_tiny", "MAP-Elites archive")
    cells = at.selectbox(key="archive_cell")
    second = cells.options[1]
    cells.set_value(second).run()
    _clean(at)
    elite_id = second.split(" · ")[1]
    at.button(key="archive_open").click().run()
    _clean(at)
    assert at.radio(key="section").value == "Creature inspection"
    assert at.selectbox(key="inspect_pick").value == elite_id
    assert any(f"**{elite_id}**" in m.value for m in at.markdown)


def test_generation_history_browses_steps(monkeypatch, results, cache_dir):
    at = _open(_app(monkeypatch, results, cache_dir), "ga_tiny", "Generation history")
    at.select_slider(key="history_step").set_value(0).run()
    _clean(at)
    first = at.selectbox(key="history_pick").options[0]
    at.button(key="history_open").click().run()
    assert at.selectbox(key="inspect_pick").value == first


def test_comparison_puts_creatures_side_by_side(monkeypatch, results, cache_dir):
    at = _open(_app(monkeypatch, results, cache_dir), "ga_tiny", "Comparison")
    pick = at.multiselect(key="compare_pick")
    pick.set_value(pick.options[:3]).run()
    _clean(at)
    table = at.dataframe[0].value
    assert table.shape[1] == 3 and "forward_displacement" in table.index


def test_empty_results_folder(monkeypatch, tmp_path, cache_dir):
    at = _app(monkeypatch, tmp_path, cache_dir)
    _clean(at)
    assert "No runs found" in at.info[0].value
    at = _app(monkeypatch, tmp_path / "nowhere", cache_dir)
    assert "There is no results folder" in at.info[0].value


def test_damaged_run_gets_a_plain_message(monkeypatch, tmp_path, cache_dir):
    copy_runs(tmp_path, ("ga_tiny", "me_tiny"))
    (tmp_path / "ga_tiny" / "config.yaml").write_text("sim: [unclosed")
    at = _open(_app(monkeypatch, tmp_path, cache_dir), "ga_tiny", "Evolution viewer")
    assert not at.exception
    assert "is damaged and cannot be read" in at.error[0].value
    assert "Traceback" not in at.error[0].value
    at = _open(at, "me_tiny", "Evolution viewer")  # the other run still works
    _clean(at)


def test_damaged_generation_file(monkeypatch, tmp_path, cache_dir):
    copy_runs(tmp_path, ("me_tiny",))
    (tmp_path / "me_tiny" / "generations" / "gen_0001.json").write_text("[{")
    at = _open(_app(monkeypatch, tmp_path, cache_dir), "me_tiny", "Creature inspection")
    assert not at.exception
    assert "generation 1" in at.error[0].value and "damaged" in at.error[0].value


@pytest.mark.parametrize("section", SECTIONS)
def test_run_in_progress(monkeypatch, tmp_path, cache_dir, section):
    copy_runs(tmp_path, ("ga_tiny",))
    make_in_progress(tmp_path / "ga_tiny", keep_steps=2)
    at = _open(_app(monkeypatch, tmp_path, cache_dir), "ga_tiny", section)
    _clean(at)
    if section == "Evolution viewer":
        assert "in progress" in at.info[0].value
        assert any("could not be read" in w.value for w in at.warning)


def test_run_that_has_only_just_started(monkeypatch, tmp_path, cache_dir):
    copy_runs(tmp_path, ("me_tiny",))
    run = tmp_path / "me_tiny"
    config = (run / "config.yaml").read_text()
    for child in list(run.iterdir()):
        if child.is_dir():
            for p in sorted(child.rglob("*"), reverse=True):
                p.rmdir() if p.is_dir() else p.unlink()
            child.rmdir()
        else:
            child.unlink()
    (run / "config.yaml").write_text(config)
    for section in SECTIONS:
        at = _open(_app(monkeypatch, tmp_path, cache_dir), "me_tiny", section)
        _clean(at)
    assert json.dumps(at.session_state["run_pick"]) == '"me_tiny"'
