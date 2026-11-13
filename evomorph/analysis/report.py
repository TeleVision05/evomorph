"""``summary.md`` and ``stats.csv`` of an experiment, and the analysis section of a run.

The verdict lines are computed, not written by hand: a primary comparison is
called significant only when its Holm-adjusted p is below 0.05, and the
hypothesis is called supported only if C beats A *and* B on *both* primary
measures that way. Anything else is reported as not shown, with the effect
size, so a null or a reversed result reads as plainly as a positive one.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from evomorph.analysis.load import ExperimentData
from evomorph.analysis.stats import (
    ALPHA,
    MEASURES,
    Comparison,
    all_comparisons,
    bootstrap_median_ci,
    clean,
    evidence_note,
)

LETTER = {"controller": "A", "morphology": "B", "joint": "C", "joint_me": "D"}
SUMMARY_MEASURES = (
    "champ_test_fitness",
    "champ_train_fitness",
    "champ_gap",
    "set_test_fitness",
    "set_gap",
    "champ_test_speed",
    "champ_test_efficiency",
    "champ_test_tilt",
    "champ_test_failure_rate",
    "final_morph_diversity",
    "set_test_behaviour_diversity",
    "final_body_plans",
)
START, END = "<!-- analyze:start -->", "<!-- analyze:end -->"


def _fmt(x: float, digits: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}g}"


def _median_ci(values: Sequence[float]) -> str:
    v = clean(values)
    if len(v) == 0:
        return "n/a"
    lo, hi = bootstrap_median_ci(v)
    ci = "" if math.isnan(lo) else f" [{_fmt(lo)}, {_fmt(hi)}]"
    return f"{_fmt(float(np.median(v)))}{ci}"


def _condition_table(data: ExperimentData, conds: list[str]) -> list[str]:
    labels = {m.column: m.label for m in MEASURES}
    head = "| measure | " + " | ".join(f"{data.label(c)} (n={len(data.runs_of(c))})" for c in conds)
    lines = [head + " |", "|---" * (len(conds) + 1) + "|"]
    for col in SUMMARY_MEASURES:
        cells = [_median_ci([r[col] for r in data.rows if r["condition"] == c]) for c in conds]
        lines.append(f"| {labels[col]} | " + " | ".join(cells) + " |")
    return lines


def _per_seed_table(data: ExperimentData, conds: list[str]) -> list[str]:
    lines = ["| condition | seed | champion | plan | selected | train (fresh) | held-out | gap |"]
    lines.append("|---|---|---|---|---|---|---|---|")
    for c in conds:
        for r in (r for r in data.rows if r["condition"] == c):
            lines.append(
                f"| {LETTER.get(c, c)} | {r['seed']} | {r['champion_id']} | {r['champion_plan']} | "
                f"{r['champ_selected']:+.3f} | {r['champ_train_fitness']:+.3f} | "
                f"{r['champ_test_fitness']:+.3f} | {r['champ_gap']:+.3f} |"
            )
    return lines


def _comparison_lines(comps: Sequence[Comparison]) -> list[str]:
    lines = ["| measure | comparison | n | medians | A12 (effect) | p | p (Holm) |"]
    lines.append("|---|---|---|---|---|---|---|")
    for c in comps:
        tag = "**primary** " if c.measure.primary else ""
        pair = f"{LETTER.get(c.first, c.first)} vs {LETTER.get(c.second, c.second)}"
        lines.append(
            f"| {tag}{c.measure.label} | {pair} | {c.n_first} vs {c.n_second} | "
            f"{_fmt(c.median_first)} vs {_fmt(c.median_second)} | {_fmt(c.a12, 2)} ({c.effect}) | "
            f"{_fmt(c.p, 2)} | {_fmt(c.p_holm, 2) if c.measure.primary else 'secondary'} |"
        )
    return lines


def _verdicts(comps: Sequence[Comparison]) -> list[str]:
    out = []
    shown: dict[tuple[str, str, str], bool] = {}
    for c in comps:
        if not c.measure.primary:
            continue
        sig = not math.isnan(c.p_holm) and c.p_holm < ALPHA
        better = not math.isnan(c.a12) and c.a12 > 0.5
        shown[(c.first, c.second, c.measure.column)] = sig and better
        first, second = LETTER.get(c.first, c.first), LETTER.get(c.second, c.second)
        if math.isnan(c.a12):
            lean = "no data"
        elif c.a12 == 0.5:
            lean = "no lean either way"
        else:
            lean = f"leans {first if better else second}"
        status = "significant" if sig else "not significant"
        out.append(
            f"- {first} vs {second}, {c.measure.label}: a run of {first} beats a run of {second} "
            f"in {100 * c.a12:.0f}% of seed pairings, ties half (A12 {_fmt(c.a12, 2)}, {c.effect}, "
            f"{lean}); Holm p = {_fmt(c.p_holm, 2)}, {status} at {ALPHA}."
        )
    keys = [(a, b, m) for a, b in (("joint", "controller"), ("joint", "morphology"))
            for m in ("champ_test_fitness", "set_test_behaviour_diversity")]  # fmt: skip
    supported = all(shown.get(k, False) for k in keys)
    out.append("")
    out.append(
        "**Hypothesis (C more robust and more behaviourally diverse than A and B): "
        + ("SUPPORTED by this data.**" if supported else "NOT SHOWN by this data.**")
        + " It counts as supported only if C beats both A and B on both primary measures "
        "with Holm-adjusted p < 0.05."
    )
    return out


def write_stats_csv(comps: Sequence[Comparison], path: Path) -> Path:
    cols = ("measure", "primary", "first", "second", "n_first", "n_second", "median_first",
            "median_second", "a12", "effect", "p", "p_holm")  # fmt: skip
    with path.open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(cols)
        for c in comps:
            w.writerow([c.measure.column, c.measure.primary, c.first, c.second, c.n_first,
                        c.n_second, repr(c.median_first), repr(c.median_second), repr(c.a12),
                        c.effect, repr(c.p), repr(c.p_holm)])  # fmt: skip
    return path


def _design_lines(data: ExperimentData) -> list[str]:
    m = data.manifest
    exp = m["design"]["experiment"]
    inv = m["invocations"]
    conds = [
        "{} ({}, mode {})".format(c["label"], c["algorithm"], c["mode"]) for c in data.conditions
    ]
    lines = [
        f"- conditions: {'; '.join(conds)}",
        f"- budget {m['budget']} simulations per run ({m['sims_per_genome']} per genome: every "
        f"environment of the '{exp['train_suite']}' suite once); seeds {m['seeds']}",
        f"- held-out suite '{exp['test_suite']}'; champion re-scored with {exp['rescore_seeds']} "
        f"fresh spawn seeds per environment; best-{exp['set_size']} final set with "
        f"{exp['set_seeds']} fresh seed(s)",
        f"- design hash {m['design_hash']}; git {m['versions']['git_commit']}, MuJoCo "
        f"{m['versions']['mujoco']}, Python {m['versions']['python']}",
        f"- command: `{inv[-1]['command'] if inv else '?'}`",
    ]
    return lines


def _throughput_lines(data: ExperimentData) -> list[str]:
    runs = data.manifest["runs"].values()
    search = [(r.get("search_seconds"), r["sims_used"]) for r in runs if r.get("search_seconds")]
    post = [(r["posthoc_seconds"], r["posthoc_sims"]) for r in runs if "posthoc_seconds" in r]
    if not search:
        return ["No timed search in the manifest."]
    s_sec, s_sims = sum(t for t, _ in search), sum(n for _, n in search)
    p_sec, p_sims = sum(t for t, _ in post), sum(n for _, n in post)
    return [
        f"- search: {s_sims} simulations in {s_sec:.0f} s = {s_sims / s_sec:.2f} simulations/s "
        f"({len(search)} timed runs; wall time includes the champion's re-score and "
        "held-out check)",
        f"- post-hoc evaluation: {p_sims} simulations in {p_sec:.0f} s = "
        f"{p_sims / max(p_sec, 1e-9):.2f} simulations/s",
        f"- total wall time of the timed runs: {(s_sec + p_sec) / 60:.1f} min",
    ]


def experiment_summary(data: ExperimentData, plots: Sequence[Path], out: Path) -> Path:
    """Write ``summary.md`` and ``stats.csv`` for an experiment folder."""
    conds = [
        c["key"] for c in data.conditions if any(r["condition"] == c["key"] for r in data.rows)
    ]
    comps = all_comparisons(data.rows, conds)
    write_stats_csv(comps, data.path / "stats.csv")
    n = [len(data.runs_of(c)) for c in conds]
    lines = [
        f"# Comparison experiment: {data.path.name}",
        "",
        *_design_lines(data),
        "",
        "## How strong is the evidence?",
        "",
        evidence_note(n),
        "",
        "## Verdict on the primary measures",
        "",
        *_verdicts(comps),
        "",
        "## Per condition: median [bootstrap 95% CI] over seeds",
        "",
        "Fitness is dimensionless (about 1 = clean walking at 0.2 m/s). Tilt in radians. "
        "Intervals need at least 3 seeds.",
        "",
        *_condition_table(data, conds),
        "",
        "## Champions, per seed",
        "",
        "'selected' is the training fitness the champion was picked with; 'train (fresh)' "
        "re-scores it on the same terrains with new seeds; the gap is held-out minus fresh train.",
        "",
        *_per_seed_table(data, conds),
        "",
        "## Key comparisons (Mann-Whitney U, two-sided; A12 = P(first better))",
        "",
        *_comparison_lines(comps),
        "",
        "## Throughput",
        "",
        *_throughput_lines(data),
        "",
        "## Plots",
        "",
        *[f"- [{p.name}](plots/{p.name})" for p in plots],
        "",
        "Per-run numbers: results.csv; comparisons: stats.csv; every run's records: runs/.",
        "",
    ]
    out.write_text("\n".join(lines))
    return out


def run_section(run: Path, plots: Sequence[Path]) -> str:
    links = "\n".join(f"- [{p.name}](plots/{p.name})" for p in plots)
    made_by = f"Written by `python -m evomorph analyze --run {run}`."
    return f"{START}\n## Analysis plots\n\n{made_by}\n\n{links}\n{END}\n"


def add_run_section(run: Path, plots: Sequence[Path]) -> Path:
    """Insert or replace the analysis section of a run's ``summary.md``."""
    path = run / "summary.md"
    text = path.read_text() if path.exists() else f"# {run.name}\n"
    if START in text and END in text:
        head, rest = text.split(START, 1)
        tail = rest.split(END, 1)[1].lstrip("\n")
        text = head + run_section(run, plots) + tail
    else:
        text = text.rstrip("\n") + "\n\n" + run_section(run, plots)
    path.write_text(text)
    return path
