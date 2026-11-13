"""The comparison experiment: every (condition, seed) run, resumable, plus the table.

Layout of ``<results>/<name>/``::

    manifest.json     design, its hash, the runs and their status, invocations,
                      wall times and versions
    experiment.yaml   the experiment section and base engine config as resolved
    runs/<condition>_s<seed>/   a normal run directory (``evolve``'s layout)
                      plus posthoc.json (see evaluation.py)
    incomplete/       run directories an interrupted invocation left half-made,
                      moved aside (never deleted) before the run is redone
    results.csv       the combined table (see table.py)

Order and resumption
--------------------
Runs go seed by seed, every condition within a seed, so an interrupted
experiment always holds complete blocks of seeds rather than many seeds of one
condition. Re-running the same command skips every run whose ``posthoc.json``
exists, evaluates runs whose search finished but whose evaluation did not, and
redoes runs whose search was interrupted. A different design (budget, suites,
evolution settings: anything but the worker count or the number of seeds) is
refused, because its runs would not be comparable with those already there.

Equal budgets
-------------
Before a finished run is evaluated or tabulated its saved ``config.yaml`` must
equal the design's config for that condition and seed (workers aside), and its
``metrics.csv`` must show exactly the design's effective budget spent.
"""

from __future__ import annotations

import csv
import dataclasses
import os
import platform
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mujoco
import yaml

from evomorph.config import load_yaml
from evomorph.evolution.runner import run_evolution, say
from evomorph.evolution.settings import RunConfig, from_sections, to_mapping
from evomorph.experiments.baselines import get_condition
from evomorph.experiments.design import (
    ExperimentConfig,
    design_hash,
    design_mapping,
    run_config,
    run_name,
)
from evomorph.experiments.evaluation import POSTHOC_FILE, posthoc_evaluate
from evomorph.experiments.table import table_row, write_table
from evomorph.utils.serialization import dump_json, load_json, to_jsonable

MANIFEST = "manifest.json"


class DesignMismatchError(ValueError):
    """The experiment folder was made with a different design."""


class BudgetMismatchError(ValueError):
    """A run did not spend exactly the design's budget, or has a different config."""


@dataclass
class ComparisonOutcome:
    path: Path
    ran: list[str]
    skipped: list[str]
    table: Path


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _git_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def _manifest_new(exp: ExperimentConfig, base: RunConfig) -> dict[str, Any]:
    return {
        "type": "comparison",
        "design_hash": design_hash(exp, base),
        "design": design_mapping(exp, base),
        "conditions": [dataclasses.asdict(get_condition(k)) for k in exp.conditions],
        "budget": exp.effective_budget,
        "sims_per_genome": exp.sims_per_genome,
        "seeds": [],
        "runs": {},
        "invocations": [],
        "created": _now(),
        "versions": {
            "python": platform.python_version(),
            "mujoco": mujoco.__version__,
            "git_commit": _git_commit(),
        },
    }


def open_experiment(
    exp_dir: Path, exp: ExperimentConfig, base: RunConfig, command: str
) -> dict[str, Any]:
    """Create the experiment folder, or reopen it if the design matches."""
    path = exp_dir / MANIFEST
    if path.exists():
        manifest = load_json(path)
        if manifest["design_hash"] != design_hash(exp, base):
            raise DesignMismatchError(
                f"{exp_dir} holds an experiment with different settings (budget, suites or "
                "evolution settings), so its runs cannot be compared with these. Re-run it with "
                "its original settings, or give this one a new --name."
            )
    else:
        exp_dir.mkdir(parents=True, exist_ok=True)
        manifest = _manifest_new(exp, base)
        sections = {"experiment": dataclasses.asdict(exp)} | to_mapping(base)
        header = "# Resolved experiment design (written by evomorph experiment).\n"
        text = header + yaml.safe_dump(to_jsonable(sections), sort_keys=False)
        (exp_dir / "experiment.yaml").write_text(text)
    manifest["seeds"] = sorted(set(manifest["seeds"]) | set(exp.seeds))
    manifest["invocations"].append({"command": command, "started": _now(), "workers": exp.workers})
    save_manifest(exp_dir, manifest)
    return manifest


def save_manifest(exp_dir: Path, manifest: dict[str, Any]) -> None:
    tmp = exp_dir / (MANIFEST + ".tmp")
    dump_json(manifest, tmp)
    os.replace(tmp, exp_dir / MANIFEST)


def run_state(run_dir: Path) -> str:
    """``missing``, ``partial`` (search interrupted), ``searched`` or ``done``."""
    if not run_dir.exists():
        return "missing"
    if (run_dir / POSTHOC_FILE).exists():
        return "done"
    if (run_dir / "summary.md").exists():
        return "searched"
    return "partial"


def _move_aside(exp_dir: Path, run_dir: Path) -> Path:
    dest_root = exp_dir / "incomplete"
    dest_root.mkdir(exist_ok=True)
    k = 1
    while (dest := dest_root / f"{run_dir.name}-{k}").exists():
        k += 1
    os.replace(run_dir, dest)
    return dest


def _comparable(cfg: RunConfig) -> RunConfig:
    return dataclasses.replace(cfg, evolution=dataclasses.replace(cfg.evolution, workers=1))


def check_run(run_dir: Path, expected: RunConfig) -> int:
    """Raise BudgetMismatchError unless the run matches the design; return sims used."""
    saved = from_sections(load_yaml(run_dir / "config.yaml"))
    if _comparable(saved) != _comparable(expected):
        raise BudgetMismatchError(
            f"{run_dir.name} was run with a different configuration than this experiment's "
            "design, so it cannot be compared; move it out of the runs folder and re-run"
        )
    with (run_dir / "metrics.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    used = int(rows[-1]["sims"]) if rows else 0
    if used != expected.evolution.budget:
        raise BudgetMismatchError(
            f"{run_dir.name} spent {used} simulations but the design's budget is "
            f"{expected.evolution.budget}; unequal budgets cannot be compared"
        )
    return used


def _one_run(
    exp_dir: Path,
    exp: ExperimentConfig,
    cfg: RunConfig,
    name: str,
    out: Callable[[str], None],
    command: str,
) -> dict[str, Any]:
    """Search (unless already done) and evaluate one run; returns its manifest entry."""
    runs_root = exp_dir / "runs"
    run_dir = runs_root / name
    entry: dict[str, Any] = {}
    state = run_state(run_dir)
    if state == "partial":
        moved = _move_aside(exp_dir, run_dir)
        out(f"[{name}] an earlier search was interrupted; moved it to {moved} and starting again")
    if state in ("missing", "partial"):
        t0 = time.perf_counter()
        prefixed = lambda line: out(f"[{name}] {line}")  # noqa: E731
        run_evolution(cfg, runs_root, name, out=prefixed, command=command)
        entry["search_seconds"] = round(time.perf_counter() - t0, 1)
    entry["sims_used"] = check_run(run_dir, cfg)
    out(f"[{name}] re-scoring the champion and the final set on train and test suites")
    t0 = time.perf_counter()
    ph = posthoc_evaluate(run_dir, exp, workers=exp.workers)
    entry["posthoc_seconds"] = round(time.perf_counter() - t0, 1)
    entry["posthoc_sims"] = ph["simulations"]
    c = ph["champion"]
    out(
        f"[{name}] champion {c['id']} ({c['plan']}): train {c['train']['fitness']:+.3f}, "
        f"test {c['test']['fitness']:+.3f} (fresh seeds)"
    )
    return entry


def collect_table(exp_dir: Path, exp: ExperimentConfig, base: RunConfig) -> Path:
    """Write ``results.csv`` from every finished run of the design's conditions."""
    rows = []
    for key in exp.conditions:
        cond = get_condition(key)
        for seed in sorted(load_json(exp_dir / MANIFEST)["seeds"]):
            run_dir = exp_dir / "runs" / run_name(cond, seed)
            if run_state(run_dir) != "done":
                continue
            used = check_run(run_dir, run_config(exp, base, cond, seed))
            rows.append(table_row(key, seed, used, load_json(run_dir / POSTHOC_FILE)))
    return write_table(rows, exp_dir / "results.csv")


def run_comparison(
    exp: ExperimentConfig,
    base: RunConfig,
    root: str | Path,
    name: str,
    out: Callable[[str], None] = say,
    command: str = "",
) -> ComparisonOutcome:
    """Run (or resume) the comparison experiment into ``root/name``."""
    exp_dir = Path(root) / name
    manifest = open_experiment(exp_dir, exp, base, command)
    ran, skipped = [], []
    total = len(exp.seeds) * len(exp.conditions)
    for seed in exp.seeds:
        for key in exp.conditions:
            cond = get_condition(key)
            rname = run_name(cond, seed)
            cfg = run_config(exp, base, cond, seed)
            if run_state(exp_dir / "runs" / rname) == "done":
                check_run(exp_dir / "runs" / rname, cfg)
                skipped.append(rname)
                out(f"[{rname}] already finished; skipped")
                continue
            out(f"=== run {len(ran) + len(skipped) + 1}/{total}: {cond.label}, seed {seed} ===")
            entry = _one_run(exp_dir, exp, cfg, rname, out, command)
            manifest["runs"][rname] = {"condition": key, "seed": seed, **entry, "finished": _now()}
            save_manifest(exp_dir, manifest)
            ran.append(rname)
    table = collect_table(exp_dir, exp, base)
    manifest["invocations"][-1]["finished"] = _now()
    save_manifest(exp_dir, manifest)
    out(f"experiment done: {len(ran)} runs made, {len(skipped)} already finished; table {table}")
    return ComparisonOutcome(exp_dir, ran, skipped, table)
