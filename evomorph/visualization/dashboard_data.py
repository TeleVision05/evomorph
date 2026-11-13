"""Read-only access to evolution run directories for the research dashboard.

Nothing here imports Streamlit, so it is unit-tested directly. Nothing here
writes into a run directory either: the dashboard only reads what
``python -m evomorph evolve`` saved (layout in ``evomorph/evolution/rundir.py``).

Three states a run directory can be in, each with a plain-language message:

* **complete**: ``summary.md`` exists (the engine writes it last);
* **in progress**: no summary yet. The newest generation file may be half
  written at the moment we read it, so a parse failure of *that* file only is
  treated as "still being written", not as damage;
* **damaged**: ``config.yaml`` missing or unreadable, or a generation file
  that is not the newest cannot be parsed. :class:`RunReadError` carries the
  message the dashboard shows.

Experiment directories are folders holding run directories; discovery walks
up to ``MAX_DEPTH`` levels so ``results/experiments/<name>/runs/<run>`` is found too.
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from evomorph.config import FitnessConfig, SimConfig, from_mapping
from evomorph.evolution.archive import ArchiveSpec
from evomorph.morphology.genome import Genome, from_dict
from evomorph.simulation.environment import EnvSpec

# A folder is a run directory when it holds any of these (a run that has only
# just started may have config.yaml and nothing else).
RUN_MARKERS: tuple[str, ...] = ("config.yaml", "metrics.csv", "generations")
# results/<run> is depth 1. The experiment command writes
# results/experiments/<name>/runs/<run>, depth 4 (measured 2026-11-13).
MAX_DEPTH = 4
# An in-progress run whose newest file is older than this is reported as
# "may have stopped". 30 min is several generations even for research.yaml
# (one GA generation of 48 genomes x 4 runs takes 10-25 s at 8-17 sims/s).
STALE_SECONDS = 30 * 60


class RunReadError(Exception):
    """A run directory cannot be read; ``str(exc)`` is meant for the person reading the page."""


@dataclass(frozen=True)
class RunEntry:
    """A discovered run directory: its path and its name relative to the results root."""

    path: Path
    name: str


def is_run_dir(path: Path) -> bool:
    return path.is_dir() and any((path / m).exists() for m in RUN_MARKERS)


def discover_runs(root: str | Path) -> list[RunEntry]:
    """Every run directory under ``root`` (sorted by relative name); [] if root is missing."""
    base = Path(root)
    if not base.is_dir():
        return []
    found: list[RunEntry] = []

    def walk(folder: Path, depth: int) -> None:
        for child in sorted(folder.iterdir()):
            if not child.is_dir() or child.name.startswith((".", "_")):
                continue
            if is_run_dir(child):
                found.append(RunEntry(child, child.relative_to(base).as_posix()))
            elif depth < MAX_DEPTH:
                walk(child, depth + 1)

    walk(base, 1)
    return found


def _stat_key(path: Path) -> tuple[str, int, int]:
    try:
        st = path.stat()
    except OSError:
        return (path.name, -1, -1)
    return (path.name, st.st_size, st.st_mtime_ns)


def signature(path: str | Path) -> tuple:
    """Changes whenever the run writes something; used as a cache key by the dashboard."""
    p = Path(path)
    files = ("config.yaml", "metrics.csv", "summary.md", "best_genome.json")
    gens = sorted((p / "generations").glob("gen_*.json")) if (p / "generations").is_dir() else []
    last = _stat_key(gens[-1]) if gens else ("", 0, 0)
    arch = _stat_key(p / "archive" / "archive.json")
    return (str(p), *(_stat_key(p / f) for f in files), len(gens), last, arch)


# ---------------------------------------------------------------------------
# Loading one run
# ---------------------------------------------------------------------------


@dataclass
class Run:
    """What the dashboard knows about one run directory."""

    path: Path
    name: str
    config: dict[str, Any]
    metrics: pd.DataFrame
    gen_files: list[Path]
    complete: bool
    notes: list[str] = field(default_factory=list)  # plain-language caveats
    last_write: float = 0.0  # unix time of the newest file

    @property
    def evolution(self) -> dict[str, Any]:
        return dict(self.config.get("evolution") or {})

    @property
    def algorithm(self) -> str:
        return str(self.evolution.get("algorithm", "ga"))

    @property
    def budget(self) -> int | None:
        b = self.evolution.get("budget")
        return int(b) if b is not None else None

    @property
    def sim(self) -> SimConfig:
        return from_mapping(SimConfig, self.config.get("sim"))

    @property
    def fitness_config(self) -> FitnessConfig:
        return from_mapping(FitnessConfig, self.config.get("fitness"))

    @property
    def archive_spec(self) -> ArchiveSpec:
        me = self.config.get("map_elites") or {}
        default = ArchiveSpec()
        return ArchiveSpec(
            tuple(me.get("descriptors", default.descriptors)),
            tuple(me.get("bins", default.bins)),
            tuple(tuple(r) for r in me.get("ranges", default.ranges)),
        )

    @property
    def eval_runs(self) -> list[tuple[EnvSpec, int]]:
        """(environment, spawn seed) of every evaluation run, in config order."""
        runs = ((self.evolution.get("evaluation") or {}).get("runs")) or []
        out = []
        for r in runs:
            out.append((EnvSpec(**(r.get("env") or {})), int(r.get("seed", 0))))
        return out or [(EnvSpec(), 0)]

    @property
    def status(self) -> str:
        if self.complete:
            return "complete"
        age = time.time() - self.last_write if self.last_write else 0.0
        if age > STALE_SECONDS:
            return f"not finished; last written {_age(age)} ago, so it may have stopped"
        return "in progress"


def _age(seconds: float) -> str:
    if seconds < 3600:
        return f"{seconds / 60:.0f} min"
    if seconds < 86400:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} days"


def _read_config(path: Path) -> dict[str, Any]:
    cfg = path / "config.yaml"
    if not cfg.is_file():
        raise RunReadError(
            f"'{path.name}' has no config.yaml, so it is not a readable run. "
            "It may have been copied incompletely or is not an evolve output folder."
        )
    try:
        data = yaml.safe_load(cfg.read_text())
    except (OSError, yaml.YAMLError):
        raise RunReadError(
            f"The settings file of '{path.name}' (config.yaml) is damaged and cannot be read. "
            "Re-run the experiment or restore the folder from a copy."
        ) from None
    if not isinstance(data, dict):
        raise RunReadError(f"The settings file of '{path.name}' (config.yaml) is empty or damaged.")
    return data


def _num(text: str) -> Any:
    try:
        v = float(text)
    except ValueError:
        return text
    return int(v) if v.is_integer() and "." not in text and "e" not in text.lower() else v


def read_metrics(path: Path) -> tuple[pd.DataFrame, int]:
    """metrics.csv as a DataFrame plus the number of malformed lines skipped.

    A line with the wrong number of fields (for example the last line of a run
    that is writing it right now) is skipped rather than failing the page.
    """
    out = path / "metrics.csv"
    if not out.is_file():
        return pd.DataFrame(), 0
    with out.open(newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return pd.DataFrame(), 0
    header, body = rows[0], rows[1:]
    good = [[_num(c) for c in r] for r in body if len(r) == len(header)]
    return pd.DataFrame(good, columns=header), len(body) - len(good)


def load_run(entry: RunEntry) -> Run:
    """Read one run directory; raises :class:`RunReadError` with a friendly message."""
    path = entry.path
    if not path.is_dir():
        raise RunReadError(f"The run folder '{entry.name}' no longer exists.")
    config = _read_config(path)
    metrics, bad = read_metrics(path)
    gen_files = sorted((path / "generations").glob("gen_*.json"))
    complete = (path / "summary.md").is_file()
    notes = []
    if bad:
        notes.append(
            f"{bad} line(s) of metrics.csv could not be read and were left out"
            + ("." if complete else " (the run may be writing them right now).")
        )
    if not gen_files and not complete:
        notes.append("No generation has finished yet; this page fills in as the run proceeds.")
    stamps = [p.stat().st_mtime for p in [*gen_files[-1:], path / "metrics.csv"] if p.exists()]
    stamps.append((path / "config.yaml").stat().st_mtime)
    return Run(path, entry.name, config, metrics, gen_files, complete, notes, max(stamps))


# ---------------------------------------------------------------------------
# Generation records and individuals
# ---------------------------------------------------------------------------


def gen_index(path: Path) -> int:
    return int(path.stem.split("_")[1])


def load_generation(run: Run, index: int) -> list[dict[str, Any]]:
    """Records of one generation file ([] if the newest file is still being written)."""
    files = {gen_index(p): p for p in run.gen_files}
    if index not in files:
        raise RunReadError(f"Generation {index} of '{run.name}' has no saved file.")
    try:
        data = json.loads(files[index].read_text())
    except (OSError, ValueError):
        newest = max(files) == index
        if newest and not run.complete:
            return []
        raise RunReadError(
            f"The file for generation {index} of '{run.name}' is damaged and cannot be read."
        ) from None
    if not isinstance(data, list):
        raise RunReadError(f"The file for generation {index} of '{run.name}' has an odd layout.")
    return data


def evaluated_records(run: Run, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The records evaluated in that step (the GA file also repeats carried elites)."""
    if run.algorithm == "ga":
        return [r for r in records if r.get("status") == "new"]
    return list(records)


@dataclass
class Index:
    """Every recorded individual by id, with the step in which it was first evaluated."""

    records: dict[str, dict[str, Any]] = field(default_factory=dict)
    first_step: dict[str, int] = field(default_factory=dict)
    by_step: dict[int, list[str]] = field(default_factory=dict)  # evaluated ids per step
    skipped: list[int] = field(default_factory=list)  # steps still being written


def build_index(run: Run) -> Index:
    idx = Index()
    for p in run.gen_files:
        step = gen_index(p)
        records = load_generation(run, step)
        if not records:
            idx.skipped.append(step)
            continue
        new = evaluated_records(run, records)
        idx.by_step[step] = [r["id"] for r in new]
        for r in records:
            if r.get("id") and r["id"] not in idx.records:
                idx.records[r["id"]] = r
                idx.first_step[r["id"]] = step
    return idx


def record_genome(record: dict[str, Any]) -> Genome:
    """The genome stored in a record (raises RunReadError if it cannot be rebuilt)."""
    try:
        return from_dict(record["genome"])
    except (KeyError, ValueError, TypeError):
        rid = record.get("id", "?")
        raise RunReadError(f"The saved genes of creature {rid} cannot be read.") from None


def lineage(index: Index, genome_id: str, max_depth: int = 12) -> list[dict[str, Any]]:
    """Ancestors of ``genome_id`` breadth first, ``max_depth`` generations back.

    Each row: id, depth (0 = the creature), generation, origin, fitness,
    parent_ids and ``recorded`` (False for a parent whose record is not in the
    run, e.g. a frozen template).
    """
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    frontier = [(genome_id, 0)]
    while frontier:
        gid, depth = frontier.pop(0)
        if gid in seen or depth > max_depth:
            continue
        seen.add(gid)
        rec = index.records.get(gid)
        if rec is None:
            rows.append({"id": gid, "depth": depth, "recorded": False})
            continue
        parents = list(rec.get("parent_ids") or [])
        rows.append(
            {
                "id": gid,
                "depth": depth,
                "generation": rec.get("generation"),
                "origin": rec.get("origin", ""),
                "fitness": rec.get("fitness"),
                "parent_ids": parents,
                "recorded": True,
            }
        )
        frontier += [(p, depth + 1) for p in parents]
    return rows


def champion_history(run: Run) -> pd.DataFrame:
    """Per step: the run's champion (best so far) id and fitness from metrics.csv.

    ``changed`` marks the steps where a new champion took over, which is what
    the evolution strip shows.
    """
    m = run.metrics
    cols = ["generation", "sims", "champion_id", "champion_fitness", "champion_plan"]
    if m.empty or not set(cols) <= set(m.columns):
        return pd.DataFrame(columns=[*cols, "changed", "genome_file"])
    df = m[cols].copy()
    df = df[df["champion_id"].astype(str).str.len() > 0]
    df["changed"] = df["champion_id"] != df["champion_id"].shift()
    df["genome_file"] = [
        str(run.path / "genomes" / f"champion_gen_{int(g):04d}.json") for g in df["generation"]
    ]
    return df.reset_index(drop=True)


def is_finite(x: Any) -> bool:
    return isinstance(x, int | float) and math.isfinite(x)
