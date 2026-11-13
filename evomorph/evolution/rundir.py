"""Run directories (brief section 11): created fresh, never overwritten.

Layout of ``results/<name or run_NNN>/``::

    config.yaml            fully resolved config (reload it to re-score any genome)
    metrics.csv            one row per generation / batch (progress.METRICS_COLUMNS)
    timing.csv             wall time per generation (kept out of metrics.csv so
                           that file is identical whatever the worker count)
    generations/gen_NNNN.json
                           every individual of the step: id, parent ids, origin,
                           generation of origin, status, fitness, per-run fitness,
                           metrics and genes
    genomes/champion_gen_NNNN.json   best-so-far genome after each step
    genomes/final/<id>.json          the final population (GA) or archive elites
    archive/               MAP-Elites archive (archive.json + elites/); the GA's
                           passive archive is saved here too
    frozen_template.json   the frozen body / controller (controller and
                           morphology modes only)
    best_genome.json       overall champion
    best_result.json       its aggregated evaluation (fitness, metrics, per-run)
    best_heldout.json      the champion scored on held-out seeds (seed + 1000)
    summary.md             human-readable summary

Floats are written with Python's shortest round-trip repr in both the CSV and
the JSON, so every stored number reads back bit-identical.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from evomorph.evolution.progress import METRICS_COLUMNS, GenerationReport, Individual
from evomorph.morphology.genome import Genome, save_genome, to_dict
from evomorph.utils.serialization import dump_json, to_jsonable

TIMING_COLUMNS = ("generation", "sims", "evaluations", "wall_seconds", "elapsed_seconds")
_RUN_RE = re.compile(r"^run_(\d{3,})$")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class RunDirExistsError(FileExistsError):
    """The requested run directory already exists (runs are never overwritten)."""


def create_run_dir(root: str | Path, name: str | None = None) -> Path:
    """Create and return a fresh run directory under ``root``.

    With a name, ``root/name`` must not exist yet. Without one, the next free
    ``run_NNN`` is taken; ``mkdir(exist_ok=False)`` makes the claim atomic, so
    two runs started together never share a folder.
    """
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    if name is not None:
        if not _NAME_RE.match(name):
            raise ValueError(
                f"run name '{name}' may only use letters, digits, '.', '_' and '-' "
                "and must start with a letter or digit"
            )
        path = base / name
        try:
            path.mkdir()
        except FileExistsError:
            raise RunDirExistsError(
                f"{path} already exists; runs are never overwritten, so pick another name"
            ) from None
        return path
    taken = [int(m.group(1)) for p in base.iterdir() if (m := _RUN_RE.match(p.name))]
    n = max(taken, default=0) + 1
    while True:
        path = base / f"run_{n:03d}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            n += 1


def _cell(value: Any) -> str:
    if isinstance(value, float):
        return repr(value)
    return str(value)


def individual_record(ind: Individual) -> dict[str, Any]:
    """JSON-ready record of one individual (see the module doc)."""
    r = ind.result
    g = ind.genome
    return {
        "id": g.id,
        "parent_ids": list(g.parent_ids),
        "generation": g.generation,
        "origin": g.origin,
        "status": ind.status,
        "valid": r.valid,
        "failure_reason": r.failure_reason,
        "fitness": r.fitness,
        "seed_fitness": list(r.seed_fitness),
        "run_labels": list(getattr(r, "run_labels", ())),
        "metrics": r.metrics,
        "fitness_terms": r.fitness_terms,
        "genome": to_dict(g),
    }


class RunWriter:
    """Observer that writes every report into a run directory."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        for sub in ("generations", "genomes", "archive"):
            (self.path / sub).mkdir(parents=True, exist_ok=True)
        self.elapsed = 0.0

    def write_config(self, sections: dict[str, Any]) -> Path:
        out = self.path / "config.yaml"
        header = "# Fully resolved configuration of this run (written by evomorph evolve).\n"
        out.write_text(header + yaml.safe_dump(to_jsonable(sections), sort_keys=False))
        return out

    def _append_csv(self, name: str, columns: Sequence[str], row: dict[str, Any]) -> None:
        out = self.path / name
        new = not out.exists()
        with out.open("a", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            if new:
                w.writerow(columns)
            w.writerow([_cell(row[c]) for c in columns])

    def on_generation(self, rep: GenerationReport) -> None:
        self._append_csv("metrics.csv", METRICS_COLUMNS, rep.row)
        self.elapsed += rep.wall_seconds
        timing = {
            "generation": rep.index,
            "sims": rep.sims_used,
            "evaluations": len(rep.evaluated),
            "wall_seconds": round(rep.wall_seconds, 3),
            "elapsed_seconds": round(self.elapsed, 3),
        }
        self._append_csv("timing.csv", TIMING_COLUMNS, timing)
        records = [individual_record(i) for i in rep.evaluated]
        if rep.algorithm == "ga":  # carried elites too, so the file is the whole population
            records += [individual_record(i) for i in rep.current if i.status != "new"]
        dump_json(records, self.path / "generations" / f"gen_{rep.index:04d}.json")
        if rep.champion is not None:
            path = self.path / "genomes" / f"champion_gen_{rep.index:04d}.json"
            save_genome(rep.champion.genome, path)

    def save_final(self, genomes: Sequence[Genome]) -> None:
        for g in genomes:
            save_genome(g, self.path / "genomes" / "final" / f"{g.id}.json")

    def save_best(self, champion: Individual) -> None:
        save_genome(champion.genome, self.path / "best_genome.json")
        dump_json(individual_record(champion), self.path / "best_result.json")

    def save_template(self, genome: Genome) -> None:
        save_genome(genome, self.path / "frozen_template.json")

    def write_summary(self, text: str) -> Path:
        out = self.path / "summary.md"
        out.write_text(text)
        return out
