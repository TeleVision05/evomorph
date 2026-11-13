"""The combined results table: one row per (condition, seed), ``results.csv``.

Column families (every number traces back to a run's ``posthoc.json`` or
``metrics.csv``; nothing here re-simulates):

* ``champ_selected``: the champion's training fitness as stored when it was
  selected (biased upward by selection: the winner's curse).
* ``champ_<suite>_<measure>``: the champion re-scored with fresh seeds on the
  training (``train``) or held-out (``test``) suite. Measures are
  :data:`~evomorph.experiments.measures.SUMMARY_KEYS`: ``fitness``,
  ``worst_fitness``, ``failure_rate`` (fell or invalid), ``invalid_rate``,
  ``speed`` (m/s forward), ``efficiency`` (0-1), ``tilt`` (rad, mean torso
  tilt: lower is more stable) and ``slip`` (0-1).
* ``set_<suite>_<measure>``: the same measures averaged over the best-K final
  set; ``set_test_best`` is the best member's test fitness.
* ``champ_gap`` / ``set_gap``: test minus train fitness (generalisation gap;
  negative means worse on unseen terrain).
* Diversity: ``set_morph_diversity``, ``set_body_plans``,
  ``set_<suite>_behaviour_diversity`` on the best-K set; ``final_*`` on the
  whole final population or archive (training evaluations).

Floats are written with ``repr`` so the file is exact and byte-reproducible;
wall times are deliberately absent (they live in the manifest).
"""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from evomorph.experiments.measures import SUMMARY_KEYS

SUITES: tuple[str, str] = ("train", "test")

TABLE_COLUMNS: tuple[str, ...] = (
    "condition",
    "seed",
    "sims",
    "champion_id",
    "champion_plan",
    "champ_selected",
    *(f"champ_{s}_{k}" for s in SUITES for k in SUMMARY_KEYS),
    "champ_gap",
    *(f"set_{s}_{k}" for s in SUITES for k in SUMMARY_KEYS),
    "set_test_best",
    "set_gap",
    "set_size",
    "set_morph_diversity",
    "set_body_plans",
    "set_train_behaviour_diversity",
    "set_test_behaviour_diversity",
    "final_n",
    "final_n_valid",
    "final_best_fitness",
    "final_mean_fitness",
    "final_morph_diversity",
    "final_body_plans",
    "final_behaviour_diversity",
)

NON_NUMERIC: tuple[str, ...] = ("condition", "champion_id", "champion_plan")


def _nanmean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if not math.isnan(float(v))]
    return float(np.mean(vals)) if vals else math.nan


def table_row(condition: str, seed: int, sims: int, ph: dict[str, Any]) -> dict[str, Any]:
    """Flatten one run's post-hoc evaluation into a :data:`TABLE_COLUMNS` row."""
    champ, sset, final = ph["champion"], ph["set"], ph["final"]
    members = sset["members"]
    row: dict[str, Any] = {
        "condition": condition,
        "seed": int(seed),
        "sims": int(sims),
        "champion_id": champ["id"],
        "champion_plan": champ["plan"],
        "champ_selected": float(champ["selected_fitness"]),
    }
    for s in SUITES:
        for k in SUMMARY_KEYS:
            row[f"champ_{s}_{k}"] = float(champ[s][k])
            row[f"set_{s}_{k}"] = _nanmean([m[s][k] for m in members])
    row["champ_gap"] = row["champ_test_fitness"] - row["champ_train_fitness"]
    row["set_test_best"] = max((float(m["test"]["fitness"]) for m in members), default=math.nan)
    row["set_gap"] = row["set_test_fitness"] - row["set_train_fitness"]
    row |= {
        "set_size": len(members),
        "set_morph_diversity": float(sset["morph_diversity"]),
        "set_body_plans": len(sset["plans"]),
        "set_train_behaviour_diversity": float(sset["behaviour_diversity"]["train"]),
        "set_test_behaviour_diversity": float(sset["behaviour_diversity"]["test"]),
        "final_n": int(final["n"]),
        "final_n_valid": int(final["n_valid"]),
        "final_best_fitness": float(final["best_fitness"]),
        "final_mean_fitness": float(final["mean_fitness"]),
        "final_morph_diversity": float(final["morph_diversity"]),
        "final_body_plans": int(final["n_body_plans"]),
        "final_behaviour_diversity": float(final["behaviour_diversity"]),
    }
    return row


def _cell(value: Any) -> str:
    return repr(value) if isinstance(value, float) else str(value)


def write_table(rows: Sequence[dict[str, Any]], path: str | Path) -> Path:
    """Write ``rows`` (already in the wanted order) as ``results.csv``."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(TABLE_COLUMNS)
        for row in rows:
            w.writerow([_cell(row[c]) for c in TABLE_COLUMNS])
    return out


def read_table(path: str | Path) -> list[dict[str, Any]]:
    """Read ``results.csv`` back; numeric columns become floats (seed and sims ints)."""
    with Path(path).open() as fh:
        raw = list(csv.DictReader(fh))
    rows = []
    for r in raw:
        row: dict[str, Any] = {}
        for k, v in r.items():
            if k in NON_NUMERIC:
                row[k] = v
            elif k in ("seed", "sims"):
                row[k] = int(v)
            else:
                row[k] = float(v)
        rows.append(row)
    return rows
