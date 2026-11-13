"""Post-hoc evaluation of a finished run: champion and final set, train and test suites.

After a run's search has spent its budget, the same procedure is applied to
every condition (it costs simulations, but outside the budget, and the same
number for every run):

* **Champion** (the run's best genome by training fitness): re-scored on every
  environment of the training suite and of the held-out test suite, with
  ``rescore_seeds`` *fresh* spawn seeds it was never selected on. Selecting the
  maximum of many noisy scores inflates the winner's score (winner's curse); the
  fresh training re-score measures how much, and the test score measures
  generalisation.
* **Final set**: the best ``set_size`` distinct valid genomes of the final GA
  population or the final MAP-Elites archive, by stored training fitness, each
  re-scored on both suites with ``set_seeds`` fresh seeds. Equal set sizes make
  the population-level comparison fair between a GA population and an archive.
* **Whole final population or archive**: diversity and training-fitness
  statistics from the stored training evaluations (no new simulations).

Seeds: training re-scores use spawn seeds 1000+, test re-scores 2000+
(:mod:`evomorph.experiments.design`). Identical (genome, environment, seed)
tasks are simulated once. The result is written to ``posthoc.json`` in the run
directory, via a temporary file, so an interrupted evaluation never leaves a
file that looks finished.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from evomorph.config import FitnessConfig, SimConfig, load_yaml
from evomorph.evolution.archive import Archive
from evomorph.evolution.pool import clamp_workers
from evomorph.evolution.selection import body_plan
from evomorph.evolution.settings import from_sections
from evomorph.experiments.design import RESCORE_TEST_SEED, RESCORE_TRAIN_SEED, ExperimentConfig
from evomorph.experiments.measures import (
    RECORD_METRICS,
    behaviour_diversity,
    final_set_measures,
    morphology_diversity,
    plan_counts,
    suite_summary,
    top_k,
)
from evomorph.morphology.genome import Genome, from_dict, genome_hash
from evomorph.simulation.environment import EnvSpec
from evomorph.simulation.evaluator import evaluate
from evomorph.simulation.suites import get_suite
from evomorph.utils.serialization import dump_json, load_json

POSTHOC_FILE = "posthoc.json"
SUITE_ROLES: tuple[str, str] = ("train", "test")

Task = tuple[Genome, EnvSpec, SimConfig, FitnessConfig, int]


def _run_task(task: Task) -> dict[str, Any]:
    """One simulation, reduced to a compact picklable record."""
    genome, env, sim, fitness, seed = task
    r = evaluate(genome, env=env, sim=sim, fitness=fitness, seed=seed)
    metrics = {k: float(r.metrics[k]) for k in RECORD_METRICS} if r.valid else {}
    return {
        "env": env.label(),
        "seed": int(seed),
        "fitness": float(r.fitness),
        "valid": bool(r.valid),
        "fell": bool(r.valid and r.fell),
        "reason": r.failure_reason,
        "metrics": metrics,
    }


def _map(tasks: Sequence[Task], workers: int) -> list[dict[str, Any]]:
    n = min(clamp_workers(workers), max(len(tasks), 1))
    if n == 1:
        return [_run_task(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=n) as pool:
        return list(pool.map(_run_task, tasks, chunksize=1))


# ---------------------------------------------------------------------------
# Reading a finished run
# ---------------------------------------------------------------------------


def _member(genome: Genome, fitness: float, valid: bool, metrics: dict) -> dict[str, Any]:
    return {"genome": genome, "fitness": float(fitness), "valid": bool(valid), "metrics": metrics}


def final_members(run_dir: Path) -> list[dict[str, Any]]:
    """The final GA population (last generation file) or MAP-Elites archive."""
    algorithm = load_yaml(run_dir / "config.yaml")["evolution"]["algorithm"]
    if algorithm == "map_elites":
        arch = Archive.load(run_dir / "archive")
        return [_member(e.genome, e.fitness, True, e.metrics) for e in arch.elites()]
    last = sorted((run_dir / "generations").glob("gen_*.json"))[-1]
    return [
        _member(from_dict(r["genome"]), r["fitness"], r["valid"], r["metrics"])
        for r in load_json(last)
    ]


def champion_member(run_dir: Path) -> dict[str, Any]:
    rec = load_json(run_dir / "best_result.json")
    return _member(from_dict(rec["genome"]), rec["fitness"], rec["valid"], rec["metrics"])


# ---------------------------------------------------------------------------
# The evaluation
# ---------------------------------------------------------------------------


class _Plan:
    """Unique (genome, suite, environment, seed) tasks and who needs which."""

    def __init__(
        self, sim: SimConfig, fitness: FitnessConfig, envs: dict[str, tuple[EnvSpec, ...]]
    ) -> None:
        self.sim, self.fitness, self.envs = sim, fitness, envs
        self.tasks: list[Task] = []
        self.index: dict[tuple[str, str, int, int], int] = {}

    def need(self, genome: Genome, suite: str, n_seeds: int) -> list[int]:
        base = RESCORE_TRAIN_SEED if suite == "train" else RESCORE_TEST_SEED
        envs = self.envs[suite]
        out = []
        for i, env in enumerate(envs):
            for j in range(n_seeds):
                key = (genome_hash(genome), suite, i, base + j)
                if key not in self.index:
                    self.index[key] = len(self.tasks)
                    self.tasks.append((genome, env, self.sim, self.fitness, base + j))
                out.append(self.index[key])
        return out


def _scored(member: dict[str, Any], need: dict, results: list[dict[str, Any]]) -> dict[str, Any]:
    g = member["genome"]
    out: dict[str, Any] = {"id": g.id, "plan": body_plan(g), "selected_fitness": member["fitness"]}
    for suite in ("train", "test"):
        records = [results[i] for i in need[suite]]
        out[suite] = suite_summary(records)
        out[f"{suite}_runs"] = records
    return out


def posthoc_evaluate(
    run_dir: str | Path, exp: ExperimentConfig, workers: int = 1, write: bool = True
) -> dict[str, Any]:
    """Evaluate a finished run (see the module doc); writes ``posthoc.json``."""
    run_dir = Path(run_dir)
    t0 = time.perf_counter()
    cfg = from_sections(load_yaml(run_dir / "config.yaml"))
    envs = {"train": get_suite(exp.train_suite), "test": get_suite(exp.test_suite)}
    plan = _Plan(cfg.sim, cfg.fitness, envs)
    champ = champion_member(run_dir)
    members = final_members(run_dir)
    chosen = top_k(members, exp.set_size)
    champ_need = {s: plan.need(champ["genome"], s, exp.rescore_seeds) for s in SUITE_ROLES}
    set_need = [{s: plan.need(m["genome"], s, exp.set_seeds) for s in SUITE_ROLES} for m in chosen]
    results = _map(plan.tasks, workers)
    scored_set = [_scored(m, n, results) for m, n in zip(chosen, set_need, strict=True)]
    genomes = [m["genome"] for m in chosen]
    out = {
        "run": run_dir.name,
        "algorithm": cfg.evolution.algorithm,
        "mode": cfg.evolution.mode,
        "seed": cfg.evolution.seed,
        "train_suite": exp.train_suite,
        "test_suite": exp.test_suite,
        "seeds": {
            "train": [RESCORE_TRAIN_SEED + j for j in range(exp.rescore_seeds)],
            "test": [RESCORE_TEST_SEED + j for j in range(exp.rescore_seeds)],
            "set": exp.set_seeds,
        },
        "simulations": len(plan.tasks),
        "champion": _scored(champ, champ_need, results),
        "set": {
            "members": scored_set,
            "morph_diversity": morphology_diversity(genomes),
            "plans": plan_counts(genomes),
            "behaviour_diversity": {
                s: behaviour_diversity([m[s]["behaviour"] for m in scored_set])
                for s in ("train", "test")
            },
        },
        "final": final_set_measures(members),
    }
    if write:
        _write_atomic(out, run_dir / POSTHOC_FILE)
        dump_json({"wall_seconds": time.perf_counter() - t0}, run_dir / "posthoc_timing.json")
    return out


def _write_atomic(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(".json.tmp")
    dump_json(obj, tmp)
    os.replace(tmp, path)
