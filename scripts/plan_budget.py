"""Budget per run that fits a wall-time limit, from an experiment's measured throughput.

Reads the manifest of a finished experiment (e.g. the --quick one), takes its
measured search and post-hoc throughput (simulations per second of wall time,
6 workers), and prints, for each seed count, the largest per-run budget (whole
genomes of the training suite) whose runs all fit in ``--hours``.

Per run the wall time is modelled as
    (budget + check_sims) / search_rate + posthoc_sims / posthoc_rate
where check_sims (the engine's champion re-score and held-out check, 2 x 11)
is already inside the measured search time, and posthoc_sims is what the full
design's re-scoring costs (champion: 11 environments x 2 suites x rescore_seeds;
set: set_size x 11 x 2 x set_seeds; no deduplication assumed, so an upper bound).

Usage:
    .venv/bin/python scripts/plan_budget.py results/experiments/comparison_quick \
        [--hours 2.5] [--conditions 4] [--rescore-seeds 3] [--set-size 10] [--rate R]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ENVS = 11  # training and test suites both have 11 environments
CHECK_SIMS = 2 * ENVS  # engine: champion re-score + held-out-seed check


def measured_rates(manifest: dict) -> tuple[float, float, float, float]:
    runs = [r for r in manifest["runs"].values() if r.get("search_seconds")]
    s_sims = sum(r["sims_used"] + CHECK_SIMS for r in runs)
    s_sec = sum(r["search_seconds"] for r in runs)
    p_sims = sum(r["posthoc_sims"] for r in runs)
    p_sec = sum(r["posthoc_seconds"] for r in runs)
    per_run = [(r["sims_used"] + CHECK_SIMS) / r["search_seconds"] for r in runs]
    return s_sims / s_sec, p_sims / p_sec, min(per_run), max(per_run)


def budget_for(hours: float, runs: int, search: float, post: float, posthoc: int) -> int:
    per_run = hours * 3600 / runs
    sims = (per_run - posthoc / post) * search - CHECK_SIMS
    return max(0, int(sims) // ENVS * ENVS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiment", type=Path)
    ap.add_argument("--hours", type=float, default=2.5)
    ap.add_argument("--conditions", type=int, default=4)
    ap.add_argument("--rescore-seeds", type=int, default=3)
    ap.add_argument("--set-size", type=int, default=10)
    ap.add_argument("--rate", type=float, help="assume this search rate instead (sims/s)")
    args = ap.parse_args()
    manifest = json.loads((args.experiment / "manifest.json").read_text())
    search, post, lo, hi = measured_rates(manifest)
    print(f"measured search rate {search:.2f} sims/s (per run {lo:.2f}-{hi:.2f}), "
          f"post-hoc rate {post:.2f} sims/s, over {len(manifest['runs'])} runs")  # fmt: skip
    if args.rate:
        search, post = args.rate, args.rate
        print(f"assuming {args.rate:.2f} sims/s for both instead")
    posthoc = 2 * ENVS * args.rescore_seeds + 2 * ENVS * args.set_size
    print(f"post-hoc simulations per run (upper bound): {posthoc}")
    for seeds in (3, 4, 5, 6, 8):
        b = budget_for(args.hours, seeds * args.conditions, search, post, posthoc)
        print(f"{seeds} seeds x {args.conditions} conditions in {args.hours:g} h: "
              f"budget {b} simulations per run = {b // ENVS} genomes")  # fmt: skip


if __name__ == "__main__":
    main()
