"""Re-score every elite of a saved archive and check it reproduces its stored numbers.

Usage:
    .venv/bin/python scripts/check_archive.py results/<run> [--workers 6]

Loads ``config.yaml`` and ``archive/`` from the run, re-evaluates each elite
with the saved evaluation spec, and compares fitness, every metric and the
cell. Prints one line per mismatch and a final count; exit code 1 if any
elite fails to reproduce.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from evomorph.config import load_yaml
from evomorph.evolution.archive import Archive
from evomorph.evolution.evalspec import SpecEvaluator
from evomorph.evolution.pool import BatchRunner
from evomorph.evolution.settings import from_sections


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    cfg = from_sections(load_yaml(args.run / "config.yaml"))
    archive = Archive.load(args.run / "archive")
    cells = archive.occupied()
    elites = [archive.get(c) for c in cells]
    t0 = time.perf_counter()
    with BatchRunner(SpecEvaluator(cfg.eval_spec), args.workers) as runner:
        results = runner.map([e.genome for e in elites])
    bad = 0
    for cell, e, r in zip(cells, elites, results, strict=True):
        same_cell = r.valid and archive.spec.cell(archive.spec.descriptor(r.metrics)) == cell
        if not (r.fitness == e.fitness and r.metrics == e.metrics and same_cell):
            bad += 1
            print(f"MISMATCH {cell} {e.genome.id}: stored {e.fitness!r}, now {r.fitness!r}")
    wall = time.perf_counter() - t0
    print(f"{len(elites) - bad}/{len(elites)} elites reproduced exactly ({wall:.1f}s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
