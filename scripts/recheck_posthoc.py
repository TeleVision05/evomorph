"""Re-run the post-hoc evaluation of every run of an experiment with the current code.

Usage:
    .venv/bin/python scripts/recheck_posthoc.py results/experiments/comparison [--workers 4]

Nothing is written. Each run's stored ``posthoc.json`` is compared, as JSON, with a
fresh ``posthoc_evaluate`` of the same run directory. IDENTICAL for every run means
the evaluation, fitness and simulation code now in the tree gives exactly the numbers
the experiment recorded, which is how docs/RESULTS.md shows that runs made in
different sessions (and at different commits) can be compared. It does not repeat
the evolutionary search itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evomorph.config import load_yaml
from evomorph.experiments.design import experiment_from_mapping
from evomorph.experiments.evaluation import posthoc_evaluate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("experiment", type=Path, help="experiment folder holding experiment.yaml")
    ap.add_argument("--workers", type=int, default=4, help="processes (capped at 6)")
    args = ap.parse_args()
    exp = experiment_from_mapping(load_yaml(args.experiment / "experiment.yaml")["experiment"])
    different = 0
    runs = sorted((args.experiment / "runs").iterdir())
    for run in runs:
        new = posthoc_evaluate(run, exp, workers=args.workers, write=False)
        old = json.loads((run / "posthoc.json").read_text())
        same = json.dumps(new, sort_keys=True) == json.dumps(old, sort_keys=True)
        different += not same
        print(run.name, "IDENTICAL" if same else "DIFFERENT", flush=True)
    print(f"{len(runs) - different} identical, {different} different")
    return 1 if different else 0


if __name__ == "__main__":
    sys.exit(main())
