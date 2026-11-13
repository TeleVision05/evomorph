"""Slow end-to-end check: a real (short-episode) GA run improves on its random start.

Run with ``pytest -m slow``. Uses 3 s episodes and a 1500-simulation budget so
it finishes in a few minutes on 6 workers; the full-length evidence is the
quick-config run documented in docs/EVOLUTION.md.
"""

from __future__ import annotations

import csv

import pytest

from evomorph.evolution.runner import run_evolution
from evomorph.evolution.settings import from_sections

SECTIONS = {
    "sim": {"duration": 3.0, "settle_time": 0.5},
    "evolution": {
        "budget": 1500,
        "population_size": 24,
        "workers": 6,
        "evaluation": {
            "runs": [{"env": {"friction": 0.6}, "seed": 0}, {"env": {"friction": 1.0}, "seed": 1}]
        },
    },
}


@pytest.mark.slow
def test_real_ga_improves_best_and_mean_fitness(tmp_path) -> None:
    out = run_evolution(from_sections(SECTIONS), tmp_path, "slow", out=None)
    rows = list(csv.DictReader((out.path / "metrics.csv").open()))
    first, last = rows[0], rows[-1]
    assert out.sims_used == 1500 and out.reproduced is True
    assert float(last["best_fitness"]) > float(first["best_fitness"])
    assert float(last["mean_fitness"]) > float(first["mean_fitness"])
    m = out.champion.result.metrics
    assert m["fell"] == 0.0 and m["foot_support_fraction"] > 0.9  # walks on its feet
