"""``summary.md`` of a run: what was run, what came out, and how to reproduce it.

Every number here is read from the run's own records (the outcome and
``metrics.csv``); nothing is computed that is not also stored elsewhere in the
run directory.
"""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

from evomorph.evolution.selection import body_plan

if TYPE_CHECKING:
    from evomorph.evolution.rundir import RunWriter
    from evomorph.evolution.runner import RunOutcome
    from evomorph.evolution.settings import RunConfig

# Metrics shown for the champion: progress, and the exploit checks (does it walk
# on its feet, grip rather than skate, stay on the ground, keep straight).
CHAMPION_METRICS: tuple[str, ...] = (
    "forward_displacement",
    "sustained_velocity",
    "energy_efficiency",
    "cost_of_transport",
    "fell",
    "foot_support_fraction",
    "slip_fraction",
    "airborne_fraction",
    "max_height_gain",
    "max_lateral_deviation",
    "yaw_drift",
    "mean_tilt",
)


def _rows(writer: RunWriter) -> list[dict[str, str]]:
    path = writer.path / "metrics.csv"
    if not path.exists():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def _f(value: str | float) -> str:
    try:
        return f"{float(value):+.3f}"
    except (TypeError, ValueError):
        return str(value)


def _progress_lines(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return ["No generation completed."]
    first, last = rows[0], rows[-1]
    cols = (
        "sims",
        "best_fitness",
        "mean_fitness",
        "median_fitness",
        "n_valid",
        "n_fell",
        "mean_gene_distance",
        "n_body_plans",
        "behaviour_spread",
        "archive_coverage",
        "qd_score",
    )
    lines = ["| | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for label, r in (("first", first), ("last", last)):
        lines.append(
            f"| {label} (step {r['generation']}) | " + " | ".join(r[c] for c in cols) + " |"
        )
    return lines


def _champion_lines(outcome: RunOutcome) -> list[str]:
    ch = outcome.champion
    if ch is None:
        return ["No valid genome was found."]
    g, r = ch.genome, ch.result
    c = g.controller
    lines = [
        f"- id `{g.id}` (generation {g.generation}, origin {g.origin}, "
        f"parents {', '.join(g.parent_ids) or 'none'})",
        f"- body plan {body_plan(g)}, gait {c.gait}, frequency {c.frequency:.2f} Hz",
        f"- fitness {r.fitness:+.4f} (per run: {', '.join(f'{x:+.3f}' for x in r.seed_fitness)})",
    ]
    if outcome.reproduced is not None:
        verdict = "identical" if outcome.reproduced else "DIFFERENT (reproducibility broken)"
        lines.append(f"- re-scored with the saved config: {verdict}")
    if outcome.heldout is not None:
        h = outcome.heldout.result
        lines.append(
            f"- held-out seeds (+1000): fitness {h.fitness:+.4f} "
            f"({'valid' if h.valid else 'INVALID: ' + h.failure_reason})"
        )
    lines.append("")
    lines.append("| metric (mean over runs) | value |")
    lines.append("|---|---|")
    lines += [f"| {k} | {_f(r.metrics.get(k, float('nan')))} |" for k in CHAMPION_METRICS]
    return lines


def write_summary(
    writer: RunWriter,
    cfg: RunConfig,
    outcome: RunOutcome,
    workers: int,
    sims_per_eval: int,
    command: str,
) -> None:
    """Write ``summary.md`` into the run directory."""
    evo = cfg.evolution
    rate = outcome.sims_used / outcome.wall_seconds if outcome.wall_seconds > 0 else 0.0
    runs = ", ".join(cfg.eval_spec.labels())
    lines = [
        f"# {outcome.path.name}",
        "",
        f"- algorithm **{evo.algorithm}**, mode **{evo.mode}**, seed {evo.seed}",
        f"- budget {evo.budget} simulations; used {outcome.sims_used} "
        f"({sims_per_eval} per genome: {runs}; aggregate {cfg.eval_spec.aggregate})",
        f"- {outcome.steps} {'generations' if evo.algorithm == 'ga' else 'batches'}, "
        f"{outcome.wall_seconds:.1f} s wall with {workers} workers ({rate:.1f} simulations/s)",
        "",
        "## Progress",
        "",
        *_progress_lines(_rows(writer)),
        "",
        "## Champion",
        "",
        *_champion_lines(outcome),
        "",
        "## Archive",
        "",
        f"{len(outcome.archive)} of {outcome.archive.spec.n_cells} cells filled "
        f"({outcome.archive.coverage:.0%}), QD-score {outcome.archive.qd_score:.2f} "
        f"(offset {outcome.archive.qd_offset:g}); "
        f"axes {', '.join(outcome.archive.spec.descriptors)}.",
        "",
        "## Reproduce",
        "",
        "Command that made this run, and the equivalent from the resolved config",
        "(every override already applied; the new run gets a fresh directory):",
        "",
        "```",
        command or "(run from Python, not the command line)",
        f"python -m evomorph evolve --config {outcome.path / 'config.yaml'}",
        "```",
        "",
    ]
    writer.write_summary("\n".join(lines))
