"""Tables that describe one creature, built from its genome and stored results.

Pure functions (no Streamlit), so they are unit-tested. Body numbers come
from the same generator helpers the simulator uses (built foot radius, mass
estimate, reach, servo torque limit), so the page shows the body that was
actually simulated, not just the raw genes.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from evomorph.controllers.cpg import gait_phases
from evomorph.evolution.selection import body_plan
from evomorph.morphology.generator import (
    describe_layout,
    estimate_mass,
    foot_radius,
    limb_reach,
    sole_length,
    torque_limit,
)
from evomorph.morphology.genome import GENE_BOUNDS, MORPHOLOGY_GENES, Genome
from evomorph.simulation.metrics import METRIC_DOCS

UNITS = {
    "hip_splay": "deg",
    "lower_splay": "deg",
    "hip_axis_tilt": "deg",
    "hip_range": "deg (+/-)",
    "knee_range": "deg (+/-)",
    "torso_mass_scale": "x 500 kg/m^3",
    "limb_mass_scale": "x 1200 kg/m^3",
}


def body_facts(genome: Genome) -> dict[str, str]:
    """Headline numbers of the built body."""
    m = genome.morphology
    return {
        "body plan": f"{body_plan(genome)} (limbs x segments)",
        "limbs": f"{genome.n_limbs} ({m.n_pairs} symmetric pairs)",
        "segments per limb": str(m.n_segments),
        "actuated joints": str(genome.n_joints),
        "estimated mass": f"{estimate_mass(genome):.2f} kg",
        "limb reach": f"{limb_reach(genome):.3f} m",
        "servo torque limit": f"{torque_limit(genome):.2f} N m",
        "foot": (
            f"capsule sole {sole_length(genome):.3f} m, radius {foot_radius(genome):.3f} m"
            if sole_length(genome) > 0
            else f"sphere, radius {foot_radius(genome):.3f} m"
        ),
    }


def morphology_table(genome: Genome) -> pd.DataFrame:
    """Every morphology gene with its legal range (lengths in m unless stated)."""
    rows = []
    for name in MORPHOLOGY_GENES:
        b = GENE_BOUNDS[name]
        value = getattr(genome.morphology, name)
        rng = f"{b.values}" if b.is_discrete else f"{b.low:g} to {b.high:g}"
        unit = UNITS.get(name, "" if b.is_discrete else "m")
        rows.append({"gene": name, "value": _fmt(value), "unit": unit, "legal range": rng})
    return pd.DataFrame(rows)


def controller_table(genome: Genome) -> pd.DataFrame:
    """Scalar controller genes (angles in rad, frequency in Hz)."""
    c = genome.controller
    rows = [
        {"parameter": "controller kind", "value": c.kind},
        {"parameter": "gait pattern", "value": c.gait},
        {"parameter": "frequency", "value": f"{c.frequency:.3f} Hz"},
        {"parameter": "hip amplitude", "value": f"{c.hip_amplitude:.3f} rad"},
        {"parameter": "knee amplitude", "value": f"{c.knee_amplitude:.3f} rad"},
        {"parameter": "knee phase offset", "value": f"{c.knee_phase_offset:+.3f} rad"},
    ]
    if len(c.params):
        rows.append({"parameter": "parameter vector", "value": f"{len(c.params)} values"})
    return pd.DataFrame(rows)


def joint_table(genome: Genome) -> pd.DataFrame:
    """One row per limb: its joints and its controller phase / bias genes."""
    layout = describe_layout(genome)
    c = genome.controller
    base = gait_phases(c.gait, genome.morphology.n_pairs)
    rows = []
    for limb in layout.limbs:
        i = limb.limb_index
        rows.append(
            {
                "limb": i,
                "pair": limb.pair,
                "side": limb.side,
                "hip joint": limb.hip_joint,
                "knee joint": limb.knee_joint or "none (one segment)",
                "gait phase (deg)": round(math.degrees(base[i]), 1),
                "phase offset (rad)": round(float(c.phase_offsets[i]), 3),
                "hip bias (rad)": round(float(c.hip_bias[i]), 3),
                "knee bias (rad)": round(float(c.knee_bias[i]), 3),
            }
        )
    return pd.DataFrame(rows)


def metric_table(record: dict[str, Any]) -> pd.DataFrame:
    """Every stored metric (mean over the evaluation runs) with its meaning."""
    metrics = record.get("metrics") or {}
    rows = [
        {"metric": k, "value": _fmt(v), "meaning": METRIC_DOCS.get(k, "")}
        for k, v in metrics.items()
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "meaning"])


def run_table(record: dict[str, Any]) -> pd.DataFrame:
    """Fitness of each evaluation run (environment / spawn seed)."""
    fits = list(record.get("seed_fitness") or [])
    labels = list(record.get("run_labels") or [])
    labels += [f"run {i}" for i in range(len(labels), len(fits))]
    return pd.DataFrame({"run": labels[: len(fits)], "fitness": [_fmt(f) for f in fits]})


def terms_table(record: dict[str, Any]) -> pd.DataFrame:
    """Fitness terms; they sum exactly to the fitness (core fitness contract)."""
    terms = record.get("fitness_terms") or {}
    return pd.DataFrame({"term": list(terms), "contribution": [_fmt(v) for v in terms.values()]})


def _fmt(value: Any) -> str:
    if isinstance(value, np.ndarray | list | tuple):
        return "[" + ", ".join(_fmt(v) for v in value) + "]"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:+.4g}" if math.isfinite(value) else "n/a"
    return str(value)


def short_label(record: dict[str, Any]) -> str:
    """'g0005-003 · +0.499 · 6x2 · gen 5' for pickers and captions."""
    m = (record.get("genome") or {}).get("morphology") or {}
    plan = f"{2 * int(m.get('n_pairs', 0))}x{int(m.get('n_segments', 0))}" if m else "?"
    fit = record.get("fitness")
    fit_s = f"{fit:+.3f}" if isinstance(fit, int | float) and record.get("valid") else "invalid"
    return f"{record.get('id', '?')} · {fit_s} · {plan} · gen {record.get('generation', '?')}"
