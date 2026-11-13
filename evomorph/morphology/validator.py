"""Genome validation and safe MJCF compilation.

``validate_genome`` checks a genome against ``GENE_BOUNDS`` without touching
MuJoCo. ``compile_model`` turns MJCF text into an ``MjModel`` and converts any
compiler failure into :class:`ModelCompileError`, the one exception type the
evaluator catches. An invalid genome therefore becomes an invalid EvalResult with
a reason, never a crash.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from evomorph.morphology.genome import (
    CONTROLLER_GENES,
    GENE_BOUNDS,
    MORPHOLOGY_GENES,
    Genome,
)


class ModelCompileError(RuntimeError):
    """MuJoCo rejected a generated model."""


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok


def _check_scalar(name: str, value: object, errors: list[str]) -> None:
    b = GENE_BOUNDS[name]
    if b.is_discrete:
        if value not in b.values:
            errors.append(f"{name}={value!r} not in {b.values}")
        return
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        errors.append(f"{name}={value!r} is not a number")
        return
    if not math.isfinite(v):
        errors.append(f"{name} is not finite")
    elif not b.low <= v <= b.high:
        errors.append(f"{name}={v:.4g} outside [{b.low:.4g}, {b.high:.4g}]")


def _check_array(name: str, value: object, n_limbs: int | None, errors: list[str]) -> None:
    """Per-limb array (``n_limbs`` entries) or free-length vector (``n_limbs=None``)."""
    b = GENE_BOUNDS[name]
    try:
        arr = np.asarray(value, dtype=float).ravel()
    except (TypeError, ValueError):
        errors.append(f"{name} is not a numeric array")
        return
    if n_limbs is not None and arr.size != n_limbs:
        errors.append(f"{name} has {arr.size} entries, body has {n_limbs} limbs")
    if not np.all(np.isfinite(arr)):
        errors.append(f"{name} has non-finite entries")
    elif arr.size and (arr.min() < b.low or arr.max() > b.high):
        errors.append(f"{name} entries outside [{b.low:.4g}, {b.high:.4g}]")


def validate_genome(genome: Genome) -> ValidationResult:
    """Check every gene against ``GENE_BOUNDS`` and the per-limb array sizes."""
    errors: list[str] = []
    if not isinstance(genome, Genome):
        return ValidationResult(False, [f"not a Genome: {type(genome).__name__}"])
    for name in MORPHOLOGY_GENES:
        _check_scalar(name, getattr(genome.morphology, name, None), errors)
    n_pairs = genome.morphology.n_pairs
    n_limbs = 2 * n_pairs if n_pairs in GENE_BOUNDS["n_pairs"].values else -1
    for name in CONTROLLER_GENES:
        value = getattr(genome.controller, name, None)
        if GENE_BOUNDS[name].kind == "vector":
            _check_array(name, value, None, errors)
        elif GENE_BOUNDS[name].per_limb:
            if n_limbs > 0:
                _check_array(name, value, n_limbs, errors)
        else:
            _check_scalar(name, value, errors)
    return ValidationResult(not errors, errors)


def compile_model(xml: str) -> mujoco.MjModel:
    """Compile MJCF text; any failure is raised as :class:`ModelCompileError`."""
    try:
        return mujoco.MjModel.from_xml_string(xml)
    except Exception as exc:  # mujoco raises ValueError; be defensive about others
        raise ModelCompileError(
            str(exc).strip().splitlines()[0] if str(exc) else repr(exc)
        ) from exc
