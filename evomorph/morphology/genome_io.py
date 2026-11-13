"""Genome serialization: JSON-safe dicts, files, and a content hash.

Import these from :mod:`evomorph.morphology.genome`, which re-exports them.

Loading is strict about names and lenient about absence: an unknown gene or
top-level key is an error (a typo must not silently fall back to a default),
while a missing gene takes its default so files written before a gene existed
still load. Missing per-limb arrays are sized to the body's limb count.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from evomorph.morphology.genome import (
    CONTROLLER_GENES,
    GENE_BOUNDS,
    MORPHOLOGY_GENES,
    PER_LIMB_GENES,
    ControllerGenes,
    Genome,
    MorphologyGenes,
)
from evomorph.utils.serialization import dump_json, load_json

SCHEMA_VERSION = 1
TOP_LEVEL_KEYS = ("schema", "id", "parent_ids", "generation", "origin", "morphology", "controller")
_ARRAY_GENES = tuple(
    n for n in CONTROLLER_GENES if GENE_BOUNDS[n].per_limb or GENE_BOUNDS[n].kind == "vector"
)


def _genes(genome: Genome) -> dict[str, dict[str, Any]]:
    m = genome.morphology
    morph = {n: (int(v) if isinstance(v := getattr(m, n), (int, np.integer)) else v)
             for n in MORPHOLOGY_GENES}  # fmt: skip
    ctrl: dict[str, Any] = {}
    for n in CONTROLLER_GENES:
        val = getattr(genome.controller, n)
        ctrl[n] = [float(x) for x in np.asarray(val).ravel()] if n in _ARRAY_GENES else val
    return {"morphology": morph, "controller": ctrl}


def to_dict(genome: Genome) -> dict[str, Any]:
    """JSON-safe dict. Floats keep full precision (exact round trip)."""
    return {
        "schema": SCHEMA_VERSION,
        "id": genome.id,
        "parent_ids": list(genome.parent_ids),
        "generation": int(genome.generation),
        "origin": genome.origin,
        **_genes(genome),
    }


def _reject_unknown(where: str, keys: Any, allowed: tuple[str, ...]) -> None:
    unknown = sorted(set(keys) - set(allowed))
    if unknown:
        raise ValueError(f"genome {where}: unknown keys {unknown} (allowed: {list(allowed)})")


def from_dict(data: dict[str, Any]) -> Genome:
    """Inverse of :func:`to_dict`. Raises ValueError on unknown keys (see module doc)."""
    _reject_unknown("file", data, TOP_LEVEL_KEYS)
    morph_in = dict(data.get("morphology", {}))
    ctrl_in = dict(data.get("controller", {}))
    _reject_unknown("morphology", morph_in, MORPHOLOGY_GENES)
    _reject_unknown("controller", ctrl_in, CONTROLLER_GENES)
    morph = MorphologyGenes(**morph_in)
    for name in _ARRAY_GENES:
        if name in ctrl_in:
            ctrl_in[name] = np.asarray(ctrl_in[name], dtype=float)
        elif name in PER_LIMB_GENES:
            ctrl_in[name] = np.zeros(2 * int(morph.n_pairs))
    return Genome(
        id=str(data.get("id", "unnamed")),
        morphology=morph,
        controller=ControllerGenes(**ctrl_in),
        parent_ids=tuple(str(p) for p in data.get("parent_ids", ())),
        generation=int(data.get("generation", 0)),
        origin=str(data.get("origin", "loaded")),
    )


def genome_hash(genome: Genome) -> str:
    """Stable hex digest of the genes only (not id, parents or origin).

    Two genomes with identical genes hash the same, whatever they are called,
    so it serves for de-duplication and caching evaluations.
    """
    text = json.dumps(_genes(genome), sort_keys=True, separators=(",", ":"), allow_nan=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def save_genome(genome: Genome, path: str | Path) -> Path:
    return dump_json(to_dict(genome), path)


def load_genome(path: str | Path) -> Genome:
    return from_dict(load_json(path))
