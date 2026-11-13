"""Genetic representation of a creature: body (morphology) plus controller.

``GENE_BOUNDS`` is the single source of truth for every gene's legal range. The
evolution stage mutates against it, :func:`random_genome` samples from it and
:func:`repair` clips/snaps into it. Lengths are metres, angles are degrees in
morphology genes (human-readable body plans) and radians in controller genes
(they feed ``sin`` directly).

Per-limb arrays (``phase_offsets``, ``hip_bias``, ``knee_bias``) are ordered by
limb index ``2 * pair + side`` with pair 0 at the front and side 0 = left,
1 = right. Their length must equal ``n_limbs``; :func:`repair` resizes them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from evomorph.utils.serialization import dump_json, load_json

GAITS: tuple[str, ...] = ("alternating", "diagonal", "tripod", "metachronal")
CONTROLLER_KINDS: tuple[str, ...] = ("cpg",)
MIN_SOLE_LENGTH = 0.01  # m; shorter foot_length values build a sphere (point) foot


@dataclass(frozen=True)
class GeneBound:
    """Legal values of one gene.

    ``kind`` is ``"float"`` (continuous in [low, high]), ``"int"`` or ``"choice"``
    (one of ``values``). ``per_limb`` genes are float arrays of length n_limbs,
    each element bounded by [low, high].
    """

    group: str  # "morphology" or "controller"
    kind: str
    low: float = 0.0
    high: float = 0.0
    values: tuple[Any, ...] = ()
    per_limb: bool = False

    @property
    def is_discrete(self) -> bool:
        return self.kind in ("int", "choice")

    @property
    def span(self) -> float:
        return self.high - self.low


def _f(group: str, low: float, high: float, per_limb: bool = False) -> GeneBound:
    return GeneBound(group=group, kind="float", low=low, high=high, per_limb=per_limb)


_M, _C = "morphology", "controller"

GENE_BOUNDS: dict[str, GeneBound] = {
    # --- morphology: discrete ---------------------------------------------
    # 1..4 symmetric pairs = 2/4/6/8 limbs, the counts the brief asks for.
    "n_pairs": GeneBound(_M, "int", 1, 4, values=(1, 2, 3, 4)),
    # One segment = hip only; two = hip + knee. Changes the real joint count.
    "n_segments": GeneBound(_M, "int", 1, 2, values=(1, 2)),
    # --- morphology: torso (m) -------------------------------------------
    "torso_length": _f(_M, 0.15, 0.80),  # mouse-robot to small-dog length; fits 4 hip pairs
    "torso_width": _f(_M, 0.08, 0.40),  # narrower tips over; wider than ~0.4 is a plank
    "torso_height": _f(_M, 0.03, 0.15),  # thin plate to boxy body; taller raises CoM
    "torso_mass_scale": _f(_M, 0.5, 2.0),  # x 500 kg/m^3 hollow shell: foam to dense payload
    # --- morphology: limbs (m) -------------------------------------------
    "upper_limb_length": _f(_M, 0.05, 0.35),  # stub to long femur; >0.35 is unstable lever
    "lower_limb_length": _f(_M, 0.05, 0.35),  # same envelope for the shank
    "limb_thickness": _f(_M, 0.010, 0.035),  # capsule radius: wire-thin to stout
    "foot_radius": _f(_M, 0.015, 0.050),  # contact sphere: point foot to pad
    # Sole length along the body axis (m). Below MIN_SOLE_LENGTH the foot is a
    # sphere; above, a capsule of radius foot_radius lying fore-aft. Two mirrored
    # point feet share one x, so a biped's support is a line and it pitches over
    # like a seesaw; a sole gives it an area. 0.15 m is a long plantigrade foot.
    "foot_length": _f(_M, 0.0, 0.15),
    "limb_mass_scale": _f(_M, 0.5, 2.0),  # x 1200 kg/m^3 (motors + tube): light to heavy legs
    # --- morphology: posture and joints (deg) -----------------------------
    # Upper segment angle below horizontal: 0 = sprawled (ant/lizard), 80 = nearly
    # vertical pillar (mammal). Steeper than 80 leaves a 1-segment leg no lever to push.
    "hip_splay": _f(_M, 0.0, 80.0),
    # Lower segment angle below horizontal. Below 20 the shank cannot carry load;
    # at 90 the knee's first-order motion is purely lateral, so 85 is the cap.
    "lower_splay": _f(_M, 20.0, 85.0),
    # Tilts the hip axis toward the body's long axis so a forward hip swing also
    # lifts the foot. This is what lets a one-segment (hip only) leg step rather
    # than only scrape; +-35 deg keeps most of the swing in the walking direction.
    "hip_axis_tilt": _f(_M, -35.0, 35.0),
    "hip_range": _f(_M, 15.0, 80.0),  # +- joint limit; <15 barely swings, >80 folds into body
    "knee_range": _f(_M, 15.0, 80.0),  # +- joint limit around the rest bend
    # --- controller --------------------------------------------------------
    "kind": GeneBound(_C, "choice", values=CONTROLLER_KINDS),
    "gait": GeneBound(_C, "choice", values=GAITS),
    # Animals of 0.2-1 m stride at roughly 0.5-3 Hz; above 3 Hz an open-loop
    # sine mostly makes the body buzz rather than step.
    "frequency": _f(_C, 0.3, 3.0),
    "hip_amplitude": _f(_C, 0.0, 0.9),  # rad; actual swing is also clipped by hip_range
    "knee_amplitude": _f(_C, 0.0, 0.9),  # rad; unused by one-segment limbs
    "knee_phase_offset": _f(_C, -math.pi, math.pi),  # rad; +pi/2 lifts during forward swing
    "phase_offsets": _f(_C, -math.pi / 2, math.pi / 2, per_limb=True),  # rad, added to gait
    "hip_bias": _f(_C, -0.35, 0.35, per_limb=True),  # rad posture offset (~20 deg)
    "knee_bias": _f(_C, -0.35, 0.35, per_limb=True),  # rad posture offset (~20 deg)
}

MORPHOLOGY_GENES: tuple[str, ...] = tuple(k for k, b in GENE_BOUNDS.items() if b.group == _M)
CONTROLLER_GENES: tuple[str, ...] = tuple(k for k, b in GENE_BOUNDS.items() if b.group == _C)
PER_LIMB_GENES: tuple[str, ...] = tuple(k for k, b in GENE_BOUNDS.items() if b.per_limb)


@dataclass
class MorphologyGenes:
    """Body plan. Defaults are a mid-range quadruped."""

    n_pairs: int = 2
    n_segments: int = 2
    torso_length: float = 0.45
    torso_width: float = 0.22
    torso_height: float = 0.07
    torso_mass_scale: float = 1.0
    upper_limb_length: float = 0.16
    lower_limb_length: float = 0.20
    limb_thickness: float = 0.02
    foot_radius: float = 0.025
    foot_length: float = 0.0  # point (sphere) foot; JSON saved before this gene loads as 0
    limb_mass_scale: float = 1.0
    hip_splay: float = 10.0
    lower_splay: float = 60.0
    hip_axis_tilt: float = 0.0
    hip_range: float = 45.0
    knee_range: float = 45.0

    @property
    def n_limbs(self) -> int:
        return 2 * int(self.n_pairs)

    @property
    def n_joints(self) -> int:
        return self.n_limbs * int(self.n_segments)


@dataclass
class ControllerGenes:
    """CPG parameters. Per-limb arrays have one entry per limb (see module doc)."""

    kind: str = "cpg"
    gait: str = "diagonal"
    frequency: float = 1.0
    hip_amplitude: float = 0.4
    knee_amplitude: float = 0.3
    knee_phase_offset: float = math.pi / 2
    phase_offsets: np.ndarray = field(default_factory=lambda: np.zeros(4))
    hip_bias: np.ndarray = field(default_factory=lambda: np.zeros(4))
    knee_bias: np.ndarray = field(default_factory=lambda: np.zeros(4))


@dataclass
class Genome:
    """One individual. ``origin`` records how it was made (random, reference, mutation...)."""

    id: str
    morphology: MorphologyGenes
    controller: ControllerGenes
    parent_ids: tuple[str, ...] = ()
    generation: int = 0
    origin: str = "random"

    @property
    def n_limbs(self) -> int:
        return self.morphology.n_limbs

    @property
    def n_joints(self) -> int:
        return self.morphology.n_joints

    def to_dict(self) -> dict[str, Any]:
        return to_dict(self)


# ----------------------------------------------------------------------------
# Sampling and repair
# ----------------------------------------------------------------------------


def _sample(bound: GeneBound, rng: np.random.Generator, n_limbs: int) -> Any:
    if bound.is_discrete:
        return bound.values[int(rng.integers(len(bound.values)))]
    if bound.per_limb:
        return rng.uniform(bound.low, bound.high, size=n_limbs)
    return float(rng.uniform(bound.low, bound.high))


def random_genome(
    rng: np.random.Generator, genome_id: str | None = None, generation: int = 0
) -> Genome:
    """Sample every gene uniformly within ``GENE_BOUNDS``.

    Joint bias offsets start at zero (a neutral posture) rather than uniform, so a
    random creature is not born crouched or splayed by an arbitrary offset; they
    are still free to evolve within bounds.
    """
    n_pairs = _sample(GENE_BOUNDS["n_pairs"], rng, 0)
    n_limbs = 2 * n_pairs
    morph: dict[str, Any] = {"n_pairs": n_pairs}
    for name in MORPHOLOGY_GENES:
        if name != "n_pairs":
            morph[name] = _sample(GENE_BOUNDS[name], rng, n_limbs)
    ctrl: dict[str, Any] = {}
    for name in CONTROLLER_GENES:
        ctrl[name] = _sample(GENE_BOUNDS[name], rng, n_limbs)
    ctrl["hip_bias"] = np.zeros(n_limbs)
    ctrl["knee_bias"] = np.zeros(n_limbs)
    if genome_id is None:
        genome_id = f"rnd-{int(rng.integers(0, 2**32)):08x}"
    return Genome(
        id=genome_id,
        morphology=MorphologyGenes(**morph),
        controller=ControllerGenes(**ctrl),
        generation=generation,
        origin="random",
    )


def _snap(value: Any, bound: GeneBound) -> Any:
    """Snap a discrete gene to the nearest allowed value (by index for choices)."""
    if value in bound.values:
        return value
    if bound.kind == "int":
        try:
            v = float(value)
        except (TypeError, ValueError):
            return bound.values[0]
        if not math.isfinite(v):
            return bound.values[0]
        return min(bound.values, key=lambda allowed: abs(allowed - v))
    return bound.values[0]


def _clip(value: Any, bound: GeneBound) -> float:
    v = float(value) if value is not None else math.nan
    if not math.isfinite(v):
        return (bound.low + bound.high) / 2.0
    return float(min(max(v, bound.low), bound.high))


def _resize(arr: Any, n: int, bound: GeneBound) -> np.ndarray:
    a = np.asarray(arr if arr is not None else [], dtype=float).ravel()
    out = np.zeros(n)
    k = min(n, a.size)
    out[:k] = a[:k]
    out = np.where(np.isfinite(out), out, 0.0)
    return np.clip(out, bound.low, bound.high)


def repair(genome: Genome) -> Genome:
    """Return a valid copy: snap discrete genes, clip continuous ones, resize arrays.

    New per-limb entries (when the limb count grew) are 0, i.e. the limb follows
    the gait pattern exactly with a neutral posture. Never raises for any finite
    or non-finite numeric content.
    """
    m, c = genome.morphology, genome.controller
    morph: dict[str, Any] = {}
    for name in MORPHOLOGY_GENES:
        b = GENE_BOUNDS[name]
        val = getattr(m, name)
        morph[name] = int(_snap(val, b)) if b.is_discrete else _clip(val, b)
    n_limbs = 2 * morph["n_pairs"]
    ctrl: dict[str, Any] = {}
    for name in CONTROLLER_GENES:
        b = GENE_BOUNDS[name]
        val = getattr(c, name)
        if b.per_limb:
            ctrl[name] = _resize(val, n_limbs, b)
        elif b.is_discrete:
            ctrl[name] = _snap(val, b)
        else:
            ctrl[name] = _clip(val, b)
    return replace(
        genome,
        morphology=MorphologyGenes(**morph),
        controller=ControllerGenes(**ctrl),
        parent_ids=tuple(str(p) for p in genome.parent_ids),
        generation=int(genome.generation),
    )


# ----------------------------------------------------------------------------
# Hand-designed baselines
# ----------------------------------------------------------------------------


def reference_quadruped() -> Genome:
    """Ant-like quadruped: sprawled thighs, angled shanks, diagonal (trot) gait.

    Hip swings the leg fore/aft about a near-vertical axis; the knee lifts the foot
    a quarter cycle ahead of the hip so each foot is up while it swings forward.
    """
    morph = MorphologyGenes(
        n_pairs=2,
        n_segments=2,
        torso_length=0.40,
        torso_width=0.20,
        torso_height=0.06,
        torso_mass_scale=1.0,
        upper_limb_length=0.14,
        lower_limb_length=0.20,
        limb_thickness=0.018,
        foot_radius=0.025,
        limb_mass_scale=1.0,
        hip_splay=10.0,
        lower_splay=60.0,
        hip_axis_tilt=0.0,
        hip_range=45.0,
        knee_range=45.0,
    )
    ctrl = ControllerGenes(
        kind="cpg",
        gait="diagonal",
        frequency=1.5,
        hip_amplitude=0.45,
        knee_amplitude=0.40,
        knee_phase_offset=math.pi / 2,
        phase_offsets=np.zeros(4),
        hip_bias=np.zeros(4),
        knee_bias=np.zeros(4),
    )
    return Genome(id="ref-quadruped", morphology=morph, controller=ctrl, origin="reference")


def reference_hexapod() -> Genome:
    """Insect-like hexapod with an alternating-tripod gait."""
    morph = MorphologyGenes(
        n_pairs=3,
        n_segments=2,
        torso_length=0.50,
        torso_width=0.16,
        torso_height=0.05,
        torso_mass_scale=1.0,
        upper_limb_length=0.12,
        lower_limb_length=0.18,
        limb_thickness=0.016,
        foot_radius=0.022,
        limb_mass_scale=1.0,
        hip_splay=10.0,
        lower_splay=60.0,
        hip_axis_tilt=0.0,
        hip_range=40.0,
        knee_range=45.0,
    )
    ctrl = ControllerGenes(
        kind="cpg",
        gait="tripod",
        frequency=1.5,
        hip_amplitude=0.40,
        knee_amplitude=0.40,
        knee_phase_offset=math.pi / 2,
        phase_offsets=np.zeros(6),
        hip_bias=np.zeros(6),
        knee_bias=np.zeros(6),
    )
    return Genome(id="ref-hexapod", morphology=morph, controller=ctrl, origin="reference")


# ----------------------------------------------------------------------------
# Serialization
# ----------------------------------------------------------------------------

SCHEMA_VERSION = 1


def to_dict(genome: Genome) -> dict[str, Any]:
    """JSON-safe dict. Floats keep full precision (exact round trip)."""
    m = {name: getattr(genome.morphology, name) for name in MORPHOLOGY_GENES}
    c: dict[str, Any] = {}
    for name in CONTROLLER_GENES:
        val = getattr(genome.controller, name)
        c[name] = [float(x) for x in np.asarray(val).ravel()] if GENE_BOUNDS[name].per_limb else val
    return {
        "schema": SCHEMA_VERSION,
        "id": genome.id,
        "parent_ids": list(genome.parent_ids),
        "generation": int(genome.generation),
        "origin": genome.origin,
        "morphology": {
            k: (int(v) if isinstance(v, (int, np.integer)) else v) for k, v in m.items()
        },
        "controller": c,
    }


def from_dict(data: dict[str, Any]) -> Genome:
    """Inverse of :func:`to_dict`. Missing genes take dataclass defaults."""
    morph_in = dict(data.get("morphology", {}))
    ctrl_in = dict(data.get("controller", {}))
    for name in PER_LIMB_GENES:
        if name in ctrl_in:
            ctrl_in[name] = np.asarray(ctrl_in[name], dtype=float)
    morph = MorphologyGenes(**{k: v for k, v in morph_in.items() if k in MORPHOLOGY_GENES})
    ctrl = ControllerGenes(**{k: v for k, v in ctrl_in.items() if k in CONTROLLER_GENES})
    return Genome(
        id=str(data.get("id", "unnamed")),
        morphology=morph,
        controller=ctrl,
        parent_ids=tuple(str(p) for p in data.get("parent_ids", ())),
        generation=int(data.get("generation", 0)),
        origin=str(data.get("origin", "loaded")),
    )


def save_genome(genome: Genome, path: str | Path) -> Path:
    return dump_json(to_dict(genome), path)


def load_genome(path: str | Path) -> Genome:
    return from_dict(load_json(path))
