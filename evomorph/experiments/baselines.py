"""The four conditions of the comparison experiment, and condition B's fixed controller.

Brief section 9 compares what is allowed to evolve. Every condition runs the
same engine (:mod:`evomorph.evolution`), on the same training suite, with the
same budget in simulations and the same search seeds; only the gene groups that
may change, and for D the search algorithm, differ:

====  =================  ==========  ===========================================
key   name               algorithm   what evolves
====  =================  ==========  ===========================================
A     controller-only    GA          controller genes; body frozen to the
                                     reference quadruped
B     morphology-only    GA          body genes; controller frozen to the
                                     baseline below
C     joint              GA          everything
D     joint MAP-Elites   MAP-Elites  everything
====  =================  ==========  ===========================================

Condition B's baseline controller
---------------------------------
B's controller is the reference quadruped's hand-designed CPG, unchanged:

* gait ``diagonal``, frequency 1.5 Hz, hip amplitude 0.45 rad, knee amplitude
  0.40 rad, knee phase offset +pi/2 (the foot lifts during the forward swing);
* every per-limb gene (phase offset, hip bias, knee bias) is 0.

It is adapted to each limb count by the core alone, with no tuning: the CPG
computes each limb's phase from the gait pattern and the limb count
(:func:`evomorph.controllers.cpg.gait_phase`), and ``repair`` sizes the per-limb
arrays (all zero). ``diagonal`` gives the standard alternating gait of every
limb count: a 2-limb alternating step, a 4-limb trot, the 6-limb alternating
tripod (diagonal equals tripod for 6 limbs) and an 8-limb alternating
tetrapod. In each, every leg is in antiphase with its neighbours, so half the
feet are down at any time.

Why this is a fair baseline:

1. It is the other half of the same reference creature. A freezes the
   reference body and evolves its controller; B freezes the reference
   controller and evolves the body. Neither condition starts with an advantage
   the other was denied.
2. It was fixed before any experiment ran (it is the core's reference
   controller, written with the core), so it was not tuned to make bodies look
   good or bad.
3. It is not degenerate: it walks the reference quadruped, and its amplitudes
   are angles, so a body that evolves longer legs gets a longer stride from the
   same controller, which is exactly what B can exploit.
4. Its main limit is stated, not hidden: one frequency and one phase pattern
   for every body. If a body needs a different gait to walk, B cannot find it.
   That is the cost of freezing the controller, which is the effect B measures.

How the baseline actually performs on a body of each limb count is measured by
``scripts/baseline_check.py`` (numbers in docs/EXPERIMENT.md).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from evomorph.evolution.modes import Freezer
from evomorph.morphology.genome import Genome, reference_quadruped

# The frozen body of A and the frozen controller of B both come from this
# reference (see the module doc); the engine resolves the name.
REFERENCE = "quadruped"


@dataclass(frozen=True)
class Condition:
    """One experimental condition."""

    key: str  # stable identifier used in folder names and tables
    letter: str  # A-D, as in the brief and the docs
    label: str  # for plots and tables
    algorithm: str  # "ga" | "map_elites"
    mode: str  # "controller" | "morphology" | "joint"


CONDITIONS: tuple[Condition, ...] = (
    Condition("controller", "A", "A controller-only", "ga", "controller"),
    Condition("morphology", "B", "B morphology-only", "ga", "morphology"),
    Condition("joint", "C", "C joint (GA)", "ga", "joint"),
    Condition("joint_me", "D", "D joint (MAP-Elites)", "map_elites", "joint"),
)
CONDITION_KEYS: tuple[str, ...] = tuple(c.key for c in CONDITIONS)


def get_condition(key: str) -> Condition:
    """The condition called ``key``; ValueError with the valid names otherwise."""
    for c in CONDITIONS:
        if c.key == key:
            return c
    raise ValueError(f"there is no condition '{key}'; choose from {', '.join(CONDITION_KEYS)}")


def baseline_controller_genome(body: Genome) -> Genome:
    """``body`` driven by condition B's baseline controller (see the module doc).

    Uses the engine's own :class:`~evomorph.evolution.modes.Freezer`, i.e. the
    exact code path condition B runs, so what this returns is what B evaluates.
    """
    freezer = Freezer("morphology", reference_quadruped(), "cpg")
    return replace(freezer.apply(body), id=f"{body.id}+baseline", origin="baseline")
