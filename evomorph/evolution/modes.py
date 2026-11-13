"""What may evolve: ``joint``, ``controller`` or ``morphology`` (brief section 9).

* ``joint``: every gene evolves (Experiment C).
* ``controller``: the body is frozen to a template genome's morphology; only
  controller genes evolve (Experiment A, a fixed quadruped).
* ``morphology``: the controller is frozen to a baseline controller; only body
  genes evolve (Experiment B). When the limb count changes, the baseline is
  adapted by the core's :func:`~evomorph.morphology.genome.repair` (per-limb
  arrays resized, new limbs get 0, i.e. they follow the gait pattern exactly).

The controller ``kind`` gene is never evolved in any mode: which controller
family is under test is an experimental choice, not a mutation.

:class:`Freezer` is applied to every genome the engine creates, so frozen genes
are fixed by construction, not merely by the operators' good behaviour.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

from evomorph.morphology.genome import (
    CONTROLLER_GENES,
    MORPHOLOGY_GENES,
    Genome,
    load_genome,
    reference_hexapod,
    reference_quadruped,
    repair,
)

MODES: tuple[str, ...] = ("joint", "controller", "morphology")
REFERENCE_TEMPLATES = {"quadruped": reference_quadruped, "hexapod": reference_hexapod}
NEVER_EVOLVED: tuple[str, ...] = ("kind",)


def mutable_genes(mode: str) -> tuple[str, ...]:
    """Gene names the operators may change in ``mode``."""
    if mode == "joint":
        names = MORPHOLOGY_GENES + CONTROLLER_GENES
    elif mode == "controller":
        names = CONTROLLER_GENES
    elif mode == "morphology":
        names = MORPHOLOGY_GENES
    else:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    return tuple(n for n in names if n not in NEVER_EVOLVED)


def load_template(source: str) -> Genome:
    """A reference name (``quadruped``, ``hexapod``) or a genome JSON path."""
    if source in REFERENCE_TEMPLATES:
        return REFERENCE_TEMPLATES[source]()
    try:
        return load_genome(source)
    except FileNotFoundError:
        raise ValueError(
            f"template '{source}' is neither a built-in reference "
            f"({', '.join(REFERENCE_TEMPLATES)}) nor an existing genome file"
        ) from None


@dataclass(frozen=True)
class Freezer:
    """Re-imposes the frozen gene group of ``mode`` from ``template``.

    ``template`` is the frozen body (controller mode) or the frozen baseline
    controller (morphology mode); it is ignored in joint mode. ``kind`` always
    comes from ``kind`` (the controller family under test).
    """

    mode: str = "joint"
    template: Genome | None = None
    kind: str = "cpg"

    def __post_init__(self) -> None:
        mutable_genes(self.mode)  # validates the mode
        if self.mode != "joint" and self.template is None:
            raise ValueError(f"mode '{self.mode}' needs a template genome to freeze")

    def apply(self, genome: Genome) -> Genome:
        """A repaired copy of ``genome`` with the frozen genes set from the template."""
        g = genome
        if self.mode == "controller":
            g = replace(g, morphology=copy.deepcopy(self.template.morphology))
        elif self.mode == "morphology":
            g = replace(g, controller=copy.deepcopy(self.template.controller))
        g = replace(g, controller=replace(g.controller, kind=self.kind))
        return repair(g)
