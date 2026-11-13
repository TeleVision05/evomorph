"""Controller interface shared by every locomotion controller.

The simulator only ever calls ``reset(layout)`` once and then ``act(t, obs)`` at
the control rate. ``act`` returns joint position targets in radians, one per
actuator in ``ActuatorLayout`` order. A new controller (e.g. a neural policy)
subclasses :class:`Controller`, registers itself with :func:`register_controller`
under a ``genome.controller.kind`` name, and needs no simulator changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from evomorph.morphology.generator import ActuatorLayout
from evomorph.morphology.genome import Genome, add_controller_kind


def _zeros3() -> np.ndarray:
    return np.zeros(3)


@dataclass
class Observation:
    """What a controller may sense at time ``time`` (s since measurement start).

    Joint arrays follow actuator order. ``torso_quat`` is MuJoCo's (w, x, y, z).
    ``foot_contacts`` has one bool per limb in limb order: whether that foot
    touched the ground at any physics step of the last control interval.
    ``torso_pos`` and ``torso_linvel`` are world frame; ``torso_angvel`` is in
    the torso's own frame (MuJoCo free-joint convention), so a gyro-like sensor.
    """

    time: float
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    torso_quat: np.ndarray
    foot_contacts: np.ndarray
    torso_pos: np.ndarray = field(default_factory=_zeros3)
    torso_linvel: np.ndarray = field(default_factory=_zeros3)
    torso_angvel: np.ndarray = field(default_factory=_zeros3)


class Controller(ABC):
    """Base class: maps (time, observation) to joint position targets."""

    layout: ActuatorLayout | None = None

    def reset(self, layout: ActuatorLayout) -> None:
        """Bind to a body. Called once before the first ``act``."""
        self.layout = layout

    @abstractmethod
    def act(self, t: float, obs: Observation) -> np.ndarray:
        """Joint position targets (rad), shape ``(layout.n_actuators,)``."""


ControllerFactory = Callable[[Genome], Controller]
_REGISTRY: dict[str, ControllerFactory] = {}


def register_controller(kind: str) -> Callable[[ControllerFactory], ControllerFactory]:
    """Decorator registering a factory ``genome -> Controller`` for ``kind``.

    Registering also makes ``kind`` a legal value of the genome's ``kind`` gene
    (``GENE_BOUNDS["kind"]``), so validation and repair accept it. A controller
    that needs its own parameters (e.g. network weights) keeps them in
    ``genome.controller.params``, a free-length vector gene.
    """

    def deco(factory: ControllerFactory) -> ControllerFactory:
        _REGISTRY[kind] = factory
        add_controller_kind(kind)
        return factory

    return deco


def registered_kinds() -> tuple[str, ...]:
    _ensure_builtin()
    return tuple(sorted(_REGISTRY))


def _ensure_builtin() -> None:
    # Importing the module registers the built-in CPG controller.
    import evomorph.controllers.cpg  # noqa: F401


def make_controller(genome: Genome) -> Controller:
    """Instantiate the controller named by ``genome.controller.kind``."""
    _ensure_builtin()
    kind = genome.controller.kind
    if kind not in _REGISTRY:
        raise KeyError(f"no controller registered for kind '{kind}' (have {sorted(_REGISTRY)})")
    return _REGISTRY[kind](genome)
