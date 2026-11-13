"""Open-loop Central Pattern Generator.

For limb ``i`` (pair ``p``, side ``s``; s = 0 left, 1 right; ``n`` pairs)::

    phi_i       = gait_phase(gait, p, s, n) + phase_offsets[i]
    hip_i(t)    = r(t) * (hip_bias[i]  + A_hip  * sin(2 pi f t + phi_i))
    knee_i(t)   = r(t) * (knee_bias[i] + A_knee * sin(2 pi f t + phi_i + knee_offset))
    r(t)        = 0.5 - 0.5 cos(pi t) for t < 1 s, else 1

``r`` ramps amplitude *and* posture bias in smoothly over the first second, so a
creature can never launch itself off a step change in its targets. Targets are
clipped to the joint ranges. One-segment limbs ignore the knee terms.

Gait patterns (``gait_phase``), with a phase of 0 or pi meaning in step or in
antiphase. Larger phase = that limb leads in time.

``alternating``  phi = pi*s. Left and right antiphase, all limbs on one side in
                 step (a pace). With 2 limbs this is a plain alternating walk.
``diagonal``     phi = pi*((p + s) mod 2). Checkerboard: diagonal neighbours in
                 step (a trot on 4 limbs).
``tripod``       phi = pi*((p + s) mod 2) + pi/2 * [n even and p >= n/2].
                 6 limbs: the classic alternating tripod (L0, R1, L2 vs R0, L1, R2).
                 4 limbs: the rear pair lags a quarter cycle, giving a four-beat
                 walk (LF, RH, RF, LH) that keeps three feet down, which is what
                 "tripod" means for a quadruped. 8 limbs: two alternating
                 tetrapods per half-body, the rear half a quarter cycle behind.
``metachronal``  phi = pi*s + 2 pi (n - 1 - p) / n. A wave running rear to front
                 on each side, sides in antiphase (millipede/centipede ripple).

Where the limb count makes patterns coincide:

* 2 limbs: alternating, diagonal, tripod and metachronal are all phi = pi*s.
* 4 limbs: metachronal is diagonal shifted by a global half cycle, i.e. the same
  trot; alternating and tripod differ.
* 6 limbs: tripod equals diagonal; alternating and metachronal differ.
* 8 limbs: all four patterns differ.
"""

from __future__ import annotations

import math

import numpy as np

from evomorph.controllers.base import Controller, Observation, register_controller
from evomorph.morphology.generator import ActuatorLayout
from evomorph.morphology.genome import GAITS, Genome

RAMP_TIME = 1.0  # s; ~1-3 CPG cycles, slow enough that no target ever steps


def gait_phase(gait: str, pair: int, side: int, n_pairs: int) -> float:
    """Base phase (rad, in [0, 2 pi)) of limb (pair, side) under ``gait``."""
    if gait == "alternating":
        phi = math.pi * side
    elif gait == "diagonal":
        phi = math.pi * ((pair + side) % 2)
    elif gait == "tripod":
        lag = 0.5 * math.pi if (n_pairs % 2 == 0 and n_pairs > 1 and pair >= n_pairs // 2) else 0.0
        phi = math.pi * ((pair + side) % 2) + lag
    elif gait == "metachronal":
        phi = math.pi * side + 2.0 * math.pi * (n_pairs - 1 - pair) / n_pairs
    else:
        raise ValueError(f"unknown gait '{gait}', expected one of {GAITS}")
    return phi % (2.0 * math.pi)


def gait_phases(gait: str, n_pairs: int) -> np.ndarray:
    """Base phases of all limbs in limb order (0L, 0R, 1L, ...)."""
    return np.array([gait_phase(gait, p, s, n_pairs) for p in range(n_pairs) for s in (0, 1)])


def ramp(t: float) -> float:
    """Smooth 0 -> 1 envelope over ``RAMP_TIME`` (zero slope at both ends)."""
    if t <= 0.0:
        return 0.0
    if t >= RAMP_TIME:
        return 1.0
    return 0.5 - 0.5 * math.cos(math.pi * t / RAMP_TIME)


class CPGController(Controller):
    """Sinusoidal CPG driven entirely by ``genome.controller``."""

    def __init__(self, genome: Genome) -> None:
        c = genome.controller
        self.gait = c.gait
        self.frequency = float(c.frequency)
        self.hip_amplitude = float(c.hip_amplitude)
        self.knee_amplitude = float(c.knee_amplitude)
        self.knee_offset = float(c.knee_phase_offset)
        self._offsets = np.asarray(c.phase_offsets, dtype=float)
        self._hip_bias = np.asarray(c.hip_bias, dtype=float)
        self._knee_bias = np.asarray(c.knee_bias, dtype=float)
        self._n_limbs = genome.n_limbs
        self._built = False

    def reset(self, layout: ActuatorLayout) -> None:
        super().reset(layout)
        n = layout.n_limbs
        if n != self._n_limbs or self._offsets.size != n:
            raise ValueError(
                f"controller genes sized for {self._offsets.size} limbs, body has {n}; "
                "call genome.repair() first"
            )
        self.phases = gait_phases(self.gait, layout.n_pairs) + self._offsets
        self._hip_idx = np.array([limb.hip_index for limb in layout.limbs])
        knees = [
            (i, limb.knee_index)
            for i, limb in enumerate(layout.limbs)
            if limb.knee_index is not None
        ]
        self._knee_limbs = np.array([i for i, _ in knees], dtype=int)
        self._knee_idx = np.array([k for _, k in knees], dtype=int)
        self._built = True

    def act(self, t: float, obs: Observation | None = None) -> np.ndarray:
        if not self._built or self.layout is None:
            raise RuntimeError("CPGController.act called before reset(layout)")
        layout = self.layout
        r = ramp(t)
        omega_t = 2.0 * math.pi * self.frequency * t
        out = np.zeros(layout.n_actuators)
        hip = self._hip_bias + self.hip_amplitude * np.sin(omega_t + self.phases)
        out[self._hip_idx] = np.clip(r * hip, -layout.hip_range, layout.hip_range)
        if self._knee_idx.size:
            k = self._knee_limbs
            knee = self._knee_bias[k] + self.knee_amplitude * np.sin(
                omega_t + self.phases[k] + self.knee_offset
            )
            out[self._knee_idx] = np.clip(r * knee, -layout.knee_range, layout.knee_range)
        return out


@register_controller("cpg")
def _make_cpg(genome: Genome) -> Controller:
    return CPGController(genome)
