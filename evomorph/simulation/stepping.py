"""Advance a Body one control interval at physics resolution, and the glitch guards.

``IntervalStepper.advance`` runs the physics steps of one control interval one at
a time, adding each step's mechanical power, torso acceleration, contact loads
and foot slip into a :class:`StepStats` row, and tracking the peak speed, spin
and contact penetration so the guards see what happened *between* control
steps, not only at their boundaries.

Contact loads and slip come straight from the constraint solver. With MuJoCo's
default pyramidal friction cone and ``condim=3`` each contact owns four
constraint rows along the pyramid edges ``n +- mu t1`` and ``n +- mu t2``. Their
forces sum to the normal force, and their constraint-space velocities
(``efc_vel``) differ pairwise by ``2 mu v.t``, which gives the tangential slip
velocity of the creature's material point at the contact. ``point_velocity`` is
the slower, general cross-check used in the tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from evomorph.config import SimConfig
from evomorph.simulation.body import GEOM_FOOT, GEOM_GROUND, GEOM_TORSO, Body
from evomorph.simulation.stepstats import SLIP_SPEED, StepStats


class SimulationGlitchError(RuntimeError):
    """The physics produced something no real body can do (NaN, runaway, launch)."""


@dataclass
class Peaks:
    """Extremes over the physics steps of one interval (what the guards check)."""

    speed: float = 0.0  # m/s, torso linear speed
    spin: float = 0.0  # rad/s, torso angular speed
    penetration: float = 0.0  # m, deepest creature-ground contact


_EDGES = np.arange(4)


def contact_force_and_slip(body: Body, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(normal force N, tangential slip speed m/s) of contacts ``idx`` against static ground.

    Fast path for pyramidal condim-3 contacts (every contact the generator and
    terrain produce); anything else goes through ``mj_contactForce`` and
    ``point_velocity``.
    """
    m, d = body.model, body.data
    con = d.contact
    adr = con.efc_address[idx]
    if body.fast_contacts and np.all(adr >= 0):
        rows = adr[:, None] + _EDGES
        fn = d.efc_force[rows].sum(axis=1)
        ev = d.efc_vel[rows]
        slip = np.hypot(ev[:, 0] - ev[:, 1], ev[:, 2] - ev[:, 3]) / (2.0 * con.friction[idx, 0])
        return fn, slip
    fn, f6 = np.zeros(idx.size), np.zeros(6)
    for j, i in enumerate(idx):
        if adr[j] >= 0:
            mujoco.mj_contactForce(m, d, int(i), f6)
            fn[j] = f6[0]
    geom = np.max(con.geom[idx], axis=1)
    vel = point_velocity(body, m.geom_bodyid[geom], con.pos[idx])
    nrm = con.frame[idx, 0:3]
    vt = vel - np.sum(vel * nrm, axis=1, keepdims=True) * nrm
    return fn, np.sqrt(np.sum(vt * vt, axis=1))


def point_velocity(body: Body, bodies: np.ndarray, points: np.ndarray) -> np.ndarray:
    """World linear velocity of material points ``points`` (k, 3) fixed to ``bodies`` (k,)."""
    d = body.data
    cvel = d.cvel[bodies]  # (k, 6): angular, linear about the root subtree com
    centre = d.subtree_com[body.torso_body]
    return cvel[:, 3:] + np.cross(cvel[:, :3], points - centre)


class IntervalStepper:
    """Steps one Body and fills a StepStats row per control interval."""

    def __init__(self, body: Body, sim: SimConfig, n_rows: int) -> None:
        self.body = body
        self.n_substeps = sim.n_substeps
        self.dt = float(body.model.opt.timestep)
        self.stats = StepStats.zeros(n_rows, self.dt)
        self.feet = np.zeros(body.layout.n_limbs, dtype=bool)
        self._prev_v = body.data.qvel[0:3].copy()

    def advance(self, row: int) -> Peaks:
        """Run one control interval into ``stats`` row ``row``; return its peaks."""
        b, d, m = self.body, self.body.data, self.body.model
        dt, dof = self.dt, b.dof_adr
        self.feet[:] = False
        acc = np.zeros(10)  # energy, accel_sq, airborne, torso, nonfoot, imp x3, slip, sliding
        speed2 = spin2 = pen = 0.0
        prev = self._prev_v
        for _ in range(self.n_substeps):
            mujoco.mj_step(m, d)
            qv = d.qvel
            acc[0] += float(np.abs(d.actuator_force) @ np.abs(qv[dof]))
            v, w = qv[0:3], qv[3:6]
            dv = v - prev
            acc[1] += float(dv @ dv)
            prev = v.copy()
            speed2 = max(speed2, float(v @ v))
            spin2 = max(spin2, float(w @ w))
            if d.ncon == 0:
                acc[2] += 1.0
            else:
                pen = max(pen, self._contacts(acc))
        self._prev_v = prev
        acc[0] *= dt
        acc[1] /= dt
        acc[5:10] *= dt
        s = self.stats
        s.n_steps[row] = self.n_substeps
        for i, name in enumerate(_ACC_FIELDS):
            getattr(s, name)[row] = acc[i]
        s.max_penetration[row] = pen
        return Peaks(speed=math.sqrt(speed2), spin=math.sqrt(spin2), penetration=pen)

    def _contacts(self, acc: np.ndarray) -> float:
        """Add one step's creature-ground contacts into ``acc`` (impulses without dt).

        Returns the deepest penetration. Ground geoms belong to the world body,
        which MuJoCo numbers first, so the creature geom of a mixed pair is the
        larger id (checked when the Body is built).
        """
        b, d = self.body, self.body.data
        con = d.contact
        geom = con.geom
        cls = b.geom_class[geom]
        ground = cls == GEOM_GROUND
        mixed = ground[:, 0] != ground[:, 1]
        if mixed.all():
            idx = None
            ccls, cgeom = cls.max(axis=1), geom.max(axis=1)
            adr, dist = con.efc_address, con.dist
        elif mixed.any():
            idx = np.flatnonzero(mixed)
            ccls, cgeom = cls[idx].max(axis=1), geom[idx].max(axis=1)
            adr, dist = con.efc_address[idx], con.dist[idx]
        else:
            acc[2] += 1.0
            return 0.0
        if b.fast_contacts and adr.min() >= 0:
            rows = adr[:, None] + _EDGES
            fn = d.efc_force[rows].sum(axis=1)
            ev = d.efc_vel[rows]
            mu = con.friction[:, 0] if idx is None else con.friction[idx, 0]
            slip = np.hypot(ev[:, 0] - ev[:, 1], ev[:, 2] - ev[:, 3]) / (2.0 * mu)
        else:
            fn, slip = contact_force_and_slip(b, np.arange(d.ncon) if idx is None else idx)
        acc[5:8] += np.bincount(ccls, weights=fn, minlength=4)[1:4]  # foot, limb, torso
        is_foot = ccls == GEOM_FOOT
        n_foot = int(is_foot.sum())
        acc[3] += float(GEOM_TORSO in ccls)
        acc[4] += float(n_foot < ccls.size)
        if n_foot:
            ffn, fslip = fn[is_foot], slip[is_foot]
            acc[8] += float(ffn @ fslip)
            acc[9] += float(ffn[fslip > SLIP_SPEED].sum())
            self.feet[b.foot_of_geom[cgeom[is_foot]]] = True
        return max(0.0, -float(dist.min()))


_ACC_FIELDS = (
    "energy",
    "accel_sq",
    "airborne",
    "torso_touch",
    "nonfoot_touch",
    "foot_impulse",
    "limb_impulse",
    "torso_impulse",
    "slip_impulse",
    "sliding_impulse",
)

_BAD_STATE_WARNINGS = (
    (mujoco.mjtWarning.mjWARN_BADQACC, "integrator instability (bad qacc)"),
    (mujoco.mjtWarning.mjWARN_BADQVEL, "integrator instability (bad qvel)"),
    (mujoco.mjtWarning.mjWARN_BADQPOS, "integrator instability (bad qpos)"),
)


def speed_limit(body: Body, sim: SimConfig) -> float:
    """m/s: the absolute cap or ``max_speed_per_size`` body sizes per second, if lower."""
    return min(sim.max_linear_speed, sim.max_speed_per_size * body.size)


def height_gain_limit(body: Body, sim: SimConfig) -> float:
    """m: the absolute launch cap or ``max_height_gain_per_size`` body sizes, if lower."""
    return min(sim.max_height_gain, sim.max_height_gain_per_size * body.size)


def torso_clearance(body: Body) -> float:
    """Torso centre height above the terrain directly beneath it (m)."""
    x, y, z = (float(v) for v in body.data.qpos[0:3])
    return z - body.ground_height(x, y)


def check_glitch(body: Body, sim: SimConfig, clearance_start: float, peaks: Peaks) -> None:
    """Raise SimulationGlitchError if the interval was not physically believable.

    ``clearance_start`` is the torso's height above local ground when measurement
    began; launch and clipping are judged against the terrain under the torso
    (``Body.ground_height``), so slopes and bumps do not trip them.
    """
    d = body.data
    if not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))):
        raise SimulationGlitchError("non-finite state (NaN/inf)")
    for warning, message in _BAD_STATE_WARNINGS:
        if d.warning[warning].number > 0:
            raise SimulationGlitchError(message)
    if peaks.speed > speed_limit(body, sim):
        raise SimulationGlitchError(
            f"runaway linear speed {peaks.speed:.1f} m/s (limit {speed_limit(body, sim):.1f})"
        )
    if peaks.spin > sim.max_angular_speed:
        raise SimulationGlitchError(f"runaway angular speed {peaks.spin:.1f} rad/s")
    if peaks.penetration > sim.max_contact_penetration:
        raise SimulationGlitchError(f"geom sank {peaks.penetration * 1000:.0f} mm into the ground")
    h = torso_clearance(body)
    if h - clearance_start > height_gain_limit(body, sim):
        raise SimulationGlitchError(f"launched {h - clearance_start:.2f} m above start height")
    if h < -sim.max_ground_penetration:
        raise SimulationGlitchError("torso clipped through the ground")
