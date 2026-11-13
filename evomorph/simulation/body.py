"""A compiled creature (``Body``): build, spawn, contact classification, observation.

Split out of ``evaluator.py``; ``evaluator`` re-exports every name here, so
``from evomorph.simulation.evaluator import build_body, spawn`` keeps working.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from evomorph.config import SimConfig
from evomorph.controllers.base import Observation
from evomorph.morphology.generator import (
    ActuatorLayout,
    build_mjcf,
    describe_layout,
    limb_reach,
    torque_limit,
)
from evomorph.morphology.genome import Genome
from evomorph.morphology.validator import ModelCompileError, compile_model
from evomorph.simulation.environment import EnvSpec, WorldFragments, build_world

# Geom classes for contact bookkeeping (``Body.geom_class``).
GEOM_GROUND, GEOM_FOOT, GEOM_LIMB, GEOM_TORSO = 0, 1, 2, 3


@dataclass
class Body:
    """A compiled creature plus the index maps the run loop needs."""

    model: mujoco.MjModel
    data: mujoco.MjData
    layout: ActuatorLayout
    torso_body: int
    torso_geom: int
    qpos_adr: np.ndarray  # joint qpos address per actuator
    dof_adr: np.ndarray  # joint dof address per actuator
    foot_of_geom: np.ndarray  # geom id -> limb index, -1 if not a foot
    is_ground: np.ndarray  # geom id -> belongs to the world body
    geom_class: np.ndarray  # geom id -> GEOM_GROUND / FOOT / LIMB / TORSO
    total_mass: float
    torque_limit: float
    ground_z: float  # terrain height under the spawn point
    world: WorldFragments
    size: float  # m, torso length + 2 x limb reach: the creature's largest possible extent
    fast_contacts: bool = True  # every contact is pyramidal condim 3 (see stepping.py)

    def ground_height(self, x: float, y: float) -> float:
        """Terrain height under world point (x, y)."""
        return self.world.ground_height(x, y)


def _geom_classes(model: mujoco.MjModel, torso_geom: int, foot_of_geom: np.ndarray) -> np.ndarray:
    cls = np.full(model.ngeom, GEOM_LIMB, dtype=np.int8)
    cls[model.geom_bodyid == 0] = GEOM_GROUND
    cls[foot_of_geom >= 0] = GEOM_FOOT
    cls[torso_geom] = GEOM_TORSO
    return cls


def _check_geom_order(model: mujoco.MjModel) -> None:
    """Contact bookkeeping relies on every ground geom having a lower id than any creature geom."""
    world = model.geom_bodyid == 0
    if not (world.any() and (~world).any()):
        return
    if np.flatnonzero(world).max() > np.flatnonzero(~world).min():
        raise ModelCompileError("ground geoms must precede creature geoms in the model")


def _fast_contacts(model: mujoco.MjModel) -> bool:
    collidable = (model.geom_contype | model.geom_conaffinity) != 0
    return bool(
        model.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL
        and np.all(model.geom_condim[collidable] == 3)
        and model.npair == 0
    )


def build_body(genome: Genome, env: EnvSpec, sim: SimConfig) -> Body:
    """Compile ``genome`` into a Body. Raises ModelCompileError on failure."""
    world = build_world(env)
    model = compile_model(build_mjcf(genome, env, timestep=sim.timestep))
    layout = describe_layout(genome)
    if model.nu != layout.n_actuators:
        raise ModelCompileError(f"model has {model.nu} actuators, layout {layout.n_actuators}")
    _check_geom_order(model)
    jid = model.actuator_trnid[:, 0]
    foot_of_geom = np.full(model.ngeom, -1, dtype=int)
    for limb in layout.limbs:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, limb.foot_geom)
        foot_of_geom[gid] = limb.limb_index
    torso_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    torso_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "torso_geom")
    return Body(
        model=model,
        data=mujoco.MjData(model),
        layout=layout,
        torso_body=torso_body,
        torso_geom=torso_geom,
        qpos_adr=model.jnt_qposadr[jid].copy(),
        dof_adr=model.jnt_dofadr[jid].copy(),
        foot_of_geom=foot_of_geom,
        is_ground=model.geom_bodyid == 0,
        geom_class=_geom_classes(model, torso_geom, foot_of_geom),
        total_mass=float(model.body_subtreemass[torso_body]),
        torque_limit=torque_limit(genome),
        ground_z=world.spawn_height,
        world=world,
        size=genome.morphology.torso_length + 2.0 * limb_reach(genome),
        fast_contacts=_fast_contacts(model),
    )


def lowest_point(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Lowest world z over every collidable creature geom at the current pose."""
    low = math.inf
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == 0 or model.geom_contype[g] == 0:
            continue
        z, rot, size = data.geom_xpos[g][2], data.geom_xmat[g].reshape(3, 3), model.geom_size[g]
        gtype = model.geom_type[g]
        if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            low = min(low, z - size[0])
        elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
            low = min(low, z - abs(rot[2, 2]) * size[1] - size[0])
        elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
            low = min(low, z - float(np.sum(np.abs(rot[2, :]) * size[:3])))
        else:
            low = min(low, z - model.geom_rbound[g])
    return low


def spawn(body: Body, sim: SimConfig, rng: np.random.Generator) -> None:
    """Rest pose (+ seeded joint noise), lowered so the lowest geom sits just above ground."""
    m, d = body.model, body.data
    mujoco.mj_resetData(m, d)
    if sim.init_noise > 0:
        noise = rng.normal(0.0, sim.init_noise, size=body.qpos_adr.size)
        lo, hi = m.jnt_range[m.actuator_trnid[:, 0]].T
        d.qpos[body.qpos_adr] = np.clip(noise, lo, hi)
    mujoco.mj_forward(m, d)
    d.qpos[2] += body.ground_z + sim.spawn_clearance - lowest_point(m, d)
    mujoco.mj_forward(m, d)


def ground_contacts(body: Body) -> tuple[np.ndarray, bool, bool]:
    """Instantaneous (per-limb foot contact, torso contact, any creature-ground contact)."""
    d = body.data
    feet = np.zeros(body.layout.n_limbs, dtype=bool)
    n = d.ncon
    if n == 0:
        return feet, False, False
    g1, g2 = d.contact.geom1[:n], d.contact.geom2[:n]
    gr1, gr2 = body.is_ground[g1], body.is_ground[g2]
    mixed = gr1 ^ gr2
    if not np.any(mixed):
        return feet, False, False
    creature = np.where(gr1, g2, g1)[mixed]
    limbs = body.foot_of_geom[creature]
    feet[limbs[limbs >= 0]] = True
    return feet, bool(np.any(creature == body.torso_geom)), True


def observe(t: float, body: Body, feet: np.ndarray) -> Observation:
    """Controller observation at measured time ``t`` (s since measurement start)."""
    d = body.data
    return Observation(
        time=t,
        joint_pos=d.qpos[body.qpos_adr].copy(),
        joint_vel=d.qvel[body.dof_adr].copy(),
        torso_quat=d.qpos[3:7].copy(),
        foot_contacts=feet.copy(),
        torso_pos=d.qpos[0:3].copy(),
        torso_linvel=d.qvel[0:3].copy(),
        torso_angvel=d.qvel[3:6].copy(),
    )
