"""Evaluate one genome: build, spawn, settle, run the controller, measure.

``evaluate`` is a top-level function whose arguments and result are picklable,
so it can be mapped over a process pool. It never raises for a bad genome: any
problem becomes ``EvalResult(valid=False, failure_reason=...)``.

Outcome semantics
-----------------
* ``valid=False``: no trustworthy result (invalid genome, compile error, NaN,
  runaway velocity, launch, ground clipping). ``fitness = INVALID_FITNESS``.
* ``valid=True`` with a non-empty ``failure_reason``: a behavioural failure (the
  creature fell). Metrics are real and the fitness carries the fall penalty.
* ``valid=True`` with an empty ``failure_reason``: completed the run.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

from evomorph.config import FitnessConfig, SimConfig
from evomorph.controllers.base import Controller, Observation, make_controller
from evomorph.morphology.generator import (
    ActuatorLayout,
    build_mjcf,
    describe_layout,
    torque_limit,
)
from evomorph.morphology.genome import Genome
from evomorph.morphology.validator import ModelCompileError, compile_model, validate_genome
from evomorph.simulation.environment import EnvSpec, build_world
from evomorph.simulation.metrics import (
    INVALID_FITNESS,
    Trajectory,
    compute_fitness,
    compute_metrics,
)
from evomorph.utils.random_state import make_rng


@dataclass(frozen=True)
class RecordSpec:
    """Optional frame capture during evaluation."""

    width: int = 640
    height: int = 480
    fps: float = 30.0
    camera: str = "track"  # named camera in the MJCF (tracks the creature's CoM)
    include_settle: bool = False


@dataclass
class EvalResult:
    genome_id: str
    valid: bool
    failure_reason: str
    fitness: float
    metrics: dict[str, float]
    fitness_terms: dict[str, float]
    env: EnvSpec
    seed: int
    sim_seconds: float
    wall_seconds: float
    frames: list[np.ndarray] | None = field(default=None, repr=False)

    @property
    def fell(self) -> bool:
        return bool(self.metrics.get("fell", 0.0))


class SimulationGlitchError(RuntimeError):
    """The physics produced something no real body can do (NaN, runaway, launch)."""


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
    total_mass: float
    torque_limit: float
    ground_z: float


def build_body(genome: Genome, env: EnvSpec, sim: SimConfig) -> Body:
    """Compile ``genome`` into a Body. Raises ModelCompileError on failure."""
    model = compile_model(build_mjcf(genome, env, timestep=sim.timestep))
    layout = describe_layout(genome)
    if model.nu != layout.n_actuators:
        raise ModelCompileError(f"model has {model.nu} actuators, layout {layout.n_actuators}")
    jid = model.actuator_trnid[:, 0]
    foot_of_geom = np.full(model.ngeom, -1, dtype=int)
    for limb in layout.limbs:
        foot_of_geom[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, limb.foot_geom)] = (
            limb.limb_index
        )
    torso_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    return Body(
        model=model,
        data=mujoco.MjData(model),
        layout=layout,
        torso_body=torso_body,
        torso_geom=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "torso_geom"),
        qpos_adr=model.jnt_qposadr[jid].copy(),
        dof_adr=model.jnt_dofadr[jid].copy(),
        foot_of_geom=foot_of_geom,
        is_ground=model.geom_bodyid == 0,
        total_mass=float(model.body_subtreemass[torso_body]),
        torque_limit=torque_limit(genome),
        ground_z=build_world(env).spawn_height,
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
    """(per-limb foot contact, torso contact, any creature-ground contact)."""
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


def check_glitch(body: Body, sim: SimConfig, z_start: float) -> None:
    """Raise SimulationGlitchError if the state is not physically believable."""
    d = body.data
    if not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))):
        raise SimulationGlitchError("non-finite state (NaN/inf)")
    if d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0:
        raise SimulationGlitchError("integrator instability (bad qacc)")
    speed = float(np.linalg.norm(d.qvel[0:3]))
    if speed > sim.max_linear_speed:
        raise SimulationGlitchError(f"runaway linear speed {speed:.1f} m/s")
    spin = float(np.linalg.norm(d.qvel[3:6]))
    if spin > sim.max_angular_speed:
        raise SimulationGlitchError(f"runaway angular speed {spin:.1f} rad/s")
    z = float(d.qpos[2])
    if z - z_start > sim.max_height_gain:
        raise SimulationGlitchError(f"launched {z - z_start:.2f} m above start height")
    if z < body.ground_z - sim.max_ground_penetration:
        raise SimulationGlitchError("torso clipped through the ground")


def settle(body: Body, sim: SimConfig, frames: _FrameSink | None) -> None:
    """Hold the rest pose (all targets 0) for ``settle_time``."""
    body.data.ctrl[:] = 0.0
    z0 = float(body.data.qpos[2])
    n = round(sim.settle_time * sim.control_rate)
    for _ in range(n):
        mujoco.mj_step(body.model, body.data, nstep=sim.n_substeps)
        check_glitch(body, sim, z0)
        if frames is not None and frames.include_settle:
            frames.maybe_capture(body.data)


class _FrameSink:
    """Renders every k-th control step into a list of RGB arrays."""

    def __init__(self, body: Body, spec: RecordSpec, control_rate: float) -> None:
        self.renderer = mujoco.Renderer(body.model, height=spec.height, width=spec.width)
        self.every = max(1, round(control_rate / spec.fps))
        self.camera = spec.camera
        self.include_settle = spec.include_settle
        self.frames: list[np.ndarray] = []
        self._count = 0

    def maybe_capture(self, data: mujoco.MjData) -> None:
        if self._count % self.every == 0:
            self.renderer.update_scene(data, camera=self.camera)
            self.frames.append(self.renderer.render().copy())
        self._count += 1

    def close(self) -> None:
        self.renderer.close()


@dataclass
class _Recorder:
    """Preallocated per-control-step buffers that become a Trajectory."""

    n: int
    n_joints: int
    n_limbs: int
    k: int = 0

    def __post_init__(self) -> None:
        n, j = self.n, self.n_joints
        self.time = np.zeros(n)
        self.pos, self.linvel = np.zeros((n, 3)), np.zeros((n, 3))
        self.quat = np.zeros((n, 4))
        self.jvel, self.force = np.zeros((n, j)), np.zeros((n, j))
        self.feet = np.zeros((n, self.n_limbs), dtype=bool)
        self.torso_c = np.zeros(n, dtype=bool)
        self.any_c = np.zeros(n, dtype=bool)

    def add(self, t: float, body: Body, contacts: tuple[np.ndarray, bool, bool]) -> None:
        d, k = body.data, self.k
        self.time[k] = t
        self.pos[k] = d.qpos[0:3]
        self.quat[k] = d.qpos[3:7]
        self.linvel[k] = d.qvel[0:3]
        self.jvel[k] = d.qvel[body.dof_adr]
        self.force[k] = d.actuator_force
        self.feet[k], self.torso_c[k], self.any_c[k] = contacts
        self.k += 1

    def trajectory(self, body: Body, duration: float, fall_time: float | None) -> Trajectory:
        s = slice(0, self.k)
        return Trajectory(
            time=self.time[s],
            torso_pos=self.pos[s],
            torso_quat=self.quat[s],
            torso_linvel=self.linvel[s],
            joint_vel=self.jvel[s],
            actuator_force=self.force[s],
            foot_contacts=self.feet[s],
            torso_contact=self.torso_c[s],
            any_contact=self.any_c[s],
            duration=duration,
            total_mass=body.total_mass,
            torque_limit=body.torque_limit,
            n_limbs=body.layout.n_limbs,
            fell=fall_time is not None,
            fall_time=fall_time,
        )


def observe(t: float, body: Body, feet: np.ndarray) -> Observation:
    d = body.data
    return Observation(
        time=t,
        joint_pos=d.qpos[body.qpos_adr].copy(),
        joint_vel=d.qvel[body.dof_adr].copy(),
        torso_quat=d.qpos[3:7].copy(),
        foot_contacts=feet.copy(),
    )


def _fall_reason(body: Body, sim: SimConfig, torso_time: float) -> str | None:
    q = body.data.qpos[3:7]
    up_z = 1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2)
    if up_z < math.cos(math.radians(sim.fall_tilt_deg)):
        return "flipped over"
    if torso_time >= sim.fall_contact_time:
        return "torso resting on the ground"
    return None


def run(
    body: Body, controller: Controller, sim: SimConfig, frames: _FrameSink | None
) -> tuple[Trajectory, str | None]:
    """Run the measured episode. Returns (trajectory, fall reason or None)."""
    controller.reset(body.layout)
    n_ctrl = round(sim.duration * sim.control_rate)
    dt = 1.0 / sim.control_rate
    rec = _Recorder(n_ctrl + 1, body.layout.n_actuators, body.layout.n_limbs)
    contacts = ground_contacts(body)
    rec.add(0.0, body, contacts)
    z0, torso_time = float(body.data.qpos[2]), 0.0
    fall_time, reason = None, None
    for k in range(n_ctrl):
        t = k * dt
        body.data.ctrl[:] = controller.act(t, observe(t, body, contacts[0]))
        mujoco.mj_step(body.model, body.data, nstep=sim.n_substeps)
        check_glitch(body, sim, z0)
        contacts = ground_contacts(body)
        rec.add(t + dt, body, contacts)
        if frames is not None:
            frames.maybe_capture(body.data)
        torso_time = torso_time + dt if contacts[1] else 0.0
        reason = _fall_reason(body, sim, torso_time)
        if reason is not None:
            fall_time = t + dt
            break
    return rec.trajectory(body, sim.duration, fall_time), reason


def _invalid(genome_id: str, reason: str, env: EnvSpec, seed: int, t0: float) -> EvalResult:
    return EvalResult(
        genome_id=genome_id,
        valid=False,
        failure_reason=reason,
        fitness=INVALID_FITNESS,
        metrics={},
        fitness_terms={},
        env=env,
        seed=seed,
        sim_seconds=0.0,
        wall_seconds=time.perf_counter() - t0,
    )


def evaluate(
    genome: Genome,
    env: EnvSpec | None = None,
    sim: SimConfig | None = None,
    fitness: FitnessConfig | None = None,
    seed: int = 0,
    record: RecordSpec | bool | None = None,
) -> EvalResult:
    """Simulate ``genome`` on ``env`` and score it. Deterministic in all arguments.

    ``seed`` drives the spawn joint-angle noise (``sim.init_noise``); terrain
    randomness is driven by ``env.seed``. ``record`` (a RecordSpec or True for the
    default) captures RGB frames into ``EvalResult.frames``.
    """
    env, sim, fitness = env or EnvSpec(), sim or SimConfig(), fitness or FitnessConfig()
    t0 = time.perf_counter()
    gid = getattr(genome, "id", "unknown")
    check = validate_genome(genome)
    if not check.ok:
        return _invalid(gid, "invalid genome: " + "; ".join(check.errors), env, seed, t0)
    try:
        body = build_body(genome, env, sim)
        controller = make_controller(genome)
    except (ModelCompileError, NotImplementedError, KeyError, ValueError) as exc:
        return _invalid(gid, f"build failed: {exc}", env, seed, t0)
    spec = RecordSpec() if record is True else (record or None)
    frames = _FrameSink(body, spec, sim.control_rate) if spec else None
    try:
        spawn(body, sim, make_rng(seed))
        settle(body, sim, frames)
        traj, fall = run(body, controller, sim, frames)
    except SimulationGlitchError as exc:
        result = _invalid(gid, f"simulation glitch: {exc}", env, seed, t0)
        result.sim_seconds = float(body.data.time)
        return result
    finally:
        if frames is not None:
            frames.close()
    metrics = compute_metrics(traj)
    score, terms = compute_fitness(metrics, fitness)
    return EvalResult(
        genome_id=gid,
        valid=True,
        failure_reason=f"fell at t={traj.survival_time:.2f}s: {fall}" if fall else "",
        fitness=score,
        metrics=metrics,
        fitness_terms=terms,
        env=env,
        seed=seed,
        sim_seconds=float(body.data.time),
        wall_seconds=time.perf_counter() - t0,
        frames=frames.frames if frames is not None else None,
    )
