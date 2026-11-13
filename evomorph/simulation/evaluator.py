"""Evaluate one genome: build, spawn, settle, run the controller, measure.

``evaluate`` is a top-level function whose arguments and result are picklable,
so it can be mapped over a process pool. It never raises for a bad genome: any
problem becomes ``EvalResult(valid=False, failure_reason=...)``.

The measured run advances one physics step at a time (``stepping.py``):
energy, acceleration, airborne time, contact loads and foot slip are integrated
at the physics rate, and the glitch guards see the peak over every step.

Outcome semantics
-----------------
* ``valid=False``: no trustworthy result (invalid genome, compile error, NaN,
  integrator blow-up, runaway velocity, launch, ground clipping or tunnelling).
  ``fitness = INVALID_FITNESS``.
* ``valid=True`` with a non-empty ``failure_reason``: a behavioural failure (the
  creature fell). Metrics are real and the fitness carries the fall penalty.
* ``valid=True`` with an empty ``failure_reason``: completed the run.

One seed is one noise realisation. Selection takes the maximum over many noisy
scores, so anything that decides survival (elites, archive cells, champions)
should be scored with :func:`evaluate_robust` over several seeds.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import mujoco
import numpy as np

from evomorph.config import FitnessConfig, SimConfig
from evomorph.controllers.base import Controller, make_controller
from evomorph.morphology.genome import Genome
from evomorph.morphology.validator import ModelCompileError, validate_genome
from evomorph.simulation.body import (  # noqa: F401  (re-exported helpers)
    Body,
    build_body,
    ground_contacts,
    lowest_point,
    observe,
    spawn,
)
from evomorph.simulation.environment import EnvSpec
from evomorph.simulation.metrics import (
    INVALID_FITNESS,
    Trajectory,
    compute_fitness,
    compute_metrics,
)
from evomorph.simulation.stepping import (  # re-exported: check_glitch, SimulationGlitchError
    IntervalStepper,
    Peaks,
    SimulationGlitchError,
    check_glitch,
    torso_clearance,
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
    # evaluate_robust only: per-seed fitness, in the order of the seeds evaluated
    seed_fitness: tuple[float, ...] = ()

    @property
    def fell(self) -> bool:
        return bool(self.metrics.get("fell", 0.0))


def _instant_peaks(body: Body) -> Peaks:
    d = body.data
    pen = max(0.0, -float(d.contact.dist.min())) if d.ncon else 0.0
    return Peaks(float(np.linalg.norm(d.qvel[0:3])), float(np.linalg.norm(d.qvel[3:6])), pen)


def settle(body: Body, sim: SimConfig, frames: _FrameSink | None) -> None:
    """Hold the rest pose (all targets 0) for ``settle_time``.

    Nothing is measured while settling, so physics runs in C (``nstep``) and the
    guards check the state at each control step.
    """
    body.data.ctrl[:] = 0.0
    h0 = torso_clearance(body)
    for _ in range(round(sim.settle_time * sim.control_rate)):
        mujoco.mj_step(body.model, body.data, nstep=sim.n_substeps)
        check_glitch(body, sim, h0, _instant_peaks(body))
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

    def add(self, t: float, body: Body, feet: np.ndarray, torso: bool, any_c: bool) -> None:
        d, k = body.data, self.k
        self.time[k] = t
        self.pos[k] = d.qpos[0:3]
        self.quat[k] = d.qpos[3:7]
        self.linvel[k] = d.qvel[0:3]
        self.jvel[k] = d.qvel[body.dof_adr]
        self.force[k] = d.actuator_force
        self.feet[k], self.torso_c[k], self.any_c[k] = feet, torso, any_c
        self.k += 1

    def trajectory(
        self, body: Body, stepper: IntervalStepper, duration: float, fall_time: float | None
    ) -> Trajectory:
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
            extra={"rest_exclusions": int(body.model.nexclude)},
            stats=stepper.stats.truncated(self.k),
        )


class _FallDetector:
    """Flip test plus torso-ground contact accumulated over a sliding window."""

    def __init__(self, sim: SimConfig) -> None:
        self.cos_limit = math.cos(math.radians(sim.fall_tilt_deg))
        self.limit = sim.fall_contact_time
        self.window = np.zeros(max(1, round(sim.fall_contact_window * sim.control_rate)))
        self.k = 0

    def update(self, body: Body, torso_seconds: float) -> str | None:
        """Feed one interval's torso-contact time; return a fall reason or None."""
        self.window[self.k % self.window.size] = torso_seconds
        self.k += 1
        q = body.data.qpos[3:7]
        if 1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2) < self.cos_limit:
            return "flipped over"
        if float(np.sum(self.window)) >= self.limit - 1e-9:
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
    stepper = IntervalStepper(body, sim, n_ctrl + 1)
    feet, torso, touching = ground_contacts(body)
    rec.add(0.0, body, feet, torso, touching)
    h0, falls = torso_clearance(body), _FallDetector(sim)
    fall_time, reason = None, None
    for k in range(n_ctrl):
        t = k * dt
        body.data.ctrl[:] = controller.act(t, observe(t, body, feet))
        check_glitch(body, sim, h0, stepper.advance(k + 1))
        s = stepper.stats
        feet = stepper.feet.copy()
        rec.add(t + dt, body, feet, s.torso_touch[k + 1] > 0, s.airborne[k + 1] < s.n_steps[k + 1])
        if frames is not None:
            frames.maybe_capture(body.data)
        reason = falls.update(body, s.torso_touch[k + 1] * s.dt)
        if reason is not None:
            fall_time = t + dt
            break
    return rec.trajectory(body, stepper, sim.duration, fall_time), reason


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


def evaluate_robust(
    genome: Genome,
    env: EnvSpec | None = None,
    sim: SimConfig | None = None,
    fitness: FitnessConfig | None = None,
    seeds: Sequence[int] = (0, 1, 2),
    aggregate: str = "mean",
    sims: Sequence[SimConfig] = (),
) -> EvalResult:
    """Score ``genome`` over several noise realisations to defeat the winner's curse.

    Runs one evaluation per seed in ``seeds`` with ``sim``, plus one per extra
    config in ``sims`` (for example a halved timestep) using seed ``seeds[0]``.
    Any invalid run makes the whole result invalid with that run's reason.
    ``fitness`` is the mean (``aggregate="mean"``) or minimum (``"min"``) of the
    per-run scores; ``metrics`` and ``fitness_terms`` are per-key means;
    ``failure_reason`` reports how many runs fell; ``seed_fitness`` lists every
    run's score. Picklable like :func:`evaluate`.
    """
    if aggregate not in ("mean", "min"):
        raise ValueError(f"aggregate must be 'mean' or 'min', got {aggregate!r}")
    if not seeds:
        raise ValueError("evaluate_robust needs at least one seed")
    t0 = time.perf_counter()
    runs = [evaluate(genome, env, sim, fitness, seed=s) for s in seeds]
    runs += [evaluate(genome, env, extra, fitness, seed=seeds[0]) for extra in sims]
    bad = next((r for r in runs if not r.valid), None)
    if bad is not None:
        out = _invalid(bad.genome_id, bad.failure_reason, bad.env, bad.seed, t0)
        out.seed_fitness = tuple(r.fitness for r in runs)
        return out
    scores = [r.fitness for r in runs]
    n_fell = sum(r.fell for r in runs)
    return EvalResult(
        genome_id=runs[0].genome_id,
        valid=True,
        failure_reason=f"fell in {n_fell}/{len(runs)} runs" if n_fell else "",
        fitness=float(np.mean(scores) if aggregate == "mean" else np.min(scores)),
        metrics={k: float(np.mean([r.metrics[k] for r in runs])) for k in runs[0].metrics},
        fitness_terms={
            k: float(np.mean([r.fitness_terms[k] for r in runs])) for k in runs[0].fitness_terms
        },
        env=runs[0].env,
        seed=int(seeds[0]),
        sim_seconds=float(sum(r.sim_seconds for r in runs)),
        wall_seconds=time.perf_counter() - t0,
        seed_fitness=tuple(scores),
    )
