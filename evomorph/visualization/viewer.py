"""Interactive MuJoCo viewer for one genome.

Command (works with the plain venv python on macOS; no mjpython needed):

    .venv/bin/python -m evomorph inspect --genome path/to/genome.json --view
    .venv/bin/python -m evomorph inspect --reference quadruped --view

This uses the blocking ``mujoco.viewer.launch``, which owns the main thread and
its own window, with the CPG driven from ``mujoco.set_mjcb_control``. The
non-blocking ``launch_passive`` is not used because on macOS it refuses to run
outside ``mjpython``. The creature spawns exactly as in ``evaluate`` (same
spawn height and seeded noise), stands still for ``settle_time``, then the
controller starts with its smooth one-second ramp. Targets update at the
configured control rate, matching evaluation. Press Backspace in the viewer to
reset: the model's reference pose (``qpos0``) is set to the spawn pose, so the
creature respawns standing on the ground rather than dropping from the height it
was built at, and the controller restarts because it is driven by simulation time.
"""

from __future__ import annotations

import math

import mujoco
import mujoco.viewer

from evomorph.config import SimConfig
from evomorph.controllers.base import make_controller
from evomorph.morphology.genome import Genome
from evomorph.simulation.environment import EnvSpec
from evomorph.simulation.evaluator import build_body, ground_contacts, observe, spawn
from evomorph.utils.random_state import make_rng


def make_control_callback(genome: Genome, body, sim: SimConfig):
    """mjcb_control callback: zero targets while settling, then the controller at its rate."""
    controller = make_controller(genome)
    controller.reset(body.layout)
    state = {"tick": -1}

    def callback(model: mujoco.MjModel, data: mujoco.MjData) -> None:
        t = data.time - sim.settle_time
        if t < 0:
            data.ctrl[:] = 0.0
            state["tick"] = -1
            return
        tick = math.floor(t * sim.control_rate + 1e-9)
        if tick != state["tick"]:
            state["tick"] = tick
            t_ctrl = tick / sim.control_rate
            body.data = data  # the viewer may hand us its own MjData after a reload
            feet = ground_contacts(body)[0]
            data.ctrl[:] = controller.act(t_ctrl, observe(t_ctrl, body, feet))

    return callback


def prepare_body(genome: Genome, env: EnvSpec | None, sim: SimConfig, seed: int):
    """Build and spawn the creature exactly as ``evaluate`` does; reset returns to the spawn."""
    body = build_body(genome, env or EnvSpec(), sim)
    spawn(body, sim, make_rng(seed))
    body.model.qpos0[:] = body.data.qpos  # Backspace (mj_resetData) respawns here
    return body


def view(
    genome: Genome, env: EnvSpec | None = None, sim: SimConfig | None = None, seed: int = 0
) -> None:
    """Open the interactive viewer (blocks until the window is closed)."""
    sim = sim or SimConfig()
    body = prepare_body(genome, env, sim, seed)
    mujoco.set_mjcb_control(make_control_callback(genome, body, sim))
    try:
        mujoco.viewer.launch(body.model, body.data)
    finally:
        mujoco.set_mjcb_control(None)
