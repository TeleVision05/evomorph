"""Render the reference quadruped on every terrain type into docs/media/terrains.

Usage (from the repo root; offscreen only, no window opens):
    .venv/bin/python scripts/render_terrains.py [--out docs/media/terrains]

Per terrain writes <label>.png (iso/side/front/top views at rest after settling,
the core's ``render_still``), <label>_overview.png (a wide still after the run's
first 4 s, showing the terrain features around the creature), <label>.mp4 (the
full evaluation from the tracking camera) and <label>_sheet.png (8 frames of it).
Prints each run's outcome so the frames can be checked against the numbers.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import imageio.v2 as imageio
import mujoco

from evomorph.config import SimConfig
from evomorph.controllers.base import make_controller
from evomorph.morphology.genome import Genome, reference_quadruped
from evomorph.simulation.environment import EnvSpec
from evomorph.simulation.evaluator import RecordSpec, build_body, settle, spawn
from evomorph.simulation.evaluator import run as run_episode
from evomorph.simulation.suites import TRAIN
from evomorph.utils.random_state import make_rng
from evomorph.visualization.renderer import contact_sheet, render_still, render_video

TERRAINS: tuple[EnvSpec, ...] = (
    EnvSpec(),
    EnvSpec(kind="slope", slope_deg=10.0),
    EnvSpec(kind="slope", slope_deg=-10.0),
    EnvSpec(kind="uneven", roughness=0.04, n_obstacles=20, seed=2),
    EnvSpec(kind="friction", friction=0.2),
    EnvSpec(kind="friction", friction=0.8, friction_spread=2.0, seed=3),
    TRAIN[-2],  # the first random training terrain (seed 4)
)


def overview(genome: Genome, env: EnvSpec, path: Path, seconds: float = 4.0) -> Path:
    """Wide still of the creature on its terrain after ``seconds`` of walking."""
    sim = replace(SimConfig(), duration=seconds)
    body = build_body(genome, env, sim)
    spawn(body, sim, make_rng(0))
    settle(body, sim, None)
    run_episode(body, make_controller(genome), sim, None)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = body.data.xpos[body.torso_body]
    cam.distance, cam.azimuth, cam.elevation = 3.0, 120.0, -22.0
    renderer = mujoco.Renderer(body.model, height=540, width=960)
    try:
        renderer.update_scene(body.data, camera=cam)
        image = renderer.render().copy()
    finally:
        renderer.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, image)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("docs/media/terrains"))
    args = ap.parse_args()
    g = reference_quadruped()
    for env in TERRAINS:
        name = env.label()
        render_still(g, args.out / f"{name}.png", env=env)
        overview(g, env, args.out / f"{name}_overview.png")
        spec = RecordSpec(width=640, height=480, fps=30, include_settle=True)
        r = render_video(g, args.out / f"{name}.mp4", env=env, spec=spec)
        contact_sheet(r.frames, args.out / f"{name}_sheet.png", n=8, cols=4)
        m = r.metrics
        status = r.failure_reason or "completed"
        if not r.valid:
            print(f"{name}: INVALID {status}")
            continue
        print(
            f"{name}: fitness {r.fitness:+.3f} fwd {m['forward_displacement']:+.2f} m "
            f"slip {m['slip_fraction']:.2f} {status}"
        )


if __name__ == "__main__":
    main()
