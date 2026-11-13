"""Offscreen rendering: mp4 of a full evaluation, multi-view stills, contact sheets.

Uses ``mujoco.Renderer`` (offscreen OpenGL; on macOS this works from plain
``python``, no window or mjpython needed). Every image corresponds to a real
simulation of a real genome: videos come from ``evaluate(..., record=...)``.
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from evomorph.config import FitnessConfig, SimConfig
from evomorph.morphology.generator import limb_reach
from evomorph.morphology.genome import Genome
from evomorph.simulation.environment import EnvSpec
from evomorph.simulation.evaluator import (
    EvalResult,
    RecordSpec,
    build_body,
    settle,
    spawn,
)
from evomorph.utils.random_state import make_rng

# (label, azimuth deg, elevation deg) of a free camera. Checked on renders: azimuth
# 90 shows the creature's left side, 180 looks at its nose (front, shows mirror
# symmetry), elevation -89 is top-down.
VIEWS: tuple[tuple[str, float, float], ...] = (
    ("iso", 135.0, -25.0),
    ("side", 90.0, -5.0),
    ("front", 180.0, -5.0),
    ("top", 90.0, -89.0),
)


def write_video(frames: list[np.ndarray], path: str | Path, fps: float = 30.0) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out, fps=fps, codec="libx264", quality=7, macro_block_size=1) as w:
        for frame in frames:
            w.append_data(frame)
    return out


def render_video(
    genome: Genome,
    path: str | Path,
    env: EnvSpec | None = None,
    sim: SimConfig | None = None,
    fitness: FitnessConfig | None = None,
    seed: int = 0,
    spec: RecordSpec | None = None,
) -> EvalResult:
    """Evaluate ``genome`` while recording, write an mp4, return the EvalResult."""
    from evomorph.simulation.evaluator import evaluate

    spec = spec or RecordSpec(include_settle=True)
    result = evaluate(genome, env=env, sim=sim, fitness=fitness, seed=seed, record=spec)
    if result.frames:
        write_video(result.frames, path, fps=spec.fps)
    return result


def _free_camera(lookat: np.ndarray, distance: float, azimuth: float, elev: float):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elev
    return cam


def render_views(
    genome: Genome,
    env: EnvSpec | None = None,
    sim: SimConfig | None = None,
    size: tuple[int, int] = (480, 360),
    settled: bool = True,
) -> np.ndarray:
    """Side-by-side iso/side/front/top views of the creature at rest (after settling)."""
    sim = sim or SimConfig()
    body = build_body(genome, env or EnvSpec(), sim)
    spawn(body, sim, make_rng(0))
    if settled:
        settle(body, sim, None)
    m = genome.morphology
    span = max(m.torso_length, m.torso_width + 2 * limb_reach(genome))
    lookat = body.data.xpos[body.torso_body].copy()
    lookat[2] *= 0.6
    width, height = size
    renderer = mujoco.Renderer(body.model, height=height, width=width)
    tiles = []
    try:
        for _, az, el in VIEWS:
            cam = _free_camera(lookat, 1.6 * span + 0.3, az, el)
            renderer.update_scene(body.data, camera=cam)
            tiles.append(renderer.render().copy())
    finally:
        renderer.close()
    return np.concatenate(tiles, axis=1)


def render_still(genome: Genome, path: str | Path, env: EnvSpec | None = None) -> Path:
    """Write the multi-view rest-pose image to ``path`` (png)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out, render_views(genome, env))
    return out


def contact_sheet(
    frames: list[np.ndarray], path: str | Path, n: int = 8, cols: int = 4
) -> Path | None:
    """Grid of ``n`` evenly spaced frames, for checking motion without a video player."""
    if not frames:
        return None
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    picked = [frames[i] for i in idx]
    rows = []
    for r in range(0, len(picked), cols):
        row = picked[r : r + cols]
        row += [np.zeros_like(picked[0])] * (cols - len(row))
        rows.append(np.concatenate(row, axis=1))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out, np.concatenate(rows, axis=0))
    return out
