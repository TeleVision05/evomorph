"""Offscreen render worker for the dashboard, run as its own process.

    python -m evomorph.visualization.dashboard_render JOBS.json

``JOBS.json`` is a list of jobs written by ``dashboard_media``. Each job names
a genome (as a dict), the environment and spawn seed, the run's sim and
fitness config, an output path and a kind:

* ``thumb``: the iso view of the creature at rest after settling (png);
* ``still``: iso / side / front / top views at rest (png);
* ``clip``: the full scored simulation (settle + run) re-simulated and
  recorded (mp4), plus a JSON next to it with the fitness and key metrics of
  that re-simulation, so the page can show they match the stored numbers.

Why a separate process: Streamlit runs page code on worker threads, and an
OpenGL context or a crash inside MuJoCo should not take the web server down.
Rendering uses ``mujoco.Renderer`` (offscreen; no window opens). Each output
is written to a temporary name and renamed, so a half-written file is never
served from the cache.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")  # never open a matplotlib window

CLIP_SIZE = (480, 360)  # px; small enough that a 330-frame mp4 stays near 1 MB
CLIP_FPS = 30
THUMB_SIZE = (400, 300)  # px per view
CLIP_METRICS = (
    "forward_displacement",
    "sustained_velocity",
    "energy_efficiency",
    "slip_fraction",
    "fell",
)


def _tmp(out: Path) -> Path:
    return out.with_name(f".tmp-{os.getpid()}-{out.name}")


def _configs(job: dict[str, Any]):
    from evomorph.config import FitnessConfig, SimConfig, from_mapping
    from evomorph.morphology.genome import from_dict
    from evomorph.simulation.environment import EnvSpec

    genome = from_dict(job["genome"])
    env = EnvSpec(**(job.get("env") or {}))
    sim = from_mapping(SimConfig, job.get("sim"))
    fitness = from_mapping(FitnessConfig, job.get("fitness"))
    return genome, env, sim, fitness


def render_image(job: dict[str, Any]) -> Path:
    import imageio.v2 as imageio

    from evomorph.visualization.renderer import render_views

    genome, env, sim, _ = _configs(job)
    tiles = render_views(genome, env, sim, size=THUMB_SIZE)
    if job["kind"] == "thumb":
        tiles = tiles[:, : THUMB_SIZE[0]]  # the first view is the iso view
    out = Path(job["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(out)
    imageio.imwrite(tmp, tiles, format="png")
    tmp.replace(out)
    return out


def render_clip(job: dict[str, Any]) -> Path:
    from evomorph.simulation.evaluator import RecordSpec, evaluate
    from evomorph.visualization.renderer import write_video

    genome, env, sim, fitness = _configs(job)
    rec = RecordSpec(width=CLIP_SIZE[0], height=CLIP_SIZE[1], fps=CLIP_FPS, include_settle=True)
    result = evaluate(genome, env, sim, fitness, seed=int(job.get("seed", 0)), record=rec)
    out = Path(job["out"])
    meta = {
        "genome_id": genome.id,
        "env": env.label(),
        "seed": int(job.get("seed", 0)),
        "valid": result.valid,
        "failure_reason": result.failure_reason,
        "fitness": result.fitness,
        "metrics": {k: result.metrics.get(k) for k in CLIP_METRICS},
    }
    meta_path = out.with_suffix(".json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    if result.frames:
        tmp = _tmp(out)
        write_video(result.frames, tmp, fps=rec.fps)
        tmp.replace(out)
    meta_path.write_text(json.dumps(meta, indent=2))
    return out


def run_job(job: dict[str, Any]) -> str:
    """Render one job; returns a one-line status (errors are reported, not raised)."""
    try:
        path = render_clip(job) if job["kind"] == "clip" else render_image(job)
    except Exception as exc:  # one bad creature must not stop the batch
        return f"failed {job.get('out')}: {type(exc).__name__}: {exc}"
    return f"wrote {path}"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m evomorph.visualization.dashboard_render JOBS.json")
        return 2
    jobs = json.loads(Path(args[0]).read_text())
    failed = 0
    for job in jobs:
        line = run_job(job)
        failed += line.startswith("failed")
        print(line, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
