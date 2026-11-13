"""Rendered images and clips for the dashboard, cached on disk.

Every picture is made by ``dashboard_render`` in a child process from a real
genome (offscreen ``mujoco.Renderer``; no window). Files are cached under
``<results root>/.dashboard_cache/`` (or ``$EVOMORPH_DASHBOARD_CACHE``),
never inside a run directory, so the dashboard does not change any run.

The cache key covers everything that changes the picture: the genes (via
``genome_hash``, so two ids with identical genes share a render), the
environment, the spawn seed, the sim config (settle time etc.), the fitness
config for clips (their side-car JSON holds a fitness), the render kind and
``RENDER_VERSION`` (bump it when the render code changes).

The interactive viewer is a separate process too: ``python -m evomorph
inspect --genome F --view`` with the run's environment and config.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from evomorph.morphology.genome import from_dict, genome_hash
from evomorph.simulation.environment import EnvSpec

RENDER_VERSION = 2  # 2: clips play in real time (capture-rate fps)
KINDS = {"thumb": ".png", "still": ".png", "clip": ".mp4"}
# Child-process time limit: fixed start-up (importing MuJoCo) plus a generous
# per-job allowance. A 10 s clip re-simulates in a few seconds on this laptop,
# a thumbnail in well under one; the margin covers a heavily loaded machine.
STARTUP_SECONDS = 60
SECONDS_PER_JOB = {"thumb": 15, "still": 15, "clip": 120}


@dataclass(frozen=True)
class Target:
    """A creature to draw: its genes and the run settings it was scored with."""

    genome: dict[str, Any]
    env: EnvSpec = field(default_factory=EnvSpec)
    seed: int = 0
    sim: dict[str, Any] = field(default_factory=dict)
    fitness: dict[str, Any] = field(default_factory=dict)

    @property
    def genome_id(self) -> str:
        return str(self.genome.get("id", "?"))


def cache_dir(results_root: str | Path) -> Path:
    env = os.environ.get("EVOMORPH_DASHBOARD_CACHE")
    return Path(env) if env else Path(results_root) / ".dashboard_cache"


def media_key(target: Target, kind: str) -> str:
    """Stable 20-hex key of one render (see the module doc for what it covers)."""
    parts = {
        "v": RENDER_VERSION,
        "kind": kind,
        "genes": genome_hash(from_dict(target.genome)),
        "env": asdict(target.env),
        "seed": target.seed if kind == "clip" else 0,  # stills always settle with seed 0
        "sim": target.sim,
        "fitness": target.fitness if kind == "clip" else {},
    }
    text = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:20]


def media_path(cache: Path, target: Target, kind: str) -> Path:
    return cache / kind / f"{media_key(target, kind)}{KINDS[kind]}"


def cached(cache: Path, target: Target, kind: str) -> Path | None:
    p = media_path(cache, target, kind)
    return p if p.is_file() else None


def clip_meta(path: Path) -> dict[str, Any] | None:
    """The re-simulation numbers saved next to a clip, if any."""
    meta = path.with_suffix(".json")
    try:
        return json.loads(meta.read_text())
    except (OSError, ValueError):
        return None


def _job(cache: Path, target: Target, kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "genome": target.genome,
        "env": asdict(target.env),
        "seed": target.seed,
        "sim": target.sim,
        "fitness": target.fitness,
        "out": str(media_path(cache, target, kind)),
    }


@dataclass
class RenderReport:
    paths: dict[str, Path | None]  # media key -> file (None if it could not be made)
    errors: list[str]


def render(cache: Path, targets: list[Target], kind: str) -> RenderReport:
    """Make every missing ``kind`` render of ``targets`` in one child process."""
    todo, seen = [], set()
    for t in targets:
        key = media_key(t, kind)
        if key not in seen and cached(cache, t, kind) is None:
            seen.add(key)
            todo.append(_job(cache, t, kind))
    errors: list[str] = []
    if todo:
        errors = _run_worker(cache, todo, kind)
    paths = {media_key(t, kind): cached(cache, t, kind) for t in targets}
    return RenderReport(paths, errors)


def _run_worker(cache: Path, jobs: list[dict[str, Any]], kind: str) -> list[str]:
    jobs_dir = cache / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps([j["out"] for j in jobs]).encode()).hexdigest()[:16]
    jobs_file = jobs_dir / f"{kind}-{key}.json"
    jobs_file.write_text(json.dumps(jobs, default=str))
    timeout = STARTUP_SECONDS + SECONDS_PER_JOB[kind] * len(jobs)
    env = {**os.environ, "MPLBACKEND": "Agg"}
    cmd = [sys.executable, "-m", "evomorph.visualization.dashboard_render", str(jobs_file)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return [f"rendering took longer than {timeout} s and was stopped"]
    finally:
        jobs_file.unlink(missing_ok=True)
    errors = [ln for ln in proc.stdout.splitlines() if ln.startswith("failed")]
    if proc.returncode not in (0, 1):
        errors.append(f"the render process stopped unexpectedly (exit code {proc.returncode})")
    return errors


# ---------------------------------------------------------------------------
# Interactive viewer (a separate process with its own window)
# ---------------------------------------------------------------------------


def genome_file(cache: Path, genome: dict[str, Any]) -> Path:
    """Write ``genome`` to a JSON file the CLI can load; returns its path."""
    out = cache / "genomes" / f"{genome_hash(from_dict(genome))}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(genome, indent=2))
    return out


def viewer_command(genome_path: Path, env: EnvSpec, seed: int, config: Path | None) -> list[str]:
    """The ``evomorph inspect --view`` command for one creature in one environment."""
    cmd = [sys.executable, "-m", "evomorph", "inspect", "--genome", str(genome_path), "--view"]
    if config is not None:
        cmd += ["--config", str(config)]
    cmd += ["--env", env.kind, "--friction", repr(env.friction), "--seed", str(seed)]
    cmd += ["--slope", repr(env.slope_deg), "--roughness", repr(env.roughness)]
    cmd += ["--obstacles", str(env.n_obstacles), "--env-seed", str(env.seed)]
    cmd += ["--friction-spread", repr(env.friction_spread)]
    return cmd


def launch_viewer(cmd: list[str]) -> subprocess.Popen:
    """Start the viewer detached from the dashboard (its own session; output discarded)."""
    return subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
