"""Shared fixtures and helpers."""

from __future__ import annotations

import functools
import itertools
import os
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from evomorph.morphology.genome import Genome, reference_quadruped, repair

LIMB_PAIRS = (1, 2, 3, 4)
SEGMENTS = (1, 2)
ALL_CONFIGS = list(itertools.product(LIMB_PAIRS, SEGMENTS))

# Makes one 16x16 offscreen frame. Run in a child process so that a broken
# OpenGL driver (which can abort rather than raise) cannot take pytest down.
_GL_PROBE = (
    "import mujoco\n"
    "m = mujoco.MjModel.from_xml_string('<mujoco><worldbody/></mujoco>')\n"
    "r = mujoco.Renderer(m, 16, 16)\n"
    "r.update_scene(mujoco.MjData(m)); r.render(); r.close()\n"
)


@functools.cache
def offscreen_gl_available() -> bool:
    """True if MuJoCo can make an offscreen OpenGL context on this machine.

    CI runners have no display: they need MUJOCO_GL=egl or osmesa. The 60 s
    timeout is generous for one tiny frame; a hang counts as unavailable.
    """
    try:
        done = subprocess.run(
            [sys.executable, "-c", _GL_PROBE], capture_output=True, timeout=60, check=False
        )
    except subprocess.TimeoutExpired:
        return False
    return done.returncode == 0


def pytest_collection_modifyitems(config, items) -> None:
    """Skip tests marked ``gl`` when no offscreen context exists, with a counted reason."""
    gl_items = [item for item in items if "gl" in item.keywords]
    if not gl_items or offscreen_gl_available():
        return
    backend = os.environ.get("MUJOCO_GL") or "default"
    skip = pytest.mark.skip(
        reason=f"no offscreen OpenGL context (MUJOCO_GL={backend}); rendering not tested"
    )
    for item in gl_items:
        item.add_marker(skip)


def configured_genome(n_pairs: int, n_segments: int, **morph) -> Genome:
    """Reference quadruped re-shaped to n_pairs x n_segments, repaired to valid."""
    base = reference_quadruped()
    m = replace(base.morphology, n_pairs=n_pairs, n_segments=n_segments, **morph)
    g = replace(base, id=f"cfg-{n_pairs}x{n_segments}", morphology=m)
    return repair(g)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)
