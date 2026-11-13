"""Terrain kinds: compile, spawn clear of geometry, height_at == collision surface, determinism."""

from __future__ import annotations

import math
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from evomorph.config import SimConfig
from evomorph.morphology.genome import random_genome
from evomorph.simulation import terrain
from evomorph.simulation.body import build_body, spawn
from evomorph.simulation.environment import EnvSpec, build_world
from evomorph.utils.random_state import make_rng
from tests.conftest import ALL_CONFIGS, configured_genome

KIND_SPECS = {
    "slope_up": EnvSpec(kind="slope", slope_deg=12.0),
    "slope_down": EnvSpec(kind="slope", slope_deg=-12.0),
    "uneven": EnvSpec(kind="uneven", roughness=0.05, n_obstacles=25),
    "friction": EnvSpec(kind="friction", friction=0.4),
    "patches": EnvSpec(kind="friction", friction=0.8, friction_spread=3.0),
    "random": EnvSpec(
        kind="random",
        slope_deg=15.0,
        roughness=0.05,
        n_obstacles=25,
        friction=0.8,
        friction_spread=2.0,
    ),
}
# The largest legal rest footprint: longest torso, longest sprawled legs, long soles.
BIGGEST = {
    "torso_length": 0.8,
    "torso_width": 0.4,
    "upper_limb_length": 0.35,
    "lower_limb_length": 0.35,
    "hip_splay": 0.0,
    "lower_splay": 20.0,
    "foot_length": 0.15,
}
SEEDS = (0, 7, 123)


def _with_seed(env: EnvSpec, seed: int) -> EnvSpec:
    return replace(env, seed=seed)


def _world_model(env: EnvSpec, spheres: np.ndarray | None = None, radius: float = 0.03):
    """The terrain alone (plus optional free spheres that collide only with the ground)."""
    w = build_world(env)
    bodies = ""
    for x, y, z in spheres if spheres is not None else []:
        bodies += (
            f'<body pos="{x} {y} {z}"><freejoint/><geom type="sphere" size="{radius}" '
            f'contype="2" conaffinity="1" condim="3" density="1000"/></body>\n'
        )
    xml = (
        f'<mujoco><compiler angle="radian"/><option timestep="0.002" integrator="implicitfast"/>'
        f"<asset>{w.assets}</asset><worldbody>{w.worldbody}{bodies}</worldbody></mujoco>"
    )
    return w, mujoco.MjModel.from_xml_string(xml)


def _probe_points(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Points over the course, dense where features change: landing edges, obstacles, borders."""
    xs = [rng.uniform(-12, 20, n), rng.uniform(-1.2, 1.2, n), rng.uniform(1.0, 12.0, n)]
    ys = [rng.uniform(-6, 6, n), rng.uniform(-1.5, 1.5, n), rng.uniform(-1.5, 1.5, n)]
    xs.append(rng.uniform(55, 59.9, n // 4))  # far end of the course (it stops at 60 m)
    ys.append(rng.uniform(-6, 6, n // 4))
    return np.concatenate(xs), np.concatenate(ys)


@pytest.mark.parametrize("name", sorted(KIND_SPECS))
@pytest.mark.parametrize("seed", SEEDS)
def test_every_kind_compiles_for_every_limb_configuration(name, seed) -> None:
    env = _with_seed(KIND_SPECS[name], seed)
    sim = SimConfig()
    for n_pairs, n_segments in ALL_CONFIGS:
        body = build_body(configured_genome(n_pairs, n_segments), env, sim)
        # The landing is z = 0; a heightfield stores it in float32 (micrometres off).
        assert body.ground_z == pytest.approx(0.0, abs=1e-5)
        assert body.world.height_at is not None


@pytest.mark.parametrize("name", sorted(KIND_SPECS))
def test_height_at_matches_a_downward_ray_cast(name) -> None:
    for seed in SEEDS:
        env = _with_seed(KIND_SPECS[name], seed)
        w, m = _world_model(env)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        xs, ys = _probe_points(np.random.default_rng(seed), 400)
        expected = w.height_at.many(xs, ys)
        gid = np.zeros(1, dtype=np.int32)
        top = 100.0
        for x, y, h in zip(xs, ys, expected, strict=True):
            dist = mujoco.mj_ray(m, d, np.array([x, y, top]), np.array([0.0, 0.0, -1.0]),
                                 None, 1, -1, gid)  # fmt: skip
            assert dist >= 0, f"ray at ({x:.2f}, {y:.2f}) hit nothing"
            assert top - dist == pytest.approx(h, abs=2e-5), (name, seed, x, y)


@pytest.mark.parametrize("name", ["slope_up", "slope_down", "uneven", "random"])
def test_dropped_spheres_rest_on_height_at(name) -> None:
    """Collision (not just ray) geometry agrees: spheres settle at height_at + radius."""
    env = _with_seed(KIND_SPECS[name], 3)
    rng = np.random.default_rng(5)
    radius = 0.03
    # On the landing, its edges, the bumpy lane, near obstacles; 1.5 m apart in y so
    # neighbours never interact (they do not collide with each other anyway).
    xs = np.concatenate([[0.0, 0.55, -0.55, 0.7, -0.7], rng.uniform(1.0, 12.0, 25)])
    ys = np.concatenate([[0.0, 0.3, -0.3, 0.5, -0.5], rng.uniform(-1.4, 1.4, 25)])
    w = build_world(env)
    z0 = w.height_at.many(xs, ys) + radius + 0.002
    _, m = _world_model(env, np.column_stack([xs, ys, z0]), radius)
    d = mujoco.MjData(m)
    mujoco.mj_step(m, d, nstep=150)  # 0.3 s: settles on flat or bumpy ground
    for i in range(xs.size):
        x, y, z = d.qpos[7 * i : 7 * i + 3]
        # Rolling spheres move: compare against the height where they are now.
        sink = w.height_at(x, y) + radius - z
        assert -2e-3 < sink < 4e-3, (i, x, y, z, w.height_at(x, y))


@pytest.mark.parametrize("name", sorted(KIND_SPECS))
def test_no_creature_starts_inside_geometry(name) -> None:
    """After spawn, no creature geom penetrates the terrain (landing is flat and big enough)."""
    env = _with_seed(KIND_SPECS[name], 11)
    sim = SimConfig()
    genomes = [configured_genome(p, s) for p, s in ALL_CONFIGS]
    rng = np.random.default_rng(99)
    genomes += [random_genome(rng) for _ in range(20)]
    genomes.append(configured_genome(4, 2, **BIGGEST))
    for g in genomes:
        body = build_body(g, env, sim)
        spawn(body, sim, make_rng(0))
        d = body.data
        ground = [i for i in range(d.ncon) if body.is_ground[d.contact.geom[i]].any()]
        dists = [d.contact.dist[i] for i in ground]
        assert not dists or min(dists) > -1e-6, (g.id, min(dists))


def test_landing_is_flat_and_covers_the_largest_rest_footprint() -> None:
    w = build_world(KIND_SPECS["random"])
    xs, ys = np.meshgrid(np.linspace(-0.6, 0.6, 25), np.linspace(-1.2, 1.2, 49))
    assert np.all(np.abs(w.height_at.many(xs, ys)) < 1e-5)  # float32 heightfield storage
    body = build_body(configured_genome(4, 2, **BIGGEST), EnvSpec(), SimConfig())
    spawn(body, SimConfig(), make_rng(0))
    creature = body.model.geom_bodyid > 0
    reach = body.model.geom_rbound[creature]
    pos = body.data.geom_xpos[creature]
    assert np.max(np.abs(pos[:, 0]) + reach) < terrain.PAD_HALF_X
    assert np.max(np.abs(pos[:, 1]) + reach) < terrain.PAD_HALF_Y


@pytest.mark.parametrize("name", sorted(KIND_SPECS))
def test_cameras_do_not_clip_the_ground_near_the_creature(name) -> None:
    """MuJoCo scales the clip planes by the model extent, which 120 m terrain boxes
    inflate to ~200 m: the near plane moved to ~2 m and cut the ground out of the
    tracking camera's foreground (seen in docs/media/terrains renders, 2026-11-13)."""
    m = build_body(configured_genome(2, 2), _with_seed(KIND_SPECS[name], 0), SimConfig()).model
    near, far = m.vis.map.znear * m.stat.extent, m.vis.map.zfar * m.stat.extent
    assert near <= 0.1 and far >= 2 * terrain.COURSE_HALF_LENGTH


def test_course_outlasts_the_speed_cap() -> None:
    sim = SimConfig()
    reach = sim.max_linear_speed * (sim.settle_time + sim.duration)
    assert reach < terrain.COURSE_HALF_LENGTH and reach < terrain.FIELD_X[1]
    # Heightfield kinks of the slope profile sit on grid lines (exact triangulation).
    for edge in (-terrain.PAD_HALF_X, terrain.PAD_HALF_X):
        k = (edge - terrain.FIELD_X[0]) / terrain.GRID
        assert abs(k - round(k)) < 1e-9


@pytest.mark.parametrize("name", ["uneven", "patches", "random"])
def test_terrain_is_deterministic_per_seed_and_differs_across_seeds(name) -> None:
    env = KIND_SPECS[name]
    first = build_world(_with_seed(env, 4)).worldbody + build_world(_with_seed(env, 4)).assets
    terrain.build_terrain.cache_clear()
    again = build_world(_with_seed(env, 4)).worldbody + build_world(_with_seed(env, 4)).assets
    other = build_world(_with_seed(env, 5)).worldbody + build_world(_with_seed(env, 5)).assets
    assert first == again and first != other


def test_random_kind_draws_within_its_stated_ranges() -> None:
    spec = KIND_SPECS["random"]
    draws = [terrain.resolve(_with_seed(spec, s)) for s in range(200)]
    slopes = np.array([p.slope_deg for p in draws])
    rough = np.array([p.roughness for p in draws])
    obst = np.array([p.n_obstacles for p in draws])
    mu = np.array([p.friction for p in draws])
    assert np.all(np.abs(slopes) <= 15.0) and slopes.min() < -10 and slopes.max() > 10
    assert np.all((rough >= 0) & (rough <= 0.05))
    assert set(obst) == set(range(26))
    assert np.all((mu >= 0.4 - 1e-12) & (mu <= 1.6 + 1e-12))
    assert terrain.resolve(_with_seed(spec, 9)) == terrain.resolve(_with_seed(spec, 9))


def test_the_first_obstacles_of_a_seed_do_not_depend_on_the_count() -> None:
    few = build_world(EnvSpec(kind="uneven", roughness=0.02, n_obstacles=3, seed=2))
    many = build_world(EnvSpec(kind="uneven", roughness=0.02, n_obstacles=10, seed=2))
    assert np.array_equal(few.height_at.obstacles, many.height_at.obstacles[:3])


@pytest.mark.parametrize(
    "env",
    [
        EnvSpec(kind="slope", slope_deg=5, roughness=0.02),
        EnvSpec(kind="uneven", roughness=0.02, slope_deg=5),
        EnvSpec(kind="friction", friction=0.5, n_obstacles=3),
        EnvSpec(kind="slope", slope_deg=60),
        EnvSpec(kind="uneven", roughness=-0.01),
        EnvSpec(kind="friction", friction_spread=0.5),
        EnvSpec(kind="random", slope_deg=-5),
    ],
)
def test_misused_fields_are_rejected_in_plain_words(env) -> None:
    with pytest.raises(ValueError):
        build_world(env)


def test_slope_rises_along_plus_x_beyond_the_landing() -> None:
    h = build_world(EnvSpec(kind="slope", slope_deg=10.0)).height_at
    t = math.tan(math.radians(10.0))
    assert h(0.6, 0) == 0.0 and h(5.6, 3.0) == pytest.approx(5.0 * t)
    assert h(-5.6, -3.0) == pytest.approx(-5.0 * t)


def test_labels_carry_the_parameters() -> None:
    assert EnvSpec().label() == "flat"
    assert EnvSpec(kind="slope", slope_deg=-7.5).label() == "slope-7.5"
    assert EnvSpec(kind="uneven", roughness=0.03, n_obstacles=10, seed=4).label() == (
        "uneven_r0.03_o10_s4"
    )
    assert EnvSpec(kind="friction", friction=0.25).label() == "friction_mu0.25"
    patches = EnvSpec(kind="friction", friction=0.8, friction_spread=2, seed=1)
    assert patches.label() == "friction_mu0.8x2_s1"
