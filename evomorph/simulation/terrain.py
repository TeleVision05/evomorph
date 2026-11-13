"""Terrain geometry behind ``EnvSpec``: slopes, uneven ground, obstacles, friction patches.

``build_terrain(env)`` returns the :class:`WorldFragments` for every kind except
``flat`` (which stays in ``environment.py``). Everything is a deterministic
function of the EnvSpec: the same spec always yields the same XML and the same
height function.

Layout (all kinds share it; +x is forward, see ``generator.py``)
----------------------------------------------------------------
* **Spawn landing.** ``|x| <= PAD_HALF_X``, ``|y| <= PAD_HALF_Y`` is flat at
  z = 0, so no creature starts inside geometry. The largest legal creature
  reaches |x| <= 0.47 m and |y| <= 0.95 m at rest (torso 0.8 m long, hips at
  85% of it, 0.7 m sprawled legs, 7.5 cm soles), so the landing leaves >= 0.13 m
  of margin. Roughness fades in over ``BLEND`` beyond it.
* **Slope profile.** The ground rises (``slope_deg`` > 0) or falls (< 0) along
  +x at a constant angle beyond the landing, on both sides: the landing is a
  ledge cut into one long incline, so walking backward on an uphill course is
  downhill. The profile is three boxes (back ramp, landing, front ramp), exact
  and cheap to collide with, spanning ``+-COURSE_HALF_LENGTH`` in x and y.
* **Roughness.** A MuJoCo heightfield over ``FIELD_X`` x ``+-FIELD_HALF_Y`` adds
  smooth random bumps (Gaussian-filtered noise, correlation ``CORRELATION``)
  scaled so the ground stays within ``+-roughness`` of the profile. Outside the
  field the profile boxes continue ``roughness + 1 cm`` lower (so they never
  poke through the field's dips): leaving the field means stepping off a curb
  of at most that height, 5 m off-axis.
* **Obstacles.** Low boxes (``OBSTACLE_HEIGHT``) with flat tops, sunk into the
  ground, scattered over the lane a walker actually covers (``OBSTACLE_LANE_X``,
  ``+-OBSTACLE_LANE_HALF_Y``).
* **Friction patches.** ``friction_spread`` > 1 on the ``friction`` kind cuts
  flat ground into 1 m strips across the course, each with friction
  ``friction * spread**u``, u ~ U(-1, 1); the landing strip keeps ``friction``.

``TerrainHeight`` reproduces the collision surface exactly: the profile
analytically and the heightfield with MuJoCo's own triangulation (each grid cell
split along the (r, c)-(r+1, c+1) diagonal; checked against ``mj_ray`` and a
dropped sphere in tests/test_terrain.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.ndimage import gaussian_filter

from evomorph.simulation.environment import (
    _ASSETS,
    _LIGHTS,
    FLOOR_HALF_SIZE,
    EnvSpec,
    WorldFragments,
    _ground_attrs,
)

# Course length: the flat floor's reasoning carries over. At the absolute speed
# cap (5 m/s) a creature covers 55 m in settle + duration (11 s); faster is a
# glitch. So every terrain extends 60 m in each direction.
COURSE_HALF_LENGTH = FLOOR_HALF_SIZE
PAD_HALF_X, PAD_HALF_Y = 0.6, 1.2  # m, flat spawn landing (see module doc)
BLEND = 0.4  # m over which roughness fades in beyond the landing
GRID = 0.1  # m heightfield spacing: 2.5 samples per bump correlation length
# Field extent: 60 m forward (the speed cap), 10 m back (walking backward is
# penalised anyway; beyond it the curb-lowered profile continues), +-5 m sideways
# (a 5 m sideways drift costs most of the lateral factor already).
FIELD_X = (-10.0, 60.0)
FIELD_HALF_Y = 5.0
EDGE_FADE = 0.5  # m: bumps fade to zero at the field border so the curb is <= roughness+1 cm
# Bump correlation length (Gaussian sigma, m): about one stride of a 0.4 m
# quadruped, so a bump is something a foot lands on, not sub-foot texture.
CORRELATION = 0.25
CURB = 0.01  # m the off-field ground sits below the field's lowest possible point
BOX_HALF_THICKNESS = 0.5  # m: ramps and strips are thick slabs, far deeper than any sinking
# Obstacles: low blocks. 2-6 cm is below the ~7 cm foot lift of the reference
# quadruped's gait at the high end and a stub toe at the low end; footprints
# from a shoe-box to a plank. Scattered over the first 12 m of a 3 m lane,
# where a 0.2-1 m/s walker spends its 10 s run.
OBSTACLE_HEIGHT = (0.02, 0.06)
OBSTACLE_SIZE_X = (0.05, 0.25)
OBSTACLE_SIZE_Y = (0.10, 0.50)
OBSTACLE_LANE_X = (PAD_HALF_X + BLEND, 12.0)
OBSTACLE_LANE_HALF_Y = 1.5
OBSTACLE_SINK = 0.05  # m the block extends below the lowest ground under it
STRIP_LENGTH = 1.0  # m friction patch length along x
STRIP_X = (-10.0 - PAD_HALF_X, COURSE_HALF_LENGTH)
MAX_SLOPE_DEG = 40.0  # steeper than this is a wall for these creatures, not a slope
MAX_ROUGHNESS = 0.2  # m

_EXTRA_ASSETS = """
    <material name="rock" rgba="0.55 0.42 0.3 1" reflectance="0.02"/>
"""


@dataclass(frozen=True)
class TerrainParams:
    """Concrete terrain parameters (the ``random`` kind's draws, already made)."""

    kind: str
    slope_deg: float
    roughness: float
    n_obstacles: int
    friction: float
    friction_spread: float
    seed: int


def _rngs(seed: int) -> list[np.random.Generator]:
    """Independent streams: 0 random-kind draws, 1 bumps, 2 obstacles, 3 friction strips.

    Separate streams keep, for example, the bump pattern of seed s identical
    whether or not the spec also asks for obstacles.
    """
    children = np.random.SeedSequence([int(seed), 0x7E77A1]).spawn(4)
    return [np.random.default_rng(c) for c in children]


def _check_ranges(env: EnvSpec) -> None:
    if abs(env.slope_deg) > MAX_SLOPE_DEG:
        raise ValueError(f"slope must be within +-{MAX_SLOPE_DEG:g} degrees")
    if not 0.0 <= env.roughness <= MAX_ROUGHNESS:
        raise ValueError(f"roughness must be between 0 and {MAX_ROUGHNESS:g} m")
    if env.n_obstacles < 0:
        raise ValueError("the number of obstacles cannot be negative")
    if env.friction_spread < 1.0:
        raise ValueError("friction_spread is a factor and must be at least 1")


def _forbid(env: EnvSpec, *names: str) -> None:
    defaults = EnvSpec()
    used = [n for n in names if getattr(env, n) != getattr(defaults, n)]
    if used:
        raise ValueError(
            f"terrain kind '{env.kind}' does not use {', '.join(used)}; "
            "use kind 'random' to combine terrain features"
        )


def resolve(env: EnvSpec) -> TerrainParams:
    """The concrete parameters ``env`` describes; for ``random`` this draws them by seed.

    ``random`` reads its fields as ranges: slope ~ U(-slope_deg, +slope_deg),
    roughness ~ U(0, roughness), obstacles ~ uniform integer in [0, n_obstacles],
    friction = friction x friction_spread**u with u ~ U(-1, 1) (log-uniform).
    """
    _check_ranges(env)
    p = TerrainParams(
        env.kind, env.slope_deg, env.roughness, env.n_obstacles, env.friction, 1.0, env.seed
    )
    if env.kind == "flat":
        _forbid(env, "slope_deg", "roughness", "n_obstacles", "friction_spread")
    elif env.kind == "slope":
        _forbid(env, "roughness", "n_obstacles", "friction_spread")
    elif env.kind == "uneven":
        _forbid(env, "slope_deg", "friction_spread")
    elif env.kind == "friction":
        _forbid(env, "slope_deg", "roughness", "n_obstacles")
        p = TerrainParams(env.kind, 0.0, 0.0, 0, env.friction, env.friction_spread, env.seed)
    elif env.kind == "random":
        if env.slope_deg < 0:
            raise ValueError("for kind 'random', slope_deg is the largest |slope| and must be >= 0")
        u = _rngs(env.seed)[0].uniform(size=4)
        p = TerrainParams(
            kind="random",
            slope_deg=float(env.slope_deg * (2.0 * u[0] - 1.0)),
            roughness=float(env.roughness * u[1]),
            n_obstacles=min(int(u[2] * (env.n_obstacles + 1)), env.n_obstacles),
            friction=float(env.friction * env.friction_spread ** (2.0 * u[3] - 1.0)),
            friction_spread=1.0,
            seed=env.seed,
        )
    return p


# ----------------------------------------------------------------------------
# Height function
# ----------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class HeightGrid:
    """A heightfield as MuJoCo stores it: normalised float32 data plus its placement."""

    data: np.ndarray  # (nrow, ncol) float32 in [0, 1]; row = y, column = x
    x0: float  # x of column 0
    y0: float  # y of row 0
    dx: float
    dy: float
    z0: float  # world z of elevation 0
    zscale: float  # world height of elevation 1 above z0

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        nrow, ncol = self.data.shape
        u, v = (x - self.x0) / self.dx, (y - self.y0) / self.dy
        return (u >= 0) & (u <= ncol - 1) & (v >= 0) & (v <= nrow - 1)

    def heights(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Surface z at points inside the field, on MuJoCo's triangles."""
        nrow, ncol = self.data.shape
        u, v = (x - self.x0) / self.dx, (y - self.y0) / self.dy
        c = np.clip(np.floor(u).astype(int), 0, ncol - 2)
        r = np.clip(np.floor(v).astype(int), 0, nrow - 2)
        fu, fv = u - c, v - r
        d = self.data.astype(np.float64)
        z00, z01 = d[r, c], d[r, c + 1]
        z10, z11 = d[r + 1, c], d[r + 1, c + 1]
        lower = z00 + fu * (z01 - z00) + fv * (z11 - z01)  # triangle (r,c)(r,c+1)(r+1,c+1)
        upper = z00 + fv * (z10 - z00) + fu * (z11 - z10)  # triangle (r,c)(r+1,c)(r+1,c+1)
        return self.z0 + self.zscale * np.where(fu >= fv, lower, upper)


def profile(x: np.ndarray, tan_a: float) -> np.ndarray:
    """Slope profile: flat landing on ``|x| <= PAD_HALF_X``, constant grade beyond it."""
    x = np.asarray(x, dtype=float)
    return tan_a * (np.maximum(x - PAD_HALF_X, 0.0) + np.minimum(x + PAD_HALF_X, 0.0))


@dataclass(frozen=True, eq=False)
class TerrainHeight:
    """``height_at(x, y)``: the top of the collision geometry under world point (x, y)."""

    tan_a: float = 0.0
    drop: float = 0.0  # m the profile boxes sit below the nominal profile (the curb)
    grid: HeightGrid | None = None
    obstacles: np.ndarray = field(default_factory=lambda: np.zeros((0, 5)))  # x0 x1 y0 y1 top

    def many(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Vectorised surface height."""
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        h = profile(x, self.tan_a) - self.drop
        if self.grid is not None:
            inside = self.grid.contains(x, y)
            if np.any(inside):
                h = np.where(inside, self.grid.heights(x, y), h)
        for x0, x1, y0, y1, top in self.obstacles:
            on = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
            h = np.where(on, np.maximum(h, top), h)
        return h

    def __call__(self, x: float, y: float) -> float:
        return float(self.many(np.array([x]), np.array([y]))[0])


# ----------------------------------------------------------------------------
# Geometry builders
# ----------------------------------------------------------------------------


def _quat_y(theta: float) -> str:
    return f"{math.cos(theta / 2):.9g} 0 {math.sin(theta / 2):.9g} 0"


def _box(name: str, centre: tuple[float, float, float], half: tuple, extra: str) -> str:
    c, h = " ".join(f"{v:.9g}" for v in centre), " ".join(f"{v:.9g}" for v in half)
    return f'    <geom name="{name}" type="box" pos="{c}" size="{h}" {extra}/>\n'


def _ramp(name: str, xa: float, xb: float, xk: float, a: float, drop: float, attrs: str) -> str:
    """Box whose top face is z = tan(a) (x - xk) - drop for x in [xa, xb]."""
    t = math.tan(a)
    xm = 0.5 * (xa + xb)
    zm = t * (xm - xk) - drop
    half_len = 0.5 * (xb - xa) / math.cos(a)
    nx, nz = -math.sin(a), math.cos(a)  # surface normal
    th = BOX_HALF_THICKNESS
    centre = (xm - th * nx, 0.0, zm - th * nz)
    half = (half_len, COURSE_HALF_LENGTH, th)
    return _box(name, centre, half, f'quat="{_quat_y(-a)}" {attrs}')


def _profile_xml(slope_deg: float, drop: float, attrs: str) -> str:
    """The landing plus two ramps (or one infinite plane when the slope is 0)."""
    if slope_deg == 0.0:
        return (
            f'    <geom name="floor" type="plane" pos="0 0 {-drop:.9g}" '
            f'size="{COURSE_HALF_LENGTH:g} {COURSE_HALF_LENGTH:g} 0.1" {attrs}/>\n'
        )
    a, length, pad, th = (
        math.radians(slope_deg),
        COURSE_HALF_LENGTH,
        PAD_HALF_X,
        BOX_HALF_THICKNESS,
    )
    # At a valley the ramp's end face leans away from the landing and would leave
    # a thin wedge gap under the surface; the landing slab reaches under the
    # ramp there instead (hidden below the ramp's surface). Crests need nothing.
    ext_front, ext_back = (0.5, 0.0) if slope_deg > 0 else (0.0, 0.5)
    lo, hi = -pad - ext_back, pad + ext_front
    landing = _box(
        "landing", (0.5 * (lo + hi), 0.0, -drop - th), (0.5 * (hi - lo), length, th), attrs
    )
    return (
        _ramp("ramp_back", -length, -pad, -pad, a, drop, attrs)
        + landing
        + _ramp("ramp_front", pad, length, pad, a, drop, attrs)
    )


def _bumps(rng: np.random.Generator, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Unit-peak smooth noise on the grid, zero on the landing and at the field border."""
    noise = gaussian_filter(rng.standard_normal((ys.size, xs.size)), CORRELATION / GRID)
    noise /= np.abs(noise).max()
    gx, gy = np.meshgrid(xs, ys)
    d_pad = np.hypot(np.maximum(np.abs(gx) - PAD_HALF_X, 0), np.maximum(np.abs(gy) - PAD_HALF_Y, 0))
    d_edge = np.minimum.reduce([gx - xs[0], xs[-1] - gx, gy - ys[0], ys[-1] - gy])
    w = _smoothstep(d_pad / BLEND) * _smoothstep(d_edge / EDGE_FADE)
    return noise * w


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _field(p: TerrainParams, rng: np.random.Generator, attrs: str) -> tuple[HeightGrid, str, str]:
    """Heightfield of profile + bumps: (grid, asset xml, geom xml)."""
    ncol = round((FIELD_X[1] - FIELD_X[0]) / GRID) + 1
    nrow = round(2 * FIELD_HALF_Y / GRID) + 1
    xs = np.linspace(FIELD_X[0], FIELD_X[1], ncol)
    ys = np.linspace(-FIELD_HALF_Y, FIELD_HALF_Y, nrow)
    z = profile(xs, math.tan(math.radians(p.slope_deg)))[None, :] + p.roughness * _bumps(
        rng, xs, ys
    )
    zmin, zmax = float(z.min()), float(z.max())
    # MuJoCo reads the elevation text top row first, i.e. the first row is y = +ry
    # (measured: the stored data is the text flipped up-down), so write it flipped.
    text = " ".join(f"{v:.7g}" for v in np.flipud((z - zmin) / (zmax - zmin)).ravel())
    # What MuJoCo will hold: the same text parsed to float32 (already spans [0, 1]).
    data = np.flipud(np.array(text.split(), dtype=np.float32).reshape(nrow, ncol)).copy()
    rx, cx = 0.5 * (FIELD_X[1] - FIELD_X[0]), 0.5 * (FIELD_X[0] + FIELD_X[1])
    grid = HeightGrid(
        data, FIELD_X[0], -FIELD_HALF_Y, xs[1] - xs[0], ys[1] - ys[0], zmin, zmax - zmin
    )
    asset = (
        f'    <hfield name="terrain" nrow="{nrow}" ncol="{ncol}" '
        f'size="{rx:g} {FIELD_HALF_Y:g} {zmax - zmin:.9g} 0.5" elevation="{text}"/>\n'
    )
    geom = (
        f'    <geom name="terrain" type="hfield" hfield="terrain" pos="{cx:g} 0 {zmin:.9g}" '
        f'material="ground" {attrs}/>\n'
    )
    return grid, asset, geom


def _obstacles(p: TerrainParams, rng: np.random.Generator, ground: TerrainHeight, attrs: str):
    """Obstacle table (x0 x1 y0 y1 top) and its geoms. Draws row by row, so the
    first k obstacles of a seed are the same whatever ``n_obstacles`` is."""
    rows, xml = [], []
    for i, u in enumerate(rng.uniform(size=(p.n_obstacles, 5))):
        cx = OBSTACLE_LANE_X[0] + u[0] * (OBSTACLE_LANE_X[1] - OBSTACLE_LANE_X[0])
        cy = OBSTACLE_LANE_HALF_Y * (2.0 * u[1] - 1.0)
        hx = 0.5 * (OBSTACLE_SIZE_X[0] + u[2] * (OBSTACLE_SIZE_X[1] - OBSTACLE_SIZE_X[0]))
        hy = 0.5 * (OBSTACLE_SIZE_Y[0] + u[3] * (OBSTACLE_SIZE_Y[1] - OBSTACLE_SIZE_Y[0]))
        height = OBSTACLE_HEIGHT[0] + u[4] * (OBSTACLE_HEIGHT[1] - OBSTACLE_HEIGHT[0])
        gx, gy = np.meshgrid(np.linspace(cx - hx, cx + hx, 21), np.linspace(cy - hy, cy + hy, 21))
        under = ground.many(gx, gy)
        top, bottom = float(under.max()) + height, float(under.min()) - OBSTACLE_SINK
        rows.append((cx - hx, cx + hx, cy - hy, cy + hy, top))
        centre, half = (cx, cy, 0.5 * (top + bottom)), (hx, hy, 0.5 * (top - bottom))
        xml.append(_box(f"obstacle_{i}", centre, half, f'material="rock" {attrs}'))
    return np.array(rows, dtype=float).reshape(-1, 5), "".join(xml)


def _strips(p: TerrainParams, rng: np.random.Generator) -> tuple[str, str]:
    """Friction patches: (materials xml, geoms xml). Strip k covers x in
    [PAD_HALF_X + k, PAD_HALF_X + k + 1) forward and mirrored backward."""
    th, length = BOX_HALF_THICKNESS, COURSE_HALF_LENGTH
    n_fwd = math.ceil((STRIP_X[1] - PAD_HALF_X) / STRIP_LENGTH)
    n_back = math.ceil((-PAD_HALF_X - STRIP_X[0]) / STRIP_LENGTH)
    spans = [(-PAD_HALF_X, PAD_HALF_X, p.friction)]
    mus = p.friction * p.friction_spread ** rng.uniform(-1.0, 1.0, size=n_fwd + n_back)
    for k in range(n_fwd):
        lo = PAD_HALF_X + k * STRIP_LENGTH
        spans.append((lo, min(lo + STRIP_LENGTH, length), float(mus[k])))
    for k in range(n_back):
        hi = -PAD_HALF_X - k * STRIP_LENGTH
        spans.append((max(hi - STRIP_LENGTH, STRIP_X[0]), hi, float(mus[n_fwd + k])))
    spans.append((-length, STRIP_X[0], p.friction))
    mats, geoms = [], []
    for i, (lo, hi, mu) in enumerate(spans):
        mats.append(_tinted_material(f"strip{i}", mu))
        centre, half = (0.5 * (lo + hi), 0.0, -th), (0.5 * (hi - lo), length, th)
        geoms.append(_box(f"strip_{i}", centre, half, f'material="strip{i}" {_ground_attrs(mu)}'))
    return "".join(mats), "".join(geoms)


def _tinted_material(name: str, mu: float) -> str:
    """Checker ground tinted icy blue (slippery) to sandy (grippy), by log friction."""
    t = float(np.clip(0.5 + math.log(mu) / math.log(9.0), 0.0, 1.0))  # mu 1/3 -> 0, 3 -> 1
    rgb = np.array([0.55, 0.75, 1.0]) * (1 - t) + np.array([1.0, 0.85, 0.6]) * t
    return (
        f'    <material name="{name}" texture="grid" texrepeat="2 2" texuniform="true" '
        f'reflectance="0.02" rgba="{rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f} 1"/>\n'
    )


@lru_cache(maxsize=64)
def build_terrain(env: EnvSpec) -> WorldFragments:
    """World fragments for every non-flat kind. Cached: the generator asks twice per build."""
    p = resolve(env)
    attrs = _ground_attrs(p.friction)
    ground_attrs = f'material="ground" {attrs}'
    _, bump_rng, obstacle_rng, strip_rng = _rngs(p.seed)
    assets, geoms = _ASSETS + _EXTRA_ASSETS, ""
    if p.friction_spread > 1.0:
        mats, geoms = _strips(p, strip_rng)
        assets += mats
        height = TerrainHeight()
    elif p.roughness > 0.0:
        grid, asset, geom = _field(p, bump_rng, attrs)
        drop = p.roughness + CURB
        assets += asset
        geoms = _profile_xml(p.slope_deg, drop, ground_attrs) + geom
        height = TerrainHeight(math.tan(math.radians(p.slope_deg)), drop, grid)
    else:
        geoms = _profile_xml(p.slope_deg, 0.0, ground_attrs)
        height = TerrainHeight(math.tan(math.radians(p.slope_deg)))
    if p.n_obstacles:
        table, xml = _obstacles(p, obstacle_rng, height, attrs)
        geoms += xml
        height = TerrainHeight(height.tan_a, height.drop, height.grid, table)
    return WorldFragments(
        assets=assets, worldbody=_LIGHTS + geoms, spawn_height=height(0.0, 0.0), height_at=height
    )
