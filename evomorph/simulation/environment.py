"""World (terrain) construction.

``build_world(env)`` returns the MJCF fragments that the morphology generator
splices into every creature model: assets, worldbody geometry and lights, plus
the ground height under the spawn point.

Collision convention every terrain must follow (see ``GROUND_CONTYPE``): terrain
geoms use ``contype=1 conaffinity=2 priority=1``; creature geoms use
``contype=2 conaffinity=3``. Creatures collide with terrain and with their own
other limbs (the generator excludes pairs that overlap at rest), terrain never
collides with terrain, and the terrain's friction wins every terrain contact
(``priority``), so ``EnvSpec.friction`` is exactly what the feet experience.

Only the flat terrain is implemented here. Every other kind is routed to
:func:`build_terrain_extension`, which hands it to ``terrain.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

ENV_KINDS: tuple[str, ...] = ("flat", "slope", "uneven", "friction", "random")

# Bitmasks: terrain-creature and creature-creature collide; terrain-terrain does not.
GROUND_CONTYPE, GROUND_CONAFFINITY = 1, 2
CREATURE_CONTYPE, CREATURE_CONAFFINITY = 2, 3
# Floor is large enough that a 10 s run at the glitch speed cap stays on it.
FLOOR_HALF_SIZE = 60.0


@dataclass(frozen=True)
class EnvSpec:
    """Which terrain to evaluate on. ``seed`` drives any random terrain features.

    Kinds and the fields each one reads (any other field must stay at its
    default; geometry and numbers are in ``terrain.py`` and docs/TERRAINS.md):

    * ``flat``: friction.
    * ``slope``: slope_deg (> 0 rises along +x, < 0 falls), friction.
    * ``uneven``: roughness (ground within +-roughness m), n_obstacles, friction, seed.
    * ``friction``: friction; friction_spread > 1 adds 1 m patches with friction
      ``friction * spread**u``, u ~ U(-1, 1), placed by seed.
    * ``random``: every field is a range, drawn by seed: slope ~ U(-slope_deg,
      +slope_deg), roughness ~ U(0, roughness), obstacles in [0, n_obstacles],
      friction = friction * friction_spread**u (``terrain.resolve`` shows the draw).
    """

    kind: str = "flat"
    slope_deg: float = 0.0
    friction: float = 1.0  # sliding friction coefficient of the ground
    roughness: float = 0.0  # m, amplitude of terrain height variation
    n_obstacles: int = 0
    seed: int = 0
    friction_spread: float = 1.0  # >= 1, multiplicative friction range (1 = uniform)

    def label(self) -> str:
        """Short human-readable id, e.g. ``flat``, ``slope+5`` or ``uneven_r0.02_o10_s3``."""
        if self.kind == "flat":
            return "flat" if self.friction == 1.0 else f"flat_mu{self.friction:g}"
        if self.kind == "slope":
            mu = "" if self.friction == 1.0 else f"_mu{self.friction:g}"
            return f"slope{self.slope_deg:+g}{mu}"
        if self.kind == "uneven":
            mu = "" if self.friction == 1.0 else f"_mu{self.friction:g}"
            return f"uneven_r{self.roughness:g}_o{self.n_obstacles}{mu}_s{self.seed}"
        if self.kind == "friction":
            if self.friction_spread == 1.0:
                return f"friction_mu{self.friction:g}"
            return f"friction_mu{self.friction:g}x{self.friction_spread:g}_s{self.seed}"
        return f"{self.kind}_s{self.seed}"


@dataclass(frozen=True)
class WorldFragments:
    """MJCF pieces for one terrain, plus how high the ground is anywhere.

    ``height_at(x, y)`` returns the terrain surface z under world point (x, y).
    The evaluator's glitch guards (torso clipped into the ground, launched above
    it) measure against this, so a slope or bumpy terrain must supply it. When it
    is None the ground is flat at ``spawn_height``.
    """

    assets: str
    worldbody: str
    spawn_height: float = 0.0  # ground z under the creature's spawn point (x=y=0)
    height_at: Callable[[float, float], float] | None = None

    def ground_height(self, x: float, y: float) -> float:
        return self.spawn_height if self.height_at is None else float(self.height_at(x, y))


def _ground_attrs(friction: float) -> str:
    # condim 3 = sliding friction only; torsional/rolling friction would let a
    # spinning foot grip unrealistically on a hard floor.
    return (
        f'contype="{GROUND_CONTYPE}" conaffinity="{GROUND_CONAFFINITY}" priority="1" '
        f'condim="3" friction="{friction:g} 0.005 0.0001"'
    )


_ASSETS = """
    <texture name="grid" type="2d" builtin="checker" rgb1="0.42 0.45 0.48"
             rgb2="0.33 0.36 0.39" width="512" height="512" mark="edge" markrgb="0.25 0.27 0.3"/>
    <material name="ground" texture="grid" texrepeat="2 2" texuniform="true" reflectance="0.02"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.55 0.68 0.82" rgb2="0.1 0.12 0.16"
             width="256" height="256"/>
"""

_LIGHTS = """
    <light name="sun" directional="true" pos="0 0 5" dir="-0.3 0.2 -1" diffuse="0.6 0.6 0.6"
           specular="0.1 0.1 0.1" castshadow="true"/>
    <light name="fill" directional="true" pos="0 0 5" dir="0.3 -0.4 -1" diffuse="0.15 0.15 0.15"
           castshadow="false"/>
"""


def _flat(env: EnvSpec) -> WorldFragments:
    floor = (
        f'    <geom name="floor" type="plane" size="{FLOOR_HALF_SIZE:g} {FLOOR_HALF_SIZE:g} 0.1" '
        f'material="ground" {_ground_attrs(env.friction)}/>\n'
    )
    return WorldFragments(assets=_ASSETS, worldbody=_LIGHTS + floor, spawn_height=0.0)


# ============================================================================
# TERRAIN EXTENSION POINT -- implemented by terrain.build_terrain. Contract:
# return WorldFragments for kinds 'slope', 'uneven', 'friction' and
# 'random', reuse _ground_attrs() / the collision bitmasks above for every
# terrain geom, keep the creature's forward direction along +x, set
# spawn_height to the ground z at (0, 0), and set height_at(x, y) to the
# surface height anywhere the creature can reach (a slope of angle a rising
# along +x is ``lambda x, y: x * tan(a)``). Without height_at the evaluator
# assumes flat ground at spawn_height and will flag a downhill walker as
# "clipped through the ground" (tests/test_evaluator_guards.py shows how).
# ============================================================================
def build_terrain_extension(env: EnvSpec) -> WorldFragments:
    """Slope, uneven, friction and random terrain: implemented in ``terrain.py``."""
    from evomorph.simulation.terrain import build_terrain  # terrain imports this module

    return build_terrain(env)


def build_world(env: EnvSpec) -> WorldFragments:
    """Return the world fragments for ``env``."""
    if env.kind not in ENV_KINDS:
        raise ValueError(f"unknown terrain kind '{env.kind}', expected one of {ENV_KINDS}")
    if env.friction <= 0:
        raise ValueError("friction must be positive")
    if env.kind == "flat":
        return _flat(env)
    return build_terrain_extension(env)
