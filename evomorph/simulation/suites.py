"""Named terrain suites: a training suite, a held-out test suite, and diagnostics.

Generalisation is measured as performance on ``test`` after selection on
``train``. The two never share an environment, and every axis of variation is
shifted, so a creature cannot score on ``test`` by having memorised a training
terrain:

* slope: train +5, -5, +10 deg; test +7.5 (between), +15 (steeper uphill),
  -12 (steeper downhill).
* friction: train 0.4 and 1.6 (plus 1.0 on every other terrain); test 0.2
  (lower), 0.7 (between), 2.5 (higher). 0.2 is ice-on-rubber territory, where a
  gait that skates instead of planting its feet should break down.
* uneven: train roughness 0.02 m / 10 obstacles and 0.04 m / 20; test 0.03 m /
  15 (between) and 0.06 m / 30 (rougher, more cluttered).
* friction patches: train 0.8 x/ 2 (0.4-1.6); test 0.6 x/ 3 (0.2-1.8).
* random: train draws |slope| <= 8 deg, roughness <= 0.03 m, <= 15 obstacles,
  friction 0.8 x/ 2; test draws from the wider |slope| <= 14, <= 0.05 m, <= 25,
  0.7 x/ 3.
* terrain seeds: train uses 1-5, test 101-105, so even a test terrain whose
  parameters fell inside the training range has bumps, obstacles and patches
  the creature has never stood on.

Cost: one evaluation is ~0.2-0.3 s of CPU for a 10 s run, so ``train`` (11
environments) costs ~3 s per genome per spawn seed. The diagnostic suites
isolate one factor each, for plots of performance against that factor.
"""

from __future__ import annotations

from evomorph.simulation.environment import EnvSpec


def _slope(deg: float) -> EnvSpec:
    return EnvSpec(kind="slope", slope_deg=deg)


def _mu(mu: float, spread: float = 1.0, seed: int = 0) -> EnvSpec:
    return EnvSpec(kind="friction", friction=mu, friction_spread=spread, seed=seed)


def _uneven(roughness: float, obstacles: int, seed: int) -> EnvSpec:
    return EnvSpec(kind="uneven", roughness=roughness, n_obstacles=obstacles, seed=seed)


def _random(slope: float, rough: float, obst: int, mu: float, spread: float, seed: int) -> EnvSpec:
    return EnvSpec(
        kind="random",
        slope_deg=slope,
        roughness=rough,
        n_obstacles=obst,
        friction=mu,
        friction_spread=spread,
        seed=seed,
    )


TRAIN: tuple[EnvSpec, ...] = (
    EnvSpec(),
    _slope(5.0),
    _slope(-5.0),
    _slope(10.0),
    _mu(0.4),
    _mu(1.6),
    _uneven(0.02, 10, seed=1),
    _uneven(0.04, 20, seed=2),
    _mu(0.8, spread=2.0, seed=3),
    _random(8.0, 0.03, 15, 0.8, 2.0, seed=4),
    _random(8.0, 0.03, 15, 0.8, 2.0, seed=5),
)

TEST: tuple[EnvSpec, ...] = (
    _slope(7.5),
    _slope(15.0),
    _slope(-12.0),
    _mu(0.2),
    _mu(0.7),
    _mu(2.5),
    _uneven(0.03, 15, seed=101),
    _uneven(0.06, 30, seed=102),
    _mu(0.6, spread=3.0, seed=103),
    _random(14.0, 0.05, 25, 0.7, 3.0, seed=104),
    _random(14.0, 0.05, 25, 0.7, 3.0, seed=105),
)

SUITES: dict[str, tuple[EnvSpec, ...]] = {
    "train": TRAIN,
    "test": TEST,
    # --- diagnostics: one factor at a time -----------------------------------
    "flat": (EnvSpec(),),
    # Slope 0 is the slope kind's own baseline (identical ground to flat).
    "slope": tuple(_slope(d) for d in (-20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30)),
    "friction": tuple(_mu(m) for m in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.5)),
    # Roughness alone (same bump pattern, seed 1, scaled), then obstacles alone.
    # 0.12-0.20 were added after the first validation run showed no speed effect
    # up to 0.08 m on either reference (docs/TERRAINS.md); 0.20 is MAX_ROUGHNESS.
    "uneven": tuple(_uneven(r, 0, seed=1) for r in (0.0, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.2))
    + tuple(_uneven(0.0, n, seed=1) for n in (10, 20, 40)),
    # One uneven setting across five terrain seeds: how much the layout alone matters.
    "uneven-seeds": tuple(_uneven(0.04, 20, seed=s) for s in range(5)),
}

SUITE_DOCS: dict[str, str] = {
    "train": "training terrains (selection happens here)",
    "test": "held-out terrains: shifted parameters, unseen terrain seeds",
    "flat": "flat ground, friction 1",
    "slope": "slope sweep -20..+30 deg (uphill > 0)",
    "friction": "uniform friction sweep 0.1..2.5",
    "uneven": "roughness sweep 0..0.2 m, then 10/20/40 obstacles on flat",
    "uneven-seeds": "roughness 0.04 m + 20 obstacles over terrain seeds 0-4",
}


def get_suite(name: str) -> tuple[EnvSpec, ...]:
    """The named suite. Raises ValueError listing the known names."""
    try:
        return SUITES[name]
    except KeyError:
        known = ", ".join(sorted(SUITES))
        raise ValueError(
            f"there is no terrain suite called '{name}'; choose one of: {known}"
        ) from None
