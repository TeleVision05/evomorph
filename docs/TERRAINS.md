# Terrains and robustness evaluation

Brief section 8 (environments A-E) and the robustness parts of section 9. Code:
`evomorph/simulation/terrain.py` (geometry), `evomorph/simulation/suites.py`
(named suites), `evomorph/experiments/robustness.py` and
`python -m evomorph robustness` (evaluation). Every number below comes from the
files in `docs/validation/terrains/`, produced by `sh scripts/validate_terrains.sh`
(MuJoCo 4.0.1, default `SimConfig`/`FitnessConfig`, spawn seeds 0-2), on 2026-11-13.

## The environments

| brief | `EnvSpec.kind` | fields it reads | geometry |
|---|---|---|---|
| A flat | `flat` | friction | infinite plane (the core's) |
| B sloped | `slope` | slope_deg (> 0 rises along +x), friction | three exact boxes: back ramp, flat landing, front ramp |
| C uneven | `uneven` | roughness, n_obstacles, friction, seed | heightfield of smooth bumps + low box obstacles |
| D friction | `friction` | friction, friction_spread, seed | plane (uniform), or 1 m strips across the course with friction `friction * spread**u`, u ~ U(-1, 1) |
| E randomized | `random` | all fields, read as **ranges** | slope ~ U(-slope_deg, +slope_deg), roughness ~ U(0, roughness), obstacles uniform in [0, n_obstacles], friction `friction * spread**u`; drawn by seed |

A field a kind does not read must stay at its default; anything else is a
plain-language `ValueError` (for example "terrain kind 'slope' does not use
roughness; use kind 'random' to combine terrain features"). `terrain.resolve(env)`
returns the concrete parameters, including the random kind's draws, and the
robustness JSON records them per environment.

Shared layout, with the reason for each number (also in the module docstring):

| what | value | why |
|---|---|---|
| course | +-60 m in x and y | 5 m/s (the glitch speed cap) x 11 s (settle + run) = 55 m |
| flat landing at spawn | \|x\| <= 0.6 m, \|y\| <= 1.2 m, z = 0 | the largest legal creature reaches \|x\| <= 0.47 m, \|y\| <= 0.95 m at rest; nobody starts inside geometry (tested for every limb configuration, 20 random genomes and the largest footprint on every kind) |
| slope profile | constant grade beyond the landing, both directions | the landing is a ledge cut into one incline: walking backward on an uphill course is downhill |
| heightfield | x in [-10, 60] m, \|y\| <= 5 m, 0.1 m grid | bumps fade in over 0.4 m beyond the landing and fade out 0.5 m before the field border |
| bump shape | Gaussian-filtered noise, sigma 0.25 m, peak \|z\| = roughness | 0.25 m is about one stride of a 0.4 m quadruped |
| off the field | the profile, `roughness + 1 cm` lower | a curb at most that high, 5 m off-axis, never poking through the field's dips |
| obstacles | 2-6 cm high, 5-25 x 10-50 cm, sunk 5 cm, flat tops; in x 1.0-12 m, \|y\| <= 1.5 m | low enough to step over, placed where a 0.2-1 m/s walker spends its 10 s |
| friction patches | 1 m strips; the landing keeps `friction` | tinted icy blue (slippery) to sandy (grippy) in renders |

**Roughness is a peak, not an RMS.** `roughness` is the largest bump anywhere
on the 70 x 10 m field. Measured on seed 1 over x 1.5-40 m, \|y\| <= 4 m, the RMS
height is **0.17 x roughness** (3.5 mm at 0.02 m, 13.9 mm at 0.08 m) and the 99th
percentile of \|z\| is 0.45 x roughness. The train and test bumps (0.02-0.06 m
peak) are therefore gentle; most of their difficulty comes from the obstacles.

**`height_at` is the collision surface.** It is what the core's glitch guards and
spawn height read. The slope profile is analytic; the heightfield reproduces
MuJoCo's own triangulation (each cell split along its (r, c)-(r+1, c+1)
diagonal). Two facts were measured, not assumed: MuJoCo reads the `elevation`
text with its first row at +y (the stored data is the text flipped
up-down), and it renormalises the data to [0, 1]. `tests/test_terrain.py`
compares `height_at` with a downward `mj_ray` at about 1300 points per kind and
seed (tolerance 0.02 mm) and with 30 dropped spheres per terrain (resting
within -2/+4 mm). The ray test was mutation-checked: a wrong diagonal and the
unflipped rows both fail it.

**Forward, falls and guards on slopes.** Forward stays +x (horizontal), so
uphill progress is horizontal progress. The clip and launch guards measure the
torso against the local ground, and the fall test (75 deg tilt or torso
contact) is unchanged. In 330 validation runs (55 environment rows x 3 seeds x 2
references) no run was invalid: there were no false clip or launch flags.

## Train/test split and why it measures generalisation

Selection happens on `train`; generalisation is the score on `test`. The two
suites share no environment, and **every axis of variation is shifted**: some
test values lie between training values (interpolation), most lie outside them
(extrapolation), and every seeded test terrain uses a seed (101-105) that
training never used (1-5). A creature cannot do well on `test` by memorising
training terrains, whether their exact layout (seeds differ) or their exact
parameter values (none repeat). It must have a gait that works on terrain it has
not been selected on. Because the suites hold the same kinds in the same
proportions, a train-test gap is attributable to the shift, not to a different
mix of terrain types. `tests/test_suites.py` checks the disjointness on every
axis and the unseen seeds.

| axis | train | test |
|---|---|---|
| slope (deg) | +5, -5, +10 | +7.5 (between), +15 (steeper up), -12 (steeper down) |
| uniform friction | 0.4, 1.6 (and 1.0 everywhere else) | 0.2 (lower), 0.7 (between), 2.5 (higher) |
| uneven (peak m / obstacles) | 0.02 / 10, 0.04 / 20 | 0.03 / 15 (between), 0.06 / 30 (rougher) |
| friction patches | 0.8 x/ 2 (0.4-1.6) | 0.6 x/ 3 (0.2-1.8) |
| random ranges | \|slope\| <= 8, <= 0.03 m, <= 15 obstacles, 0.8 x/ 2 | \|slope\| <= 14, <= 0.05 m, <= 25, 0.7 x/ 3 |
| terrain seeds | 1-5 | 101-105 |

Low friction is in training on purpose: the core's open issues record that the
references skate (70-80% of stance load slides), and a skater should pay for it
during selection. The suites were fixed before anything was measured on them.

Exact parameters (from `python -m evomorph robustness --list`):

### train

| environment | kind | slope deg | roughness m | obstacles | friction | spread | terrain seed |
|---|---|---|---|---|---|---|---|
| flat | flat | 0 | 0 | 0 | 1 | 1 | 0 |
| slope+5 | slope | 5 | 0 | 0 | 1 | 1 | 0 |
| slope-5 | slope | -5 | 0 | 0 | 1 | 1 | 0 |
| slope+10 | slope | 10 | 0 | 0 | 1 | 1 | 0 |
| friction_mu0.4 | friction | 0 | 0 | 0 | 0.4 | 1 | 0 |
| friction_mu1.6 | friction | 0 | 0 | 0 | 1.6 | 1 | 0 |
| uneven_r0.02_o10_s1 | uneven | 0 | 0.02 | 10 | 1 | 1 | 1 |
| uneven_r0.04_o20_s2 | uneven | 0 | 0.04 | 20 | 1 | 1 | 2 |
| friction_mu0.8x2_s3 | friction | 0 | 0 | 0 | 0.8 | 2 | 3 |
| random_s4 | random | 8 | 0.03 | 15 | 0.8 | 2 | 4 |
| random_s5 | random | 8 | 0.03 | 15 | 0.8 | 2 | 5 |

### test

| environment | kind | slope deg | roughness m | obstacles | friction | spread | terrain seed |
|---|---|---|---|---|---|---|---|
| slope+7.5 | slope | 7.5 | 0 | 0 | 1 | 1 | 0 |
| slope+15 | slope | 15 | 0 | 0 | 1 | 1 | 0 |
| slope-12 | slope | -12 | 0 | 0 | 1 | 1 | 0 |
| friction_mu0.2 | friction | 0 | 0 | 0 | 0.2 | 1 | 0 |
| friction_mu0.7 | friction | 0 | 0 | 0 | 0.7 | 1 | 0 |
| friction_mu2.5 | friction | 0 | 0 | 0 | 2.5 | 1 | 0 |
| uneven_r0.03_o15_s101 | uneven | 0 | 0.03 | 15 | 1 | 1 | 101 |
| uneven_r0.06_o30_s102 | uneven | 0 | 0.06 | 30 | 1 | 1 | 102 |
| friction_mu0.6x3_s103 | friction | 0 | 0 | 0 | 0.6 | 3 | 103 |
| random_s104 | random | 14 | 0.05 | 25 | 0.7 | 3 | 104 |
| random_s105 | random | 14 | 0.05 | 25 | 0.7 | 3 | 105 |

Realised random draws (`terrain_draws` in the JSON):

| terrain | slope deg | roughness m | obstacles | friction |
|---|---|---|---|---|
| random_s4 (train) | -2.83 | 0.016 | 2 | 1.07 |
| random_s5 (train) | -3.68 | 0.011 | 12 | 1.01 |
| random_s104 (test) | +10.14 | 0.039 | 6 | 0.43 |
| random_s105 (test) | +5.95 | 0.024 | 9 | 0.43 |

Two draws per suite is a small sample: both training draws happen to be mild
downhill on normal friction, and both test draws are uphill on low friction. The
random rows of `test` are harder than the ranges alone suggest.

Diagnostic suites (one factor each): `flat`; `slope` -20 to +30 deg in 5 deg steps;
`friction` 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2.5; `uneven` roughness 0, 0.02, 0.04,
0.06, 0.08, 0.12, 0.16, 0.2 m (no obstacles, seed 1), then 10, 20, 40 obstacles on
flat; `uneven-seeds` 0.04 m + 20 obstacles over seeds 0-4. The 0.12-0.2 m
roughness rows were **added after the first validation run** showed no speed
effect up to 0.08 m; the other rows are unchanged and came out bit-identical
on the rerun.

Cost: one 10 s evaluation is about 0.2-0.3 s of CPU, so `train` (11
environments) costs about 3 s per genome per spawn seed.

## Validation with the reference creatures

Speeds are `avg_forward_velocity` (m/s), means over spawn seeds 0-2. "fell" is
falls out of 3 runs; slip is `slip_fraction` (share of foot load delivered while
sliding faster than 5 cm/s). Sources: `docs/validation/terrains/ref-*_<suite>.csv`.

**Slopes.** Uphill is slower for both, monotonically, and steep enough slopes
make them fail:

| slope deg | -20 | -10 | 0 | +5 | +10 | +15 | +20 | +25 | +30 |
|---|---|---|---|---|---|---|---|---|---|
| quadruped m/s | 0.581 | 0.601 | 0.447 | 0.338 | 0.243 | 0.139 | 0.070 | 0.047 | 0.041 |
| quadruped fell | 1/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | **3/3** | **3/3** |
| hexapod m/s | 0.605 | 0.538 | 0.456 | 0.404 | 0.381 | 0.323 | 0.205 | 0.096 | 0.061 |
| hexapod fell | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | **1/3** |

The quadruped falls on every run from +25 deg. The hexapod never falls through
+25 deg and falls once in three runs at +30 deg, by which point its speed is 13%
of flat.

**Friction.** Low friction raises slip for both. It reduces progress clearly
for the hexapod, but only modestly for the quadruped, whose progress saturates
below mu = 0.5:

| friction | 0.1 | 0.2 | 0.3 | 0.5 | 0.75 | 1.0 | 1.5 | 2.5 |
|---|---|---|---|---|---|---|---|---|
| quadruped m/s | 0.384 | 0.378 | 0.376 | 0.387 | 0.414 | 0.447 | 0.488 | 0.571 |
| quadruped slip | 0.954 | 0.926 | 0.885 | 0.860 | 0.805 | 0.721 | 0.629 | 0.630 |
| hexapod m/s | 0.258 | 0.304 | 0.337 | 0.393 | 0.428 | 0.456 | 0.505 | 0.501 |
| hexapod slip | 0.974 | 0.967 | 0.958 | 0.951 | 0.865 | 0.812 | 0.687 | 0.793 |

The quadruped still makes 86% of its flat-ground progress at mu = 0.1. This is
not a wiring fault: the contacts carry exactly the terrain's friction (checked
in the solver's contact list at mu 0.1 and 2.5). It is a property of the gait:
its feet already slide on 72% of stance load at mu = 1. Its fitness drops more
than its speed (0.450 at mu 1 against 0.335 at mu 0.3), because the slip factor
charges the skating. Whether training across friction makes evolved skaters lose
is a question for the evolution experiments; this run shows the terrain can
express it.

**Rough ground and obstacles.** For roughness alone the result is **null over
the train/test range**. Up to 0.08 m peak (RMS 1.4 cm), neither reference slows
down (quadruped 0.444-0.456 against 0.447 m/s; hexapod 0.454-0.470 against
0.456). The quadruped slows only from 0.16 m peak (0.408 m/s) and at 0.20 m
(0.368, -18%). The hexapod does not slow even at 0.20 m (0.466). The quadruped's
fitness does fall steadily with roughness (0.450 at 0 to 0.053 at 0.20 m) through
the posture and smoothness factors; the hexapod's does not (0.401-0.535).
Obstacles do reduce progress, the quadruped's strongly: 0.329 m/s with 20 and
0.243 with 40 (against 0.447). The hexapod is barely affected: 0.448 with 40
(against 0.456). Ten obstacles on seed 1 change nothing for either: none of
them lies on the path (the first one within \|y\| < 0.5 m, x < 5 m is obstacle
number 11).

**Determinism.** Repeating a run with the same seeds gives bit-identical results:
the flat row is identical in the `flat`, `slope` (slope 0), `friction` (mu 1) and
`uneven` (r 0, 0 obstacles) suites, and the rerun `uneven` rows matched the first
run exactly. Results differ across terrain seeds. With 0.04 m peak and 20
obstacles, quadruped fitness over seeds 0-4 is 0.162, 0.150, 0.419, 0.338 and
0.291; the hexapod's is 0.378, 0.445, 0.493, 0.343 and 0.486. The tests check
the same thing (`test_terrain_is_deterministic_per_seed_and_differs_across_seeds`,
`test_terrain_seed_changes_results_and_spawn_seed_is_repeatable`).

**Train against test.** Neither reference was selected on `train`; they are
hand-designed. So these numbers measure how much harder `test` is, not
generalisation:

| | quadruped train | quadruped test | hexapod train | hexapod test |
|---|---|---|---|---|
| mean fitness | 0.357 | 0.181 | 0.455 | 0.278 |
| worst environment | 0.024 (random_s4) | -0.675 (uneven_r0.06_o30_s102) | 0.344 (slope+10) | 0.004 (random_s104) |
| failure rate | 0% | 6% (2 falls on uneven_r0.06_o30) | 0% | 0% |
| mean m/s | 0.419 | 0.338 | 0.447 | 0.364 |

## Known limitations (not fixed here)

- **Upright factor on slopes.** The fitness's upright factor and `mean_tilt`
  measure tilt from world vertical, so standing parallel to a 10 deg slope costs
  10 deg of "tilt". That is the same for every creature on a given slope, but
  it lowers absolute slope scores. Fitness is core code; this is reported, not
  changed.
- **Energy efficiency ignores potential energy.** `energy_efficiency` and the
  cost of transport credit horizontal distance only, so climbing looks
  inefficient and descending looks efficient.
- **Torso on an obstacle counts as a fall** (0.25 s of torso contact within 1 s).
  Being high-centred on a rock is a real failure mode, but it is scored the same
  as tipping over.
- **Very slow creatures never leave the landing.** Below about 0.06 m/s a
  creature stays on the flat 0.6 m landing for the whole run, so terrain cannot
  affect it. Such creatures are already penalised as stationary below 0.1 m.
- **Clip planes.** The 120 m terrain boxes raised MuJoCo's model extent from 12
  to about 200, which moved the camera's near plane to about 2 m and cut the
  ground out of the tracking camera's foreground. `build_body` now sets
  absolute clip planes (5 cm, 300 m; `environment.fit_clip_planes`). This
  affects rendering only, and a test guards it.

## Media

`docs/media/terrains/` holds renders of the reference quadruped on flat ground,
slopes of +10 and -10 deg, uneven ground (0.04 m, 20 obstacles, seed 2), mu 0.2,
friction patches (0.8 x/ 2) and `random_s4`. Each terrain has a rest-pose
multi-view png, a wide `_overview.png` taken after 4 s of walking, the full run
as an mp4 from the tracking camera, and a `_sheet.png` of 8 frames. Regenerate
them with `.venv/bin/python scripts/render_terrains.py` (offscreen; no window
opens).
