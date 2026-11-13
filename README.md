# EVOMORPH

**A software-only evolutionary robotics lab.** Creatures with evolved bodies
(2 to 8 legs, one or two segments per leg, evolved proportions, masses and
joint ranges) and evolved central-pattern-generator gaits learn to walk in
MuJoCo physics, by a genetic algorithm or by MAP-Elites, across slopes,
rough ground, obstacles and slippery patches. Nothing is animated: every
picture is a saved genome re-simulated offscreen.

It was built to answer one question, and to be able to answer it with "no":

> **Does evolving morphology and controller together give more robust and more
> behaviourally diverse locomotion than evolving a controller for a fixed body?**

<!-- RESULTS-HEADLINE -->

[![Champions of one joint-evolution run, generation 0 to the last](docs/media/hero/evolution.jpg)](docs/media/hero/evolution.mp4)

*Click for the clip: the champion after each generation of one real run,
re-simulated from its saved genome (details in [docs/media/hero/NOTES.md](docs/media/hero/NOTES.md)).*

## Contents

- [Install](#install)
- [Commands](#commands)
- [Architecture](#architecture)
- [The fitness function](#the-fitness-function)
- [Research results](#research-results)
- [Dashboard](#dashboard)
- [Limits and known issues](#limits-and-known-issues)
- [Reproduce everything](#reproduce-everything)

## Install

Needs Python 3.12 (3.11 also works) and [uv](https://docs.astral.sh/uv/)
(`curl -LsSf https://astral.sh/uv/install.sh | sh` or `brew install uv`).
Everything runs on the CPU; no GPU is used.

```bash
git clone https://github.com/TeleVision05/evomorph.git
cd evomorph
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements-lock.txt -e ".[dev,dashboard]"
.venv/bin/pre-commit install        # optional: ruff + size checks on every commit
.venv/bin/python -m pytest -q       # fast suite
```

`requirements-lock.txt` pins the exact versions the results were made with
(macOS arm64, Python 3.12.13, MuJoCo 4.0.1). Runs are bit-identical across
processes and worker counts, but only within one MuJoCo version.

On a Linux machine without a display, set `MUJOCO_GL=egl` or `MUJOCO_GL=osmesa`
for offscreen rendering (CI uses OSMesa). The six tests that render are skipped,
and counted as skipped, when no OpenGL context can be made.

## Commands

The brief's command list (section 13), each run in this repository with its
real output (trimmed where marked).

<!-- COMMANDS -->

## Architecture

```
evomorph/
  morphology/    genome (GENE_BOUNDS: every gene is 'morphology' or 'controller'),
                 procedural MJCF generator, validator
  controllers/   Controller interface + registry; the CPG controller
  simulation/    terrain (flat, slope, uneven, friction patches, random), suites
                 (train / held-out test), evaluator (physics-rate metrics,
                 glitch guards), metrics, fitness
  evolution/     GA, MAP-Elites, operators, selection and diversity, archive,
                 run directories, process pool (max 6 workers)
  experiments/   the A-D comparison: conditions, design, post-hoc re-scoring,
                 measures, results table, terrain robustness
  analysis/      statistics (bootstrap CIs, Mann-Whitney, A12, Holm), plots
                 (Agg backend), offscreen gallery, summaries
  visualization/ offscreen renderer, interactive viewer, Streamlit dashboard
  commands/      one CLI module per command, discovered automatically
configs/         quick.yaml, research.yaml, comparison.yaml
scripts/         rendering, plotting, budget planning, validation helpers
```

How a creature is made and scored:

1. A **genome** holds body genes (limb pairs 1-4, segments 1-2, torso size and
   mass, limb lengths, thickness, foot size, splay, hip-axis tilt, joint
   ranges) and controller genes (gait pattern, frequency, hip and knee
   amplitudes, knee phase offset, per-limb phase offsets and posture biases).
2. The **generator** writes an MJCF model with a free root joint, symmetric
   limbs, hinge joints, position servos, contact feet and mass from density.
   The validator compiles it before anything is simulated; an invalid genome
   is scored, never crashed on.
3. The **CPG** drives each joint with `A sin(2 pi f t + phase)`, the phases
   coming from the gait pattern (alternating, diagonal, tripod, metachronal)
   for whatever limb count the body has, plus evolved per-limb offsets.
4. The **evaluator** settles the creature for 1 s, then runs 10 s at a 2 ms
   physics step. Energy, contacts, slip and airborne time are integrated at
   every physics step; guards reject non-finite states, runaway speed or spin,
   launches and tunnelling.
5. **Evolution** scores every genome on all 11 training environments (11
   simulations per genome), so every elite and archive cell carries an
   11-run score rather than one lucky run. Budgets are counted in simulations.

Design documents: [docs/EVOLUTION.md](docs/EVOLUTION.md) (engine),
[docs/TERRAINS.md](docs/TERRAINS.md) (terrains and suites),
[docs/EXPERIMENT.md](docs/EXPERIMENT.md) (the comparison design),
[docs/DASHBOARD.md](docs/DASHBOARD.md), [docs/RESULTS.md](docs/RESULTS.md),
and the original brief [docs/SPEC.md](docs/SPEC.md).

## The fitness function

```
fitness = (sustained_velocity / 0.2 m/s)
          x upright x lateral x rotation x energy x slip x support x airborne x effort x smoothness
          - 1.0 x fell - 0.2 x stationary
```

| factor | 1 when | falls toward 0 when |
|---|---|---|
| upright | torso level | the body leans or rolls |
| lateral | the path is straight | it crabs sideways or wanders |
| rotation | heading holds | it spins or circles |
| energy | cost of transport is low | it thrashes |
| slip | feet grip | feet slide while loaded (skating) |
| support | only feet touch the ground | shins, knees or belly drag |
| airborne | at most 5% of the time in the air | it hops, bounds or launches |
| effort | servos are not saturated | it strains |
| smoothness | accelerations are gentle | it vibrates or slams |

Why it is built this way:

- **Multiplicative, not additive.** An earlier additive version capped each
  penalty, so any creature faster than about 0.7 m/s won however badly it
  moved. Here every flaw scales progress down: halving a factor halves the
  score at any speed, so speed cannot buy back a bad gait.
- **Sustained velocity, not displacement.** A creature that lunges and stops
  scores its steady speed, not its lunge.
- **Backward motion and falling are penalties, bounded.** Backward progress is
  not softened by the quality factors and is capped at -2, so every valid score
  is at least -3.2 and an invalid genome (-20) always ranks last.
- **Metrics are kept separately from the score.** 36 raw metrics are stored
  for every run, and the score's terms sum exactly to the fitness, so any
  champion can be audited term by term.

The scales were calibrated on a 300-genome random sweep plus two hand-built
reference creatures, never on evolved champions (`evomorph/simulation/fitness.py`,
`configs/*.yaml`).

## Research results

<!-- RESULTS -->

## Dashboard

```bash
.venv/bin/streamlit run evomorph/visualization/dashboard.py
```

Six sections over any run directory, including runs still in progress:
evolution viewer, creature inspection (with an on-demand clip and a button
that opens the MuJoCo viewer), the MAP-Elites archive as a clickable heat map,
statistics, side-by-side comparison and generation history. Every image is a
saved genome rendered offscreen in a child process; every clip is a
re-simulation whose fitness is shown next to the stored one.
Details: [docs/DASHBOARD.md](docs/DASHBOARD.md).

| Evolution viewer | MAP-Elites archive |
|---|---|
| ![Evolution viewer](docs/media/dashboard/evolution_viewer.png) | ![MAP-Elites archive](docs/media/dashboard/map_elites_archive.png) |
| **Creature inspection** | **Comparison** |
| ![Creature inspection](docs/media/dashboard/creature_inspection.png) | ![Comparison](docs/media/dashboard/comparison.png) |

The screenshots show two quick single-seed runs (`configs/quick.yaml`,
6000 simulations each), not the comparison experiment.

## Limits and known issues

<!-- LIMITS -->

## Reproduce everything

<!-- REPRODUCE -->

## License

MIT, copyright 2026 Tarun Vadapalli. See [LICENSE](LICENSE).
