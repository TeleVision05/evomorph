# EVOMORPH

Software-only evolutionary robotics lab: creatures with evolved bodies and CPG
gaits, simulated in MuJoCo. Brief: [docs/SPEC.md](docs/SPEC.md).

This repository currently holds the core: genome, procedural MJCF generator,
CPG controller, flat terrain, evaluator, metrics and fitness. Evolution,
MAP-Elites, extra terrains, experiments and the dashboard build on it.

## Setup

Needs Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/)
(`curl -LsSf https://astral.sh/uv/install.sh | sh`, or `brew install uv`).

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements-lock.txt -e ".[dev]"
.venv/bin/pre-commit install
```

`requirements-lock.txt` pins the exact versions the tests ran with. Results are
bit-identical across runs and processes, but only within one MuJoCo version.

## Worked example

```bash
# 1. a few random creatures, written as JSON genomes
.venv/bin/python -m evomorph random --n 5 --seed 0 --out results/demo
# 2. describe and validate one, and render its rest pose
.venv/bin/python -m evomorph inspect --genome results/demo/rnd0-0000.json --png results/demo/0.png
# 3. simulate it: metrics table, fitness terms, optional video
.venv/bin/python -m evomorph evaluate --genome results/demo/rnd0-0000.json --video results/demo/0.mp4
# 4. export a reference creature as a genome file and evaluate it on slippery ground
.venv/bin/python -m evomorph inspect --reference quadruped --save results/demo/quad.json
.venv/bin/python -m evomorph evaluate --genome results/demo/quad.json --friction 0.3
```

Other commands:

```bash
.venv/bin/python -m evomorph inspect --reference hexapod --view       # interactive viewer
.venv/bin/python -m evomorph evaluate --reference hexapod --config configs/quick.yaml
.venv/bin/python scripts/random_sweep.py --n 300 --seed 0 --workers 8 --out results/sweep
.venv/bin/python scripts/render_media.py --no-references results/sweep/genomes/rnd0-0001.json
```

The viewer uses the blocking `mujoco.viewer.launch`, which runs under the plain
venv python on macOS (`launch_passive` would need `mjpython`). Close the window
to return; Backspace respawns the creature at its spawn pose.

## How a creature is scored

`evaluate()` settles the creature for 1 s, then runs the controller for 10 s,
stepping physics one 2 ms step at a time. Energy, acceleration, airborne time,
ground loads per body part and foot slip are integrated at every physics step,
and the glitch guards (NaN, integrator blow-up, runaway speed or spin, launch,
tunnelling) check the peak over every step. Raw metrics (`METRIC_KEYS`) are kept
separate from the score.

```
fitness = (sustained_velocity / 0.2 m/s) x upright x lateral x rotation x energy
          x slip x support x airborne x effort x smoothness  - fall - stationary
```

Each quality factor is 1 for a flawless gait and falls toward 0 as its flaw
grows, so no speed buys back drift, spinning, brute force, skating, shin-dragging
or hopping. `evomorph/simulation/fitness.py` has the formulas;
`configs/quick.yaml` has the scales.

One evaluation is one noise realisation. Score anything that decides survival
(elites, archive cells, champions) with `evaluate_robust(genome, seeds=(0, 1, 2))`,
which averages several seeds (and optionally a perturbed timestep), or the search
will select lucky seeds.

## Tests

```bash
.venv/bin/python -m pytest          # default suite, about 10 s
.venv/bin/python -m pytest -m slow  # 300-genome random sweep
```

## Extending

| to add | do this |
|---|---|
| a CLI command | new `evomorph/commands/<name>.py` exposing `register(subparsers)` |
| a controller | subclass `controllers.base.Controller` and decorate a factory with `@register_controller("<kind>")`; that also makes `<kind>` a legal `kind` gene. Keep its weights in `genome.controller.params` (a free-length vector gene) |
| a terrain | implement `simulation.environment.build_terrain_extension`, including `height_at(x, y)` so the guards measure against the local ground |
| a config section | append it to `configs/quick.yaml`, read it with `config.load_section(path, "<name>", YourDataclass)` |

Mutation and crossover must draw ranges from `morphology.genome.GENE_BOUNDS` and
finish with `repair(genome)`, which always returns a valid genome. `vector`
genes keep their length; mutate them element-wise.

## Media

`docs/media/` holds a still (`.png`), a contact sheet (`_sheet.png`) and an mp4 for
the two reference creatures and for `rnd0-0276`, the best of the 300-genome seed-0
sweep (a hexapod on short soles, alternating gait, 1.98 m in 10 s). Its genome is
`docs/media/rnd0-0276.json`; re-render with
`scripts/render_media.py --no-references docs/media/rnd0-0276.json`.
