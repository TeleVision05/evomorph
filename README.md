# EVOMORPH

Software-only evolutionary robotics lab: creatures with evolved bodies and CPG
gaits, simulated in MuJoCo. Brief: [docs/SPEC.md](docs/SPEC.md).

This repository currently holds the core: genome, procedural MJCF generator,
CPG controller, flat terrain, evaluator, metrics and fitness. Evolution,
MAP-Elites, extra terrains, experiments and the dashboard build on it.

## Setup

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/pre-commit install
```

## Commands

```bash
.venv/bin/python -m evomorph inspect --reference quadruped            # body plan + validation
.venv/bin/python -m evomorph inspect --genome g.json --png g.png      # iso/side/front/top still
.venv/bin/python -m evomorph inspect --reference hexapod --view       # interactive viewer
.venv/bin/python -m evomorph evaluate --reference hexapod --config configs/quick.yaml
.venv/bin/python -m evomorph evaluate --genome g.json --video run.mp4 --seed 3
.venv/bin/python -m evomorph random --n 20 --seed 0 --out results/random20
.venv/bin/python scripts/random_sweep.py --n 300 --seed 0 --workers 8 --out results/sweep
.venv/bin/python scripts/render_media.py docs/media/rnd0-0282.json
```

The viewer uses the blocking `mujoco.viewer.launch`, which runs under the plain
venv python on macOS (`launch_passive` would need `mjpython`). Close the window
to return.

## Tests

```bash
.venv/bin/python -m pytest          # default suite, a few seconds
.venv/bin/python -m pytest -m slow  # 300-genome random sweep
```

## Extending

| to add | do this |
|---|---|
| a CLI command | new `evomorph/commands/<name>.py` exposing `register(subparsers)` |
| a controller | subclass `controllers.base.Controller`, decorate a factory with `@register_controller("<kind>")`, add the kind to `GENE_BOUNDS["kind"]` |
| a terrain | implement `simulation.environment.build_terrain_extension` |
| a config section | append it to `configs/quick.yaml`, read it with `config.load_section(path, "<name>", YourDataclass)` |

Mutation and crossover must draw ranges from `morphology.genome.GENE_BOUNDS` and
finish with `repair(genome)`, which always returns a valid genome.

## Media

`docs/media/` holds a still (`.png`), a contact sheet (`_sheet.png`) and an mp4 for
the two reference creatures and the best of 300 random genomes (`rnd0-0282`).
