# Evolution engine

How EVOMORPH evolves creatures (brief sections 5, 7 and 11), and what the
first runs on this laptop showed. Code: `evomorph/evolution/`. Command:
`python -m evomorph evolve`.

## Running it

```bash
# GA, joint morphology + controller evolution, quick laptop budget (~6-8 min)
.venv/bin/python -m evomorph evolve --config configs/quick.yaml
# MAP-Elites with the same budget
.venv/bin/python -m evomorph evolve --config configs/quick.yaml --algorithm map_elites
# Controller-only on the reference quadruped, or body-only with its controller
.venv/bin/python -m evomorph evolve --config configs/quick.yaml --mode controller
.venv/bin/python -m evomorph evolve --config configs/quick.yaml --mode morphology
# The GA without its diversity mechanism
.venv/bin/python -m evomorph evolve --config configs/quick.yaml --no-diversity
```

Options: `--algorithm ga|map_elites`, `--mode joint|controller|morphology`,
`--workers N` (capped at 6), `--budget SIMS`, `--seed N`, `--name DIR`,
`--results ROOT`, `--no-diversity`. One progress line is printed per
generation (GA) or batch (MAP-Elites).

## The contract other stages build on

| piece | what it is |
|---|---|
| `EvalSpec(runs, sim, fitness, aggregate)` | the `(EnvSpec, seed)` runs one genome is scored on; `aggregate` is `mean` or `min` |
| `evaluate_spec(genome, spec)` / `SpecEvaluator(spec)` | runs the core `evaluate()` once per run; fitness aggregated, metrics and terms averaged (`fell` becomes the fraction of runs that fell), per-run values kept in `seed_fitness`, `run_labels`, `run_metrics`; invalid if any run is invalid |
| evaluator | any picklable `Genome -> EvalResult` callable; its cost is its `n_sims` attribute (or `sims_per_eval=`) |
| budget | simulations = genomes x runs; both algorithms stop at it and never exceed it |
| mode | `joint` (all genes), `controller` (body frozen to `frozen_body`), `morphology` (controller frozen to `frozen_controller`, resized to the limb count by the core's `repair`); the `kind` gene never evolves |
| `run_evolution(cfg, results_root, name, evaluator=None, sims_per_eval=None)` | one run into a fresh directory; returns `RunOutcome` |

## One GA generation

1. Score the population in one batch over the process pool (at most 6
   workers). Results return in input order and every random draw happens in
   the parent process, so `metrics.csv` is byte-identical with 1 or 6 workers
   (tested).
2. Elites survive unchanged and are not re-scored (evaluation is
   deterministic): the global top `elite_count`, plus with diversity on the
   best of every body plan (limbs x segments).
3. The rest are children of two tournament winners (size 3): uniform
   crossover with probability `crossover_prob`, then mutation, then the
   mode's frozen genes are re-imposed, then the core's `repair` and
   `validate_genome`. A small fraction are random immigrants.

Selection pressure with diversity on is a rank score divided by the niche
count on normalised gene distance (fitness sharing on ranks, because fitness
can be negative). `--no-diversity` switches off sharing, body-plan elites and
immigrants together.

## MAP-Elites

A 10 x 10 archive (quick) over `sustained_velocity` (0-0.8 m/s) and
`energy_efficiency` (0-0.7). Ranges come from `scripts/descriptor_probe.py
--n 200 --seed 7 --workers 6`: valid, non-fallen random creatures reached at
most 0.64 m/s and efficiency 0.55, and the references sit near 0.4 m/s and
0.45. A cell keeps the best valid genome that fell in none of its runs;
replacement needs strictly higher fitness. 192 random genomes start it, then
each batch mutates (and with probability 0.2 crosses) elites drawn uniformly
from the occupied cells. QD-score is the sum over filled cells of fitness +
3.2 (the magnitude of the lowest valid fitness), so filling a cell never
lowers it.

## Run directory

`results/<name or run_NNN>/` is created fresh (a taken name is refused with a
plain message; `run_NNN` claims the next free number atomically):

| file | contents |
|---|---|
| `config.yaml` | fully resolved config; `evolve --config` on it re-runs the experiment, and re-scoring a saved genome with it reproduces its stored numbers exactly (tested) |
| `metrics.csv` | one row per generation/batch: sims, evaluations, best/mean/median/std/p10/p90 fitness, valid/fell counts, gene distance, body plans, behaviour spread, archive cells/coverage/QD-score, champion id/fitness/plan/speed/efficiency |
| `timing.csv` | wall seconds per step and elapsed (kept out of metrics.csv so that file is deterministic) |
| `generations/gen_NNNN.json` | every individual: id, parent ids, origin, generation of origin, status, fitness, per-run fitness, metrics, terms, genes |
| `genomes/champion_gen_NNNN.json`, `genomes/final/` | champion after each step; final population (GA) or archive elites (MAP-Elites) |
| `archive/` | `archive.json` + `elites/cell_II_JJ.json` (the GA keeps a passive archive of everything it scored, so both algorithms report coverage and QD-score) |
| `best_genome.json`, `best_result.json`, `best_heldout.json` | champion, its aggregated evaluation, and its score on held-out seeds (seed + 1000, outside the budget) |
| `frozen_template.json` | the frozen body or controller (controller and morphology modes) |
| `summary.md` | what ran, first vs last generation, champion with exploit checks, re-score check, held-out-seed score, archive, the command to reproduce |

Rendering and plots (offscreen, files only):

```bash
.venv/bin/python scripts/render_run.py results/<run> --elites 6
.venv/bin/python scripts/plot_runs.py --out results/plots results/<runA> results/<runB>
```

## Results on this laptop

All runs below: `configs/quick.yaml` as committed (48 genomes, 3 runs per
genome on flat ground at friction 0.6/0.8/1.0 with spawn seeds 0/1/2, budget
6000 simulations, search seed 0, 6 workers). The laptop was shared with other
jobs during every run (load average 20-33 on 12 cores), so wall times are
pessimistic.

### 1. GA, joint evolution

`python -m evomorph evolve --config configs/quick.yaml --name quick_ga_joint_s0`
(51 generations, 6000 simulations, 433.8 s).

| generation | sims | best | mean | median | fell (of 48) | gene distance | body plans |
|---|---|---|---|---|---|---|---|
| 0 | 144 | +0.318 | -0.746 | -0.722 | 21 | 0.322 | 8 |
| 10 | 1320 | +0.750 | -0.108 | +0.010 | 5 | 0.285 | 8 |
| 20 | 2490 | +0.750 | -0.024 | +0.041 | 2 | 0.254 | 8 |
| 30 | 3660 | +0.750 | -0.069 | +0.020 | 2 | 0.255 | 8 |
| 40 | 4842 | +0.760 | -0.093 | +0.010 | 2 | 0.265 | 8 |
| 50 | 6000 | +1.021 | +0.123 | +0.078 | 0 | 0.226 | 8 |

Generation 0 mostly cannot walk: median forward displacement -0.003 m and 4
of 48 move a metre (mean over the three runs). Best and mean fitness both
rise, but not smoothly: the mean climbs from -0.75 to about 0 in ten
generations and then plateaus, and the best sat at 0.750 from generation 8 to
47 before a mutation of an 8-legged elite reached 1.021 in generation 48. With
sharing, body-plan elites and immigrants, much of the population stays
exploratory rather than converging, which keeps the mean low.

Champion `g0048-009` ([video](media/evolution/ga_champion.mp4),
[frames](media/evolution/ga_champion_sheet.png),
[genome](media/evolution/ga_champion.json)): 8 legs, two segments, long soled
feet, metachronal gait at 2.24 Hz. Per run (friction 0.6 / 0.8 / 1.0):

| check | 0.6 | 0.8 | 1.0 | reading |
|---|---|---|---|---|
| fitness | +1.039 | +1.050 | +0.974 | |
| forward displacement (m, 10 s) | 6.81 | 6.64 | 6.42 | about 0.65 m/s |
| foot support fraction | 1.000 | 1.000 | 1.000 | all ground load on feet: no belly or shin dragging |
| airborne fraction | 0.001 | 0.001 | 0.001 | never hops |
| max height gain (m) | 0.014 | 0.014 | 0.014 | no launch |
| max lateral deviation (m) | 0.33 | 0.09 | 0.20 | straight within 5% of distance |
| mean tilt (rad) | 0.034 | 0.035 | 0.035 | torso level (2 degrees) |
| slip fraction | 0.76 | 0.68 | 0.67 | feet skate (see below) |

Re-scored from the saved `config.yaml`: identical. On held-out spawn seeds
(+1000): +1.028, so no winner's-curse drop. The frames show it walking on its
feet with a visible leg wave; the torso never touches the ground.

**Finding: skating is not punished by the friction range.** Evaluated
across friction (`scripts/friction_sweep.py docs/media/evolution/ga_champion.json
--reference quadruped`, seed 0, default sim), the champion's distance hardly
depends on it: 6.27 m at friction 0.2 (93% of stance load sliding), 6.89 m at
0.45, 6.43 m at 1.0, 5.86 m at 1.5. The hand-designed reference quadruped
behaves the same way (3.77 m at 0.2, 4.46 m at 1.0). The creature walks, but
its feet slide through most of each stance, and training on friction 0.6-1.0
did not select against that because the slower feet cost no speed. The only
charge is the slip factor (at most half the progress). This is not a bug in
the evolution code and not a loophole that makes a non-walker win; it is the
known core weakness (core open issues: stiff position servos, incompatible
stance arcs). Making skating lose needs either a stronger `slip_penalty` or a
contact or controller change in the core; neither was made here, because
re-tuning the score after seeing the winner is exactly what the house rules
forbid. Also noted: the champion's `torso_mass_scale` is at its upper bound
(2.0); joint torque scales with total mass, so a heavy torso buys strength
for light legs. The population median moved only from 1.33 to 1.46, so this
is a property of this champion, not a population-wide drift.

### 2. MAP-Elites, joint evolution

`python -m evomorph evolve --config configs/quick.yaml --algorithm map_elites --name quick_me_joint_s0`
(42 batches of 48 genomes, 6000 simulations, 722.6 s).

| batch | sims | coverage | QD-score | best | mean of elites | body plans in archive | behaviour spread |
|---|---|---|---|---|---|---|---|
| 0 | 144 | 5% | 15.6 | +0.318 | -0.084 | 3 | 0.329 |
| 10 | 1584 | 42% | 139.5 | +0.484 | +0.121 | 3 | 0.386 |
| 20 | 3024 | 56% | 191.5 | +0.825 | +0.220 | 3 | 0.417 |
| 30 | 4464 | 60% | 205.5 | +0.828 | +0.225 | 3 | 0.424 |
| 41 | 6000 | 62% | 213.7 | +0.828 | +0.247 | 3 | 0.432 |

Of the 192 random initial genomes, 70 fell in at least one run and could not
enter the archive. Every one of the 62 final elites re-scores to its stored
fitness, metrics and cell (`scripts/check_archive.py
results/quick_me_joint_s0`: 62/62).

[Contact sheet](media/evolution/me_elites_sheet.png) of six elites from
cells chosen far apart (greedy farthest point from the best cell, rendered by
`scripts/render_run.py results/quick_me_joint_s0 --elites 6`; one row each,
five frames over 10 s at friction 0.6): a slow, efficient long-bodied walker
(cell (3,9), 0.30 m/s, efficiency 0.65, the champion, slip 0.36), a small
compact body that twists and skates fast (cell (9,2), 0.87 m/s, slip 0.97,
fitness 0.11), a near-stationary shuffler (cell (0,0)), an alternating-gait
turner (cell (4,3)), a 0.56 Hz creeper (cell (0,5)) and a mid-speed walker
(cell (5,6)). They are different behaviours and proportions.

**Finding: MAP-Elites kept behaviours apart but not bodies.** All six of
those elites are 6-leg two-segment bodies, and so are 54 of the 62 elites
(8x2: 6, 4x2: 2); 53 use the diagonal gait. One body plan was good enough to
win almost every behaviour cell. The GA's passive archive at the same budget
holds 6x2: 26, 8x2: 25, 4x2: 13, 4x1: 1. Speed and efficiency are behaviour
descriptors; they do not ask for different bodies, so a QD archive on them
is not by itself evidence of morphological diversity.

**GA vs MAP-Elites at equal budget (one seed, so no significance claim).**
MAP-Elites covered more of the archive early (42% vs 38% at ~1.5k
simulations, 56% vs 51% at ~3k), but by 6000 simulations the GA's passive
archive was ahead on coverage (65% vs 62%), QD-score (227.9 vs 213.7) and
best fitness (1.021 vs 0.828). MAP-Elites' champion is the better gripper
(slip 0.36 vs 0.70) and more efficient (0.65 vs 0.53) but half as fast
(0.30 vs 0.63 m/s). Several seeds per condition are needed before either
algorithm can be called better.
