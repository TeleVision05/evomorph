# Evolution engine

How EVOMORPH evolves creatures (brief sections 5, 7 and 11), and what the
first runs on this laptop showed. Code: `evomorph/evolution/`. Command:
`python -m evomorph evolve`.

## Running it

```bash
# GA, joint morphology + controller evolution, quick budget (6-12 min measured here)
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
47 before a mutant of the 8-legged elite `g0045-032` (+0.660) reached 1.021
in generation 48. With
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
did not select against that because skating costs it almost no speed. The only
charge is the slip factor (at most half the progress). This is not a bug in
the evolution code and not a loophole that makes a non-walker win; it is the
known core weakness (core open issues: stiff position servos, incompatible
stance arcs). Making skating lose needs either a stronger `slip_penalty` or a
contact or controller change in the core; neither was made here, because
re-tuning the score after seeing the winner is exactly what the house rules
forbid. Also noted: joint torque scales with total mass, so a heavy torso
buys strength for the legs. The champion of generations 8-47 (`g0008-034`)
had `torso_mass_scale` at its upper bound (2.0) and the final champion sits at
1.83, but the population median moved only from 1.33 (generation 0) to 1.46
(generation 35), so this is not a population-wide drift. Watch it in longer
runs.

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

### 3. Does the diversity mechanism do anything?

Same config, seed and budget with it switched off:
`python -m evomorph evolve --config configs/quick.yaml --no-diversity --name quick_ga_joint_s0_nodiv`
(44 generations, 6000 simulations, 657.5 s). Generation 0 is identical in
both runs (same random draws), so the curves start together. Plots:
`scripts/plot_runs.py --out docs/media/evolution/plots results/quick_ga_joint_s0
results/quick_ga_joint_s0_nodiv results/quick_me_joint_s0` →
[diversity](media/evolution/plots/diversity.png),
[fitness](media/evolution/plots/fitness.png), [QD](media/evolution/plots/qd.png).

| sims | | best | mean | gene distance | body plans | behaviour spread | coverage | QD-score |
|---|---|---|---|---|---|---|---|---|
| ~1500 | on | +0.750 | -0.057 | 0.274 | 8 | 0.358 | 38% | 128.0 |
| | off | +0.497 | +0.075 | 0.094 | 3 | 0.289 | 34% | 114.7 |
| ~3000 | on | +0.750 | -0.015 | 0.262 | 8 | 0.371 | 50% | 172.7 |
| | off | +1.218 | +0.193 | 0.126 | 3 | 0.408 | 50% | 177.3 |
| ~4500 | on | +0.760 | -0.059 | 0.258 | 8 | 0.383 | 57% | 198.0 |
| | off | +1.301 | +0.232 | 0.118 | 4 | 0.452 | 59% | 215.5 |
| 6000 | on | +1.021 | +0.123 | 0.226 | 8 | 0.501 | 65% | 227.9 |
| | off | +1.588 | +0.598 | 0.083 | 1 | 0.285 | 68% | 252.3 |

It does what it is built to do: after the first 1500 simulations the
population keeps a mean gene distance of 0.255 (vs 0.117 without) and all 8
body plans in every generation (vs 3.5 on average without, down to a single
plan at 3318 and 6000 simulations). Behaviour spread, though, is not
consistently higher with it.

**Negative result for this seed:** without diversity the GA found the better
walker. Its champion `g0038-028` ([video](media/evolution/ga_nodiv_champion.mp4),
[frames](media/evolution/ga_nodiv_champion_sheet.png)) is a tall-legged
hexapod in an alternating tripod at 1.77 Hz: fitness +1.588 (per run +1.520,
+1.604, +1.641; held-out seeds +1.602), about 7.9 m in 10 s, foot support
1.000, airborne 0.0-0.8%, height gain 2 cm, tilt 2 degrees, slip 0.41-0.64.
It walks on its feet; no exploit found. Its archive coverage and QD-score are
also higher. With one seed per condition this does not show that diversity
hurts, but it does show the mechanism, as configured (sharing radius 0.15,
one elite per body plan, 5% immigrants), spends a large share of the
population on plans that are not competitive at this budget: mean fitness
stays between -0.12 and +0.12 after the first 1500 simulations with it,
against +0.01 to +0.60 without.

### 4. Throughput and practical budgets

| measurement | command | result |
|---|---|---|
| 404 independent simulations | `scripts/descriptor_probe.py --n 200 --seed 7 --workers 6` | 10.1/s (load average 26-33); an earlier identical probe on a quieter machine: 17.2/s |
| GA, 6000 simulations | section 1 | 13.8/s (433.8 s) |
| GA without diversity, 6000 | section 3 | 9.1/s (657.5 s) |
| MAP-Elites, 6000 | section 2 | 8.3/s (722.6 s), load average up to 56 |

One simulation is a 1 s settle plus a 10 s run (about 0.2 s of CPU on an idle
core). On this 12-core laptop, shared with the owner's work and capped at 6
workers, plan on 8-17 simulations/s:

| budget (simulations) | wall time at 8-17/s | use |
|---|---|---|
| 6000 | 6-12 min | `quick.yaml`: one run per condition |
| 30000 | 30-60 min | `research.yaml`: one condition, one seed |
| 12 x 30000 | 6-12 h | 3 modes + MAP-Elites, 3 seeds each: overnight |

One seed per condition (as above) is enough to see mechanisms work and too
few to rank algorithms; the comparisons in sections 2 and 3 need at least 3
seeds per condition, which is what the 12-run overnight budget buys.

## Open issues (2026-11-13)

- Every comparison above is one seed per condition; none is a ranking.
- Feet skate and the friction range 0.6-1.0 does not make that cost speed
  (section 1). Needs a core decision: slip penalty, contact model or a
  closed-loop controller.
- MAP-Elites on speed x efficiency kept one body plan in 54 of 62 cells
  (section 2). For morphological diversity add a body descriptor or run the
  archive per body plan.
- The diversity mechanism, as configured, kept 8 body plans but cost best and
  mean fitness on this seed (section 3). Sharing radius, plan elites and
  immigrants have not been tuned.
- Controller-only and morphology-only modes are tested (operators, CLI) but
  have not had a full quick run yet.
- Pre-existing core test `tests/test_evaluator.py::test_single_process_throughput`
  (a 0.5 CPU-second limit for one 10 s evaluation) failed in 2 of 10 full
  suite runs on this machine under load average ~30 (0.549 s measured). It
  is a timing threshold in the core, unchanged on this branch.
