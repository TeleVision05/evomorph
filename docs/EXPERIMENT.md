# The comparison experiment

Brief section 9: **does jointly evolving morphology and controller give more
robust and more behaviourally diverse locomotion than evolving a controller
for a fixed body?** This document describes the experiment that answers it,
how to run and analyse it, the smoke-test result and the measured cost of the
full run. Code: `evomorph/experiments/` and `evomorph/analysis/`. Config:
`configs/comparison.yaml`.

## Running it

```bash
# Smoke test of the whole pipeline (every condition, post-hoc evaluation, table)
.venv/bin/python -m evomorph experiment --type comparison --quick
.venv/bin/python -m evomorph analyze --run results/experiments/comparison_quick

# The full design (see "Budget for the full run" before starting it)
.venv/bin/python -m evomorph experiment --type comparison
.venv/bin/python -m evomorph analyze --run results/experiments/comparison

# analyze also works on any single run directory
.venv/bin/python -m evomorph analyze --run results/experiments/comparison/runs/joint_s0
```

`experiment` options: `--config` (default `configs/comparison.yaml`),
`--quick`, `--budget SIMS` (per run), `--seeds N`, `--seed-start N`,
`--workers N` (capped at 6), `--name`, `--results` (default
`results/experiments`). `analyze` options: `--run DIR`, `--no-render` (skip the
rendered gallery).

**Resuming.** Re-run the same command. Finished runs are skipped; a run whose
search finished but whose evaluation did not is only evaluated; a run whose
search was interrupted is moved to `incomplete/` (never deleted) and redone.
Asking for more seeds of the same design adds runs. A different design
(budget, suites, evolution settings) in the same folder is refused with a
plain message, because its runs could not be compared with those already
there. Runs go seed by seed, all conditions within a seed, so an interrupted
experiment always holds complete blocks of seeds.

## The design

### Conditions

| | condition | algorithm | what evolves | fixed |
|---|---|---|---|---|
| A | controller-only | GA | controller genes | body = reference quadruped |
| B | morphology-only | GA | body genes | controller = baseline below |
| C | joint | GA | everything | nothing |
| D | joint MAP-Elites | MAP-Elites | everything | nothing |

Everything else is shared and identical: the engine and its settings
(`evolution` and `map_elites` sections), the training suite, the budget in
simulations, the search seeds, the population size, and the post-hoc
evaluation. The `kind` gene (controller family) never evolves.

**B's baseline controller** is the reference quadruped's own hand-designed CPG,
unchanged: diagonal gait, 1.5 Hz, hip amplitude 0.45 rad, knee amplitude
0.40 rad, knee phase offset +pi/2, every per-limb phase offset and posture bias
0. It adapts to each limb count through the core alone: the CPG derives each
limb's phase from the gait pattern and the limb count, and `repair` sizes the
per-limb arrays. Diagonal is the standard alternating gait at every count: an
alternating step for 2 limbs, a trot for 4, the alternating tripod for 6
(diagonal equals tripod there), an alternating tetrapod for 8.

Why it is fair: (1) it is the other half of the same reference creature, so A
(reference body, evolved controller) and B (reference controller, evolved body)
are mirror images; (2) it was fixed with the core, before any experiment, so
it was not tuned to make bodies look good or bad; (3) its amplitudes are
angles, so a body that evolves longer legs gets a longer stride from it, which
is exactly what B can exploit; (4) its limit is stated rather than hidden: one
frequency and one phase pattern for every body. If a body needs a different
gait, B cannot find it. That cost is what B measures. How the baseline drives
bodies of every limb count is measured below.

### Training and test

* **Training suite** `train` (11 environments: flat, slopes +5/-5/+10 deg,
  friction 0.4 and 1.6, two uneven fields, friction patches, two random
  terrains; `evomorph/simulation/suites.py`, `docs/TERRAINS.md`). Every genome
  is scored on every training environment once, environment i with spawn seed
  i, and its fitness is the mean. So every elite, archive cell and champion
  carries an 11-run score, never a single lucky run; friction from 0.4 to 1.6
  is in it, so feet that skate lose fitness through the slip factor.
* **Held-out suite** `test` (11 environments, none shared with training,
  shifted parameters, unseen terrain seeds 101-105). Never seen during the
  search.

### Equal budgets

Budgets are counted in simulations, not generations. One genome costs 11
simulations in every condition. The engine stops when the next genome would
not fit, so every run spends exactly the budget rounded down to whole genomes.
Before a run is evaluated or tabulated, its saved `config.yaml` must equal the
design's config for that condition and seed (worker count aside) and its
`metrics.csv` must show exactly that budget spent; otherwise the experiment
stops with an error (tested). `evolution.max_generations` must be null.

MAP-Elites' random initialisation and its batch size are both set to the GA's
population size, so D starts from as many random creatures as the GA's
generation 0 and makes as many children per step as a GA generation. With the
same search seed, C and D draw the *same* random generation-0 creatures (both
use one generator seeded with the run seed); they diverge from the first
selection on. This is a common-random-numbers design, not an accident; the
tests are still unpaired (see Statistics).

### Measures (per condition and seed)

After the search, outside the budget and identically for every run:

* **Champion** (best genome by training fitness): re-scored on all 11
  training and all 11 held-out environments with 3 *fresh* spawn seeds each
  (1000-1002 on train, 2000-2002 on test), seeds it was never selected on.
  Comparing its stored (selected) fitness with the fresh training score shows
  the winner's curse; the test score is robustness.
* **Best-K final set**: the 10 best distinct valid genomes of the final GA
  population or MAP-Elites archive, by training fitness, each re-scored on
  both suites with one fresh seed. Equal set sizes make a population and an
  archive comparable.
* **Whole final population or archive**: diversity and training statistics
  from the stored training evaluations.

| measure | definition |
|---|---|
| best / mean fitness | champion's fitness / best-K set's mean fitness, on train and test |
| forward speed | `avg_forward_velocity`, m/s, mean over valid runs |
| energy efficiency | `energy_efficiency` = 1/(1 + cost of transport), 0-1 |
| stability | `mean_tilt`, rad (plots in degrees), mean torso tilt; lower is steadier |
| failure rate | share of runs that fell **or** were invalid; invalid share reported separately |
| generalisation gap | test minus train fitness (fresh seeds both); negative = worse on unseen terrain |
| morphological diversity | mean pairwise normalised distance over the *morphology* genes (0-1) and the number of body plans; A has one body, so 0 by construction |
| behavioural diversity | mean pairwise distance between behaviour vectors / sqrt(6), 0-1. A vector is sustained velocity [0, 0.8] m/s, efficiency [0, 0.7], mean tilt [0, 60 deg], slip fraction [0, 1], airborne fraction [0, 1], yaw drift [0, pi], scaled by these fixed ranges (chosen from the metric definitions before any run) |

Invalid runs score `INVALID_FITNESS` (-20), exactly as evolution saw them.
Every number is in `results.csv` (one row per run; column meanings in
`evomorph/experiments/table.py`) and traceable to `runs/<run>/posthoc.json`.

### Statistics

`evomorph/analysis/stats.py`. For every condition and measure: the per-seed
values, the median and a percentile-bootstrap 95% interval of the median
(10000 resamples, seeded; needs 3 or more seeds, and with few seeds the
interval is little more than the range). For C vs A, C vs B and D vs C: a
two-sided Mann-Whitney U test and the Vargha-Delaney A12 effect size, the
probability that a run of the first condition is better than one of the
second (0.5 = no difference; 0.56/0.64/0.71 = small/medium/large).

* **Primary measures** (the two halves of the hypothesis, fixed before any
  result): champion fitness on held-out terrain (robustness) and behavioural
  diversity of the best-K set on held-out terrain. Their six tests are
  Holm-corrected. Everything else is secondary and labelled so.
* **Unpaired** tests: seed k of two conditions shares a number, not a random
  state that makes the runs comparable pair by pair.
* **The verdict line is computed**: the hypothesis is "supported" only if C
  beats both A and B on both primary measures with Holm-adjusted p < 0.05;
  otherwise "not shown".
* **Few seeds.** With n seeds per condition the smallest two-sided p the exact
  test can produce is 2 / C(2n, n): 0.33 for n = 2, 0.10 for n = 3, 0.029 for
  n = 4, 0.0079 for n = 5. Holm multiplies the smallest primary p by 6, so
  with 4 or fewer seeds no primary comparison can be significant, and
  `summary.md` says the evidence is weak.

## B's baseline on every limb count

`.venv/bin/python scripts/baseline_check.py` (flat ground, friction 1, spawn
seeds 0-2; the reference quadruped's body with only the limb count and segment
count changed, plus the reference hexapod's body; all driven by B's baseline):

| body | fitness | speed m/s | fell |
|---|---|---|---|
| quadruped body as 2x1 | -0.200 | -0.001 | 0/3 |
| quadruped body as 2x2 | -1.183 | -0.001 | 3/3 |
| quadruped body as 4x1 | -0.261 | -0.005 | 0/3 |
| quadruped body as 4x2 (the reference) | +0.450 | +0.447 | 0/3 |
| quadruped body as 6x1 | -0.203 | +0.000 | 0/3 |
| quadruped body as 6x2 | +0.049 | +0.251 | 0/3 |
| quadruped body as 8x1 | -0.182 | +0.001 | 0/3 |
| quadruped body as 8x2 | +0.006 | +0.219 | 0/3 |
| reference hexapod body (6x2) | +0.322 | +0.416 | 0/3 |

So the baseline walks every two-segment body with 4 or more legs at 0.22-0.45
m/s, but on these bodies it moves **no one-segment body at all** and the
two-segment biped falls. A one-segment leg needs its hip axis tilted to lift
the foot while it swings (the core's `hip_axis_tilt` gene, 0 on the reference
quadruped). B can evolve that gene, and the quick run's B champion of seed 0
is a 4x1 that does move (0.443 m/s on training terrain, 0.373 held-out), but
B starts with one-segment bodies at a disadvantage that C, which can also tune
the gait, does not have. This is part
of what freezing the controller costs, and it is why B's body-plan diversity
should be read together with its fitness.

## Smoke test: the --quick result (2 seeds; it cannot answer the question)

Command and size: `.venv/bin/python -m evomorph experiment --type comparison
--quick` (configs/comparison.yaml `quick`: 2 seeds, 660 simulations per run =
60 genomes = 5 generations of 12, champion re-scored with 1 fresh seed per
environment, best-4 set), then `.venv/bin/python -m evomorph analyze --run
results/experiments/comparison_quick`. Made with commit c632d26 (experiment)
and analysed with this branch's analysis code. A copy of the table, the
statistics, the summary, the manifest and the plots is in
`docs/media/experiment/quick/`.

- **Wall time: 1579 s (26.3 min)** for the 8 runs (`/usr/bin/time`: real
  1578.97), on this laptop while other work kept the load average at 43-81. I
  also ran the test suite for about 3 minutes in the middle of it.
- Every run spent exactly 660 simulations (checked by the experiment).
- Re-running the same command skipped all 8 runs: `experiment done: 0 runs
  made, 8 already finished` (0.54 s).
- `analyze` wrote the 7 plots and `summary.md`; I opened every plot and each
  shows what its title says.

Medians over the 2 seeds (no intervals: they need 3 or more seeds):

| measure | A controller | B morphology | C joint GA | D joint ME |
|---|---|---|---|---|
| champion fitness, held-out | 0.111 | 0.206 | 0.100 | 0.109 |
| champion fitness, training (fresh seeds) | 0.196 | 0.307 | 0.127 | 0.156 |
| champion gap (held-out - training) | -0.086 | -0.102 | -0.027 | -0.047 |
| champion speed, held-out (m/s) | 0.170 | 0.591 | 0.071 | 0.215 |
| champion failure rate, held-out | 0 | 0 | 0 | 0 |
| morphological diversity, final population | 0 | 0.291 | 0.265 | 0.198 |
| behavioural diversity, best-4 set, held-out | 0.046 | 0.103 | 0.148 | 0.096 |
| body plans, final population | 1 | 5.5 | 6 | 2.5 |

Champions per seed (selected training fitness / fresh training / held-out):
A 0.219/0.215/0.160 and 0.146/0.177/0.061; B 0.123/0.112/0.082 and
0.547/0.503/0.329; C 0.025/0.025/0.001 and 0.230/0.230/0.199; D
0.022/0.019/0.029 and 0.296/0.294/0.189.

What this does and does not say:

- **Nothing about the hypothesis.** With 2 seeds the smallest possible p is
  0.33 (1 after Holm), and the computed verdict is "NOT SHOWN". The largest
  primary effect, C over A in behavioural diversity, is A12 = 1.0 (p = 0.33);
  on held-out champion fitness C beats B in only 25% of seed pairings. These
  are hints at most.
- **The search is far too short.** After 60 genomes most champions are worse
  than the hand-designed reference quadruped (training 0.357, held-out 0.181,
  seeds 0-2, docs/TERRAINS.md). Only three of eight beat its held-out
  number: B seed 1 (0.329), C seed 1 (0.199) and D seed 1 (0.189).
- **Every champion lost fitness on held-out terrain except D seed 0**
  (+0.010), and the winner's curse at this size is small: fresh training
  re-scores are within 0.05 of the selected scores.
- **The fastest creature skates.** B seed 1's champion (a long-legged 4x2
  trotter, `docs/media/experiment/quick/b_seed1_champion_sheet.png`) covers
  11.1 m in 10 s on flat ground and 1.04 m/s averaged over the training suite
  (more than twice the reference's 0.42), on its feet (foot support 0.90,
  never fell), but 86% of its foot load is delivered while sliding and it is
  12% airborne (training-suite means, `scripts/render_run.py` on the run).
  The multiplicative fitness charges each of these, which is why its fitness
  is 0.50-0.55 rather than the ~4.8 its progress term alone would give. This is the core's known skating weakness, now seen on terrain; I
  did not change any score.
- **MAP-Elites kept few cells**: 6 and 16 of 100 after 60 genomes (the heat
  map), because only never-fallen genomes enter the archive.

## Budget for the full run

`.venv/bin/python scripts/plan_budget.py results/experiments/comparison_quick`
reads the quick run's measured throughput and prints the largest per-run
budget that fits (model in the script's docstring; post-hoc cost counted as an
upper bound of 286 simulations per run):

```
measured search rate 3.89 sims/s (per run 2.50-6.88), post-hoc rate 3.99 sims/s, over 8 runs
3 seeds x 4 conditions in 2.5 h: budget 2618 simulations per run = 238 genomes
4 seeds x 4 conditions in 2.5 h: budget 1881 simulations per run = 171 genomes
5 seeds x 4 conditions in 2.5 h: budget 1441 simulations per run = 131 genomes
6 seeds x 4 conditions in 2.5 h: budget 1155 simulations per run = 105 genomes
8 seeds x 4 conditions in 2.5 h: budget 792 simulations per run = 72 genomes
```

**The committed full design is 5 seeds x 1441 simulations (131 genomes,
population 24, about 5.5 generations).** Five seeds because it is the fewest
with which a primary comparison can pass the Holm correction at all (5 vs 5:
smallest p 0.0079, x 6 = 0.048; 4 vs 4: 0.029, x 6 = 0.17), and then only with
complete separation between conditions. The population was cut from the
engine's 48 to 24 so the small budget still gives about five generations.

Read this honestly before running it:

- **Throughput depends on what else the laptop is doing.** The per-run rate
  varied 2.5-6.9 simulations/s with the load. At the best measured rate
  (`--rate 6.65`) 2.5 h fits 5 seeds x 2684 simulations; on a quiet machine it
  would be more. Override with `--budget`; resuming is safe only with the same
  budget.
- **131 genomes is very little evolution.** The quick run's 60 genomes mostly
  did not beat the hand-designed quadruped, and the engine's own flat-ground
  GA found its best walker (+1.02) only near the end of a 6000-simulation run
  (docs/EVOLUTION.md). A 2.5 h run will
  most likely compare under-searched conditions; a null result from it would
  say more about the budget than about the hypothesis. With an overnight
  window, `--hours 8` gives 5 seeds x 5302 simulations (482 genomes), which is
  the run I would trust more.
- The ordering of runs (seed by seed) means an interrupted full run still
  yields complete blocks of seeds, and resuming picks up where it stopped.

## Limitations

- **Measures inherited from the core.** Tilt is measured from world vertical,
  so slopes add a constant tilt; energy efficiency ignores potential energy;
  torso contact with an obstacle counts as a fall (docs/TERRAINS.md). They are
  the same for every condition, so they do not bias the comparison, but they
  shape absolute numbers.
- **Skating** is charged by the slip factor, not prevented; fast skaters can
  still win (above). Changing that is a core fitness decision, not an
  experiment setting.
- **Behaviour space**: the six behaviour descriptors and their ranges are a
  choice. They were fixed before any run, but another choice could rank the
  conditions differently.
- **A's morphological diversity is 0 by construction** and B's controller
  diversity is 0 by construction; only behavioural diversity compares all four.
- **Champion vs set**: the champion is the all-time best of a run; the set is
  the final population or archive. For the GA the champion is usually an elite
  of the final population, for MAP-Elites it is always in the archive.
- Post-hoc evaluation uses 3 fresh seeds per environment for champions and 1
  for set members; set measures are therefore noisier per genome.
