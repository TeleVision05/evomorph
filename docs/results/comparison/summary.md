# Comparison experiment: comparison

- conditions: A controller-only (ga, mode controller); B morphology-only (ga, mode morphology); C joint (GA) (ga, mode joint); D joint (MAP-Elites) (map_elites, mode joint)
- budget 1441 simulations per run (11 per genome: every environment of the 'train' suite once); seeds [0, 1, 2, 3, 4]
- held-out suite 'test'; champion re-scored with 3 fresh spawn seeds per environment; best-10 final set with 1 fresh seed(s)
- design hash 9ae4752f2ef32431; git 4b61992, MuJoCo 4.0.1, Python 3.12.13
- command: `python -m evomorph experiment --type comparison --config configs/comparison.yaml --workers 4`

## How strong is the evidence?

5 seeds per condition (smallest group). With 5 vs 5 seeds the smallest p-value a two-sided Mann-Whitney test can give is 0.00794, and after the Holm correction over the 6 primary tests 0.0476. Only near-complete separation between conditions can be significant with this many seeds; a non-significant comparison is not evidence that two conditions are equal.

## Verdict on the primary measures

- C vs A, champion fitness, held-out terrain: a run of C beats a run of A in 48% of seed pairings, ties half (A12 0.48, negligible, leans A); Holm p = 1, not significant at 0.05.
- C vs B, champion fitness, held-out terrain: a run of C beats a run of B in 12% of seed pairings, ties half (A12 0.12, large, leans B); Holm p = 0.33, not significant at 0.05.
- D vs C, champion fitness, held-out terrain: a run of D beats a run of C in 64% of seed pairings, ties half (A12 0.64, medium, leans D); Holm p = 1, not significant at 0.05.
- C vs A, behavioural diversity, held-out (0-1): a run of C beats a run of A in 36% of seed pairings, ties half (A12 0.36, medium, leans A); Holm p = 1, not significant at 0.05.
- C vs B, behavioural diversity, held-out (0-1): a run of C beats a run of B in 64% of seed pairings, ties half (A12 0.64, medium, leans C); Holm p = 1, not significant at 0.05.
- D vs C, behavioural diversity, held-out (0-1): a run of D beats a run of C in 68% of seed pairings, ties half (A12 0.68, medium, leans D); Holm p = 1, not significant at 0.05.

**Hypothesis (C more robust and more behaviourally diverse than A and B): NOT SHOWN by this data.** It counts as supported only if C beats both A and B on both primary measures with Holm-adjusted p < 0.05.

## Per condition: median [bootstrap 95% CI] over seeds

Fitness is dimensionless (about 1 = clean walking at 0.2 m/s). Tilt in radians. Intervals need at least 3 seeds.

| measure | A controller-only (n=5) | B morphology-only (n=5) | C joint (GA) (n=5) | D joint (MAP-Elites) (n=5) |
|---|---|---|---|---|
| champion fitness, held-out terrain | 0.0778 [0.0336, 0.158] | 0.327 [0.223, 0.505] | 0.0578 [-2.36e-05, 0.356] | 0.12 [0.041, 0.284] |
| champion fitness, training terrain (fresh seeds) | 0.196 [0.116, 0.324] | 0.64 [0.447, 0.892] | 0.103 [0.0244, 0.507] | 0.172 [0.0708, 0.353] |
| champion generalisation gap (test - train) | -0.0686 [-0.233, -0.0451] | -0.32 [-0.417, -0.15] | -0.0523 [-0.152, -0.011] | -0.0521 [-0.108, -0.0298] |
| best-K set mean fitness, held-out | 0.0367 [0.0219, 0.0679] | 0.253 [0.149, 0.339] | 0.00805 [-0.0211, 0.0358] | -0.0236 [-0.0463, 0.0988] |
| best-K set generalisation gap (test - train) | -0.0489 [-0.0639, -0.0306] | -0.162 [-0.299, -0.103] | -0.029 [-0.0589, -0.0204] | -0.0473 [-0.0991, -0.031] |
| champion forward speed, held-out (m/s) | 0.157 [0.122, 0.286] | 0.653 [0.453, 0.963] | 0.288 [0.0643, 0.386] | 0.203 [0.143, 0.391] |
| champion energy efficiency, held-out (0-1) | 0.431 [0.215, 0.48] | 0.443 [0.415, 0.519] | 0.306 [0.193, 0.494] | 0.326 [0.251, 0.496] |
| champion mean tilt, held-out (rad) | 0.113 [0.091, 0.162] | 0.108 [0.0791, 0.151] | 0.0928 [0.0542, 0.177] | 0.102 [0.0728, 0.122] |
| champion failure rate, held-out | 0 [0, 0.0606] | 0 [0, 0.0909] | 0 [0, 0.0303] | 0 [0, 0] |
| morphological diversity, final population (0-1) | 0 [0, 0] | 0.281 [0.271, 0.297] | 0.308 [0.3, 0.312] | 0.221 [0.194, 0.232] |
| behavioural diversity, held-out (0-1) | 0.17 [0.129, 0.175] | 0.111 [0.0821, 0.184] | 0.157 [0.101, 0.186] | 0.158 [0.125, 0.258] |
| body plans in the final population | 1 [1, 1] | 8 [8, 8] | 8 [8, 8] | 4 [3, 4] |

## Champions, per seed

'selected' is the training fitness the champion was picked with; 'train (fresh)' re-scores it on the same terrains with new seeds; the gap is held-out minus fresh train.

| condition | seed | champion | plan | selected | train (fresh) | held-out | gap |
|---|---|---|---|---|---|---|---|
| A | 0 | g0004-018 | 4x2 | +0.329 | +0.324 | +0.091 | -0.233 |
| A | 1 | g0004-001 | 4x2 | +0.113 | +0.116 | +0.048 | -0.069 |
| A | 2 | g0003-010 | 4x2 | +0.123 | +0.123 | +0.078 | -0.045 |
| A | 3 | g0005-013 | 4x2 | +0.208 | +0.207 | +0.158 | -0.049 |
| A | 4 | g0004-005 | 4x2 | +0.205 | +0.196 | +0.034 | -0.163 |
| B | 0 | g0007-006 | 4x2 | +0.409 | +0.447 | +0.298 | -0.150 |
| B | 1 | g0007-003 | 6x2 | +0.931 | +0.892 | +0.505 | -0.387 |
| B | 2 | g0006-012 | 4x2 | +0.889 | +0.640 | +0.223 | -0.417 |
| B | 3 | g0005-012 | 4x2 | +0.834 | +0.761 | +0.442 | -0.320 |
| B | 4 | g0004-003 | 4x2 | +0.530 | +0.572 | +0.327 | -0.245 |
| C | 0 | g0007-000 | 8x2 | +0.069 | +0.069 | +0.058 | -0.011 |
| C | 1 | g0005-005 | 6x2 | +0.471 | +0.507 | +0.356 | -0.152 |
| C | 2 | g0001-012 | 4x2 | +0.026 | +0.024 | -0.000 | -0.024 |
| C | 3 | g0003-014 | 8x2 | +0.138 | +0.139 | +0.086 | -0.052 |
| C | 4 | g0005-013 | 4x1 | +0.098 | +0.103 | +0.049 | -0.054 |
| D | 0 | b0004-016 | 4x1 | +0.069 | +0.071 | +0.041 | -0.030 |
| D | 1 | b0005-001 | 4x2 | +0.278 | +0.279 | +0.171 | -0.108 |
| D | 2 | b0004-011 | 6x2 | +0.353 | +0.353 | +0.284 | -0.069 |
| D | 3 | b0003-005 | 6x2 | +0.184 | +0.172 | +0.120 | -0.052 |
| D | 4 | b0004-021 | 4x1 | +0.176 | +0.118 | +0.086 | -0.033 |

## Key comparisons (Mann-Whitney U, two-sided; A12 = P(first better))

| measure | comparison | n | medians | A12 (effect) | p | p (Holm) |
|---|---|---|---|---|---|---|
| **primary** champion fitness, held-out terrain | C vs A | 5 vs 5 | 0.0578 vs 0.0778 | 0.48 (negligible) | 1 | 1 |
| **primary** champion fitness, held-out terrain | C vs B | 5 vs 5 | 0.0578 vs 0.327 | 0.12 (large) | 0.056 | 0.33 |
| **primary** champion fitness, held-out terrain | D vs C | 5 vs 5 | 0.12 vs 0.0578 | 0.64 (medium) | 0.55 | 1 |
| **primary** behavioural diversity, held-out (0-1) | C vs A | 5 vs 5 | 0.157 vs 0.17 | 0.36 (medium) | 0.55 | 1 |
| **primary** behavioural diversity, held-out (0-1) | C vs B | 5 vs 5 | 0.157 vs 0.111 | 0.64 (medium) | 0.55 | 1 |
| **primary** behavioural diversity, held-out (0-1) | D vs C | 5 vs 5 | 0.158 vs 0.157 | 0.68 (medium) | 0.42 | 1 |
| champion fitness, training terrain (fresh seeds) | C vs A | 5 vs 5 | 0.103 vs 0.196 | 0.28 (large) | 0.31 | secondary |
| champion fitness, training terrain (fresh seeds) | C vs B | 5 vs 5 | 0.103 vs 0.64 | 0.04 (large) | 0.016 | secondary |
| champion fitness, training terrain (fresh seeds) | D vs C | 5 vs 5 | 0.172 vs 0.103 | 0.68 (medium) | 0.42 | secondary |
| champion generalisation gap (test - train) | C vs A | 5 vs 5 | -0.0523 vs -0.0686 | 0.72 (large) | 0.31 | secondary |
| champion generalisation gap (test - train) | C vs B | 5 vs 5 | -0.0523 vs -0.32 | 0.96 (large) | 0.016 | secondary |
| champion generalisation gap (test - train) | D vs C | 5 vs 5 | -0.0521 vs -0.0523 | 0.44 (small) | 0.84 | secondary |
| best-K set mean fitness, held-out | C vs A | 5 vs 5 | 0.00805 vs 0.0367 | 0.08 (large) | 0.032 | secondary |
| best-K set mean fitness, held-out | C vs B | 5 vs 5 | 0.00805 vs 0.253 | 0 (large) | 0.0079 | secondary |
| best-K set mean fitness, held-out | D vs C | 5 vs 5 | -0.0236 vs 0.00805 | 0.4 (small) | 0.69 | secondary |
| best-K set mean fitness, training | C vs A | 5 vs 5 | 0.0294 vs 0.0819 | 0.08 (large) | 0.032 | secondary |
| best-K set mean fitness, training | C vs B | 5 vs 5 | 0.0294 vs 0.415 | 0 (large) | 0.0079 | secondary |
| best-K set mean fitness, training | D vs C | 5 vs 5 | 0.0755 vs 0.0294 | 0.64 (medium) | 0.55 | secondary |
| best-K set generalisation gap (test - train) | C vs A | 5 vs 5 | -0.029 vs -0.0489 | 0.84 (large) | 0.095 | secondary |
| best-K set generalisation gap (test - train) | C vs B | 5 vs 5 | -0.029 vs -0.162 | 1 (large) | 0.0079 | secondary |
| best-K set generalisation gap (test - train) | D vs C | 5 vs 5 | -0.0473 vs -0.029 | 0.2 (large) | 0.15 | secondary |
| champion forward speed, held-out (m/s) | C vs A | 5 vs 5 | 0.288 vs 0.157 | 0.72 (large) | 0.31 | secondary |
| champion forward speed, held-out (m/s) | C vs B | 5 vs 5 | 0.288 vs 0.653 | 0 (large) | 0.0079 | secondary |
| champion forward speed, held-out (m/s) | D vs C | 5 vs 5 | 0.203 vs 0.288 | 0.48 (negligible) | 1 | secondary |
| champion energy efficiency, held-out (0-1) | C vs A | 5 vs 5 | 0.306 vs 0.431 | 0.32 (medium) | 0.42 | secondary |
| champion energy efficiency, held-out (0-1) | C vs B | 5 vs 5 | 0.306 vs 0.443 | 0.16 (large) | 0.095 | secondary |
| champion energy efficiency, held-out (0-1) | D vs C | 5 vs 5 | 0.326 vs 0.306 | 0.68 (medium) | 0.42 | secondary |
| champion mean tilt, held-out (rad) | C vs A | 5 vs 5 | 0.0928 vs 0.113 | 0.68 (medium) | 0.42 | secondary |
| champion mean tilt, held-out (rad) | C vs B | 5 vs 5 | 0.0928 vs 0.108 | 0.64 (medium) | 0.55 | secondary |
| champion mean tilt, held-out (rad) | D vs C | 5 vs 5 | 0.102 vs 0.0928 | 0.4 (small) | 0.69 | secondary |
| champion failure rate, held-out | C vs A | 5 vs 5 | 0 vs 0 | 0.52 (negligible) | 1 | secondary |
| champion failure rate, held-out | C vs B | 5 vs 5 | 0 vs 0 | 0.52 (negligible) | 1 | secondary |
| champion failure rate, held-out | D vs C | 5 vs 5 | 0 vs 0 | 0.6 (small) | 0.42 | secondary |
| best-K set failure rate, held-out | C vs A | 5 vs 5 | 0 vs 0 | 0.56 (small) | 0.81 | secondary |
| best-K set failure rate, held-out | C vs B | 5 vs 5 | 0 vs 0 | 0.56 (small) | 0.81 | secondary |
| best-K set failure rate, held-out | D vs C | 5 vs 5 | 0 vs 0 | 0.58 (small) | 0.7 | secondary |
| morphological diversity, best-K set (0-1) | C vs A | 5 vs 5 | 0.284 vs 0 | 1 (large) | 0.0075 | secondary |
| morphological diversity, best-K set (0-1) | C vs B | 5 vs 5 | 0.284 vs 0.191 | 1 (large) | 0.0079 | secondary |
| morphological diversity, best-K set (0-1) | D vs C | 5 vs 5 | 0.206 vs 0.284 | 0 (large) | 0.0079 | secondary |
| morphological diversity, final population (0-1) | C vs A | 5 vs 5 | 0.308 vs 0 | 1 (large) | 0.0075 | secondary |
| morphological diversity, final population (0-1) | C vs B | 5 vs 5 | 0.308 vs 0.281 | 1 (large) | 0.0079 | secondary |
| morphological diversity, final population (0-1) | D vs C | 5 vs 5 | 0.221 vs 0.308 | 0 (large) | 0.0079 | secondary |
| behavioural diversity, final population (0-1) | C vs A | 5 vs 5 | 0.237 vs 0.237 | 0.44 (small) | 0.84 | secondary |
| behavioural diversity, final population (0-1) | C vs B | 5 vs 5 | 0.237 vs 0.253 | 0.4 (small) | 0.69 | secondary |
| behavioural diversity, final population (0-1) | D vs C | 5 vs 5 | 0.195 vs 0.237 | 0.4 (small) | 0.69 | secondary |
| body plans in the final population | C vs A | 5 vs 5 | 8 vs 1 | 1 (large) | 0.004 | secondary |
| body plans in the final population | C vs B | 5 vs 5 | 8 vs 8 | 0.5 (negligible) | 1 | secondary |
| body plans in the final population | D vs C | 5 vs 5 | 4 vs 8 | 0 (large) | 0.0065 | secondary |

## Throughput

- search: 28820 simulations in 5041 s = 5.72 simulations/s (20 timed runs; wall time includes the champion's re-score and held-out check)
- post-hoc evaluation: 5192 simulations in 820 s = 6.33 simulations/s
- total wall time of the timed runs: 97.7 min

## Plots

- [fitness_vs_simulations.png](plots/fitness_vs_simulations.png)
- [train_vs_test.png](plots/train_vs_test.png)
- [locomotion.png](plots/locomotion.png)
- [diversity.png](plots/diversity.png)
- [body_plans.png](plots/body_plans.png)
- [archive_heatmap.png](plots/archive_heatmap.png)
- [gallery.png](plots/gallery.png)

Per-run numbers: results.csv; comparisons: stats.csv; every run's records: runs/.
