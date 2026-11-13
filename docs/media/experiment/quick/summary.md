# Comparison experiment: comparison_quick

- conditions: A controller-only (ga, mode controller); B morphology-only (ga, mode morphology); C joint (GA) (ga, mode joint); D joint (MAP-Elites) (map_elites, mode joint)
- budget 660 simulations per run (11 per genome: every environment of the 'train' suite once); seeds [0, 1]
- held-out suite 'test'; champion re-scored with 1 fresh spawn seeds per environment; best-4 final set with 1 fresh seed(s)
- design hash 3aaadba1a4abfcb8; git c632d26, MuJoCo 4.0.1, Python 3.12.13
- command: `python -m evomorph experiment --type comparison --config configs/comparison.yaml --quick`

## How strong is the evidence?

2 seeds per condition (smallest group). With 2 vs 2 seeds the smallest p-value a two-sided Mann-Whitney test can give is 0.333, and after the Holm correction over the 6 primary tests 1, so no primary comparison can reach p < 0.05: the evidence is WEAK whatever the data look like. Treat every difference below as a hint for a larger run, not a result.

## Verdict on the primary measures

- C vs A, champion fitness, held-out terrain: a run of C beats a run of A in 50% of seed pairings, ties half (A12 0.5, negligible, no lean either way); Holm p = 1, not significant at 0.05.
- C vs B, champion fitness, held-out terrain: a run of C beats a run of B in 25% of seed pairings, ties half (A12 0.25, large, leans B); Holm p = 1, not significant at 0.05.
- D vs C, champion fitness, held-out terrain: a run of D beats a run of C in 50% of seed pairings, ties half (A12 0.5, negligible, no lean either way); Holm p = 1, not significant at 0.05.
- C vs A, behavioural diversity, held-out (0-1): a run of C beats a run of A in 100% of seed pairings, ties half (A12 1, large, leans C); Holm p = 1, not significant at 0.05.
- C vs B, behavioural diversity, held-out (0-1): a run of C beats a run of B in 50% of seed pairings, ties half (A12 0.5, negligible, no lean either way); Holm p = 1, not significant at 0.05.
- D vs C, behavioural diversity, held-out (0-1): a run of D beats a run of C in 25% of seed pairings, ties half (A12 0.25, large, leans C); Holm p = 1, not significant at 0.05.

**Hypothesis (C more robust and more behaviourally diverse than A and B): NOT SHOWN by this data.** It counts as supported only if C beats both A and B on both primary measures with Holm-adjusted p < 0.05.

## Per condition: median [bootstrap 95% CI] over seeds

Fitness is dimensionless (about 1 = clean walking at 0.2 m/s). Tilt in radians. Intervals need at least 3 seeds.

| measure | A controller-only (n=2) | B morphology-only (n=2) | C joint (GA) (n=2) | D joint (MAP-Elites) (n=2) |
|---|---|---|---|---|
| champion fitness, held-out terrain | 0.111 | 0.206 | 0.1 | 0.109 |
| champion fitness, training terrain (fresh seeds) | 0.196 | 0.307 | 0.127 | 0.156 |
| champion generalisation gap (test - train) | -0.0856 | -0.102 | -0.0274 | -0.0473 |
| best-K set mean fitness, held-out | 0.104 | 0.184 | 0.0655 | 0.0799 |
| best-K set generalisation gap (test - train) | -0.0546 | -0.126 | -0.0379 | -0.0325 |
| champion forward speed, held-out (m/s) | 0.17 | 0.591 | 0.0705 | 0.215 |
| champion energy efficiency, held-out (0-1) | 0.417 | 0.395 | 0.32 | 0.303 |
| champion mean tilt, held-out (rad) | 0.107 | 0.179 | 0.0395 | 0.118 |
| champion failure rate, held-out | 0 | 0 | 0 | 0 |
| morphological diversity, final population (0-1) | 0 | 0.291 | 0.265 | 0.198 |
| behavioural diversity, held-out (0-1) | 0.0463 | 0.103 | 0.148 | 0.0962 |
| body plans in the final population | 1 | 5.5 | 6 | 2.5 |

## Champions, per seed

'selected' is the training fitness the champion was picked with; 'train (fresh)' re-scores it on the same terrains with new seeds; the gap is held-out minus fresh train.

| condition | seed | champion | plan | selected | train (fresh) | held-out | gap |
|---|---|---|---|---|---|---|---|
| A | 0 | g0005-005 | 4x2 | +0.219 | +0.215 | +0.160 | -0.055 |
| A | 1 | g0003-003 | 4x2 | +0.146 | +0.177 | +0.061 | -0.116 |
| B | 0 | g0007-000 | 4x1 | +0.123 | +0.112 | +0.082 | -0.030 |
| B | 1 | g0008-003 | 4x2 | +0.547 | +0.503 | +0.329 | -0.173 |
| C | 0 | g0006-004 | 4x1 | +0.025 | +0.025 | +0.001 | -0.024 |
| C | 1 | g0008-000 | 8x2 | +0.230 | +0.230 | +0.199 | -0.031 |
| D | 0 | b0004-011 | 4x1 | +0.022 | +0.019 | +0.029 | +0.010 |
| D | 1 | b0004-010 | 4x2 | +0.296 | +0.294 | +0.189 | -0.104 |

## Key comparisons (Mann-Whitney U, two-sided; A12 = P(first better))

| measure | comparison | n | medians | A12 (effect) | p | p (Holm) |
|---|---|---|---|---|---|---|
| **primary** champion fitness, held-out terrain | C vs A | 2 vs 2 | 0.1 vs 0.111 | 0.5 (negligible) | 1 | 1 |
| **primary** champion fitness, held-out terrain | C vs B | 2 vs 2 | 0.1 vs 0.206 | 0.25 (large) | 0.67 | 1 |
| **primary** champion fitness, held-out terrain | D vs C | 2 vs 2 | 0.109 vs 0.1 | 0.5 (negligible) | 1 | 1 |
| **primary** behavioural diversity, held-out (0-1) | C vs A | 2 vs 2 | 0.148 vs 0.0463 | 1 (large) | 0.33 | 1 |
| **primary** behavioural diversity, held-out (0-1) | C vs B | 2 vs 2 | 0.148 vs 0.103 | 0.5 (negligible) | 1 | 1 |
| **primary** behavioural diversity, held-out (0-1) | D vs C | 2 vs 2 | 0.0962 vs 0.148 | 0.25 (large) | 0.67 | 1 |
| champion fitness, training terrain (fresh seeds) | C vs A | 2 vs 2 | 0.127 vs 0.196 | 0.5 (negligible) | 1 | secondary |
| champion fitness, training terrain (fresh seeds) | C vs B | 2 vs 2 | 0.127 vs 0.307 | 0.25 (large) | 0.67 | secondary |
| champion fitness, training terrain (fresh seeds) | D vs C | 2 vs 2 | 0.156 vs 0.127 | 0.5 (negligible) | 1 | secondary |
| champion generalisation gap (test - train) | C vs A | 2 vs 2 | -0.0274 vs -0.0856 | 1 (large) | 0.33 | secondary |
| champion generalisation gap (test - train) | C vs B | 2 vs 2 | -0.0274 vs -0.102 | 0.75 (large) | 0.67 | secondary |
| champion generalisation gap (test - train) | D vs C | 2 vs 2 | -0.0473 vs -0.0274 | 0.5 (negligible) | 1 | secondary |
| best-K set mean fitness, held-out | C vs A | 2 vs 2 | 0.0655 vs 0.104 | 0.25 (large) | 0.67 | secondary |
| best-K set mean fitness, held-out | C vs B | 2 vs 2 | 0.0655 vs 0.184 | 0.25 (large) | 0.67 | secondary |
| best-K set mean fitness, held-out | D vs C | 2 vs 2 | 0.0799 vs 0.0655 | 0.75 (large) | 0.67 | secondary |
| best-K set mean fitness, training | C vs A | 2 vs 2 | 0.103 vs 0.158 | 0.25 (large) | 0.67 | secondary |
| best-K set mean fitness, training | C vs B | 2 vs 2 | 0.103 vs 0.31 | 0.25 (large) | 0.67 | secondary |
| best-K set mean fitness, training | D vs C | 2 vs 2 | 0.112 vs 0.103 | 0.75 (large) | 0.67 | secondary |
| best-K set generalisation gap (test - train) | C vs A | 2 vs 2 | -0.0379 vs -0.0546 | 0.75 (large) | 0.67 | secondary |
| best-K set generalisation gap (test - train) | C vs B | 2 vs 2 | -0.0379 vs -0.126 | 0.75 (large) | 0.67 | secondary |
| best-K set generalisation gap (test - train) | D vs C | 2 vs 2 | -0.0325 vs -0.0379 | 0.5 (negligible) | 1 | secondary |
| champion forward speed, held-out (m/s) | C vs A | 2 vs 2 | 0.0705 vs 0.17 | 0 (large) | 0.33 | secondary |
| champion forward speed, held-out (m/s) | C vs B | 2 vs 2 | 0.0705 vs 0.591 | 0 (large) | 0.33 | secondary |
| champion forward speed, held-out (m/s) | D vs C | 2 vs 2 | 0.215 vs 0.0705 | 1 (large) | 0.33 | secondary |
| champion energy efficiency, held-out (0-1) | C vs A | 2 vs 2 | 0.32 vs 0.417 | 0.25 (large) | 0.67 | secondary |
| champion energy efficiency, held-out (0-1) | C vs B | 2 vs 2 | 0.32 vs 0.395 | 0.5 (negligible) | 1 | secondary |
| champion energy efficiency, held-out (0-1) | D vs C | 2 vs 2 | 0.303 vs 0.32 | 0.5 (negligible) | 1 | secondary |
| champion mean tilt, held-out (rad) | C vs A | 2 vs 2 | 0.0395 vs 0.107 | 1 (large) | 0.33 | secondary |
| champion mean tilt, held-out (rad) | C vs B | 2 vs 2 | 0.0395 vs 0.179 | 1 (large) | 0.33 | secondary |
| champion mean tilt, held-out (rad) | D vs C | 2 vs 2 | 0.118 vs 0.0395 | 0 (large) | 0.33 | secondary |
| champion failure rate, held-out | C vs A | 2 vs 2 | 0 vs 0 | 0.5 (negligible) | 1 | secondary |
| champion failure rate, held-out | C vs B | 2 vs 2 | 0 vs 0 | 0.5 (negligible) | 1 | secondary |
| champion failure rate, held-out | D vs C | 2 vs 2 | 0 vs 0 | 0.5 (negligible) | 1 | secondary |
| best-K set failure rate, held-out | C vs A | 2 vs 2 | 0 vs 0 | 0.5 (negligible) | 1 | secondary |
| best-K set failure rate, held-out | C vs B | 2 vs 2 | 0 vs 0 | 0.5 (negligible) | 1 | secondary |
| best-K set failure rate, held-out | D vs C | 2 vs 2 | 0 vs 0 | 0.5 (negligible) | 1 | secondary |
| morphological diversity, best-K set (0-1) | C vs A | 2 vs 2 | 0.215 vs 0 | 1 (large) | 0.22 | secondary |
| morphological diversity, best-K set (0-1) | C vs B | 2 vs 2 | 0.215 vs 0.146 | 0.75 (large) | 0.67 | secondary |
| morphological diversity, best-K set (0-1) | D vs C | 2 vs 2 | 0.213 vs 0.215 | 0.5 (negligible) | 1 | secondary |
| morphological diversity, final population (0-1) | C vs A | 2 vs 2 | 0.265 vs 0 | 1 (large) | 0.22 | secondary |
| morphological diversity, final population (0-1) | C vs B | 2 vs 2 | 0.265 vs 0.291 | 0.5 (negligible) | 1 | secondary |
| morphological diversity, final population (0-1) | D vs C | 2 vs 2 | 0.198 vs 0.265 | 0 (large) | 0.33 | secondary |
| behavioural diversity, final population (0-1) | C vs A | 2 vs 2 | 0.205 vs 0.199 | 0.5 (negligible) | 1 | secondary |
| behavioural diversity, final population (0-1) | C vs B | 2 vs 2 | 0.205 vs 0.259 | 0.25 (large) | 0.67 | secondary |
| behavioural diversity, final population (0-1) | D vs C | 2 vs 2 | 0.171 vs 0.205 | 0.5 (negligible) | 1 | secondary |
| body plans in the final population | C vs A | 2 vs 2 | 6 vs 1 | 1 (large) | 0.19 | secondary |
| body plans in the final population | C vs B | 2 vs 2 | 6 vs 5.5 | 0.75 (large) | 0.62 | secondary |
| body plans in the final population | D vs C | 2 vs 2 | 2.5 vs 6 | 0 (large) | 0.22 | secondary |

## Throughput

- search: 5280 simulations in 1401 s = 3.77 simulations/s (8 timed runs; wall time includes the champion's re-score and held-out check)
- post-hoc evaluation: 704 simulations in 176 s = 3.99 simulations/s
- total wall time of the timed runs: 26.3 min

## Plots

- [fitness_vs_simulations.png](plots/fitness_vs_simulations.png)
- [train_vs_test.png](plots/train_vs_test.png)
- [locomotion.png](plots/locomotion.png)
- [diversity.png](plots/diversity.png)
- [body_plans.png](plots/body_plans.png)
- [archive_heatmap.png](plots/archive_heatmap.png)
- [gallery.png](plots/gallery.png)

Per-run numbers: results.csv; comparisons: stats.csv; every run's records: runs/.
