# ga_tiny

- algorithm **ga**, mode **joint**, seed 0
- budget 64 simulations; used 64 (2 per genome: flat_mu0.6/s0, flat/s1; aggregate mean)
- 7 generations, 6.9 s wall with 2 workers (9.3 simulations/s)

## Progress

| | sims | best_fitness | mean_fitness | median_fitness | n_valid | n_fell | mean_gene_distance | n_body_plans | behaviour_spread | archive_coverage | qd_score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| first (step 0) | 16 | 0.02278 | -0.411 | -0.2285 | 8 | 1 | 0.3402 | 6 | 0.2557 | 0.04 | 12.64 |
| last (step 6) | 64 | 0.6562 | 0.1325 | 0.06379 | 8 | 1 | 0.1798 | 3 | 0.4461 | 0.14 | 46.21 |

## Champion

- id `g0006-003` (generation 6, origin mutation, parents g0005-003)
- body plan 6x2, gait diagonal, frequency 2.93 Hz
- fitness +0.6562 (per run: +0.682, +0.630)
- re-scored with the saved config: identical
- held-out seeds (+1000): fitness +0.6867 (valid)

| metric (mean over runs) | value |
|---|---|
| forward_displacement | +1.956 |
| sustained_velocity | +0.532 |
| energy_efficiency | +0.529 |
| cost_of_transport | +0.893 |
| fell | +0.000 |
| foot_support_fraction | +1.000 |
| slip_fraction | +0.791 |
| airborne_fraction | +0.042 |
| max_height_gain | +0.013 |
| max_lateral_deviation | +0.417 |
| yaw_drift | +0.204 |
| mean_tilt | +0.017 |

## Archive

14 of 100 cells filled (14%), QD-score 46.21 (offset 3.2); axes sustained_velocity, energy_efficiency.

## Reproduce

Command that made this run, and the equivalent from the resolved config
(every override already applied; the new run gets a fresh directory):

```
python -m evomorph evolve --config tests/fixtures/dashboard/tiny.yaml --workers 2 --name ga_tiny --results tests/fixtures/dashboard/runs
python -m evomorph evolve --config tests/fixtures/dashboard/runs/ga_tiny/config.yaml
```
