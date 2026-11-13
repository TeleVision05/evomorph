# me_tiny

- algorithm **map_elites**, mode **joint**, seed 0
- budget 64 simulations; used 64 (2 per genome: flat_mu0.6/s0, flat/s1; aggregate mean)
- 4 batches, 14.7 s wall with 2 workers (4.3 simulations/s)

## Progress

| | sims | best_fitness | mean_fitness | median_fitness | n_valid | n_fell | mean_gene_distance | n_body_plans | behaviour_spread | archive_coverage | qd_score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| first (step 0) | 16 | 0.02278 | -0.03908 | 0.01168 | 4 | 0 | 0.3451 | 3 | 0.3278 | 0.04 | 12.64 |
| last (step 3) | 64 | 0.07162 | -0.01572 | 0.01742 | 11 | 0 | 0.292 | 4 | 0.3022 | 0.11 | 35.03 |

## Champion

- id `b0002-002` (generation 2, origin mutation, parents b0000-003)
- body plan 6x2, gait metachronal, frequency 2.16 Hz
- fitness +0.0716 (per run: +0.066, +0.077)
- re-scored with the saved config: identical
- held-out seeds (+1000): fitness +0.0484 (valid)

| metric (mean over runs) | value |
|---|---|
| forward_displacement | +1.696 |
| sustained_velocity | +0.416 |
| energy_efficiency | +0.202 |
| cost_of_transport | +3.942 |
| fell | +0.000 |
| foot_support_fraction | +0.995 |
| slip_fraction | +0.799 |
| airborne_fraction | +0.181 |
| max_height_gain | +0.074 |
| max_lateral_deviation | +0.421 |
| yaw_drift | +0.469 |
| mean_tilt | +0.145 |

## Archive

11 of 100 cells filled (11%), QD-score 35.03 (offset 3.2); axes sustained_velocity, energy_efficiency.

## Reproduce

Command that made this run, and the equivalent from the resolved config
(every override already applied; the new run gets a fresh directory):

```
python -m evomorph evolve --config tests/fixtures/dashboard/tiny.yaml --algorithm map_elites --workers 2 --name me_tiny --results tests/fixtures/dashboard/runs
python -m evomorph evolve --config tests/fixtures/dashboard/runs/me_tiny/config.yaml
```
