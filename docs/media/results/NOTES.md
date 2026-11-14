# Media from the comparison experiment

Everything here was rendered offscreen (`mujoco.Renderer`, no window) on
2026-11-13 from genomes saved by the real comparison run
(`results/experiments/comparison`, 4 conditions x 5 seeds, 1441 simulations
per run, made with commit 4b61992; the numbers are in
[`docs/RESULTS.md`](../../RESULTS.md)). Each clip or still is a re-simulation of a
saved genome; the evolution clip prints the stored and the re-simulated fitness
for every stage and all four agree exactly ("identical").
Script: `scripts/make_media.py`. Run directories below are relative to
`results/experiments/comparison/runs/`.

| file | what it shows | run and genome | command |
|---|---|---|---|
| `evolution.mp4` (13.1 s, 1280x720, 30 fps, H.264 yuv420p, 1.0 MB) | The champion of a MAP-Elites run at each generation where it changed, 3.25 s of walking each on flat ground: gen 0 `b0000-023` (4x2, training fitness +0.013), gen 1 `b0001-023` (8x2, +0.142), gen 2 `b0002-023` (4x2, +0.260), gen 4 `b0004-011` (6x2 metachronal, +0.353). Real time; the caption on each frame gives the stage. | `joint_me_s2` (condition D, seed 2) | `python scripts/make_media.py evolution results/experiments/comparison/runs/joint_me_s2 --out DIR --name evolution --seconds 3.25 --start 2.0` |
| `evolution.jpg` | Poster: the last frame of the clip (final champion `b0004-011`). | same | same |
| `elites_grid.jpg` (1920x1080) | Nine elites from archive cells far apart in the same run, mid-walk at t = 4 s, each with its body plan, gait, frequency, fitness, speed and efficiency. Includes weak and tipped-over elites on purpose: it is an archive, not a leaderboard. | `joint_me_s2` archive | `python scripts/make_media.py elites RUN --out elites_grid.jpg --k 9 --cols 3` |
| `still_morphology_s1.jpg` | Champion of the best condition-B run (6x2, round feet), t = 4 s. | `morphology_s1`, `g0007-003` | `python scripts/make_media.py still RUN --out FILE.jpg --at 4.0` |
| `still_joint_s1.jpg` | Best condition-C champion (6x2 metachronal walker), t = 4 s. | `joint_s1`, `g0005-005` | same |
| `still_joint_me_s2.jpg` | Best condition-D champion (6x2), t = 4 s. | `joint_me_s2`, `b0004-011` | same |
| `still_controller_s3.jpg` | Best condition-A champion (reference-quadruped body), t = 4 s. | `controller_s3`, `g0005-013` | same |
| `results_main.jpg` | The main results figure: training against held-out fitness, one point per seed. | `plots/train_vs_test.png` of `evomorph analyze --run results/experiments/comparison` | `python scripts/make_media.py jpeg PLOT.png --out FILE.jpg` |
| `results_fitness_vs_simulations.jpg` | Best training fitness against simulations used. | `plots/fitness_vs_simulations.png` | same |

The clip was checked by extracting frames at 1.3 s, 4.7 s, 8 s and 11.7 s with
ffmpeg and looking at them: the creature is visible, lit, upright on its feet and
the caption names the stage. The champion of this run is not the fastest of the
experiment; it was chosen because its champion changed four times (most runs
change only two or three times, so an evolution clip from them would be short)
and because its final champion is one of the cleanest gaits by the exploit
check in RESULTS.md (airborne 0.1%, slip 0.34, no drift).
