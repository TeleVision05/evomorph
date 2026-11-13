"""Render a finished evolution run offscreen: champion video and stills, elite sheet.

Usage:
    .venv/bin/python scripts/render_run.py results/<run> [--elites 6] [--out DIR]

Writes into ``<run>/videos/`` (or ``--out``):

* ``champion.mp4``: the champion re-simulated (settle + run) on the first run
  of the saved evaluation spec, with the saved sim and fitness config;
* ``champion.png``: iso / side / front / top rest-pose views;
* ``champion_sheet.png``: 8 evenly spaced frames of the video;
* ``elites_sheet.png`` (with ``--elites N``): N archive elites from cells far
  apart (greedy farthest-point choice in normalised descriptor space), one row
  each, 5 frames per row, labelled with cell, plan, gait and metrics.

Also prints the champion's exploit checks on every run of the spec. Offscreen
only (``mujoco.Renderer``): no window opens.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from evomorph.config import load_yaml
from evomorph.evolution.archive import Archive
from evomorph.evolution.selection import body_plan
from evomorph.evolution.settings import from_sections
from evomorph.morphology.genome import load_genome
from evomorph.simulation.evaluator import RecordSpec, evaluate
from evomorph.visualization.renderer import contact_sheet, render_still, write_video

CHECKS = (
    "forward_displacement",
    "sustained_velocity",
    "foot_support_fraction",
    "slip_fraction",
    "airborne_fraction",
    "max_height_gain",
    "max_lateral_deviation",
    "yaw_drift",
    "mean_tilt",
    "fell",
)


def _champion(run: Path, out: Path, cfg) -> None:
    genome = load_genome(run / "best_genome.json")
    spec = cfg.eval_spec
    print(f"champion {genome.id} plan {body_plan(genome)} gait {genome.controller.gait}")
    for env, seed in spec.runs:
        r = evaluate(genome, env, spec.sim, spec.fitness, seed=seed)
        vals = "  ".join(f"{k}={r.metrics.get(k, float('nan')):.3f}" for k in CHECKS)
        print(f"  {env.label()}/s{seed}: fitness {r.fitness:+.3f}  {vals}")
    env, seed = spec.runs[0]
    rec = RecordSpec(width=640, height=480, fps=30, include_settle=True)
    r = evaluate(genome, env, spec.sim, spec.fitness, seed=seed, record=rec)
    write_video(r.frames, out / "champion.mp4", fps=rec.fps)
    contact_sheet(r.frames, out / "champion_sheet.png", n=8, cols=4)
    render_still(genome, out / "champion.png", env)
    print(f"wrote {out / 'champion.mp4'}, champion.png, champion_sheet.png")


def spread_cells(archive: Archive, k: int) -> list[tuple[int, int]]:
    """Greedy farthest-point choice of ``k`` occupied cells, starting from the best."""
    keys = archive.occupied()
    pts = {c: archive.spec.normalise(archive.get(c).descriptor) for c in keys}
    best = archive.best()
    chosen = [archive.spec.cell(best.descriptor)]
    while len(chosen) < min(k, len(keys)):
        far = max(
            (c for c in keys if c not in chosen),
            key=lambda c: min(np.linalg.norm(pts[c] - pts[s]) for s in chosen),
        )
        chosen.append(far)
    return chosen


def _label(img: np.ndarray, text: str) -> np.ndarray:
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    draw.rectangle((0, 0, pil.width, 18), fill=(0, 0, 0))
    draw.text((4, 3), text, fill=(255, 255, 255))
    return np.asarray(pil)


def _elites(run: Path, out: Path, cfg, k: int) -> None:
    archive = Archive.load(run / "archive")
    env, seed = cfg.eval_spec.runs[0]
    rec = RecordSpec(width=320, height=240, fps=10, include_settle=False)
    rows = []
    for cell in spread_cells(archive, k):
        e = archive.get(cell)
        g = e.genome
        r = evaluate(g, env, cfg.sim, cfg.fitness, seed=seed, record=rec)
        idx = np.linspace(0, len(r.frames) - 1, 5).round().astype(int)
        tiles = [r.frames[i] for i in idx]
        m = e.metrics
        text = (
            f"cell {cell} {body_plan(g)} {g.controller.gait} f={g.controller.frequency:.2f}Hz "
            f"fit {e.fitness:+.2f} v {m['sustained_velocity']:.2f} eff "
            f"{m['energy_efficiency']:.2f} slip {m['slip_fraction']:.2f}"
        )
        print(f"  {g.id}: {text}")
        rows.append(_label(np.concatenate(tiles, axis=1), text))
    Image.fromarray(np.concatenate(rows, axis=0)).save(out / "elites_sheet.png")
    print(f"wrote {out / 'elites_sheet.png'} ({len(rows)} elites)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("--elites", type=int, default=0, help="elite rows in the archive sheet")
    ap.add_argument("--out", type=Path, help="output folder (default <run>/videos)")
    args = ap.parse_args()
    cfg = from_sections(load_yaml(args.run / "config.yaml"))
    out = args.out or args.run / "videos"
    out.mkdir(parents=True, exist_ok=True)
    _champion(args.run, out, cfg)
    if args.elites:
        _elites(args.run, out, cfg, args.elites)


if __name__ == "__main__":
    main()
