"""Render the reference creatures (and optional genome files) to docs/media.

Usage:
    .venv/bin/python scripts/render_media.py [--out docs/media] [genome.json ...]

For each creature writes <name>.mp4 (settle + 10 s run), <name>.png (iso, side,
front and top rest-pose views) and <name>_sheet.png (8 evenly spaced frames).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evomorph.morphology.genome import load_genome, reference_hexapod, reference_quadruped
from evomorph.simulation.evaluator import RecordSpec
from evomorph.visualization.renderer import contact_sheet, render_still, render_video


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("docs/media"))
    ap.add_argument("genomes", nargs="*", type=Path)
    args = ap.parse_args()
    creatures = [reference_quadruped(), reference_hexapod()]
    creatures += [load_genome(p) for p in args.genomes]
    for g in creatures:
        name = g.id
        render_still(g, args.out / f"{name}.png")
        spec = RecordSpec(width=640, height=480, fps=30, include_settle=True)
        r = render_video(g, args.out / f"{name}.mp4", spec=spec)
        contact_sheet(r.frames, args.out / f"{name}_sheet.png", n=8, cols=4)
        m = r.metrics
        print(
            f"{name}: fitness {r.fitness:+.3f} fwd {m['forward_displacement']:+.2f} m "
            f"lat {m['lateral_displacement']:.2f} m {r.failure_reason or 'completed'}"
        )


if __name__ == "__main__":
    main()
