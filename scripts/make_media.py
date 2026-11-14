"""README and portfolio media from real run directories, rendered offscreen.

Usage (every subcommand reads saved genomes and re-simulates them; nothing is staged):

    .venv/bin/python scripts/make_media.py evolution RUN --out DIR [--name evolution]
    .venv/bin/python scripts/make_media.py elites RUN --out FILE.jpg [--k 9 --cols 3]
    .venv/bin/python scripts/make_media.py still RUN --out FILE.jpg [--at 4.0]
    .venv/bin/python scripts/make_media.py audit RUN [RUN ...] --out FILE.png
    .venv/bin/python scripts/make_media.py jpeg FIGURE.png --out FIGURE.jpg

* ``evolution``: the champion after every generation where it changed, from
  generation 0 to the last, each walking for ``--seconds`` on the run's first
  training environment with its own spawn seed. H.264, yuv420p, 1280x720, 30 fps,
  plus a JPEG poster (same basename) of the final champion.
* ``elites``: a grid of MAP-Elites elites from archive cells far apart.
* ``still``: one 1280x720 frame of the run's champion mid-walk.
* ``audit``: one row per run, 6 frames of its champion across the 10 s run plus
  the exploit metrics, for checking champions by eye.

Clips are real time. The core's ``RecordSpec`` keeps every round(50 / fps)-th
control step, so asking it for 30 fps really captures 25 frames per simulated
second (docs/DASHBOARD.md). Here each output frame is instead rendered from the
control step nearest to its own timestamp, frame j at t0 + j / 30 s, and written
straight to the encoder, so a clip is exactly as long as the motion it shows
and memory stays flat at 720p. Offscreen only (``mujoco.Renderer``).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from evomorph.config import load_yaml
from evomorph.controllers.base import make_controller
from evomorph.evolution.archive import Archive
from evomorph.evolution.selection import body_plan
from evomorph.evolution.settings import RunConfig, from_sections
from evomorph.morphology.genome import Genome, load_genome
from evomorph.simulation.evaluator import build_body, run, settle, spawn
from evomorph.simulation.fitness import compute_fitness
from evomorph.simulation.metrics import compute_metrics
from evomorph.utils.random_state import make_rng

WIDTH, HEIGHT, FPS = 1280, 720, 30
# x264 constant-quality level: 26 keeps a 12 s 720p clip of these scenes under
# the 2.5 MB budget (checked by the size print); lower is sharper and larger.
CRF = 26
AUDIT_METRICS = (
    "forward_displacement",
    "foot_support_fraction",
    "slip_fraction",
    "airborne_fraction",
    "max_height_gain",
    "max_lateral_deviation",
)


def tracking_camera(body) -> mujoco.MjvCamera:
    """A camera that follows the torso from the front-left, framed by body size.

    The MJCF's own ``track`` camera is framed for 4:3 and leaves a creature small
    in 16:9. Distance 1.4 x the body's largest extent + 0.5 m keeps the biggest
    legal body (about 2 m across) and the smallest one similarly sized in frame;
    azimuth 125 deg and elevation -18 deg show the legs from the side and the
    direction of travel (+x) to the right.
    """
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    cam.distance = 1.4 * body.size + 0.5
    cam.azimuth = 125.0
    cam.elevation = -18.0
    return cam


class _WindowSink:
    """Duck-typed frame sink for ``evaluator.run``: renders the control step nearest
    to each output timestamp inside [t0, t0 + seconds) and hands it to ``emit``."""

    include_settle = False

    def __init__(self, body, size, fps, t0, seconds, control_rate, emit) -> None:
        width, height = size
        self.renderer = mujoco.Renderer(body.model, height=height, width=width)
        self.camera = tracking_camera(body)
        n = round(seconds * fps)
        self.targets = [round((t0 + j / fps) * control_rate) for j in range(n)]
        self.emit = emit
        self.step = 0
        self.done = 0

    def maybe_capture(self, data: mujoco.MjData) -> None:
        self.step += 1  # run() calls this after control step `step`
        while self.done < len(self.targets) and self.targets[self.done] == self.step:
            self.renderer.update_scene(data, camera=self.camera)
            self.emit(self.renderer.render())
            self.done += 1

    def close(self) -> None:
        self.renderer.close()


def simulate_window(genome, cfg: RunConfig, t0, seconds, emit, size=(WIDTH, HEIGHT), fps=FPS):
    """Re-simulate ``genome`` on the run's first evaluation run; emit frames of the window.

    Returns (fitness of this one run, metrics, frames emitted). The physics is the
    evaluator's own (spawn, settle, run), so the fitness equals the stored one.
    """
    env, seed = cfg.eval_spec.runs[0]
    body = build_body(genome, env, cfg.sim)
    controller = make_controller(genome)
    sink = _WindowSink(body, size, fps, t0, seconds, cfg.sim.control_rate, emit)
    try:
        spawn(body, cfg.sim, make_rng(seed))
        settle(body, cfg.sim, None)
        traj, _ = run(body, controller, cfg.sim, sink)
    finally:
        sink.close()
    metrics = compute_metrics(traj)
    fitness, _ = compute_fitness(metrics, cfg.fitness)
    return fitness, metrics, sink.done


def _font(size: int) -> ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def caption(frame: np.ndarray, lines: list[str], size: int = 30) -> np.ndarray:
    """Draw ``lines`` in a dark band at the top-left of ``frame``."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")
    font = _font(size)
    pad, step = size // 2, int(size * 1.3)
    width = max(draw.textlength(t, font=font) for t in lines) + 2 * pad
    draw.rectangle((0, 0, width, pad * 2 + step * len(lines)), fill=(0, 0, 0, 170))
    for i, text in enumerate(lines):
        draw.text((pad, pad + i * step), text, font=font, fill=(255, 255, 255))
    return np.asarray(img)


def load_cfg(run_dir: Path) -> RunConfig:
    return from_sections(load_yaml(run_dir / "config.yaml"))


def champion_changes(run_dir: Path) -> list[tuple[int, str, float]]:
    """(generation, champion id, stored training fitness) where the champion changed."""
    out: list[tuple[int, str, float]] = []
    with open(run_dir / "metrics.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            gid = row["champion_id"]
            if not out or gid != out[-1][1]:
                out.append((int(row["generation"]), gid, float(row["champion_fitness"])))
    return out


def stored_run_fitness(run_dir: Path, genome_id: str) -> float:
    """The champion's stored fitness on the first evaluation run, from its generation file."""
    for path in sorted((run_dir / "generations").glob("gen_*.json")):
        for rec in json.loads(path.read_text()):
            if rec["id"] == genome_id and rec.get("seed_fitness"):
                return float(rec["seed_fitness"][0])
    raise LookupError(f"{genome_id} has no stored record in {run_dir}")


def _writer(path: Path):
    return imageio.get_writer(
        path,
        fps=FPS,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=16,
        output_params=["-crf", str(CRF), "-preset", "slow", "-movflags", "+faststart"],
    )


def cmd_evolution(args: argparse.Namespace) -> None:
    run_dir, cfg = args.run, load_cfg(args.run)
    changes = champion_changes(run_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    clip = args.out / f"{args.name}.mp4"
    last: list[np.ndarray] = []
    with _writer(clip) as w:
        for gen, gid, train_fit in changes:
            genome = load_genome(run_dir / "genomes" / f"champion_gen_{gen:04d}.json")
            label = [
                f"generation {gen}: champion {gid}, {body_plan(genome)} {genome.controller.gait}",
                f"training fitness {train_fit:+.3f} (mean of {cfg.eval_spec.n_sims} runs)",
            ]

            def emit(frame, label=label):
                last[:] = [frame]
                w.append_data(caption(frame, label))

            fit, m, n = simulate_window(genome, cfg, args.start, args.seconds, emit)
            stored = stored_run_fitness(run_dir, gid)
            print(
                f"gen {gen} {gid}: {n} frames, flat/s0 fitness {fit:+.4f} (stored {stored:+.4f}, "
                f"{'identical' if fit == stored else 'DIFFERENT'}), forward "
                f"{m['forward_displacement']:.2f} m"
            )
    Image.fromarray(last[0]).save(args.out / f"{args.name}.jpg", quality=88)
    print(f"wrote {clip} ({clip.stat().st_size / 1e6:.2f} MB) and {args.name}.jpg")


def _mid_frame(genome: Genome, cfg: RunConfig, at: float, size) -> np.ndarray:
    frames: list[np.ndarray] = []
    simulate_window(genome, cfg, at, 1.0 / FPS, frames.append, size=size)
    if not frames:
        raise RuntimeError(f"{genome.id} stopped before t={at} s")
    return frames[0]


def cmd_still(args: argparse.Namespace) -> None:
    cfg = load_cfg(args.run)
    genome = load_genome(args.run / "best_genome.json")
    frame = _mid_frame(genome, cfg, args.at, (WIDTH, HEIGHT))
    Image.fromarray(frame).save(args.out, quality=90)
    print(f"wrote {args.out}: {genome.id} {body_plan(genome)} at t={args.at} s")


def _grid(tiles: list[np.ndarray], cols: int) -> np.ndarray:
    blank = np.zeros_like(tiles[0])
    tiles = tiles + [blank] * (-len(tiles) % cols)
    rows = [np.concatenate(tiles[i : i + cols], axis=1) for i in range(0, len(tiles), cols)]
    return np.concatenate(rows, axis=0)


def cmd_elites(args: argparse.Namespace) -> None:
    cfg = load_cfg(args.run)
    archive = Archive.load(args.run / "archive")
    tiles = []
    for cell in archive.spread_cells(args.k):
        e = archive.get(cell)
        g, m = e.genome, e.metrics
        frame = _mid_frame(g, cfg, args.at, (640, 360))
        text = [
            f"{body_plan(g)} {g.controller.gait} {g.controller.frequency:.2f} Hz",
            f"fit {e.fitness:+.2f}  v {m['sustained_velocity']:.2f} m/s  "
            f"eff {m['energy_efficiency']:.2f}",
        ]
        print(f"cell {cell} {g.id}: {' | '.join(text)}")
        tiles.append(caption(frame, text, size=20))
    Image.fromarray(_grid(tiles, args.cols)).save(args.out, quality=88)
    print(f"wrote {args.out} ({len(tiles)} elites of {len(archive.occupied())} cells)")


def cmd_audit(args: argparse.Namespace) -> None:
    rows = []
    for run_dir in args.runs:
        cfg = load_cfg(run_dir)
        genome = load_genome(run_dir / "best_genome.json")
        tiles: list[np.ndarray] = []
        times = np.linspace(0.5, cfg.sim.duration - 0.5, 6)
        targets = {round(t * FPS) for t in times}
        frames: list[np.ndarray] = []

        def keep(frame, frames=frames, targets=targets, tiles=tiles):
            if len(frames) in targets:
                tiles.append(frame)
            frames.append(np.empty(0))  # count only; frames are not kept

        # t0 = 1 / FPS, not 0: the sink matches control steps counted from 1, so a
        # target at step 0 would never match and no frame would ever be emitted.
        fit, m, _ = simulate_window(genome, cfg, 1.0 / FPS, cfg.sim.duration, keep, size=(320, 240))
        vals = "  ".join(f"{k.replace('_fraction', '')} {m[k]:.2f}" for k in AUDIT_METRICS)
        text = f"{run_dir.name} {genome.id} {body_plan(genome)} flat fit {fit:+.2f}  {vals}"
        print(text)
        rows.append(caption(np.concatenate(tiles, axis=1), [text], size=13))
    Image.fromarray(np.concatenate(rows, axis=0)).save(args.out)
    print(f"wrote {args.out} ({len(rows)} champions)")


def cmd_jpeg(args: argparse.Namespace) -> None:
    img = Image.open(args.figure)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    img.convert("RGB").save(args.out, quality=90)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e3:.0f} kB)")


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ev = sub.add_parser("evolution")
    ev.add_argument("run", type=Path)
    ev.add_argument("--out", type=Path, required=True)
    ev.add_argument("--name", default="evolution")
    # 2 s per champion after 2 s of walking: the gait is established and five or
    # six champions fit the 10-15 s clip length.
    ev.add_argument("--seconds", type=float, default=2.0)
    ev.add_argument("--start", type=float, default=2.0)
    el = sub.add_parser("elites")
    el.add_argument("run", type=Path)
    el.add_argument("--out", type=Path, required=True)
    el.add_argument("--k", type=int, default=9)
    el.add_argument("--cols", type=int, default=3)
    el.add_argument("--at", type=float, default=4.0)
    st = sub.add_parser("still")
    st.add_argument("run", type=Path)
    st.add_argument("--out", type=Path, required=True)
    st.add_argument("--at", type=float, default=4.0)
    au = sub.add_parser("audit")
    au.add_argument("runs", type=Path, nargs="+")
    au.add_argument("--out", type=Path, required=True, help="image file to write, e.g. audit.png")
    jp = sub.add_parser("jpeg")
    jp.add_argument("figure", type=Path)
    jp.add_argument("--out", type=Path, required=True)
    return ap


COMMANDS: dict[str, Callable[[argparse.Namespace], None]] = {
    "evolution": cmd_evolution,
    "elites": cmd_elites,
    "still": cmd_still,
    "audit": cmd_audit,
    "jpeg": cmd_jpeg,
}


def _check_out(ap: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Stop before any simulation when --out cannot be written as the file it must be."""
    out = args.out
    if args.cmd == "evolution":  # --out is the folder that receives the clip and poster
        if out.exists() and not out.is_dir():
            ap.error(f"--out {out} is a file; give a folder for the clip and its poster.")
        return
    if out.is_dir() or not out.suffix:
        ap.error(
            f"--out {out} must be an image file name such as {out.name or 'sheet'}.png, "
            "not a folder."
        )


def main() -> None:
    ap = _parser()
    args = ap.parse_args()
    _check_out(ap, args)
    COMMANDS[args.cmd](args)


if __name__ == "__main__":
    main()
