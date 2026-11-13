"""``inspect``: print a genome's body plan and controller, validate it, optionally render/view."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from evomorph.commands._common import (
    add_env_args,
    add_genome_args,
    configs_from_args,
    env_from_args,
    genome_from_args,
)
from evomorph.controllers.cpg import gait_phases
from evomorph.morphology.generator import (
    build_mjcf,
    describe_layout,
    estimate_mass,
    limb_reach,
    torque_limit,
)
from evomorph.morphology.genome import (
    CONTROLLER_GENES,
    GENE_BOUNDS,
    MORPHOLOGY_GENES,
    Genome,
    save_genome,
)
from evomorph.morphology.validator import ModelCompileError, compile_model, validate_genome


def _fmt(value: object) -> str:
    if isinstance(value, np.ndarray):
        return "[" + ", ".join(f"{v:+.3f}" for v in value) + "]"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def gene_lines(genome: Genome) -> list[str]:
    """The raw genes, safe to print even for an invalid genome."""
    lines = [
        f"genome {genome.id}  origin={genome.origin} generation={genome.generation} "
        f"parents={list(genome.parent_ids) or '-'}",
        "morphology:",
    ]
    lines += [f"  {n:<20} {_fmt(getattr(genome.morphology, n, None))}" for n in MORPHOLOGY_GENES]
    lines.append("controller:")
    for n in CONTROLLER_GENES:
        unit = "rad" if GENE_BOUNDS[n].kind == "float" and n != "frequency" else ""
        lines.append(f"  {n:<20} {_fmt(getattr(genome.controller, n, None))} {unit}".rstrip())
    return lines


def summary_lines(genome: Genome) -> list[str]:
    """Human-readable description of a *valid* genome (used by inspect and tests)."""
    m = genome.morphology
    head, *genes = gene_lines(genome)
    lines = [
        head,
        f"body: {genome.n_limbs} limbs ({m.n_pairs} pairs) x {m.n_segments} segment(s) = "
        f"{genome.n_joints} actuated joints; est. mass {estimate_mass(genome):.2f} kg; "
        f"reach {limb_reach(genome):.3f} m; servo limit {torque_limit(genome):.2f} N m",
        *genes,
    ]
    c = genome.controller
    layout = describe_layout(genome)
    phases = gait_phases(c.gait, m.n_pairs)
    lines.append("limbs (actuator index; base gait phase in degrees):")
    for limb in layout.limbs:
        knee = limb.knee_index if limb.knee_index is not None else "-"
        deg = math.degrees(phases[limb.limb_index])
        lines.append(
            f"  limb {limb.limb_index} pair {limb.pair} {limb.side:<5} hip#{limb.hip_index} "
            f"knee#{knee} phase {deg:5.1f}"
        )
    return lines


def run(args: argparse.Namespace) -> int:
    genome = genome_from_args(args)
    env = env_from_args(args)
    check = validate_genome(genome)
    if not check.ok:
        print("\n".join(gene_lines(genome)))
        print("VALIDATION: invalid\n  " + "\n  ".join(check.errors))
        return 1
    print("\n".join(summary_lines(genome)))
    if args.save:
        print(f"wrote {save_genome(genome, args.save)}")
    try:
        xml = build_mjcf(genome, env)
        model = compile_model(xml)
    except (ModelCompileError, NotImplementedError) as exc:
        print(f"VALIDATION: genome ok, model failed: {exc}")
        return 1
    print(
        f"VALIDATION: ok; MuJoCo model nbody={model.nbody} njnt={model.njnt} nu={model.nu} "
        f"mass={float(model.body_subtreemass[1]):.3f} kg; "
        f"rest-pose self-collision exclusions={xml.count('<exclude')}"
    )
    if args.xml:
        Path(args.xml).write_text(xml)
        print(f"wrote {args.xml}")
    if args.png:
        from evomorph.visualization.renderer import render_still

        print(f"wrote {render_still(genome, args.png, env)}")
    if args.view:
        from evomorph.visualization.viewer import view

        sim, _ = configs_from_args(args)
        view(genome, env, sim, seed=args.seed)
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("inspect", help="describe, validate, render or view a genome")
    add_genome_args(p)
    add_env_args(p)
    p.add_argument("--png", type=Path, help="write iso/side/front/top rest-pose image")
    p.add_argument("--xml", type=Path, help="write the generated MJCF")
    p.add_argument("--save", type=Path, help="write the genome as JSON (e.g. export a reference)")
    p.add_argument("--view", action="store_true", help="open the interactive MuJoCo viewer")
    p.set_defaults(func=run)
