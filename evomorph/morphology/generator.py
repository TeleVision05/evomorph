"""Procedural MJCF generation from a genome.

Frame: +x is forward (the direction fitness rewards), +y is the creature's left,
+z is up. The torso is a box whose centre carries the free root joint.

Body plan
---------
Limb pairs sit on the torso's side walls at mid-height, evenly spaced front
(pair 0) to back and mirrored across the sagittal (x-z) plane. Each limb is::

    torso -> thigh_<tag>  (hinge hip_<tag>)   capsule along d_upper
          -> shin_<tag>   (hinge knee_<tag>)  capsule along d_lower, foot sphere at its end
    (one-segment limbs: no shin; the foot sits at the end of the thigh)

``d_upper`` points outward and down by ``hip_splay``; ``d_lower`` outward and down
by ``lower_splay``. All body frames are axis-aligned with the torso, so joint axes
are written in torso coordinates. Joint angle 0 is the rest pose.

Joint axis conventions (checked numerically in tests/test_generator.py):

* hip: axis = normalise(d_upper x x_hat), tilted toward x_hat by ``hip_axis_tilt``.
  A positive hip angle swings the foot forward (+x) on both sides; a positive
  tilt makes the forward swing also lift the foot, which is what lets a hip-only
  leg take steps instead of only scraping.
* knee: axis = +x on the left, -x on the right. A positive knee angle lifts the
  foot (and moves it outward) on both sides.

Self-collision policy: limbs collide with each other and with the torso (see the
bitmasks in ``environment.py``), because with self-collision off evolution found
long-legged creatures that swept front and rear legs straight through each other
(measured in the first 300-genome sweep, 2026-11-13). Two exceptions: MuJoCo's
built-in parent-child filter (a thigh never collides with the torso it hangs
from, nor a shin with its own thigh), and any body pair that already
interpenetrates at the rest pose, which ``build_mjcf`` detects with MuJoCo's own
collision pass and excludes, treating it as fused by design rather than letting
the overlap explode at spawn. Every creature geom collides with the terrain;
only feet count as "foot contacts", and torso contact is how falls are detected.

Mass: geom densities (``TORSO_DENSITY``, ``LIMB_DENSITY``) times the mass-scale
genes; MuJoCo derives mass and inertia from the geometry.

Actuation: one position servo per joint. Peak torque scales with body weight and
leg length (``torque_limit``) so a big creature is not underpowered and a small
one cannot fling itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from evomorph.morphology.genome import Genome
from evomorph.simulation.environment import (
    CREATURE_CONAFFINITY,
    CREATURE_CONTYPE,
    EnvSpec,
    build_world,
)

GRAVITY = 9.81
TORSO_DENSITY = 500.0  # kg/m^3: hollow shell with batteries/electronics, half of water
LIMB_DENSITY = 1200.0  # kg/m^3: polymer/aluminium tube with embedded motors
# Peak joint torque = TORQUE_FACTOR * m g * reach / n_limbs. With half the legs in
# stance each carries 2 m g / n_limbs; x1.5 headroom gives 3.0. Enough to stand
# and push, not enough to launch the body off one stroke.
TORQUE_FACTOR = 3.0
# Servo stiffness: the torque limit is reached at this tracking error (rad). A
# smaller value is a stiffer, more "geared" servo.
SERVO_SATURATION_ERROR = 0.25
# Reflected rotor inertia of a geared motor, as a fraction of m*reach^2/n_limbs.
ARMATURE_FACTOR = 0.01
SIDE_NAMES = ("left", "right")


def limb_tag(pair: int, side: int) -> str:
    """Unique, predictable limb tag: ``0L`` is the front-left limb."""
    return f"{pair}{'L' if side == 0 else 'R'}"


@dataclass(frozen=True)
class LimbLayout:
    """Where one limb's actuators live in ``data.ctrl``."""

    limb_index: int  # 2 * pair + side
    pair: int  # 0 = front
    side: str  # "left" or "right"
    hip_index: int
    knee_index: int | None
    hip_joint: str
    knee_joint: str | None
    foot_geom: str


@dataclass(frozen=True)
class ActuatorLayout:
    """Actuator map shared by the generator, controllers and evaluator."""

    limbs: tuple[LimbLayout, ...]
    n_pairs: int
    n_segments: int
    n_actuators: int
    hip_range: float  # rad, symmetric joint limit
    knee_range: float  # rad, symmetric joint limit (0 when there is no knee)

    @property
    def n_limbs(self) -> int:
        return len(self.limbs)


def describe_layout(genome: Genome) -> ActuatorLayout:
    """Actuator layout of ``genome``: hip then (optional) knee per limb, limb order 0L,0R,1L..."""
    m = genome.morphology
    limbs: list[LimbLayout] = []
    idx = 0
    for pair in range(m.n_pairs):
        for side in (0, 1):
            tag = limb_tag(pair, side)
            hip_i = idx
            idx += 1
            knee_i = None
            if m.n_segments == 2:
                knee_i = idx
                idx += 1
            limbs.append(
                LimbLayout(
                    limb_index=2 * pair + side,
                    pair=pair,
                    side=SIDE_NAMES[side],
                    hip_index=hip_i,
                    knee_index=knee_i,
                    hip_joint=f"hip_{tag}",
                    knee_joint=f"knee_{tag}" if knee_i is not None else None,
                    foot_geom=f"foot_{tag}",
                )
            )
    return ActuatorLayout(
        limbs=tuple(limbs),
        n_pairs=m.n_pairs,
        n_segments=m.n_segments,
        n_actuators=idx,
        hip_range=math.radians(m.hip_range),
        knee_range=math.radians(m.knee_range) if m.n_segments == 2 else 0.0,
    )


# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def limb_geometry(genome: Genome, pair: int, side: int) -> dict[str, np.ndarray]:
    """Rest-pose geometry of one limb in torso coordinates."""
    m = genome.morphology
    sgn = 1.0 if side == 0 else -1.0
    # Hips span 85% of the torso so the outer pairs sit just inside its ends.
    frac = 1.0 - 2.0 * pair / (m.n_pairs - 1) if m.n_pairs > 1 else 0.0
    x = 0.5 * m.torso_length * 0.85 * frac
    hip = np.array([x, sgn * 0.5 * m.torso_width, 0.0])
    th, ph, ps = (math.radians(a) for a in (m.hip_splay, m.lower_splay, m.hip_axis_tilt))
    d_up = np.array([0.0, sgn * math.cos(th), -math.sin(th)])
    d_low = np.array([0.0, sgn * math.cos(ph), -math.sin(ph)])
    x_hat = np.array([1.0, 0.0, 0.0])
    hip_axis = _unit(math.cos(ps) * _unit(np.cross(d_up, x_hat)) + sgn * math.sin(ps) * x_hat)
    knee_axis = sgn * x_hat
    return {"hip": hip, "d_up": d_up, "d_low": d_low, "hip_axis": hip_axis, "knee_axis": knee_axis}


def _capsule_mass(radius: float, length: float, density: float) -> float:
    return density * (math.pi * radius**2 * length + 4.0 / 3.0 * math.pi * radius**3)


def estimate_mass(genome: Genome) -> float:
    """Analytic total mass (kg), matching MuJoCo's density-based value closely."""
    m = genome.morphology
    torso = TORSO_DENSITY * m.torso_mass_scale * m.torso_length * m.torso_width * m.torso_height
    rho = LIMB_DENSITY * m.limb_mass_scale
    per_limb = _capsule_mass(m.limb_thickness, m.upper_limb_length, rho)
    if m.n_segments == 2:
        per_limb += _capsule_mass(m.limb_thickness, m.lower_limb_length, rho)
    per_limb += rho * 4.0 / 3.0 * math.pi * m.foot_radius**3
    return torso + m.n_limbs * per_limb


def limb_reach(genome: Genome) -> float:
    m = genome.morphology
    return m.upper_limb_length + (m.lower_limb_length if m.n_segments == 2 else 0.0)


def torque_limit(genome: Genome) -> float:
    """Peak servo torque (N m) for every joint of this creature."""
    m = genome.morphology
    return TORQUE_FACTOR * estimate_mass(genome) * GRAVITY * limb_reach(genome) / m.n_limbs


# ----------------------------------------------------------------------------
# MJCF writing
# ----------------------------------------------------------------------------


def _v(a: np.ndarray) -> str:
    return " ".join(f"{float(x):.6g}" for x in a)


def _limb_xml(genome: Genome, pair: int, side: int) -> str:
    m = genome.morphology
    g = limb_geometry(genome, pair, side)
    tag = limb_tag(pair, side)
    r, fr = m.limb_thickness, m.foot_radius
    hip_r, knee_r = math.radians(m.hip_range), math.radians(m.knee_range)
    upper_end = g["d_up"] * m.upper_limb_length
    lines = [
        f'      <body name="thigh_{tag}" pos="{_v(g["hip"])}">',
        f'        <joint name="hip_{tag}" type="hinge" axis="{_v(g["hip_axis"])}" '
        f'range="{-hip_r:.6g} {hip_r:.6g}"/>',
        f'        <geom name="thigh_{tag}_geom" type="capsule" size="{r:.6g}" '
        f'fromto="0 0 0 {_v(upper_end)}" class="limb"/>',
    ]
    if m.n_segments == 2:
        lower_end = g["d_low"] * m.lower_limb_length
        lines += [
            f'        <body name="shin_{tag}" pos="{_v(upper_end)}">',
            f'          <joint name="knee_{tag}" type="hinge" axis="{_v(g["knee_axis"])}" '
            f'range="{-knee_r:.6g} {knee_r:.6g}"/>',
            f'          <geom name="shin_{tag}_geom" type="capsule" size="{r:.6g}" '
            f'fromto="0 0 0 {_v(lower_end)}" class="limb"/>',
            f'          <geom name="foot_{tag}" type="sphere" size="{fr:.6g}" '
            f'pos="{_v(lower_end)}" class="foot"/>',
            "        </body>",
        ]
    else:
        lines.append(
            f'        <geom name="foot_{tag}" type="sphere" size="{fr:.6g}" '
            f'pos="{_v(upper_end)}" class="foot"/>'
        )
    lines.append("      </body>")
    return "\n".join(lines)


def _actuators_xml(genome: Genome) -> str:
    layout = describe_layout(genome)
    tau = torque_limit(genome)
    kp = tau / SERVO_SATURATION_ERROR
    lines = []
    for limb in layout.limbs:
        joints = [(limb.hip_joint, layout.hip_range)]
        if limb.knee_joint is not None:
            joints.append((limb.knee_joint, layout.knee_range))
        for joint, rng in joints:
            lines.append(
                f'    <position name="{joint}_act" joint="{joint}" kp="{kp:.6g}" dampratio="1" '
                f'ctrlrange="{-rng:.6g} {rng:.6g}" forcerange="{-tau:.6g} {tau:.6g}"/>'
            )
    return "\n".join(lines)


def rest_pose_overlaps(xml: str) -> list[tuple[str, str]]:
    """Body-name pairs whose creature geoms interpenetrate at the rest pose.

    The model is posed at qpos0 (torso high above the ground, all joints 0) and
    MuJoCo's collision pass reports every penetrating geom pair.
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    pairs: set[tuple[str, str]] = set()
    for i in range(data.ncon):
        b1, b2 = (int(model.geom_bodyid[g]) for g in data.contact.geom[i])
        if b1 == 0 or b2 == 0:
            continue
        n1, n2 = sorted((model.body(b1).name, model.body(b2).name))
        pairs.add((n1, n2))
    return sorted(pairs)


def build_mjcf(genome: Genome, env: EnvSpec | None = None, timestep: float = 0.002) -> str:
    """Complete MJCF document for ``genome`` standing in ``env``.

    The torso is placed at an arbitrary height; the evaluator lowers it so the
    lowest geom rests just above the ground (see ``evaluator.spawn``). Body pairs
    that overlap at rest get a ``<contact><exclude>`` (see module doc).
    """
    env = env or EnvSpec()
    xml = _document(genome, env, timestep, excludes=())
    try:
        excludes = rest_pose_overlaps(xml)
    except ValueError:
        return xml  # let compile_model report the compiler error with its own type
    return _document(genome, env, timestep, excludes) if excludes else xml


def _document(genome: Genome, env: EnvSpec, timestep: float, excludes: tuple | list) -> str:
    m = genome.morphology
    world = build_world(env)
    rho_l = LIMB_DENSITY * m.limb_mass_scale
    rho_t = TORSO_DENSITY * m.torso_mass_scale
    reach = limb_reach(genome)
    armature = ARMATURE_FACTOR * estimate_mass(genome) * reach**2 / m.n_limbs
    half = (m.torso_length / 2, m.torso_width / 2, m.torso_height / 2)
    limbs = "\n".join(_limb_xml(genome, p, s) for p in range(m.n_pairs) for s in (0, 1))
    nose_r = min(half[1], half[2]) * 0.8
    spawn_z = world.spawn_height + reach + m.torso_height + 0.5
    # Side-on tracking camera, distance scaled to the creature's footprint.
    span = max(m.torso_length, m.torso_width + 2 * reach)
    cam_d, cam_h = 2.0 * span + 0.6, 0.45 * (2.0 * span + 0.6)
    exclude_xml = "".join(f'\n    <exclude body1="{a}" body2="{b}"/>' for a, b in excludes)
    return f"""<mujoco model="{genome.id}">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{timestep:.6g}" integrator="implicitfast"/>
  <visual>
    <global offwidth="1280" offheight="960"/>
    <headlight ambient="0.25 0.25 0.25" diffuse="0.35 0.35 0.35" specular="0 0 0"/>
    <quality shadowsize="4096"/>
  </visual>
  <default>
    <joint armature="{armature:.6g}" damping="0"/>
    <geom contype="{CREATURE_CONTYPE}" conaffinity="{CREATURE_CONAFFINITY}" condim="3"/>
    <default class="limb">
      <geom density="{rho_l:.6g}" rgba="0.85 0.55 0.2 1"/>
    </default>
    <default class="foot">
      <geom density="{rho_l:.6g}" rgba="0.2 0.2 0.25 1"/>
    </default>
  </default>
  <asset>{world.assets}  </asset>
  <worldbody>
{world.worldbody}    <body name="torso" pos="0 0 {spawn_z:.6g}">
      <freejoint name="root"/>
      <geom name="torso_geom" type="box" size="{_v(np.array(half))}" density="{rho_t:.6g}"
            rgba="0.25 0.5 0.75 1"/>
      <geom name="nose" type="sphere" size="{nose_r:.6g}" pos="{half[0]:.6g} 0 0"
            contype="0" conaffinity="0" density="0" rgba="0.9 0.9 0.95 1"/>
      <camera name="track" mode="trackcom" pos="0 {-cam_d:.4g} {cam_h:.4g}"
              xyaxes="1 0 0 0 {cam_h:.4g} {cam_d:.4g}"/>
{limbs}
    </body>
  </worldbody>
  <contact>{exclude_xml}
  </contact>
  <actuator>
{_actuators_xml(genome)}
  </actuator>
</mujoco>
"""
