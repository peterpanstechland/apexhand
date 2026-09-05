"""Preset rigid objects for coin-roll / baoding-style tasks.

Mass is set in kilograms via ``MassPropertiesCfg(mass=...)`` so the Web UI can
expose weight in grams without fighting density × volume.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.utils import PresetCfg

from .token_cfg import PAN_COIN_CFG


# The hackathon's actual balls, measured on the bench: 30 mm turned wood at
# 9.55 g each. The two numbers cross-check -- a 30 mm sphere is 14.14 cm^3, so
# 9.55 g implies 676 kg/m^3, right on beech / birch.
#
# Single source of truth: the rewards' target gap and the reset spacing are both
# derived from the radius, so re-measuring the balls does not need matching
# edits in the MDP terms.
BAODING_BALL_DIAMETER_M = 0.030
BAODING_BALL_RADIUS_M = BAODING_BALL_DIAMETER_M / 2.0
BAODING_BALL_MASS_G = 9.55


def _rigid_props() -> sim_utils.RigidBodyPropertiesCfg:
    return sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=False,
        disable_gravity=False,
        enable_gyroscopic_forces=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=0,
        sleep_threshold=0.005,
        stabilization_threshold=0.0025,
        max_depenetration_velocity=1.0,
    )


def _material(
    static_friction: float,
    dynamic_friction: float,
    restitution: float,
) -> RigidBodyMaterialCfg:
    return RigidBodyMaterialCfg(
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        restitution=restitution,
    )


def make_ball(
    radius_mm: float,  # radius, not diameter. A 45 mm baoding ball → 22.5.
    mass_g: float,
    static_friction: float = 0.45,
    dynamic_friction: float = 0.40,
    restitution: float = 0.08,
    prim_name: str = "object",
    color: tuple[float, float, float] = (0.82, 0.18, 0.16),
    pos: tuple[float, float, float] = (0.0, 0.0, 0.56),
) -> RigidObjectCfg:
    radius = radius_mm / 1000.0
    mass = mass_g / 1000.0
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        spawn=sim_utils.SphereCfg(
            radius=radius,
            rigid_props=_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.003, rest_offset=0.0),
            physics_material=_material(static_friction, dynamic_friction, restitution),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.25),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=(1.0, 0.0, 0.0, 0.0)),
    )


def make_wood_ball(
    diameter_mm: float = BAODING_BALL_DIAMETER_M * 1000.0,
    mass_g: float = BAODING_BALL_MASS_G,
    prim_name: str = "object",
    color: tuple[float, float, float] = (0.85, 0.72, 0.48),
    pos: tuple[float, float, float] = (0.0, 0.0, 0.56),
) -> RigidObjectCfg:
    """A solid wooden baoding ball, defaulting to the measured bench pair.

    Friction is wood against the hand's rubber tactile shells (mu ~ 0.6-0.9),
    much grippier than the metal coin the earlier stages used. Restitution stays
    low so a 9.55 g ball does not bounce out of the cradle on contact.
    """
    return make_ball(
        diameter_mm / 2.0,
        mass_g,
        static_friction=0.75,
        dynamic_friction=0.65,
        restitution=0.05,
        prim_name=prim_name,
        color=color,
        pos=pos,
    )


def make_cube(
    size_mm: float,
    mass_g: float,
    static_friction: float = 0.60,
    dynamic_friction: float = 0.55,
    restitution: float = 0.0,
    prim_name: str = "object",
    color: tuple[float, float, float] = (0.25, 0.45, 0.75),
    pos: tuple[float, float, float] = (0.0, 0.0, 0.56),
) -> RigidObjectCfg:
    size = size_mm / 1000.0
    mass = mass_g / 1000.0
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        spawn=sim_utils.CuboidCfg(
            size=(size, size, size),
            rigid_props=_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.003, rest_offset=0.0),
            physics_material=_material(static_friction, dynamic_friction, restitution),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.4),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=(1.0, 0.0, 0.0, 0.0)),
    )


def make_rod(
    length_mm: float = 80.0,
    radius_mm: float = 6.0,
    mass_g: float = 12.0,
    static_friction: float = 0.55,
    dynamic_friction: float = 0.50,
    restitution: float = 0.0,
    prim_name: str = "object",
    color: tuple[float, float, float] = (0.55, 0.42, 0.22),
    pos: tuple[float, float, float] = (0.0, 0.0, 0.56),
) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        spawn=sim_utils.CylinderCfg(
            radius=radius_mm / 1000.0,
            height=length_mm / 1000.0,
            axis="Y",
            rigid_props=_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass_g / 1000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.003, rest_offset=0.0),
            physics_material=_material(static_friction, dynamic_friction, restitution),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.45),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=(1.0, 0.0, 0.0, 0.0)),
    )


def make_egg(
    radius_mm: float = 18.0,
    length_mm: float = 45.0,
    mass_g: float = 35.0,
    static_friction: float = 0.50,
    dynamic_friction: float = 0.45,
    restitution: float = 0.05,
    prim_name: str = "object",
    color: tuple[float, float, float] = (0.93, 0.88, 0.72),
    pos: tuple[float, float, float] = (0.0, 0.0, 0.56),
) -> RigidObjectCfg:
    # CapsuleCfg uses radius + height of the cylindrical midsection.
    mid = max(length_mm / 1000.0 - 2.0 * radius_mm / 1000.0, 0.004)
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        spawn=sim_utils.CapsuleCfg(
            radius=radius_mm / 1000.0,
            height=mid,
            axis="Z",
            rigid_props=_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass_g / 1000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.003, rest_offset=0.0),
            physics_material=_material(static_friction, dynamic_friction, restitution),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.3),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=(1.0, 0.0, 0.0, 0.0)),
    )


# The pair spawns straddling the palm centreline. Offsetting by one radius plus a
# 1 mm skin keeps the two spheres from starting interpenetrated at any diameter;
# reset_objects_in_palm re-seats them relative to the live palm frame anyway.
_PAIR_SPAWN_Y = BAODING_BALL_RADIUS_M + 0.001

# The two real balls are the same natural wood, so the policy is never allowed to
# use colour (see the identity-free pair observation). The tints differ only so
# play videos and debug renders are readable.
BAODING_WOOD_30MM = make_wood_ball(
    color=(0.87, 0.74, 0.50), pos=(-0.04, -_PAIR_SPAWN_Y, 0.545)
)
BAODING_WOOD_30MM_B = make_wood_ball(
    prim_name="object2", color=(0.78, 0.62, 0.38), pos=(-0.04, _PAIR_SPAWN_Y, 0.545)
)

# Names are diameters. make_ball takes radius in mm.
BAODING_38MM = make_ball(19.0, 40.0, color=(0.82, 0.18, 0.16), pos=(-0.04, -0.012, 0.545))
BAODING_45MM = make_ball(22.5, 70.0, color=(0.82, 0.18, 0.16), pos=(-0.04, -0.014, 0.548))
BAODING_50MM = make_ball(25.0, 90.0, color=(0.82, 0.18, 0.16), pos=(-0.04, -0.016, 0.550))
CUBE_60MM = make_cube(60.0, 40.0, pos=(-0.03, 0.0, 0.56))
ROD_80MM = make_rod(pos=(-0.03, 0.0, 0.56))
EGG_45MM = make_egg(pos=(-0.03, 0.0, 0.56))

# Second ball for the pair — green, mirrored across the palm Y axis.
BAODING_38MM_B = make_ball(19.0, 40.0, prim_name="object2", color=(0.15, 0.55, 0.28), pos=(-0.04, 0.012, 0.545))
BAODING_45MM_B = make_ball(22.5, 70.0, prim_name="object2", color=(0.15, 0.55, 0.28), pos=(-0.04, 0.014, 0.548))
BAODING_50MM_B = make_ball(25.0, 90.0, prim_name="object2", color=(0.15, 0.55, 0.28), pos=(-0.04, 0.016, 0.550))


@configclass
class ObjectPresetCfg(PresetCfg):
    """Hydra path: ``env.scene.object=baoding_45mm``."""

    pan_coin_32mm: RigidObjectCfg = PAN_COIN_CFG
    baoding_wood_30mm: RigidObjectCfg = BAODING_WOOD_30MM
    baoding_38mm: RigidObjectCfg = BAODING_38MM
    baoding_45mm: RigidObjectCfg = BAODING_45MM
    baoding_50mm: RigidObjectCfg = BAODING_50MM
    cube_60mm: RigidObjectCfg = CUBE_60MM
    rod_80mm: RigidObjectCfg = ROD_80MM
    egg_45mm: RigidObjectCfg = EGG_45MM
    default: RigidObjectCfg = pan_coin_32mm


@configclass
class Object2PresetCfg(PresetCfg):
    """Optional second body. ``env.scene.object2=baoding_45mm``."""

    none = None
    baoding_wood_30mm: RigidObjectCfg = BAODING_WOOD_30MM_B
    baoding_38mm: RigidObjectCfg = BAODING_38MM_B
    baoding_45mm: RigidObjectCfg = BAODING_45MM_B
    baoding_50mm: RigidObjectCfg = BAODING_50MM_B
    default = none


@configclass
class BaodingObjectPresetCfg(PresetCfg):
    """Default 30 mm wooden pair — the balls the hackathon actually supplies."""

    baoding_wood_30mm: RigidObjectCfg = BAODING_WOOD_30MM
    baoding_38mm: RigidObjectCfg = BAODING_38MM
    baoding_45mm: RigidObjectCfg = BAODING_45MM
    baoding_50mm: RigidObjectCfg = BAODING_50MM
    default: RigidObjectCfg = baoding_wood_30mm


@configclass
class BaodingObject2PresetCfg(PresetCfg):
    baoding_wood_30mm: RigidObjectCfg = BAODING_WOOD_30MM_B
    baoding_38mm: RigidObjectCfg = BAODING_38MM_B
    baoding_45mm: RigidObjectCfg = BAODING_45MM_B
    baoding_50mm: RigidObjectCfg = BAODING_50MM_B
    default: RigidObjectCfg = baoding_wood_30mm
