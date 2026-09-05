"""Apex Hand ArticulationCfg for Isaac Lab (PhysX implicit actuators)."""

from __future__ import annotations

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.utils import PresetCfg

from .joints import (
    ACTUATED_JOINT_NAMES,
    COUPLED_JOINT_NAMES,
    COUPLED_SOURCE_NAMES,
    PAD_BODY_NAMES,
    TIP_BODY_NAMES,
    Side,
    actuated_joint_names,
    coupled_joint_names,
    coupled_source_names,
    pad_body_names,
    tip_body_names,
)

# Curriculum token V1 (meters).
COIN_RADIUS = 0.016
COIN_THICKNESS = 0.004

_REPO_ROOT = Path(__file__).resolve().parents[3]
_USD_DIR = _REPO_ROOT / "assets" / "apex" / "usd"


def _usd_path(side: Side) -> str:
    """Resolved USD path. Isaac Sim 6 writes ``<usd_dir>/apex_hand_<side>/apex_hand_<side>.usda``."""
    candidates = [
        _USD_DIR / side / f"apex_hand_{side}" / f"apex_hand_{side}.usda",
        _USD_DIR / side / f"apex_hand_{side}" / f"apex_hand_{side}.usd",
        _USD_DIR / side / f"apex_hand_{side}.usd",
        _USD_DIR / f"apex_hand_{side}.usd",
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    return str(_USD_DIR / side / f"apex_hand_{side}" / f"apex_hand_{side}.usda")


# Palm-down knuckle-roll pose. Measured frames at zero joints (identity rot):
#   fingers point local -Z, flexor/pad side is local +X, index->pinky is local +Y.
# This quaternion (verified with scripts/probe_hand_frames.py) puts the fingers
# horizontal along world -X, the pads 6 mm *below* the finger links, and keeps
# index->pinky along +Y. So the backs of the fingers face up (world +Z) and
# gravity holds the coin on the knuckles. The conjugate gives palm-up instead.
_PALM_DOWN_ROT = (0.7071068, 0.0, -0.7071068, 0.0)
# Conjugate of the palm-down quaternion: palm faces world +Z so gravity seats
# balls in the cradle instead of on the knuckle backs.
_PALM_UP_ROT = (0.7071068, 0.0, 0.7071068, 0.0)

# Fingers stay side by side (j0 = 0) so adjacent proximal phalanges form a
# groove for the coin; mild MCP flexion keeps the knuckle bridge close to level
# while the distal segments curl away. A tight mid-limit curl (j1 = 0.611) tilts
# the bridge enough that the coin rolls back and forth along the finger instead
# of settling, so keep j1 small. ApexCoupledEMAAction offsets from this pose, so
# it is also what action 0 commands.
_KNUCKLE_ROLL_JOINT_POS = {
    ".*_thumb_j0": 0.611,
    ".*_thumb_j1": 0.262,
    ".*_thumb_j2": 0.489,
    ".*_thumb_j3": 0.349,
    ".*_thumb_j4": 0.349,
    ".*_index_j0": 0.00,
    ".*_index_j1": 0.262,
    ".*_index_j2": 0.524,
    ".*_index_j3": 0.524,
    ".*_middle_j0": 0.00,
    ".*_middle_j1": 0.262,
    ".*_middle_j2": 0.524,
    ".*_middle_j3": 0.524,
    ".*_ring_j0": 0.00,
    ".*_ring_j1": 0.262,
    ".*_ring_j2": 0.524,
    ".*_ring_j3": 0.524,
    ".*_pinky_j0": 0.00,
    ".*_pinky_j1": 0.262,
    ".*_pinky_j2": 0.524,
    ".*_pinky_j3": 0.524,
}

# Slightly open cup: abduction still near 0 (self-collision is off), MCP/PIP
# flexed just enough that two 45 mm balls rest against the pads and palm.
_CRADLE_JOINT_POS = {
    ".*_thumb_j0": 0.70,
    ".*_thumb_j1": 0.40,
    ".*_thumb_j2": 0.55,
    ".*_thumb_j3": 0.40,
    ".*_thumb_j4": 0.40,
    ".*_index_j0": 0.05,
    ".*_index_j1": 0.45,
    ".*_index_j2": 0.55,
    ".*_index_j3": 0.45,
    ".*_middle_j0": 0.00,
    ".*_middle_j1": 0.50,
    ".*_middle_j2": 0.55,
    ".*_middle_j3": 0.45,
    ".*_ring_j0": 0.00,
    ".*_ring_j1": 0.50,
    ".*_ring_j2": 0.55,
    ".*_ring_j3": 0.45,
    ".*_pinky_j0": -0.05,
    ".*_pinky_j1": 0.45,
    ".*_pinky_j2": 0.55,
    ".*_pinky_j3": 0.45,
}


def make_hand_cfg(side: Side, pose: str = "palm_down_knuckle") -> ArticulationCfg:
    """Build an ArticulationCfg. ``pose`` is ``palm_down_knuckle`` or ``palm_up_cradle``."""
    if pose == "palm_up_cradle":
        rot = _PALM_UP_ROT
        joints = _CRADLE_JOINT_POS
    else:
        rot = _PALM_DOWN_ROT
        joints = _KNUCKLE_ROLL_JOINT_POS
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=_usd_path(side),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                retain_accelerations=False,
                enable_gyroscopic_forces=False,
                angular_damping=0.01,
                max_linear_velocity=1000.0,
                max_angular_velocity=64 / math.pi * 180.0,
                max_depenetration_velocity=1000.0,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                # Must stay off: the URDF's tactile shells already overlap at rest,
                # so PhysX self-collision diverges (Gate 3 coupling error blows up
                # to ~1300 rad, only 9/16 joints move). The cost is that adjacent
                # fingers can pass through each other, so abduction authority is
                # clamped in ActionsCfg instead -- see the finger j0 scale.
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0005,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.50),
            rot=rot,
            joint_pos=joints,
        ),
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                # URDF effort is 0 and dynamics are uncalibrated; Allegro's 0.5 Nm
                # cannot drive these links to limit. Raise until a system ID pass.
                effort_limit_sim=2.0,
                stiffness=8.0,
                damping=0.3,
                friction=0.01,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )


APEX_HAND_RIGHT_CFG = make_hand_cfg("right", "palm_down_knuckle")
APEX_HAND_LEFT_CFG = make_hand_cfg("left", "palm_down_knuckle")
APEX_HAND_RIGHT_PALM_UP_CFG = make_hand_cfg("right", "palm_up_cradle")
APEX_HAND_LEFT_PALM_UP_CFG = make_hand_cfg("left", "palm_up_cradle")


@configclass
class RobotPresetCfg(PresetCfg):
    """Hydra path: ``env.scene.robot=palm_up_cradle``."""

    palm_down_knuckle: ArticulationCfg = APEX_HAND_RIGHT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    palm_up_cradle: ArticulationCfg = APEX_HAND_RIGHT_PALM_UP_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    default: ArticulationCfg = palm_down_knuckle


@configclass
class CradleRobotPresetCfg(PresetCfg):
    """Same poses, but default is palm-up so baoding does not need a post_init overwrite."""

    palm_down_knuckle: ArticulationCfg = APEX_HAND_RIGHT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    palm_up_cradle: ArticulationCfg = APEX_HAND_RIGHT_PALM_UP_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    default: ArticulationCfg = palm_up_cradle

__all__ = [
    "ACTUATED_JOINT_NAMES",
    "APEX_HAND_LEFT_CFG",
    "APEX_HAND_LEFT_PALM_UP_CFG",
    "APEX_HAND_RIGHT_CFG",
    "APEX_HAND_RIGHT_PALM_UP_CFG",
    "CradleRobotPresetCfg",
    "COIN_RADIUS",
    "COIN_THICKNESS",
    "COUPLED_JOINT_NAMES",
    "COUPLED_SOURCE_NAMES",
    "PAD_BODY_NAMES",
    "RobotPresetCfg",
    "TIP_BODY_NAMES",
    "actuated_joint_names",
    "coupled_joint_names",
    "coupled_source_names",
    "make_hand_cfg",
    "pad_body_names",
    "tip_body_names",
]
