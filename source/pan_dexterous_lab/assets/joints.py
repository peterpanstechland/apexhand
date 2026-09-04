"""Single source of truth for Apex Hand joint / frame names.

Policy index must never appear as a raw integer elsewhere. Always go through
these lists so Isaac joints and (later) the Rysen SDK stay aligned.

Official URDF (v0.2.0): 21 DoF = 16 actuated + 5 passive 1:1 couplings.
"""

from __future__ import annotations

from typing import Literal

Side = Literal["left", "right"]

FINGERS = ("index", "middle", "ring", "pinky")


def joint_names(side: Side, logical: list[str]) -> list[str]:
    """Prefix logical names (``thumb_j0``) with ``left_`` / ``right_``."""
    return [f"{side}_{name}" for name in logical]


# Logical names (no side prefix) — order is the policy action order.
ACTUATED_LOGICAL: list[str] = [
    "thumb_j0",
    "thumb_j1",
    "thumb_j2",
    "thumb_j3",
    "index_j0",
    "index_j1",
    "index_j2",
    "middle_j0",
    "middle_j1",
    "middle_j2",
    "ring_j0",
    "ring_j1",
    "ring_j2",
    "pinky_j0",
    "pinky_j1",
    "pinky_j2",
]

# Passive joints that must 1:1 follow the corresponding source joint.
COUPLED_LOGICAL: list[str] = [
    "thumb_j4",
    "index_j3",
    "middle_j3",
    "ring_j3",
    "pinky_j3",
]

COUPLED_SOURCE_LOGICAL: list[str] = [
    "thumb_j3",
    "index_j2",
    "middle_j2",
    "ring_j2",
    "pinky_j2",
]

# Official joint ranges in degrees, keyed by logical name.
JOINT_LIMITS_DEG: dict[str, tuple[float, float]] = {
    "thumb_j0": (0.0, 90.0),
    "thumb_j1": (-10.0, 60.0),
    "thumb_j2": (0.0, 80.0),
    "thumb_j3": (-20.0, 80.0),
    "thumb_j4": (-20.0, 80.0),
    "index_j0": (-25.0, 25.0),
    "index_j1": (-20.0, 90.0),
    "index_j2": (-5.0, 100.0),
    "index_j3": (-5.0, 100.0),
    "middle_j0": (-25.0, 25.0),
    "middle_j1": (-20.0, 90.0),
    "middle_j2": (-5.0, 100.0),
    "middle_j3": (-5.0, 100.0),
    "ring_j0": (-25.0, 25.0),
    "ring_j1": (-20.0, 90.0),
    "ring_j2": (-5.0, 100.0),
    "ring_j3": (-5.0, 100.0),
    "pinky_j0": (-25.0, 25.0),
    "pinky_j1": (-20.0, 90.0),
    "pinky_j2": (-5.0, 100.0),
    "pinky_j3": (-5.0, 100.0),
}

PAD_LOGICAL = ["thumb_pad", "index_pad", "middle_pad", "ring_pad", "pinky_pad"]
TIP_LOGICAL = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]

# Knuckle roll runs over the *back* of the fingers, so the contact surface is the
# dorsal side of the proximal phalanx: the segment between the MCP frame
# (``link1``) and the PIP frame (``link2``). Thumb is excluded — it only catches
# the coin at the pinky end. Index -> pinky order matches FINGERS.
KNUCKLE_PROXIMAL_LOGICAL = [f"{finger}_link1" for finger in FINGERS]
KNUCKLE_DISTAL_LOGICAL = [f"{finger}_link2" for finger in FINGERS]


def actuated_joint_names(side: Side) -> list[str]:
    return joint_names(side, ACTUATED_LOGICAL)


def coupled_joint_names(side: Side) -> list[str]:
    return joint_names(side, COUPLED_LOGICAL)


def coupled_source_names(side: Side) -> list[str]:
    return joint_names(side, COUPLED_SOURCE_LOGICAL)


def pad_body_names(side: Side) -> list[str]:
    return joint_names(side, PAD_LOGICAL)


def tip_body_names(side: Side) -> list[str]:
    return joint_names(side, TIP_LOGICAL)


def knuckle_proximal_body_names(side: Side) -> list[str]:
    return joint_names(side, KNUCKLE_PROXIMAL_LOGICAL)


def knuckle_distal_body_names(side: Side) -> list[str]:
    return joint_names(side, KNUCKLE_DISTAL_LOGICAL)


def palm_body_name(side: Side) -> str:
    return f"{side}_palm_link"


# Default policy side. Left-hand lists exist for asset sanity / later training.
DEFAULT_SIDE: Side = "right"

ACTUATED_JOINT_NAMES = actuated_joint_names(DEFAULT_SIDE)
COUPLED_JOINT_NAMES = coupled_joint_names(DEFAULT_SIDE)
COUPLED_SOURCE_NAMES = coupled_source_names(DEFAULT_SIDE)
PAD_BODY_NAMES = pad_body_names(DEFAULT_SIDE)
TIP_BODY_NAMES = tip_body_names(DEFAULT_SIDE)
KNUCKLE_PROXIMAL_BODY_NAMES = knuckle_proximal_body_names(DEFAULT_SIDE)
KNUCKLE_DISTAL_BODY_NAMES = knuckle_distal_body_names(DEFAULT_SIDE)
