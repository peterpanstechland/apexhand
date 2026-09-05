"""Rewards for two-ball baoding rotation in a palm-up cradle.

Identity-free by construction. The two real balls are the same natural wood, so
nothing downstream -- reward, observation or the camera tracker -- may depend on
telling ball A from ball B. Rotation is therefore measured from the *axis* of
the pair (see :func:`pair_geometry`), encoded as twice its in-plane angle:
swapping the balls adds pi to the axis angle, which adds 2*pi to the doubled
angle and so changes nothing.

The geometry is stateless. The one stateful piece -- unwrapping the axis angle
across steps -- lives in exactly one term, :func:`spin`, which the reward
manager evaluates once per step.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from pan_dexterous_lab.assets.joints import DEFAULT_SIDE
from pan_dexterous_lab.assets.objects import BAODING_BALL_RADIUS_M

from ._geom import episode_start_ids, pair_geometry, per_env_buf
from .commands import spin_direction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_ANGLE_TOTAL = "_pan_baoding_angle_total"
_ANGLE_PREV = "_pan_baoding_angle_prev"
_DROP_FIRED = "_pan_baoding_drop_fired"


def spin(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    object2_cfg: SceneEntityCfg = SceneEntityCfg("object2"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    side: str = DEFAULT_SIDE,
    max_dtheta: float = 0.25,
) -> torch.Tensor:
    """Progress of the pair axis in the commanded direction, per step.

    Also the single owner of the unwrapped angle: it accumulates total rotation
    and publishes it as ``extras["baoding_revolutions"]``, which is the metric
    the task is actually judged on (one revolution = each ball back where it
    started = 2*pi of axis angle).

    Clamped both ways. The ceiling stops a flick or a bounce from paying out a
    one-step jackpot; the floor at zero leaves backspin unrewarded rather than
    punished, which keeps early exploration alive.
    """
    geom = pair_geometry(env, object_cfg, object2_cfg, robot_cfg, side)
    total = per_env_buf(env, _ANGLE_TOTAL)
    prev = per_env_buf(env, _ANGLE_PREV)

    reset_ids = episode_start_ids(env)
    if len(reset_ids) > 0:
        prev[reset_ids] = geom.doubled_angle[reset_ids]
        total[reset_ids] = 0.0

    # Shortest-arc difference, so the doubled angle unwraps across the +/-pi
    # branch cut. Valid while the axis turns under 90 deg per step; at the 60 Hz
    # policy rate that ceiling is ~5400 deg/s, far past anything physical.
    step_doubled = torch.atan2(
        torch.sin(geom.doubled_angle - prev), torch.cos(geom.doubled_angle - prev)
    )
    prev[:] = geom.doubled_angle
    delta_angle = 0.5 * step_doubled
    total += delta_angle

    env.extras["baoding_revolutions"] = total / (2.0 * math.pi)
    env.extras["drop_rate"] = geom.dropped.float()

    signed = spin_direction(env) * delta_angle
    return torch.clamp(signed, min=0.0, max=max_dtheta)


def ball_gap(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    object2_cfg: SceneEntityCfg = SceneEntityCfg("object2"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    side: str = DEFAULT_SIDE,
    ball_radius: float = BAODING_BALL_RADIUS_M,
    skin: float = 0.003,
) -> torch.Tensor:
    """How far the centre distance strays from the two balls resting in contact.

    Penalising this keeps the pair together, so the policy rotates them around
    each other instead of parking one and orbiting the other.
    """
    geom = pair_geometry(env, object_cfg, object2_cfg, robot_cfg, side)
    return torch.abs(geom.gap - (2.0 * ball_radius + skin))


def pair_centering(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    object2_cfg: SceneEntityCfg = SceneEntityCfg("object2"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    side: str = DEFAULT_SIDE,
) -> torch.Tensor:
    """In-plane distance from the pair midpoint to the centre of the cup.

    Measured against the cup rather than ``mid_offset`` (which is palm-frame,
    for the policy's benefit): the palm origin is at the wrist, so penalising
    distance from *it* would reward dragging the pair backwards off the hand.
    """
    geom = pair_geometry(env, object_cfg, object2_cfg, robot_cfg, side)
    mid = 0.5 * (geom.pos_a + geom.pos_b)
    return torch.linalg.norm((mid - geom.cup)[:, :2], dim=-1)


def balls_dropped(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    object2_cfg: SceneEntityCfg = SceneEntityCfg("object2"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    side: str = DEFAULT_SIDE,
) -> torch.Tensor:
    """Termination: at least one ball has left the cradle."""
    return pair_geometry(env, object_cfg, object2_cfg, robot_cfg, side).dropped


def drop_penalty(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    object2_cfg: SceneEntityCfg = SceneEntityCfg("object2"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    side: str = DEFAULT_SIDE,
) -> torch.Tensor:
    """1 on the first step a ball falls out, 0 afterwards.

    Fires once so the cost of dropping does not scale with however many steps
    the episode happens to keep running after the fact.
    """
    dropped = pair_geometry(env, object_cfg, object2_cfg, robot_cfg, side).dropped.float()
    already = per_env_buf(env, _DROP_FIRED)
    reset_ids = episode_start_ids(env)
    if len(reset_ids) > 0:
        already[reset_ids] = 0.0
    fresh = dropped * (1.0 - already)
    already[:] = torch.maximum(already, dropped)
    return fresh


def hold_pair(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    object2_cfg: SceneEntityCfg = SceneEntityCfg("object2"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    side: str = DEFAULT_SIDE,
) -> torch.Tensor:
    """1 while both balls stay in the cradle. The survival term."""
    return (~pair_geometry(env, object_cfg, object2_cfg, robot_cfg, side).dropped).float()
