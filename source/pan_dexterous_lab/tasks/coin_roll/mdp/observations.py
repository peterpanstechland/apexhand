"""Extra observations beyond the stock in-hand kinematic set."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from pan_dexterous_lab.assets.joints import DEFAULT_SIDE, tip_body_names

from ._geom import (
    _as_torch,
    coin_and_robot,
    episode_start_ids,
    knuckle_surface_pos,
    pair_geometry,
    per_env_buf,
)
from .commands import spin_direction

_OBS_ANGLE_PREV = "_pan_baoding_obs_angle_prev"

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def fingertip_pos_w(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    side: str = DEFAULT_SIDE,
) -> torch.Tensor:
    """World-frame positions of the five fingertip frames, flattened (N, 15)."""
    robot = env.scene[robot_cfg.name]
    ids, _ = robot.find_bodies(tip_body_names(side), preserve_order=True)
    pos = _as_torch(robot.data.body_pos_w)[:, ids]
    return pos.reshape(env.num_envs, -1)


def coin_to_knuckle_rel_pos(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    side: str = DEFAULT_SIDE,
) -> torch.Tensor:
    """Coin center minus each knuckle seat, flattened (N, 12).

    The knuckle roll happens on the backs of the fingers, so the pad-relative
    version told the policy nothing about where the coin sits.
    """
    coin, robot = coin_and_robot(env, object_cfg, robot_cfg)
    seats = knuckle_surface_pos(env, robot, (0, 1, 2, 3), side)
    coin_pos = _as_torch(coin.data.root_pos_w).unsqueeze(1)
    return (coin_pos - seats).reshape(env.num_envs, -1)


def baoding_pair_obs(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    object2_cfg: SceneEntityCfg = SceneEntityCfg("object2"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    side: str = DEFAULT_SIDE,
) -> torch.Tensor:
    """(N, 6) ball-pair state restricted to what a real camera can recover.

    This is the *only* object information the deployed actor sees, so every
    channel has to be reproducible by ``real/ball_tracker.py`` from two blobs in
    an image plus a one-time palm calibration. Order matches that module:

    0:2  pair midpoint relative to the palm centre, in the palm plane
    2:4  cos / sin of the doubled axis angle
    4    centre-to-centre gap
    5    doubled axis angle rate, rad/s

    Deliberately absent:

    * *Ball identity.* The two real balls are the same turned wood and the
      tracker cannot label them, so the axis angle is doubled -- swapping the
      balls adds pi, which the doubling absorbs.
    * *Height above the palm.* One fixed camera cannot measure it. A 30 mm ball
      at ~30 cm changes apparent radius by well under a pixel over the few
      millimetres of travel available, so a z channel would be noise in sim and
      a fabrication on hardware.
    * *Absolute world position.* Everything is palm-relative, so the policy
      never has to be told where the hand is bolted.
    """
    geom = pair_geometry(env, object_cfg, object2_cfg, robot_cfg, side)

    prev = per_env_buf(env, _OBS_ANGLE_PREV)
    reset_ids = episode_start_ids(env)
    if len(reset_ids) > 0:
        prev[reset_ids] = geom.doubled_angle[reset_ids]
    step = torch.atan2(
        torch.sin(geom.doubled_angle - prev), torch.cos(geom.doubled_angle - prev)
    )
    prev[:] = geom.doubled_angle
    rate = step / env.step_dt

    return torch.cat(
        [
            geom.mid_offset[:, :2],
            torch.cos(geom.doubled_angle).unsqueeze(-1),
            torch.sin(geom.doubled_angle).unsqueeze(-1),
            geom.gap.unsqueeze(-1),
            rate.unsqueeze(-1),
        ],
        dim=-1,
    )


def baoding_spin_command(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """(N, 1) commanded rotation direction, the same signal ``spin`` scores."""
    return spin_direction(env).unsqueeze(-1)


def object2_pos_w(
    env: "ManagerBasedRLEnv",
    object2_cfg: SceneEntityCfg = SceneEntityCfg("object2"),
) -> torch.Tensor:
    """World position of the second object (baoding pair). Zeros if missing."""
    if object2_cfg.name not in env.scene.keys():
        return torch.zeros(env.num_envs, 3, device=env.device)
    body = env.scene[object2_cfg.name]
    return _as_torch(body.data.root_pos_w)


def object2_lin_vel_w(
    env: "ManagerBasedRLEnv",
    object2_cfg: SceneEntityCfg = SceneEntityCfg("object2"),
) -> torch.Tensor:
    if object2_cfg.name not in env.scene.keys():
        return torch.zeros(env.num_envs, 3, device=env.device)
    body = env.scene[object2_cfg.name]
    return _as_torch(body.data.root_lin_vel_w)
