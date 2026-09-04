"""Extra observations beyond the stock in-hand kinematic set."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from pan_dexterous_lab.assets.joints import DEFAULT_SIDE, tip_body_names

from ._geom import _as_torch, coin_and_robot, knuckle_surface_pos

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
