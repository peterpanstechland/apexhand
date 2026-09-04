"""Rewards for two-ball baoding rotation in a palm-up cradle."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from pan_dexterous_lab.assets.joints import DEFAULT_SIDE

from ._geom import _as_torch, maybe_reset_buf, resolve_palm_id

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _pair_and_palm(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg,
    object2_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    side: str,
):
    a = env.scene[object_cfg.name]
    b = env.scene[object2_cfg.name]
    robot = env.scene[robot_cfg.name]
    palm = _as_torch(robot.data.body_pos_w)[:, resolve_palm_id(robot, side)]
    pa = _as_torch(a.data.root_pos_w)
    pb = _as_torch(b.data.root_pos_w)
    return a, b, robot, palm, pa, pb


def spin(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    object2_cfg: SceneEntityCfg = SceneEntityCfg("object2"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    side: str = DEFAULT_SIDE,
    max_dtheta: float = 0.25,
) -> torch.Tensor:
    """Positive yaw increment of the pair vector around world +Z."""
    _, _, _, palm, pa, pb = _pair_and_palm(env, object_cfg, object2_cfg, robot_cfg, side)
    vec = pb - pa
    yaw = torch.atan2(vec[:, 1], vec[:, 0])
    prev = maybe_reset_buf(env, "_pan_baoding_yaw")
    reset_ids = (env.episode_length_buf <= 1).nonzero(as_tuple=False).squeeze(-1)
    if len(reset_ids) > 0:
        prev[reset_ids] = yaw[reset_ids]
    dtheta = torch.atan2(torch.sin(yaw - prev), torch.cos(yaw - prev))
    prev[:] = yaw
    env.extras["baoding_yaw"] = yaw
    return torch.clamp(dtheta, min=0.0, max=max_dtheta)


def ball_gap(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    object2_cfg: SceneEntityCfg = SceneEntityCfg("object2"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    side: str = DEFAULT_SIDE,
    target_gap: float = 0.048,
) -> torch.Tensor:
    """|‖p_b − p_a‖ − target|. Default target ≈ two 45 mm radii + 3 mm."""
    _, _, _, _, pa, pb = _pair_and_palm(env, object_cfg, object2_cfg, robot_cfg, side)
    dist = torch.linalg.norm(pb - pa, dim=-1)
    return torch.abs(dist - target_gap)


def pair_centering(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    object2_cfg: SceneEntityCfg = SceneEntityCfg("object2"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    side: str = DEFAULT_SIDE,
) -> torch.Tensor:
    """Horizontal distance from the pair midpoint to the palm."""
    _, _, _, palm, pa, pb = _pair_and_palm(env, object_cfg, object2_cfg, robot_cfg, side)
    mid = 0.5 * (pa + pb)
    delta = mid - palm
    return torch.linalg.norm(delta[:, :2], dim=-1)


def drop_penalty(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    object2_cfg: SceneEntityCfg = SceneEntityCfg("object2"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    side: str = DEFAULT_SIDE,
    drop_offset: float = 0.06,
) -> torch.Tensor:
    """1 on the first step a ball falls well below the palm."""
    dropped = balls_dropped(
        env,
        object_cfg=object_cfg,
        object2_cfg=object2_cfg,
        robot_cfg=robot_cfg,
        side=side,
        drop_offset=drop_offset,
    )
    already = maybe_reset_buf(env, "_pan_baoding_drop_fired")
    fresh = dropped.float() * (1.0 - already)
    already[:] = torch.maximum(already, dropped.float())
    return fresh


def balls_dropped(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    object2_cfg: SceneEntityCfg = SceneEntityCfg("object2"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    side: str = DEFAULT_SIDE,
    drop_offset: float = 0.06,
) -> torch.Tensor:
    _, _, _, palm, pa, pb = _pair_and_palm(env, object_cfg, object2_cfg, robot_cfg, side)
    low_a = pa[:, 2] < (palm[:, 2] - drop_offset)
    low_b = pb[:, 2] < (palm[:, 2] - drop_offset)
    far_a = torch.linalg.norm(pa - palm, dim=-1) > 0.14
    far_b = torch.linalg.norm(pb - palm, dim=-1) > 0.14
    dropped = low_a | low_b | far_a | far_b
    env.extras["drop_rate"] = dropped.float()
    return dropped


def hold_pair(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    object2_cfg: SceneEntityCfg = SceneEntityCfg("object2"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    side: str = DEFAULT_SIDE,
) -> torch.Tensor:
    """1 while both balls stay in the cradle volume."""
    return (~balls_dropped(env, object_cfg, object2_cfg, robot_cfg, side)).float()
