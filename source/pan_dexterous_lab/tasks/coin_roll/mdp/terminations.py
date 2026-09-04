"""Drop and hold-success terminations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from pan_dexterous_lab.assets.joints import DEFAULT_SIDE

from ._geom import _as_torch, coin_and_robot, knuckle_distances, knuckle_surface_pos, maybe_reset_buf

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def coin_dropped(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    drop_offset: float = 0.035,
    no_contact_s: float = 0.20,
    contact_threshold: float = 0.030,
    side: str = DEFAULT_SIDE,
) -> torch.Tensor:
    """True once the coin has fallen off the back of the hand.

    Reference is the knuckle seat plane, not the fingertip pads: palm-down, the
    pads are ~55 mm *below* the seated coin, so a pad-relative height would only
    trip long after the coin was gone.
    """
    # Reward and termination both call this in one step; cache so air-time is not double-counted.
    step = int(getattr(env, "common_step_counter", -1))
    if getattr(env, "_pan_drop_cache_step", None) == step:
        return env._pan_drop_cache

    coin, robot = coin_and_robot(env, object_cfg, robot_cfg)
    seats = knuckle_surface_pos(env, robot, (0, 1, 2, 3), side)
    coin_pos = _as_torch(coin.data.root_pos_w)
    height = coin_pos[:, 2] - seats[:, :, 2].mean(dim=-1)
    below = height < -drop_offset

    lin_vel = _as_torch(coin.data.root_lin_vel_w)
    falling = lin_vel[:, 2] < -0.15

    dist = knuckle_distances(env, coin, robot, side)
    any_contact = dist.min(dim=-1).values < contact_threshold
    air = maybe_reset_buf(env, "_pan_air_time")
    air[:] = torch.where(any_contact, torch.zeros_like(air), air + env.step_dt)
    lost = (~any_contact) & falling & (air > no_contact_s)

    dropped = below | lost
    env.extras["drop_rate"] = dropped.float()
    env._pan_drop_cache = dropped
    env._pan_drop_cache_step = step
    return dropped


def hold_ok(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_speed: float = 0.45,
    contact_threshold: float = 0.022,
    finger_indices: tuple[int, ...] = (0, 1),
    side: str = DEFAULT_SIDE,
) -> torch.Tensor:
    """Per-step hold mask: coin still seated on the knuckles and not falling.

    Mean distance to the bridged knuckle pair, so a coin sitting slightly
    off-center still counts. Requiring both knuckles every frame zeroed a long
    streak on one-frame PhysX jitter.
    """
    coin, robot = coin_and_robot(env, object_cfg, robot_cfg)
    seats = knuckle_surface_pos(env, robot, finger_indices, side).mean(dim=1)
    height = _as_torch(coin.data.root_pos_w)[:, 2] - seats[:, 2]
    speed = torch.linalg.norm(_as_torch(coin.data.root_lin_vel_w), dim=-1)
    dist = knuckle_distances(env, coin, robot, side)
    near = dist[:, list(finger_indices)].mean(dim=-1) < contact_threshold
    return near & (height > -0.020) & (height < 0.025) & (speed < max_speed)


def hold_success(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    hold_steps: int = 150,
    stable_steps: int = 30,
    max_speed: float = 0.45,
    contact_threshold: float = 0.022,
    finger_indices: tuple[int, ...] = (0, 1),
    side: str = DEFAULT_SIDE,
) -> torch.Tensor:
    """True when the coin has been held for ``hold_steps`` and is briefly stable.

    Gate 4 is “held ~3 s without dropping”. A 3 s episode is only 180 policy
    steps, so requiring 150 *consecutive* perfect frames is nearly impossible:
    one speed spike resets the streak. Instead we require the episode to have
    lasted ``hold_steps`` (~2.5 s) and the last ``stable_steps`` (~0.5 s) to
    be in the hold band.
    """
    ok = hold_ok(
        env,
        object_cfg=object_cfg,
        robot_cfg=robot_cfg,
        max_speed=max_speed,
        contact_threshold=contact_threshold,
        finger_indices=finger_indices,
        side=side,
    )
    streak = maybe_reset_buf(env, "_pan_hold_streak")
    streak[:] = torch.where(ok, streak + 1.0, torch.zeros_like(streak))
    lasted = env.episode_length_buf >= hold_steps
    done = lasted & (streak >= stable_steps)
    env.extras["success_rate"] = done.float()
    env.extras["hold_ok"] = ok.float()
    return done
