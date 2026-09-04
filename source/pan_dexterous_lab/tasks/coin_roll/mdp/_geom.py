"""Shared geometry helpers for coin-roll rewards / observations / terminations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from pan_dexterous_lab.assets.joints import (
    DEFAULT_SIDE,
    knuckle_distal_body_names,
    knuckle_proximal_body_names,
    pad_body_names,
    palm_body_name,
)

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv

_PAD_CACHE = "_pan_pad_ids"
_PALM_CACHE = "_pan_palm_id"
_KNUCKLE_CACHE = "_pan_knuckle_ids"

# Measured with scripts/debug_spawn.py --drop: dropped from 30 mm the coin settles
# with its center 13 mm above the link1/link2 bone axis, centred between the two
# knuckles it bridges. Treat that as the seated height, so distance-to-knuckle is
# ~0 when the coin is properly seated.
KNUCKLE_SEAT_HEIGHT = 0.013


def _as_torch(x):
    return x.torch if hasattr(x, "torch") else x


def resolve_pad_ids(robot: "Articulation", side: str = DEFAULT_SIDE) -> list[int]:
    ids, _ = robot.find_bodies(pad_body_names(side), preserve_order=True)
    return ids


def resolve_palm_id(robot: "Articulation", side: str = DEFAULT_SIDE) -> int:
    ids, _ = robot.find_bodies([palm_body_name(side)], preserve_order=True)
    return ids[0]


def cached_body_ids(env: "ManagerBasedRLEnv", robot: "Articulation", side: str = DEFAULT_SIDE):
    if not hasattr(env, _PAD_CACHE):
        setattr(env, _PAD_CACHE, resolve_pad_ids(robot, side))
        setattr(env, _PALM_CACHE, resolve_palm_id(robot, side))
    return getattr(env, _PAD_CACHE), getattr(env, _PALM_CACHE)


def cached_knuckle_ids(env: "ManagerBasedRLEnv", robot: "Articulation", side: str = DEFAULT_SIDE):
    """Body ids of (proximal, distal) knuckle frames per finger, index -> pinky."""
    if not hasattr(env, _KNUCKLE_CACHE):
        prox, _ = robot.find_bodies(knuckle_proximal_body_names(side), preserve_order=True)
        dist, _ = robot.find_bodies(knuckle_distal_body_names(side), preserve_order=True)
        setattr(env, _KNUCKLE_CACHE, (prox, dist))
    return getattr(env, _KNUCKLE_CACHE)


def knuckle_surface_pos(
    env: "ManagerBasedRLEnv",
    robot: "Articulation",
    finger_indices: tuple[int, ...] = (0, 1, 2, 3),
    side: str = DEFAULT_SIDE,
    surface_offset: float = KNUCKLE_SEAT_HEIGHT,
) -> torch.Tensor:
    """(N, k, 3) seated coin center above the selected proximal phalanges.

    Midpoint of the MCP (``link1``) and PIP (``link2``) frames, raised along world
    +Z. The hand is fixed palm-down, so +Z *is* the dorsal direction; the finger
    link frames are rotated relative to the palm (their local -X points along
    world +Y), so a body-local offset would land in the groove between fingers
    instead of on top of the knuckle.
    """
    prox_ids, dist_ids = cached_knuckle_ids(env, robot, side)
    body_pos = _as_torch(robot.data.body_pos_w)
    points = []
    for i in finger_indices:
        mid = 0.5 * (body_pos[:, prox_ids[i]] + body_pos[:, dist_ids[i]])
        points.append(mid + _unit_z(mid) * surface_offset)
    return torch.stack(points, dim=1)


def knuckle_distances(
    env: "ManagerBasedRLEnv",
    coin: "RigidObject",
    robot: "Articulation",
    side: str = DEFAULT_SIDE,
    surface_offset: float = KNUCKLE_SEAT_HEIGHT,
) -> torch.Tensor:
    """(N, 4) distance from the coin center to each finger's knuckle contact point."""
    points = knuckle_surface_pos(env, robot, (0, 1, 2, 3), side, surface_offset)
    coin_pos = _as_torch(coin.data.root_pos_w).unsqueeze(1)
    return torch.linalg.norm(coin_pos - points, dim=-1)


def coin_and_robot(env: "ManagerBasedRLEnv", object_cfg: SceneEntityCfg, robot_cfg: SceneEntityCfg):
    coin: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    return coin, robot


def palm_axes(env: "ManagerBasedRLEnv", robot: "Articulation", side: str = DEFAULT_SIDE):
    """Return (origin, x_hat, y_hat, z_hat) for the knuckle-roll frame, in world.

    origin: index knuckle seat, i.e. phase 0 of the index -> pinky roll
    y_hat:  index knuckle → pinky knuckle (coin travel)
    z_hat:  world +Z, the dorsal normal of the fixed palm-down hand
    x_hat:  y × z, the finger-pointing axis the coin rolls about. Rolling toward
            the pinky needs omega·x_hat > 0 (contact is below the coin center,
            so v_y = omega_x * R).

    The finger link frames are *not* aligned with the palm, so the previous
    version's ``quat_apply(palm_quat, +Z)`` did not give the dorsal normal.
    """
    seats = knuckle_surface_pos(env, robot, (0, 3), side)
    origin = seats[:, 0]
    y = seats[:, 1] - origin
    y_hat = torch.nn.functional.normalize(y, dim=-1)
    z_hat = _unit_z(origin)
    x_hat = torch.nn.functional.normalize(torch.cross(y_hat, z_hat, dim=-1), dim=-1)
    # re-orthogonalize so the triad stays right-handed if the knuckles tilt
    z_hat = torch.nn.functional.normalize(torch.cross(x_hat, y_hat, dim=-1), dim=-1)
    return origin, x_hat, y_hat, z_hat


def _unit_x(ref: torch.Tensor) -> torch.Tensor:
    x = torch.zeros_like(ref)
    x[:, 0] = 1.0
    return x


def _unit_z(ref: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(ref)
    z[:, 2] = 1.0
    return z


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v by quaternions q (wxyz)."""
    w = q[:, 0:1]
    xyz = q[:, 1:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def pad_mean_pos(
    env: "ManagerBasedRLEnv",
    robot: "Articulation",
    pad_indices: tuple[int, ...] = (1, 2),
    side: str = DEFAULT_SIDE,
) -> torch.Tensor:
    """Mean world position of the selected pads (index/middle by default)."""
    pad_ids, _ = cached_body_ids(env, robot, side)
    sel = [pad_ids[i] for i in pad_indices]
    return _as_torch(robot.data.body_pos_w)[:, sel].mean(dim=1)


def pad_distances(
    env: "ManagerBasedRLEnv",
    coin: "RigidObject",
    robot: "Articulation",
    side: str = DEFAULT_SIDE,
) -> torch.Tensor:
    """(N, 5) Euclidean distance from coin center to thumb/index/middle/ring/pinky pads."""
    pad_ids, _ = cached_body_ids(env, robot, side)
    pads = _as_torch(robot.data.body_pos_w)[:, pad_ids]
    coin_pos = _as_torch(coin.data.root_pos_w).unsqueeze(1)
    return torch.linalg.norm(coin_pos - pads, dim=-1)


def maybe_reset_buf(env: "ManagerBasedRLEnv", name: str, dim: int = 1) -> torch.Tensor:
    """Allocate or zero a per-env buffer on episode reset."""
    buf = getattr(env, name, None)
    if buf is None:
        shape = (env.num_envs, dim) if dim > 1 else (env.num_envs,)
        buf = torch.zeros(shape, device=env.device)
        setattr(env, name, buf)
    # After env.step increments the counter, the first policy step of an episode is 1.
    reset_ids = (env.episode_length_buf <= 1).nonzero(as_tuple=False).squeeze(-1)
    if len(reset_ids) > 0:
        buf[reset_ids] = 0.0
    return buf
