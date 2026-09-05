"""Stage A (hold) and Stage B (index→middle transfer) reward terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from pan_dexterous_lab.assets.apex_cfg import COIN_RADIUS
from pan_dexterous_lab.assets.joints import DEFAULT_SIDE

from ._geom import (
    _as_torch,
    cached_knuckle_ids,
    cached_lateral_axis,
    coin_and_robot,
    knuckle_distances,
    knuckle_surface_pos,
    maybe_reset_buf,
    palm_axes,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def coin_knuckle_distance(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    finger_indices: tuple[int, ...] = (0, 1),
    side: str = DEFAULT_SIDE,
) -> torch.Tensor:
    """L2 from the coin center to the mean seat of the bridged knuckles.

    ~0 when the coin lies flat across the two knuckles; grows if it slides into
    the groove between fingers, drifts along a finger, or lifts off.
    """
    coin, robot = coin_and_robot(env, object_cfg, robot_cfg)
    seats = knuckle_surface_pos(env, robot, finger_indices, side).mean(dim=1)
    coin_pos = _as_torch(coin.data.root_pos_w)
    return torch.linalg.norm(coin_pos - seats, dim=-1)


def coin_bridge_distance(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    bridges: tuple[tuple[int, int], ...] = ((0, 1), (1, 2)),
    side: str = DEFAULT_SIDE,
) -> torch.Tensor:
    """Min L2 to any listed knuckle bridge — Stage B seats migrate index→middle."""
    coin, robot = coin_and_robot(env, object_cfg, robot_cfg)
    coin_pos = _as_torch(coin.data.root_pos_w)
    dists = []
    for pair in bridges:
        seats = knuckle_surface_pos(env, robot, pair, side).mean(dim=1)
        dists.append(torch.linalg.norm(coin_pos - seats, dim=-1))
    return torch.minimum(dists[0], dists[1]) if len(dists) == 2 else torch.stack(dists).min(dim=0).values


def coin_bridge_seat_offset(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    bridges: tuple[tuple[int, int], ...] = ((0, 1), (1, 2)),
    side: str = DEFAULT_SIDE,
) -> torch.Tensor:
    """|height| above the nearest bridge seat plane."""
    coin, robot = coin_and_robot(env, object_cfg, robot_cfg)
    z = _as_torch(coin.data.root_pos_w)[:, 2]
    offs = []
    for pair in bridges:
        seats = knuckle_surface_pos(env, robot, pair, side).mean(dim=1)
        offs.append(torch.clamp((z - seats[:, 2]).abs(), 0.0, 0.06))
    stacked = torch.stack(offs, dim=0)
    return stacked.min(dim=0).values


def coin_seat_offset(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    finger_indices: tuple[int, ...] = (0, 1),
    side: str = DEFAULT_SIDE,
) -> torch.Tensor:
    """|height above the knuckle seat plane|, clipped. 0 when properly seated.

    Palm-down means being *above* the seat is as bad as being below: the coin is
    either bouncing or has been flicked off the back of the hand.
    """
    coin, robot = coin_and_robot(env, object_cfg, robot_cfg)
    seats = knuckle_surface_pos(env, robot, finger_indices, side).mean(dim=1)
    height = _as_torch(coin.data.root_pos_w)[:, 2] - seats[:, 2]
    return torch.clamp(height.abs(), 0.0, 0.06)


def hold_bonus(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    side: str = DEFAULT_SIDE,
) -> torch.Tensor:
    """1.0 for every step the coin is seated on the knuckles — the Stage A goal.

    Shares its predicate with the ``hold_success`` termination, so the reward the
    policy chases and the metric Gate 4 reads are the same condition.
    """
    from .terminations import hold_ok

    return hold_ok(env, object_cfg=object_cfg, robot_cfg=robot_cfg, side=side).float()


def desired_contact(
    env: "ManagerBasedRLEnv",
    threshold: float = 0.022,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    finger_indices: tuple[int, ...] = (0, 1),
    side: str = DEFAULT_SIDE,
) -> torch.Tensor:
    """Fraction of the required knuckles within ``threshold`` of the coin.

    Instanceable Apex USD cannot take a PhysX ContactReporter API at spawn time,
    so contact is a distance proxy. A coin bridging two knuckles measures ~11-13
    mm to each seat (half the 23 mm finger pitch), hence the 22 mm threshold.
    """
    coin, robot = coin_and_robot(env, object_cfg, robot_cfg)
    dist = knuckle_distances(env, coin, robot, side)
    hits = (dist[:, list(finger_indices)] < threshold).float()
    return hits.mean(dim=-1)


def finger_crossing(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    safe_gap: float = 0.018,
    side: str = DEFAULT_SIDE,
) -> torch.Tensor:
    """Total lateral squeeze of adjacent fingers below ``safe_gap``, 0 when apart.

    Self-collision is disabled (the URDF's tactile shells overlap at rest and
    PhysX diverges), so nothing physically stops adjacent fingers from passing
    through each other. Abduction authority is clamped in the action term, which
    should make this impossible; this term is the guard that makes it visible in
    the reward log if it ever happens again.

    Lateral is measured along the live index -> pinky axis rather than world Y,
    so the term is valid for either hand.
    """
    robot = env.scene[robot_cfg.name]
    _, link2_ids = cached_knuckle_ids(env, robot, side)
    pos = _as_torch(robot.data.body_pos_w)[:, link2_ids]
    axis = cached_lateral_axis(env, robot, side).unsqueeze(1)
    lat = (pos * axis).sum(dim=-1)
    gaps = lat[:, 1:] - lat[:, :-1]
    return torch.clamp(safe_gap - gaps, min=0.0).sum(dim=-1)


def drop_penalty(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    side: str = DEFAULT_SIDE,
) -> torch.Tensor:
    """1.0 on the step the coin falls off the knuckles.

    Deliberately takes no threshold arguments: ``coin_dropped`` caches per step
    because reward and termination both call it, so passing different thresholds
    here would silently be ignored depending on call order.
    """
    from .terminations import coin_dropped

    dropped = coin_dropped(env, object_cfg=object_cfg, robot_cfg=robot_cfg, side=side)
    already = maybe_reset_buf(env, "_pan_drop_fired")
    fresh = dropped.float() * (1.0 - already)
    already[:] = torch.maximum(already, dropped.float())
    return fresh


def progress(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    side: str = DEFAULT_SIDE,
    target_phase: float = 0.55,
) -> torch.Tensor:
    """Forward phase increment plus dense pull toward the middle→ring seat (~0.55).

    Phase 0 is the index knuckle and 1 the pinky knuckle. Index/middle spawn sits
    near ~0.17; middle/ring success lives near 0.45–0.72. Pure Δφ was too weak
    next to hold/spin rewards, so add ``1 − |φ − target|`` as a dense cue.
    """
    coin, robot = coin_and_robot(env, object_cfg, robot_cfg)
    origin, _, y_hat, _ = palm_axes(env, robot, side)
    seats = knuckle_surface_pos(env, robot, (0, 3), side)
    span = torch.linalg.norm(seats[:, 1] - seats[:, 0], dim=-1).clamp(min=1e-3)
    proj = ((_as_torch(coin.data.root_pos_w) - origin) * y_hat).sum(dim=-1) / span
    phase = torch.clamp(proj, 0.0, 1.0)
    prev = maybe_reset_buf(env, "_pan_phase")
    # seed previous phase on reset so the first step is not a huge spike
    reset_ids = (env.episode_length_buf <= 1).nonzero(as_tuple=False).squeeze(-1)
    if len(reset_ids) > 0:
        prev[reset_ids] = phase[reset_ids]
    delta = phase - prev
    prev[:] = phase
    env.extras["phase_reached"] = phase
    toward = 1.0 - torch.clamp((phase - target_phase).abs() / max(target_phase, 1e-3), 0.0, 1.0)
    # Δφ only counts when moving forward; dense term keeps the signal alive at rest.
    return torch.clamp(delta, min=0.0) * 5.0 + toward


def roll_rotation(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    side: str = DEFAULT_SIDE,
    max_omega: float = 1.0,
) -> torch.Tensor:
    """Forward roll about the finger axis, only when the coin also advances in phase.

    Uncapped ω was ~6 rad/s per step and dominated Stage B, so the policy farmed
    in-place spin with ``progress≈0``. Cap and gate by positive lateral travel.
    """
    coin, robot = coin_and_robot(env, object_cfg, robot_cfg)
    origin, x_hat, y_hat, _ = palm_axes(env, robot, side)
    omega = _as_torch(coin.data.root_ang_vel_w)
    roll = torch.clamp((omega * x_hat).sum(dim=-1), min=0.0, max=max_omega)
    s_y = ((_as_torch(coin.data.root_pos_w) - origin) * y_hat).sum(dim=-1)
    prev_s = maybe_reset_buf(env, "_pan_roll_s")
    reset_ids = (env.episode_length_buf <= 1).nonzero(as_tuple=False).squeeze(-1)
    if len(reset_ids) > 0:
        prev_s[reset_ids] = s_y[reset_ids]
    ds = s_y - prev_s
    prev_s[:] = s_y
    return roll * (ds > 1e-5).float()


def slip_penalty(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    radius: float = COIN_RADIUS,
    side: str = DEFAULT_SIDE,
) -> torch.Tensor:
    """|Δs_y − R Δθ_x| — penalize translating the coin without rolling it."""
    coin, robot = coin_and_robot(env, object_cfg, robot_cfg)
    origin, x_hat, y_hat, _ = palm_axes(env, robot, side)
    coin_pos = _as_torch(coin.data.root_pos_w)
    s_y = ((coin_pos - origin) * y_hat).sum(dim=-1)
    omega = _as_torch(coin.data.root_ang_vel_w)
    dt = env.step_dt
    dtheta = (omega * x_hat).sum(dim=-1) * dt
    prev_s = maybe_reset_buf(env, "_pan_coin_s")
    reset_ids = (env.episode_length_buf <= 1).nonzero(as_tuple=False).squeeze(-1)
    if len(reset_ids) > 0:
        prev_s[reset_ids] = s_y[reset_ids]
    ds = s_y - prev_s
    prev_s[:] = s_y
    return torch.abs(ds - radius * dtheta)


def success_bonus(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    hold_steps: int = 8,
    phase_low: float = 0.45,
    phase_high: float = 0.72,
    rot_threshold: float = 1.57,
    contact_threshold: float = 0.022,
    target_fingers: tuple[int, ...] = (1, 2),
    side: str = DEFAULT_SIDE,
) -> torch.Tensor:
    """1.0 after ``hold_steps`` consecutive frames that meet transfer success.

    Index->middle transfer means the coin ends up bridging the middle and ring
    knuckles: phase 1/3 to 2/3 of the index->pinky span, still seated, having
    accumulated >= ``rot_threshold`` of forward roll.
    """
    coin, robot = coin_and_robot(env, object_cfg, robot_cfg)
    origin, x_hat, y_hat, _ = palm_axes(env, robot, side)
    seats = knuckle_surface_pos(env, robot, (0, 3), side)
    span = torch.linalg.norm(seats[:, 1] - seats[:, 0], dim=-1).clamp(min=1e-3)
    coin_pos = _as_torch(coin.data.root_pos_w)
    phase = torch.clamp(((coin_pos - origin) * y_hat).sum(dim=-1) / span, 0.0, 1.0)
    seat_ref = knuckle_surface_pos(env, robot, target_fingers, side).mean(dim=1)
    height = coin_pos[:, 2] - seat_ref[:, 2]
    in_zone = (phase >= phase_low) & (phase <= phase_high) & (height > -0.02) & (height < 0.03)

    rot_acc = maybe_reset_buf(env, "_pan_rot_acc")
    omega = _as_torch(coin.data.root_ang_vel_w)
    rot_acc[:] = rot_acc + torch.clamp((omega * x_hat).sum(dim=-1), min=0.0) * env.step_dt
    rolled = rot_acc >= rot_threshold

    dist = knuckle_distances(env, coin, robot, side)
    contacted = dist[:, list(target_fingers)].max(dim=-1).values < contact_threshold

    ok = in_zone & rolled & contacted
    streak = maybe_reset_buf(env, "_pan_success_streak")
    streak[:] = torch.where(ok, streak + 1.0, torch.zeros_like(streak))
    done = (streak >= hold_steps).float()
    already = maybe_reset_buf(env, "_pan_success_fired")
    fresh = done * (1.0 - already)
    already[:] = torch.maximum(already, done)
    env.extras["success_rate"] = done
    return fresh
