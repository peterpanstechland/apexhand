"""Reset helpers for the coin-roll task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from pan_dexterous_lab.assets.joints import DEFAULT_SIDE
from pan_dexterous_lab.assets.objects import BAODING_BALL_RADIUS_M

from ._geom import (
    _as_torch,
    cached_lateral_axis,
    coin_and_robot,
    knuckle_surface_pos,
    palm_cup_pos,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_coin_on_knuckles(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    finger_indices: tuple[int, ...] = (0, 1),
    height: float = 0.006,
    jitter: float = 0.002,
    side: str = DEFAULT_SIDE,
) -> None:
    """Lay the coin flat across the backs of the selected proximal phalanges.

    Knuckle roll starts with the coin bridging the index and middle knuckles
    (``finger_indices=(0, 1)``), faces parallel to the back of the hand, so a
    finger lift flips it about the finger-pointing axis. Must run after the
    robot-joint reset so the knuckle frames are at the posed configuration.
    """
    if len(env_ids) == 0:
        return
    coin, robot = coin_and_robot(env, object_cfg, robot_cfg)
    points = knuckle_surface_pos(env, robot, finger_indices, side)[env_ids].mean(dim=1)
    pos = points.clone()
    pos[:, 2] += height
    if jitter > 0.0:
        pos += (torch.rand_like(pos) * 2.0 - 1.0) * jitter

    root_pose = _as_torch(coin.data.default_root_pose)[env_ids].clone()
    root_pose[:, 0:3] = pos
    # Identity: cylinder axis is local Z, so the coin lies flat with its faces
    # up and down and can roll about the finger axis.
    root_pose[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=pos.device, dtype=pos.dtype)
    vel = torch.zeros(len(env_ids), 6, device=pos.device, dtype=pos.dtype)
    coin.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
    coin.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=env_ids)


def reset_objects_in_palm(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    object2_cfg: SceneEntityCfg | None = None,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height: float = 0.018,
    pair_gap: float = 0.003,
    ball_radius: float = BAODING_BALL_RADIUS_M,
    jitter: float = 0.003,
    side: str = DEFAULT_SIDE,
) -> None:
    """Seat one or two objects in the palm cup (palm-up cradle).

    Anchored to the proximal-phalanx row, not to ``palm_link``. The palm frame's
    origin sits back at the wrist, inside a solid block: spawning there put the
    balls *inside* the collision mesh, and PhysX depenetrated them out sideways
    at exactly ``max_depenetration_velocity``, so every episode ended in six
    steps with the pair already gone. The knuckle row is the actual floor of the
    cup and is where the balls rest in the reference footage.

    ``pair_gap`` is the clearance between the two ball *surfaces* and ``height``
    is measured up from the knuckle bone axis, so the centre spacing follows
    from ``ball_radius`` and neither needs re-tuning when the balls change size.
    """
    if len(env_ids) == 0:
        return
    robot = env.scene[robot_cfg.name]
    cup = palm_cup_pos(env, robot, side)[env_ids]
    # Lateral axis measured from index -> pinky rather than assumed to be world
    # +Y, which only holds for the right hand.
    y_hat = cached_lateral_axis(env, robot, side)[env_ids]

    names = [object_cfg.name]
    if object2_cfg is not None:
        names.append(object2_cfg.name)
    elif "object2" in env.scene.keys():
        names.append("object2")

    n_obj = len(names)
    for i, name in enumerate(names):
        if name not in env.scene.keys():
            continue
        body = env.scene[name]
        pos = cup.clone()
        pos[:, 2] += height
        if n_obj == 2:
            sign = -1.0 if i == 0 else 1.0
            pos = pos + y_hat * sign * (ball_radius + pair_gap * 0.5)
        if jitter > 0.0:
            pos += (torch.rand_like(pos) * 2.0 - 1.0) * jitter
        root_pose = _as_torch(body.data.default_root_pose)[env_ids].clone()
        root_pose[:, 0:3] = pos
        root_pose[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=pos.device, dtype=pos.dtype)
        vel = torch.zeros(len(env_ids), 6, device=pos.device, dtype=pos.dtype)
        body.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        body.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=env_ids)


def randomize_lighting(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    intensity_range: tuple[float, float] = (400.0, 1400.0),
    color_lo: tuple[float, float, float] = (0.62, 0.64, 0.70),
    color_hi: tuple[float, float, float] = (1.0, 0.98, 0.92),
    prim_path: str = "/World/domeLight",
) -> None:
    """Jitter dome-light intensity and color. Shared across envs (one USD prim)."""
    del env_ids
    try:
        import omni.usd
        from pxr import Gf
    except ImportError:
        return
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    lo, hi = intensity_range
    intensity = float(lo + (hi - lo) * torch.rand(1).item())
    color = [
        float(color_lo[i] + (color_hi[i] - color_lo[i]) * torch.rand(1).item())
        for i in range(3)
    ]
    for name, value in (
        ("inputs:intensity", intensity),
        ("inputs:color", Gf.Vec3f(*color)),
    ):
        attr = prim.GetAttribute(name)
        if attr:
            attr.Set(value)


def randomize_camera_offset(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    sensor_name: str = "overhead_camera",
    pos_jitter_m: float = 0.005,
    rot_jitter_deg: float = 3.0,
) -> None:
    """Nudge a camera's world pose. No-op when the sensor is not in the scene."""
    sensors = getattr(env.scene, "sensors", {})
    cam = sensors.get(sensor_name) if isinstance(sensors, dict) else None
    if cam is None:
        try:
            cam = env.scene[sensor_name]
        except Exception:
            return
    if not hasattr(cam, "data"):
        return
    pos = _as_torch(cam.data.pos_w)[env_ids]
    quat = _as_torch(cam.data.quat_w_world)[env_ids] if hasattr(cam.data, "quat_w_world") else None
    if quat is None and hasattr(cam.data, "quat_w_ros"):
        quat = _as_torch(cam.data.quat_w_ros)[env_ids]
    if quat is None:
        return
    pos = pos + (torch.rand_like(pos) * 2.0 - 1.0) * pos_jitter_m
    # Small random yaw/pitch via adding noise to the quaternion vector part, then renormalize.
    noise = (torch.rand_like(quat) * 2.0 - 1.0) * (rot_jitter_deg / 180.0 * 0.15)
    noise[:, 0] = 0.0
    quat = torch.nn.functional.normalize(quat + noise, dim=-1)
    if hasattr(cam, "set_world_poses"):
        cam.set_world_poses(pos, quat, env_ids=env_ids)
