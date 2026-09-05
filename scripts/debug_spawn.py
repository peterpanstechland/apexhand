#!/usr/bin/env python3
"""Preview the spawn pose: print object / knuckle / pad geometry, optionally record.

Drives a constant action that holds ``init_state.joint_pos`` instead of the
policy, so the rendered pose is the calibration pose rather than whatever a
checkpoint does. Use this to verify the object actually rests where the reward
assumes before touching reward weights.

Reports knuckle-relative geometry for the single-object coin stages and
palm-relative pair geometry when the scene has an ``object2`` (baoding).
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Preview Apex coin spawn pose.")
parser.add_argument("--task", type=str, default="PAN-CoinHold-Apex-Play-v0")
parser.add_argument("--steps", type=int, default=180)
parser.add_argument("--action", choices=["init", "zero"], default="init")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=180)
parser.add_argument(
    "--drop", type=float, default=None, help="Override the coin reset height above the knuckle surface (m)."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

import gymnasium as gym
import torch

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import pan_dexterous_lab.tasks  # noqa: F401
from pan_dexterous_lab.assets.joints import FINGERS, actuated_joint_names, pad_body_names
from pan_dexterous_lab.tasks.coin_roll.hand_side import detect_hand_side
from pan_dexterous_lab.tasks.coin_roll.mdp._geom import (
    DROP_DEPTH_M,
    DROP_RADIUS_M,
    knuckle_surface_pos,
    pair_geometry,
)


def hold_init_action(uw) -> torch.Tensor:
    """Action whose rescale-to-limits target equals ``default_joint_pos``."""
    robot = uw.scene["robot"]
    ids, _ = robot.find_joints(
        actuated_joint_names(detect_hand_side(robot)), preserve_order=True
    )
    limits = robot.data.soft_joint_pos_limits.torch[:, ids]
    lo, hi = limits[..., 0], limits[..., 1]
    target = robot.data.default_joint_pos.torch[:, ids]
    return 2.0 * (target - lo) / (hi - lo).clamp(min=1e-6) - 1.0


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    if args_cli.drop is not None:
        env_cfg.events.reset_object.params["height"] = args_cli.drop
        env_cfg.events.reset_object.params["jitter"] = 0.0
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join("logs", "preview", "spawn"),
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
    env.reset()
    uw = env.unwrapped
    robot = uw.scene["robot"]
    coin = uw.scene["object"]
    side = detect_hand_side(robot)
    is_pair = "object2" in uw.scene.keys()
    pad_ids, _ = robot.find_bodies(pad_body_names(side), preserve_order=True)
    body_names = list(robot.body_names)

    action = (
        hold_init_action(uw)
        if args_cli.action == "init"
        else torch.zeros(1, env.action_space.shape[-1], device=uw.device)
    )
    print(f"[spawn] task={args_cli.task} hand={side} pair={is_pair}")

    # The cradle only works if the fingers actually curl. Print the realised
    # joint angles and palm-relative fingertip heights once, so a pose that
    # silently stays flat is obvious instead of showing up as mystery drops.
    act_names = actuated_joint_names(side)
    act_ids, _ = robot.find_joints(act_names, preserve_order=True)
    q0 = robot.data.joint_pos.torch[0, act_ids]
    target = robot.data.default_joint_pos.torch[0, act_ids]
    print("[spawn] joint  actual/default (deg):")
    for name, actual, want in zip(act_names, q0, target):
        print(f"         {name:20s} {float(actual) * 57.2958:+7.1f} / {float(want) * 57.2958:+7.1f}")
    palm0 = robot.data.body_pos_w.torch[0, robot.find_bodies([f"{side}_palm_link"])[0][0]]
    print("[spawn] body palm-rel (x fwd, y lateral, z palmar):")
    for name, p in zip(body_names, robot.data.body_pos_w.torch[0]):
        rel = p - palm0
        print(f"         {name:24s} ({rel[0]:+.3f}, {rel[1]:+.3f}, {rel[2]:+.3f})")

    # Episode length under a constant hold action is the cradle's carrying
    # capacity: if the pose cannot cradle the balls at all they roll off the
    # palm edge in a fraction of a second and every episode ends on `drop`.
    episode_lengths: list[int] = []
    prev_len = 0

    for step in range(args_cli.steps):
        current_len = int(uw.episode_length_buf[0])
        if current_len < prev_len:
            episode_lengths.append(prev_len)
        prev_len = current_len
        verbose = step % 10 == 0 or step < 5
        if verbose and is_pair:
            geom = pair_geometry(uw, side=side)
            # Cup-relative, matching what the drop predicate tests.
            off_a = (geom.pos_a - geom.cup)[0]
            off_b = (geom.pos_b - geom.cup)[0]
            all_pos = robot.data.body_pos_w.torch[0]
            near_a = torch.linalg.norm(geom.pos_a[0] - all_pos, dim=-1)
            near_b = torch.linalg.norm(geom.pos_b[0] - all_pos, dim=-1)
            ia, ib = int(near_a.argmin()), int(near_b.argmin())
            print(
                f"t={step:03d} "
                f"a_off=({off_a[0]:+.3f},{off_a[1]:+.3f},{off_a[2]:+.3f}) "
                f"b_off=({off_b[0]:+.3f},{off_b[1]:+.3f},{off_b[2]:+.3f}) "
                f"gap={float(geom.gap[0]):.3f} "
                f"|a|={float(torch.linalg.norm(off_a)):.3f} "
                f"|b|={float(torch.linalg.norm(off_b)):.3f} "
                f"dropped={bool(geom.dropped[0])} "
                f"near_a={body_names[ia]}@{float(near_a[ia]):.3f} "
                f"near_b={body_names[ib]}@{float(near_b[ib]):.3f}"
            )
        elif verbose:
            knuckles = knuckle_surface_pos(uw, robot, (0, 1, 2, 3), side)[0]
            axis = knuckle_surface_pos(uw, robot, (0, 1), side, surface_offset=0.0)[0]
            c = coin.data.root_pos_w.torch[0]
            k_dist = torch.linalg.norm(c - knuckles, dim=-1)
            pads = robot.data.body_pos_w.torch[0, pad_ids]
            p_dist = torch.linalg.norm(c - pads, dim=-1)
            all_pos = robot.data.body_pos_w.torch[0]
            all_dist = torch.linalg.norm(c - all_pos, dim=-1)
            nearest = int(all_dist.argmin())
            print(
                f"t={step:03d} coin=({c[0]:+.3f},{c[1]:+.3f},{c[2]:+.3f}) "
                f"d_knuckle idx={float(k_dist[0]):.3f} mid={float(k_dist[1]):.3f} "
                f"ring={float(k_dist[2]):.3f} pinky={float(k_dist[3]):.3f} "
                f"dz_axis={float(c[2] - axis[:, 2].mean()):+.3f} "
                f"d_pad_min={float(p_dist.min()):.3f} "
                f"nearest={body_names[nearest]}@{float(all_dist[nearest]):.3f}"
            )
        env.step(action)

    if is_pair:
        print(f"[spawn] drop thresholds: depth>{DROP_DEPTH_M} m or radius>{DROP_RADIUS_M} m")
    if episode_lengths:
        mean = sum(episode_lengths) / len(episode_lengths)
        print(
            f"[spawn] episodes={len(episode_lengths)} mean_length={mean:.1f} steps "
            f"({mean * uw.step_dt:.2f}s) lengths={episode_lengths[:12]}"
        )
    else:
        print(f"[spawn] survived all {args_cli.steps} steps without a reset")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
