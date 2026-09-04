#!/usr/bin/env python3
"""Preview the spawn pose: print coin / knuckle / pad geometry, optionally record.

Drives a constant action that holds ``init_state.joint_pos`` instead of the
policy, so the rendered pose is the calibration pose rather than whatever a
checkpoint does. Use this to verify the coin actually rests on the knuckles
before touching rewards.
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
from pan_dexterous_lab.assets.joints import ACTUATED_JOINT_NAMES, PAD_BODY_NAMES
from pan_dexterous_lab.tasks.coin_roll.mdp._geom import knuckle_surface_pos


def hold_init_action(uw) -> torch.Tensor:
    """Action whose rescale-to-limits target equals ``default_joint_pos``."""
    robot = uw.scene["robot"]
    ids, _ = robot.find_joints(ACTUATED_JOINT_NAMES, preserve_order=True)
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
    pad_ids, _ = robot.find_bodies(PAD_BODY_NAMES, preserve_order=True)
    body_names = list(robot.body_names)

    action = (
        hold_init_action(uw)
        if args_cli.action == "init"
        else torch.zeros(1, env.action_space.shape[-1], device=uw.device)
    )

    for step in range(args_cli.steps):
        knuckles = knuckle_surface_pos(uw, robot, (0, 1, 2, 3))[0]
        axis = knuckle_surface_pos(uw, robot, (0, 1), surface_offset=0.0)[0]
        c = coin.data.root_pos_w.torch[0]
        k_dist = torch.linalg.norm(c - knuckles, dim=-1)
        pads = robot.data.body_pos_w.torch[0, pad_ids]
        p_dist = torch.linalg.norm(c - pads, dim=-1)
        if step % 10 == 0 or step < 5:
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

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
