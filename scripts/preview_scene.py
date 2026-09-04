#!/usr/bin/env python3
"""Spawn the task once, write a PNG (and optional short video). No training."""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Preview a PΛN scene and write a still.")
parser.add_argument("--task", type=str, default="PAN-CoinHold-Apex-Play-v0")
parser.add_argument("--steps", type=int, default=8)
parser.add_argument("--out", type=str, default="logs/webui/preview/scene.png")
parser.add_argument("--video", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

import pan_dexterous_lab.tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def main():
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)) or ".", exist_ok=True)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env.reset()
    action = torch.zeros(1, env.action_space.shape[-1], device=env.unwrapped.device)
    frame = None
    for _ in range(max(1, args_cli.steps)):
        env.step(action)
        frame = env.render()
    if frame is None:
        frame = env.render()
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    if frame is not None:
        if frame.ndim == 4:
            frame = frame[0]
        if frame.dtype != np.uint8:
            pix = np.clip(frame, 0, 255) if frame.max() > 1.5 else np.clip(frame * 255.0, 0, 255)
            frame = pix.astype(np.uint8)
        Image.fromarray(frame[..., :3]).save(args_cli.out)
        print(f"[preview] wrote {args_cli.out} shape={frame.shape}")
    else:
        print("[preview] env.render() returned None")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
