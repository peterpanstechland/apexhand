#!/usr/bin/env python3
"""Roll out random actions and print per-term reward mean / std / max-abs."""

from __future__ import annotations

import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Probe reward term magnitudes.")
parser.add_argument("--task", type=str, default="PAN-CoinHold-Apex-Play-v0")
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--num_envs", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import pan_dexterous_lab.tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    uw = env.unwrapped
    rm = uw.reward_manager
    names = list(rm.active_terms)
    acc = {n: [] for n in names}
    for _ in range(args_cli.steps):
        action = 2.0 * torch.rand(uw.num_envs, env.action_space.shape[-1], device=uw.device) - 1.0
        env.step(action)
        step = rm._step_reward.detach()
        for i, name in enumerate(names):
            acc[name].append(step[:, i].cpu())
    report = []
    for name in names:
        cat = torch.cat([t.flatten() for t in acc[name]])
        mean = float(cat.mean())
        std = float(cat.std())
        mx = float(cat.abs().max())
        report.append(
            {
                "name": name,
                "mean": mean,
                "std": std,
                "max_abs": mx,
                "dead": mx < 1e-8,
            }
        )
    report.sort(key=lambda r: r["max_abs"], reverse=True)
    print(json.dumps({"task": args_cli.task, "steps": args_cli.steps, "terms": report}, indent=2))
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
