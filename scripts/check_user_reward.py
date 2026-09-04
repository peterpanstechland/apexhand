#!/usr/bin/env python3
"""Headless 1-env check of user_rewards.user_reward: shape, dtype, finite."""

from __future__ import annotations

import argparse
import json
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Validate user_reward in a live env.")
parser.add_argument("--task", type=str, default="PAN-CoinHold-Apex-Play-v0")
parser.add_argument("--steps", type=int, default=5)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import pan_dexterous_lab.tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from pan_dexterous_lab.tasks.coin_roll.mdp import user_reward


def main():
    report = {"ok": False, "error": None, "steps": []}
    try:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
        env = gym.make(args_cli.task, cfg=env_cfg)
        env.reset()
        uw = env.unwrapped
        action = torch.zeros(1, env.action_space.shape[-1], device=uw.device)
        for i in range(args_cli.steps):
            env.step(action)
            val = user_reward(uw)
            if not isinstance(val, torch.Tensor):
                raise TypeError(f"user_reward returned {type(val)}, expected Tensor")
            if tuple(val.shape) != (1,):
                raise ValueError(f"shape {tuple(val.shape)} != (1,)")
            if not torch.is_floating_point(val):
                raise TypeError(f"dtype {val.dtype} is not floating")
            if not torch.isfinite(val).all():
                raise ValueError(f"non-finite values: {val}")
            report["steps"].append({"step": i, "value": float(val[0])})
        report["ok"] = True
        env.close()
    except Exception as exc:
        report["error"] = f"{exc}\n{traceback.format_exc()}"
    print(json.dumps(report, indent=2))
    simulation_app.close()
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
