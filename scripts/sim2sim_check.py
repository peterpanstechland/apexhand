#!/usr/bin/env python3
"""Run the same checkpoint on PhysX and Newton/MJWarp, compare extras."""

from __future__ import annotations

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Cross-engine consistency check.")
parser.add_argument("--task", type=str, default="PAN-CoinHold-Apex-Play-v0")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--episodes", type=int, default=8)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--mapping", type=str, default="configs/sim2sim/newton_to_physx_apex.yaml")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import pan_dexterous_lab.tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import resolve_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


def _rollout(task: str, physics: str, checkpoint: str | None, episodes: int, num_envs: int) -> dict:
    env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    env_cfg = resolve_presets(env_cfg, selected=(physics,))
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.num_envs = num_envs
    if physics == "newton_mjwarp":
        env_cfg.scene.clone_in_fabric = False
    env = gym.make(task, cfg=env_cfg)
    wrapped = RslRlVecEnvWrapper(env)
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=env.unwrapped.device)
    if checkpoint and os.path.isfile(checkpoint):
        runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    obs = wrapped.get_observations()
    succ = []
    drop = []
    # episodes * horizon, but we just run a fixed number of policy steps.
    horizon = int(env.unwrapped.max_episode_length)
    for _ in range(episodes * max(horizon, 1) // max(num_envs, 1)):
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, dones, extras = wrapped.step(actions)
        extra = extras[0] if isinstance(extras, tuple) else extras
        logs = extra.get("log", extra) if isinstance(extra, dict) else {}
        if isinstance(logs, dict):
            if "success_rate" in logs:
                succ.append(float(torch.as_tensor(logs["success_rate"]).float().mean()))
            if "drop_rate" in logs:
                drop.append(float(torch.as_tensor(logs["drop_rate"]).float().mean()))
        if dones is not None and bool(torch.as_tensor(dones).any()):
            pass
    env.close()
    return {
        "physics": physics,
        "success_mean": sum(succ) / len(succ) if succ else None,
        "drop_mean": sum(drop) / len(drop) if drop else None,
        "samples": len(succ),
    }


def main():
    results = []
    for physics in ("physx", "newton_mjwarp"):
        try:
            results.append(_rollout(args_cli.task, physics, args_cli.checkpoint, args_cli.episodes, args_cli.num_envs))
        except Exception as exc:
            results.append({"physics": physics, "error": str(exc)})
    report = {"task": args_cli.task, "checkpoint": args_cli.checkpoint, "results": results}
    if (
        len(results) == 2
        and results[0].get("success_mean") is not None
        and results[1].get("success_mean") is not None
    ):
        report["success_delta"] = abs(results[0]["success_mean"] - results[1]["success_mean"])
    print(json.dumps(report, indent=2))
    simulation_app.close()


if __name__ == "__main__":
    main()
