#!/usr/bin/env python3
"""Evaluate a Coin Hold checkpoint and print termination rates (Gate 4)."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate Apex Coin Hold success rate.")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--task", type=str, default="PAN-CoinHold-Apex-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--episodes", type=int, default=3, help="Completed episodes per env (approx).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg

import pan_dexterous_lab.tasks  # noqa: F401
from pan_dexterous_lab.assets.joints import DEFAULT_SIDE


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    uw = env.unwrapped
    robot = uw.scene["robot"]
    # Self-collision is disabled on the Apex USD, so adjacent fingers can pass
    # through each other. Watch for it: index..pinky must stay ordered along the
    # lateral axis, and adjacent link2 frames must stay ~1 finger pitch apart.
    lat_ids, _ = robot.find_bodies(
        [f"{DEFAULT_SIDE}_{f}_link2" for f in ("index", "middle", "ring", "pinky")], preserve_order=True
    )
    j0_ids, _ = robot.find_joints(
        [f"{DEFAULT_SIDE}_{f}_j0" for f in ("index", "middle", "ring", "pinky")], preserve_order=True
    )
    min_gap = float("inf")
    max_abduction = 0.0
    cross_steps = 0
    measured_steps = 0

    term = uw.termination_manager
    names = list(term.active_terms)
    counts = {n: 0 for n in names}
    finished = 0
    hold_ok_sum = 0.0
    hold_ok_n = 0
    target = args_cli.num_envs * args_cli.episodes
    max_steps = int(uw.max_episode_length) * args_cli.episodes + 20

    obs = env.get_observations()
    for _ in range(max_steps):
        if finished >= target:
            break
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
        extras = getattr(uw, "extras", {})
        if "hold_ok" in extras:
            hold_ok_sum += float(extras["hold_ok"].float().mean().item())
            hold_ok_n += 1

        lat = robot.data.body_pos_w.torch[:, lat_ids, 1]
        gaps = lat[:, 1:] - lat[:, :-1]
        min_gap = min(min_gap, float(gaps.min().item()))
        cross_steps += int((gaps < 0.010).any(dim=-1).sum().item())
        measured_steps += lat.shape[0]
        max_abduction = max(max_abduction, float(robot.data.joint_pos.torch[:, j0_ids].abs().max().item()))
        done = dones.bool() if torch.is_tensor(dones) else torch.as_tensor(dones, device=uw.device).bool()
        if done.ndim > 1:
            done = done.squeeze(-1)
        if not bool(done.any()):
            continue
        for name in names:
            counts[name] += int(term.get_term(name)[done].sum().item())
        finished += int(done.sum().item())

    print(f"[EVAL] checkpoint={args_cli.checkpoint}")
    print(f"[EVAL] finished_episodes={finished}  max_episode_length={int(uw.max_episode_length)}")
    if hold_ok_n:
        print(f"[EVAL] mean_hold_ok={hold_ok_sum / hold_ok_n:.4f}")
    print(
        f"[EVAL] finger_min_lateral_gap={min_gap:.4f} m  (rest pitch 0.023)"
        f"  max_abduction={max_abduction:.4f} rad"
        f"  squeezed_env_steps={cross_steps}/{measured_steps}"
    )
    if finished == 0:
        print("[EVAL] no episodes finished")
    else:
        for name in names:
            print(f"[EVAL] {name}: {counts[name] / finished:.4f}  ({counts[name]}/{finished})")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
