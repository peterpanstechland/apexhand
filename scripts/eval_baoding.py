#!/usr/bin/env python3
"""Evaluate a Baoding Rotate checkpoint.

The headline number is revolutions per episode: one revolution means each ball
travelled all the way around the pair and back to where it started. Everything
else printed here is a way of catching a policy that games that number --
dropping a ball, flinging the pair off the palm, or exploiting the fact that
self-collision is disabled by driving fingers through each other.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate Apex baoding rotation.")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--task", type=str, default="PAN-BaodingRotate-Apex-Left-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--episodes", type=int, default=4, help="Completed episodes per env (approx).")
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
from pan_dexterous_lab.assets.joints import FINGERS
from pan_dexterous_lab.tasks.coin_roll.hand_side import detect_hand_side


def _quantiles(values: list[float]) -> str:
    t = torch.tensor(values)
    return (
        f"mean={t.mean():.3f}  p10={t.quantile(0.1):.3f}  "
        f"median={t.median():.3f}  p90={t.quantile(0.9):.3f}  max={t.max():.3f}"
    )


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
    side = detect_hand_side(robot)
    episode_s = float(uw.max_episode_length) * uw.step_dt

    link2_ids, _ = robot.find_bodies([f"{side}_{f}_link2" for f in FINGERS], preserve_order=True)
    j0_ids, _ = robot.find_joints([f"{side}_{f}_j0" for f in FINGERS], preserve_order=True)
    # Fingers must stay ordered along their own index -> pinky axis. Measuring
    # against world Y instead would just report the mirror for a left hand.
    lateral_axis = None

    term = uw.termination_manager
    names = list(term.active_terms)
    counts = {n: 0 for n in names}
    finished = 0
    revolutions: list[float] = []
    min_gap = float("inf")
    max_abduction = 0.0
    cross_steps = 0
    measured_steps = 0
    target = args_cli.num_envs * args_cli.episodes
    max_steps = int(uw.max_episode_length) * args_cli.episodes + 40

    obs = env.get_observations()
    for _ in range(max_steps):
        if finished >= target:
            break
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))

        pos = robot.data.body_pos_w.torch[:, link2_ids]
        if lateral_axis is None:
            lateral_axis = torch.nn.functional.normalize(pos[:, -1] - pos[:, 0], dim=-1)
        lat = (pos * lateral_axis.unsqueeze(1)).sum(dim=-1)
        gaps = lat[:, 1:] - lat[:, :-1]
        min_gap = min(min_gap, float(gaps.min().item()))
        cross_steps += int((gaps < 0.010).any(dim=-1).sum().item())
        measured_steps += lat.shape[0]
        max_abduction = max(
            max_abduction, float(robot.data.joint_pos.torch[:, j0_ids].abs().max().item())
        )

        done = dones.bool() if torch.is_tensor(dones) else torch.as_tensor(dones, device=uw.device).bool()
        if done.ndim > 1:
            done = done.squeeze(-1)
        if not bool(done.any()):
            continue
        # Rewards run before resets, so the extras written this step still hold
        # each finishing episode's final total.
        spun = (getattr(uw, "extras", {}) or {}).get("baoding_revolutions")
        if spun is not None:
            revolutions.extend(spun[done].detach().float().cpu().tolist())
        for name in names:
            counts[name] += int(term.get_term(name)[done].sum().item())
        finished += int(done.sum().item())

    print(f"[EVAL] checkpoint={args_cli.checkpoint}")
    print(f"[EVAL] task={args_cli.task}  hand={side}  episode={episode_s:.1f}s")
    print(f"[EVAL] finished_episodes={finished}")
    print(
        f"[EVAL] finger_min_lateral_gap={min_gap:.4f} m  (rest pitch 0.023)"
        f"  max_abduction={max_abduction:.4f} rad"
        f"  squeezed_env_steps={cross_steps}/{measured_steps}"
    )
    if finished == 0:
        print("[EVAL] no episodes finished")
    else:
        if revolutions:
            print(f"[EVAL] revolutions_per_episode: {_quantiles(revolutions)}")
            rate = torch.tensor(revolutions).mean() / episode_s
            print(f"[EVAL] mean_spin_rate={rate:.3f} rev/s")
        for name in names:
            print(f"[EVAL] {name}: {counts[name] / finished:.4f}  ({counts[name]}/{finished})")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
