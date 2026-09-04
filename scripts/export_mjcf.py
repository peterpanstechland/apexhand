#!/usr/bin/env python3
"""Spawn one env on Newton/MJWarp and write the generated MJCF."""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Export the Apex scene as MJCF via Newton.")
parser.add_argument("--task", type=str, default="PAN-CoinHold-Apex-Play-v0")
parser.add_argument("--out", type=str, default="assets/mjcf/apex_hand.xml")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import pan_dexterous_lab.tasks  # noqa: F401
from isaaclab_newton.physics import MJWarpSolverCfg
from isaaclab_tasks.utils.hydra import resolve_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


def main():
    out = os.path.abspath(args_cli.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg = resolve_presets(env_cfg, selected=("newton_mjwarp",))
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.num_envs = 1
    env_cfg.scene.clone_in_fabric = False
    physics = env_cfg.sim.physics
    if hasattr(physics, "solver_cfg") and physics.solver_cfg is not None:
        physics.solver_cfg.save_to_mjcf = out
    else:
        env_cfg.sim.physics.solver_cfg = MJWarpSolverCfg(save_to_mjcf=out)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    action = torch.zeros(1, env.action_space.shape[-1], device=env.unwrapped.device)
    env.step(action)
    print(f"[export_mjcf] requested {out} exists={os.path.isfile(out)}")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
