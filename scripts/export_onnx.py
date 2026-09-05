#!/usr/bin/env python3
"""Export the latest RSL-RL checkpoint to ONNX and write joint_map.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="PAN-CoinHold-Apex-v0")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--log-dir", type=str, default="logs/rsl_rl")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import importlib.metadata as metadata
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg, load_cfg_from_registry
from isaaclab_tasks.utils import get_checkpoint_path

import pan_dexterous_lab.tasks  # noqa: F401
from pan_dexterous_lab.assets.joints import (
    actuated_joint_names,
    coupled_joint_names,
    coupled_source_names,
    pad_body_names,
)
from pan_dexterous_lab.tasks.coin_roll.hand_side import detect_hand_side


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=2)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))

    log_root = Path(args_cli.log_dir) / agent_cfg.experiment_name
    resume = args_cli.checkpoint or get_checkpoint_path(log_root, ".*", "model_.*.pt")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=env.unwrapped.device)
    runner.load(str(resume))
    out_dir = Path(resume).parent / "exported"
    out_dir.mkdir(parents=True, exist_ok=True)
    runner.export_policy_to_onnx(path=str(out_dir), filename="policy.onnx")
    try:
        runner.export_policy_to_jit(path=str(out_dir), filename="policy.pt")
    except Exception as exc:  # noqa: BLE001 — JIT is nice-to-have
        print(f"[WARN] JIT export skipped: {exc}")
    # Record the actor's observation layout term by term. The real-robot runner
    # rebuilds the vector from this, so a term that changes size (or a task with
    # a different obs entirely) fails loudly at startup instead of silently
    # feeding the policy a misaligned vector -- which is what made the first
    # hardware run barely move.
    uw = env.unwrapped
    side = detect_hand_side(uw.scene["robot"])
    obs_manager = uw.observation_manager
    actor_group = agent_cfg.obs_groups["actor"] if hasattr(agent_cfg, "obs_groups") else ["policy"]
    obs_terms = [
        {"name": name, "dim": int(dim[0])}
        for group in actor_group
        for name, dim in zip(
            obs_manager.active_terms[group], obs_manager.group_obs_term_dim[group]
        )
    ]
    mapping = {
        "task": args_cli.task,
        "checkpoint": str(resume),
        "side": side,
        "actuated_joint_names": actuated_joint_names(side),
        "coupled_joint_names": coupled_joint_names(side),
        "coupled_source_names": coupled_source_names(side),
        "pad_body_names": pad_body_names(side),
        "actor_obs_groups": actor_group,
        "obs_terms": obs_terms,
        "obs_dim": sum(t["dim"] for t in obs_terms),
        "action_scale": {
            k: v for k, v in getattr(env_cfg.actions.joint_pos, "scale", {}).items()
        }
        if isinstance(getattr(env_cfg.actions.joint_pos, "scale", None), dict)
        else getattr(env_cfg.actions.joint_pos, "scale", None),
        "ema_alpha": getattr(env_cfg.actions.joint_pos, "alpha", None),
        "policy_hz": round(1.0 / uw.step_dt),
        "note": "Never remap these indices. Policy[i] is actuated_joint_names[side][i].",
    }
    map_path = out_dir / "joint_map.json"
    map_path.write_text(json.dumps(mapping, indent=2))
    print(f"ONNX  -> {out_dir / 'policy.onnx'}")
    print(f"map   -> {map_path}")
    print(f"obs   -> {mapping['obs_dim']} dims over {len(obs_terms)} terms:")
    for term in obs_terms:
        print(f"         {term['name']:32s} {term['dim']}")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
