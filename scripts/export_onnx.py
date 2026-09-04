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
    ACTUATED_JOINT_NAMES,
    COUPLED_JOINT_NAMES,
    COUPLED_SOURCE_NAMES,
    PAD_BODY_NAMES,
)


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
    mapping = {
        "task": args_cli.task,
        "checkpoint": str(resume),
        "actuated_joint_names": ACTUATED_JOINT_NAMES,
        "coupled_joint_names": COUPLED_JOINT_NAMES,
        "coupled_source_names": COUPLED_SOURCE_NAMES,
        "pad_body_names": PAD_BODY_NAMES,
        "action_scale": "delta from default_joint_pos; EMA alpha=0.6; rescale_to_limits=False; abduction *_j0 scale≈0.04",
        "policy_hz": 60,
        "note": "Never remap these indices. Policy[i] is ACTUATED_JOINT_NAMES[i]. Knuckle-roll Stage B policy.",
        "eval_transfer_success": "see PROGRESS.md / RESULTS.md (model_2099 ≈ 61.5% on 512 eps)",
    }
    map_path = out_dir / "joint_map.json"
    map_path.write_text(json.dumps(mapping, indent=2))
    print(f"ONNX  -> {out_dir / 'policy.onnx'}")
    print(f"map   -> {map_path}")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
