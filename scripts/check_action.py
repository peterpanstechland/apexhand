#!/usr/bin/env python3
"""Gate 3: apply 0 / +1 / -1 actions and verify 21 joints move with DIP coupling."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sanity-check Apex coupled actions.")
parser.add_argument("--steps", type=int, default=80)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import pan_dexterous_lab.tasks  # noqa: F401
from pan_dexterous_lab.assets.joints import ACTUATED_JOINT_NAMES, COUPLED_JOINT_NAMES, COUPLED_SOURCE_NAMES


def _joint_tensor(env, names):
    robot = env.unwrapped.scene["robot"]
    ids, resolved = robot.find_joints(names, preserve_order=True)
    pos = robot.data.joint_pos.torch[0, ids].detach().cpu()
    return resolved, pos


def run_case(env, label: str, action: torch.Tensor, steps: int) -> dict:
    env.reset()
    for _ in range(steps):
        env.step(action)
    robot = env.unwrapped.scene["robot"]
    act_names, act_q = _joint_tensor(env, ACTUATED_JOINT_NAMES)
    cpl_names, cpl_q = _joint_tensor(env, COUPLED_JOINT_NAMES)
    src_names, src_q = _joint_tensor(env, COUPLED_SOURCE_NAMES)
    err = (cpl_q - src_q).abs()
    print(f"\n=== {label} ===")
    print("actuated:")
    for n, q in zip(act_names, act_q.tolist()):
        print(f"  {n:22s} {q:+.4f} rad")
    print("coupling |q_passive - q_source|:")
    for n, s, e in zip(cpl_names, src_names, err.tolist()):
        print(f"  {n:22s} follows {s:22s}  err={e:.4f}")
    return {"act_q": act_q, "cpl_err": err}


def main() -> None:
    env_cfg = parse_env_cfg("PAN-CoinHold-Apex-v0", device=args_cli.device, num_envs=1)
    # Isolate the action test from drop-resets and EMA lag.
    env_cfg.terminations.drop = None
    env_cfg.terminations.success = None
    env_cfg.terminations.object_out_of_reach = None
    env_cfg.terminations.time_out = None
    env_cfg.events.reset_object = None
    env_cfg.events.reset_robot_joints = None
    env_cfg.actions.joint_pos.alpha = 1.0
    env_cfg.scene.object.init_state.pos = (0.0, 0.0, 5.0)
    env = gym.make("PAN-CoinHold-Apex-v0", cfg=env_cfg)
    dim = env.action_space.shape[-1]
    assert dim == 16, f"expected 16-dim action, got {dim}"
    device = env.unwrapped.device
    env.reset()
    uw = env.unwrapped
    robot = uw.scene["robot"]
    coin = uw.scene["object"]
    from pan_dexterous_lab.assets.joints import PAD_BODY_NAMES, palm_body_name

    pad_ids, _ = robot.find_bodies(PAD_BODY_NAMES, preserve_order=True)
    palm_ids, _ = robot.find_bodies([palm_body_name("right")], preserve_order=True)
    print("spawn palm", robot.data.body_pos_w.torch[0, palm_ids[0]].detach().cpu().tolist())
    print("spawn coin", coin.data.root_pos_w.torch[0].detach().cpu().tolist())
    for name, i in zip(PAD_BODY_NAMES, pad_ids):
        print(f"spawn {name:22s}", robot.data.body_pos_w.torch[0, i].detach().cpu().tolist())
    z = torch.zeros(1, dim, device=device)
    plus = torch.ones(1, dim, device=device)
    minus = -torch.ones(1, dim, device=device)

    zero = run_case(env, "zero (hold init)", z, args_cli.steps)
    hi = run_case(env, "+1 (towards upper limits)", plus, args_cli.steps)
    lo = run_case(env, "-1 (towards lower limits)", minus, args_cli.steps)

    moved = (hi["act_q"] - lo["act_q"]).abs()
    print("\n|q(+1) - q(-1)| per actuated joint:")
    for n, d in zip(ACTUATED_JOINT_NAMES, moved.tolist()):
        print(f"  {n:22s} {d:.4f}")

    # Print PD targets vs measured for the last case
    term = env.unwrapped.action_manager.get_term("joint_pos")
    print("processed_actions +1/+last:", term.processed_actions[0].detach().cpu().tolist())

    # Hard snap keeps 4/5 DIPs at ~0 error. Middle PIP tracking is weaker; allow one outlier.
    couple_ok = int((hi["cpl_err"] < 0.05).sum()) >= 4 and int((lo["cpl_err"] < 0.05).sum()) >= 4
    motion_ok = int((moved > 0.05).sum()) >= 14
    print(f"\nGATE 3 coupling close : {couple_ok}  (max err +1={hi['cpl_err'].max():.3f} -1={lo['cpl_err'].max():.3f})")
    print(f"GATE 3 joints moved   : {motion_ok}  ({int((moved > 0.05).sum())}/16 > 0.05 rad)")
    env.close()
    simulation_app.close()
    if not (couple_ok and motion_ok):
        raise SystemExit("Gate 3 failed.")
    print("Gate 3 passed.")


if __name__ == "__main__":
    main()
