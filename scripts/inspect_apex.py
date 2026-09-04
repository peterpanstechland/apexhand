#!/usr/bin/env python3
"""Gate 2: spawn both Apex USDs and print joints / pads / tips."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect Apex Hand USD articulations.")
parser.add_argument("--side", choices=["right", "left", "both"], default="both")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from pan_dexterous_lab.assets.apex_cfg import APEX_HAND_LEFT_CFG, APEX_HAND_RIGHT_CFG
from pan_dexterous_lab.assets.joints import (
    JOINT_LIMITS_DEG,
    actuated_joint_names,
    coupled_joint_names,
    pad_body_names,
    tip_body_names,
)


def _names(x):
    return list(x) if not isinstance(x, list) else x


def inspect(side: str, cfg, *, do_reset: bool = True, hand: Articulation | None = None):
    if hand is None:
        cfg = cfg.copy()
        cfg.prim_path = f"/World/{side}_hand"
        hand = Articulation(cfg=cfg)
        if do_reset:
            SimulationContext.instance().reset()
    hand.reset()
    joint_names = _names(hand.joint_names)
    body_names = _names(hand.body_names)
    print("=" * 72)
    print(f"SIDE {side}")
    print(f"  usd            : {cfg.spawn.usd_path}")
    print(f"  num_joints     : {hand.num_joints}")
    print(f"  num_bodies     : {len(body_names)}")
    print("  joints:")
    limits = hand.data.soft_joint_pos_limits.torch[0].detach().cpu()
    for i, name in enumerate(joint_names):
        lo, hi = limits[i].tolist()
        logical = name.split("_", 1)[-1] if name.startswith(f"{side}_") else name
        expected = JOINT_LIMITS_DEG.get(logical)
        exp = f"  official {expected[0]:.0f}~{expected[1]:.0f} deg" if expected else ""
        print(f"    [{i:02d}] {name:22s}  {math.degrees(lo):7.1f} ~ {math.degrees(hi):7.1f} deg{exp}")

    missing_act = [n for n in actuated_joint_names(side) if n not in joint_names]
    missing_cpl = [n for n in coupled_joint_names(side) if n not in joint_names]
    missing_pad = [n for n in pad_body_names(side) if n not in body_names]
    missing_tip = [n for n in tip_body_names(side) if n not in body_names]
    print(f"  missing actuated : {missing_act or 'none'}")
    print(f"  missing coupled  : {missing_cpl or 'none'}")
    print(f"  missing pads     : {missing_pad or 'none'}")
    print(f"  missing tips     : {missing_tip or 'none'}")
    print(f"  bodies           : {body_names}")
    ok = (
        hand.num_joints >= 21
        and not missing_act
        and not missing_cpl
        and not missing_pad
        and not missing_tip
    )
    print(f"  GATE 2           : {'PASS' if ok else 'FAIL'}")
    # keep a handle so GC does not despawn
    return hand, ok


def main() -> None:
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0))
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2000.0))

    sides = ["right", "left"] if args_cli.side == "both" else [args_cli.side]
    mapping = {"right": APEX_HAND_RIGHT_CFG, "left": APEX_HAND_LEFT_CFG}
    spawned = []
    for side in sides:
        cfg = mapping[side].copy()
        cfg.prim_path = f"/World/{side}_hand"
        spawned.append((side, Articulation(cfg=cfg)))
    SimulationContext.instance().reset()
    results = []
    handles = []
    for side, hand in spawned:
        _, ok = inspect(side, mapping[side], do_reset=False, hand=hand)
        handles.append(hand)
        results.append(ok)
    if not all(results):
        raise SystemExit("Gate 2 failed — check USD conversion and --merge-joints.")
    print("Gate 2 passed for:", ", ".join(sides))
    simulation_app.close()


if __name__ == "__main__":
    main()
