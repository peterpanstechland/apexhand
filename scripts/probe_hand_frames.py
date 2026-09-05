#!/usr/bin/env python3
"""Print Apex Hand body frames at a chosen pose to calibrate orientation.

Used to decide the palm-down knuckle-roll pose: which world axis the fingers
point along, which way the dorsal (knuckle) side faces, and the index->pinky
lateral span.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Probe Apex Hand body frames.")
parser.add_argument("--side", choices=["right", "left"], default="right")
parser.add_argument("--pose", choices=["zero", "default"], default="zero")
parser.add_argument(
    "--hand-pose",
    choices=["palm_down_knuckle", "palm_up_cradle"],
    default="palm_down_knuckle",
    help="Which spawn pose to build. --pose default then uses that pose's joint angles.",
)
parser.add_argument(
    "--rot",
    type=float,
    nargs=4,
    default=None,
    metavar=("W", "X", "Y", "Z"),
    help="Override init_state.rot as a wxyz quaternion.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from pan_dexterous_lab.assets.apex_cfg import make_hand_cfg

_GROUPS = ["link0", "link1", "link2", "link3", "link4", "pad", "tip"]


def main() -> None:
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0))
    sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2000.0))

    cfg = make_hand_cfg(args_cli.side, args_cli.hand_pose).copy()
    cfg.prim_path = f"/World/{args_cli.side}_hand"
    if args_cli.pose == "zero":
        cfg.init_state.joint_pos = {".*": 0.0}
    if args_cli.rot is not None:
        cfg.init_state.rot = tuple(args_cli.rot)
    hand = Articulation(cfg=cfg)
    sim.reset()
    hand.reset()
    for _ in range(4):
        sim.step()
        hand.update(sim.get_physics_dt())

    names = list(hand.body_names)
    pos = hand.data.body_pos_w.torch[0].detach().cpu()
    quat = hand.data.body_quat_w.torch[0].detach().cpu()

    print(f"# side={args_cli.side} hand_pose={args_cli.hand_pose} pose={args_cli.pose} rot={cfg.init_state.rot}")
    print(f"# root pos={cfg.init_state.pos}")
    print(f"{'body':28s} {'x':>8s} {'y':>8s} {'z':>8s}")
    for group in ["palm_link", "palm_base", *_GROUPS]:
        for i, name in enumerate(names):
            if not name.endswith(group):
                continue
            p = pos[i]
            print(f"{name:28s} {p[0]:8.4f} {p[1]:8.4f} {p[2]:8.4f}")

    def body(substr: str) -> int:
        return next(i for i, n in enumerate(names) if n.endswith(substr))

    palm = body("palm_link")
    print()
    print(f"palm_link quat (wxyz) = {[round(float(v), 4) for v in quat[palm]]}")

    def rotate(qi, vec):
        q = quat[qi]
        w, xyz = q[0], q[1:4]
        v = torch.tensor(vec, dtype=xyz.dtype)
        t = 2.0 * torch.cross(xyz, v, dim=-1)
        return v + w * t + torch.cross(xyz, t, dim=-1)

    for name in ("index_link1", "index_link2", "middle_link1"):
        i = body(name)
        minus_x = rotate(i, (-1.0, 0.0, 0.0))
        print(f"{name:14s} local -X in world = {[round(float(v), 3) for v in minus_x]}")
    for finger in ("index", "middle", "ring", "pinky"):
        p0 = pos[body(f"{finger}_link0")]
        tip = pos[body(f"{finger}_tip")]
        pad = pos[body(f"{finger}_pad")]
        print(
            f"{finger:7s} link0->tip = {[round(float(v), 4) for v in (tip - p0)]}"
            f"   pad-tip delta = {[round(float(v), 4) for v in (pad - tip)]}"
        )
    lateral = pos[body("pinky_link0")] - pos[body("index_link0")]
    print(f"index_link0 -> pinky_link0 (lateral) = {[round(float(v), 4) for v in lateral]}")
    print(f"lateral span = {float(torch.linalg.norm(lateral)):.4f} m")

    simulation_app.close()


if __name__ == "__main__":
    main()
