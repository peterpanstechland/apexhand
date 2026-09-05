#!/usr/bin/env python3
"""Webcam MediaPipe landmarks → real Apex Hand.

Default: preview only. Space arms / disarms motors. Esc quits.

  source env_real.sh
  python scripts/landmark_teleop.py
  python scripts/landmark_teleop.py --dry-run          # camera only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from real.apex_interface import DEFAULT_HAND_IP, ApexInterface
from real.joint_table import ACTUATED_LOGICAL
from real.safety import SafetyFilter
from tracking.filters import EMA
from tracking.hand_tracker import HandTracker, TrackedHand
from tracking.retarget import landmarks_to_actuated, pinch_amount


def _flip_x(hand: TrackedHand) -> TrackedHand:
    lm = hand.landmarks.copy()
    lm[:, 0] = 1.0 - lm[:, 0]
    return TrackedHand(landmarks=lm, handedness=hand.handedness, score=hand.score)


def discover_camera() -> int:
    """Prefer a USB webcam over the laptop's built-in UVC device."""
    try:
        import subprocess

        text = subprocess.check_output(["v4l2-ctl", "--list-devices"], text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return 0
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    builtin = ("USB2.0 HD UVC", "Integrated", "IR Camera")
    fallback = None
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        nodes = [ln for ln in lines[1:] if ln.startswith("/dev/video")]
        if not nodes:
            continue
        idx = int(nodes[0].rsplit("video", 1)[1])
        if fallback is None:
            fallback = idx
        if not any(tag in lines[0] for tag in builtin):
            return idx
    return fallback if fallback is not None else 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Landmark teleop for the Apex Hand")
    p.add_argument("--ip", default=DEFAULT_HAND_IP, help="Hand IP (env APEX_HAND_IP)")
    p.add_argument("--camera", type=int, default=None, help="OpenCV camera index (default: USB cam)")
    p.add_argument("--dry-run", action="store_true", help="Do not connect or move the hand")
    p.add_argument("--hz", type=float, default=45.0)
    p.add_argument("--ema", type=float, default=0.55)
    p.add_argument("--max-step", type=float, default=0.22, help="Max Δq per tick (rad)")
    p.add_argument("--max-speed", type=float, default=3.5, help="SDK joint speed cap (rad/s)")
    p.add_argument("--max-accel", type=float, default=40.0, help="SDK joint accel cap (rad/s²)")
    p.add_argument("--torque-pct", type=float, default=30.0, help="SDK finger torque 0-100")
    p.add_argument("--torque-nmm", type=float, default=160.0)
    p.add_argument("--current-limit", type=float, default=0.0, help="Trip current in A; 0 = auto from idle")
    p.add_argument("--backoff", type=float, default=0.04, help="Open flex this many rad on overcurrent")
    p.add_argument("--side", choices=("auto", "left", "right"), default="auto", help="Robot side; auto uses SDK get_hand_dir")
    p.add_argument("--arm", action=argparse.BooleanOptionalAction, default=True, help="Enable motors on connect")
    p.add_argument("--match-hand", action=argparse.BooleanOptionalAction, default=False, help="Require MediaPipe handedness == robot side")
    p.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--no-window", action="store_true")
    p.add_argument("--frames", type=int, default=0, help="Exit after N frames (0 = until Esc)")
    return p.parse_args()


def _draw(frame, tracked, connections, *, armed: bool, side: str, fps: float, q, msg: str):
    h, w = frame.shape[:2]
    if tracked is not None:
        pts = [(int(p[0] * w), int(p[1] * h)) for p in tracked.landmarks]
        for a, b in connections:
            cv2.line(frame, pts[a], pts[b], (0, 200, 80), 2)
        for x, y in pts:
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
    hud = [
        f"{'ARMED' if armed else 'SAFE'}  robot={side}  fps={fps:.0f}",
        msg,
        "Space: arm/disarm   Esc: quit",
    ]
    if q is not None:
        hud.append("  ".join(f"{n.split('_')[0][0]}{n[-1]}:{np.rad2deg(v):+.0f}" for n, v in zip(ACTUATED_LOGICAL, q)))
    color = (0, 0, 255) if armed else (220, 220, 220)
    for i, line in enumerate(hud):
        cv2.putText(frame, line, (12, 28 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def main() -> int:
    args = _parse_args()
    cam_idx = args.camera if args.camera is not None else discover_camera()
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"Cannot open camera index {cam_idx}", file=sys.stderr)
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    tracker = HandTracker()
    ema = EMA(args.ema)
    safety = SafetyFilter(args.max_step, current_limit_a=args.current_limit, backoff_rad=args.backoff)
    hand: ApexInterface | None = None
    robot_side = "dry-run"
    armed = False

    if not args.dry_run:
        hand = ApexInterface(args.ip)
        print(f"Connecting {args.ip} ...")
        hand.connect()
        hand.configure_motion(
            max_speed=args.max_speed,
            max_accel=args.max_accel,
            finger_torque_pct=args.torque_pct,
        )
        robot_side = args.side if args.side != "auto" else hand.side
        print(f"Connected. hardware side={hand.side}, teleop side={robot_side}.")
        if args.arm:
            current = np.array(hand.get_joint_positions(), dtype=np.float64)
            ema.reset(current)
            safety.reset(current)
            hand.enable()
            armed = True
            print("ARMED")

    period = 1.0 / max(args.hz, 1.0)
    t_prev = time.monotonic()
    fps = 0.0
    n = 0
    last_q = None
    status = "show your hand"

    try:
        while True:
            t0 = time.monotonic()
            ok, raw = cap.read()
            if not ok:
                print("camera frame failed", file=sys.stderr)
                break
            tracked = tracker.process(raw)
            frame = cv2.flip(raw, 1) if args.mirror else raw
            drawn = _flip_x(tracked) if tracked is not None and args.mirror else tracked
            q = None
            if tracked is None:
                status = "no hand"
                ema.reset()
            elif args.match_hand and robot_side != "dry-run" and tracked.handedness.lower() != robot_side:
                status = f"need {robot_side} hand (got {tracked.handedness})"
            else:
                currents = hand.get_finger_currents() if hand is not None else None
                pinch = pinch_amount(tracked.landmarks)
                q = safety.filter(
                    ema.update(landmarks_to_actuated(tracked.landmarks)),
                    finger_current=currents,
                    pinch=pinch,
                )
                last_q = q
                sent = False
                if armed and hand is not None:
                    hand.set_joint_positions(q, torque_nmm=args.torque_nmm)
                    sent = True
                amps = ""
                if currents:
                    amps = "  ".join(f"{k[0]}:{v:.2f}A" for k, v in currents.items())
                hit = f" CONTACT {','.join(safety.contact)}" if safety.contact else ""
                status = f"{tracked.handedness} {tracked.score:.2f} pinch={pinch:.2f} {'SENT' if sent else 'HOLD'} {amps}{hit}"

            dt = t0 - t_prev
            t_prev = t0
            if dt > 0:
                fps = 0.9 * fps + 0.1 / dt if fps else 1.0 / dt
            if not args.no_window:
                _draw(frame, drawn, tracker.connections, armed=armed, side=robot_side, fps=fps, q=last_q if q is None else q, msg=status)
                cv2.imshow("Apex landmark teleop", frame)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = 0
                if n % 15 == 0:
                    print(f"[{n}] {status}")

            if n % max(1, int(args.hz)) == 0:
                print(f"[{n}] armed={int(armed)} {status}")

            if key in (27, ord("q")):
                break
            if key in (ord(" "), ord("a")) and hand is not None:
                armed = not armed
                if armed:
                    current = np.array(hand.get_joint_positions(), dtype=np.float64)
                    ema.reset(current)
                    safety.reset(current)
                    hand.enable()
                    print("ARMED")
                else:
                    hand.disable()
                    print("DISARMED")

            n += 1
            if args.frames and n >= args.frames:
                break
            sleep = period - (time.monotonic() - t0)
            if sleep > 0:
                time.sleep(sleep)
    finally:
        if hand is not None:
            hand.close()
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
