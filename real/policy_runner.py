"""ONNX policy loop: SDK proprioception (+ camera ball tracking) → SafetyFilter.

The observation vector is rebuilt from the layout the exporter recorded in
``joint_map.json``, not from a layout hardcoded here, so a policy whose actor
needs state the hand cannot measure fails at startup instead of quietly
receiving a misaligned vector.

For the baoding task the ``pair`` term comes from ``real.ball_tracker``; pass
``--calib`` to supply the palm-plane calibration. Without a tracker the loop
refuses to arm a policy that asks for ``pair``.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np

from real.apex_interface import DEFAULT_HAND_IP, ApexInterface
from real.joint_table import ACTION_SCALE, ACTUATED_LOGICAL, DEFAULT_ACTUATED_POS
from real.obs_assembler import ObsAssembler, RuntimeState
from real.safety import SafetyFilter

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_ONNX = (
    _REPO
    / "logs/rsl_rl/pan_coin_transfer/2026-09-04_11-26-02_rebalance_v2_fin/exported/policy.onnx"
)
_running = True


def _stop(signum, _frame) -> None:
    global _running
    _running = False
    print(f"\nstop signal {signum}", flush=True)


def default_q() -> np.ndarray:
    return np.array([DEFAULT_ACTUATED_POS[n] for n in ACTUATED_LOGICAL], dtype=np.float64)


def scale_vec() -> np.ndarray:
    return np.array([ACTION_SCALE[n] for n in ACTUATED_LOGICAL], dtype=np.float64)


class OnnxPolicy:
    def __init__(self, path: Path) -> None:
        import onnxruntime as ort

        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.out = self.sess.get_outputs()[0].name

    @property
    def obs_dim(self) -> int | None:
        """Input width, or None when the model declares it dynamic."""
        shape = self.sess.get_inputs()[0].shape
        return shape[-1] if isinstance(shape[-1], int) else None

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        y = self.sess.run([self.out], {self.inp: obs.reshape(1, -1).astype(np.float32)})[0]
        return np.clip(y.reshape(-1), -1.0, 1.0).astype(np.float64)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run an exported ONNX policy on the real Apex Hand")
    p.add_argument("--ip", default=DEFAULT_HAND_IP)
    p.add_argument("--onnx", type=Path, default=_DEFAULT_ONNX)
    p.add_argument("--camera", type=int, default=0, help="Ball-tracker camera index")
    p.add_argument("--calib", type=Path, default=None, help="Palm-plane calibration JSON")
    p.add_argument("--spin", type=int, choices=[-1, 1], default=1, help="Commanded direction")
    p.add_argument("--show", action="store_true", help="Show the tracker overlay window")
    p.add_argument("--gain", type=float, default=0.30, help="Multiply policy Δq (start small)")
    p.add_argument("--hz", type=float, default=60.0)
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--warmup", type=float, default=2.0, help="Hold default pose before policy")
    p.add_argument("--ema", type=float, default=0.6, help="Same alpha as ApexCoupledEMAAction")
    p.add_argument("--max-step", type=float, default=0.08)
    p.add_argument("--torque-pct", type=float, default=30.0)
    p.add_argument("--torque-nmm", type=float, default=160.0)
    p.add_argument("--max-speed", type=float, default=2.0)
    p.add_argument("--max-accel", type=float, default=20.0)
    return p.parse_args()


def main() -> int:
    args = _parse()
    if not args.onnx.is_file():
        print(f"missing ONNX: {args.onnx}", file=sys.stderr)
        return 1
    fmap = args.onnx.parent / "joint_map.json"
    if not fmap.is_file():
        print(
            f"missing {fmap}. Re-export with scripts/export_onnx.py so the "
            "observation layout ships alongside the weights.",
            file=sys.stderr,
        )
        return 1
    meta = json.loads(fmap.read_text())
    spec = meta.get("obs_terms")
    if not spec:
        print(
            f"{fmap} has no 'obs_terms'. It predates spec-driven observations; "
            "re-export this checkpoint.",
            file=sys.stderr,
        )
        return 1
    print(f"policy {meta.get('task')}  ckpt={Path(meta.get('checkpoint', '')).name}")

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    policy = OnnxPolicy(args.onnx)
    try:
        obs_builder = ObsAssembler.from_spec(spec, onnx_dim=policy.obs_dim)
    except RuntimeError as exc:
        print(f"observation mismatch: {exc}", file=sys.stderr)
        return 1
    print(obs_builder.describe())

    tracker = None
    cap = None
    if obs_builder.needs_tracker:
        if args.calib is None:
            print(
                "this policy needs camera ball tracking; pass --calib with a "
                "palm-plane calibration JSON",
                file=sys.stderr,
            )
            return 1
        import cv2

        from real.ball_tracker import BallTracker, PalmCalibration, draw_overlay

        tracker = BallTracker(PalmCalibration.load(args.calib))
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"cannot open camera {args.camera}", file=sys.stderr)
            return 1
        print(f"ball tracker on camera {args.camera}, spin={args.spin:+d}")
    q0 = default_q()
    scales = scale_vec()
    safety = SafetyFilter(args.max_step)
    hand = ApexInterface(args.ip)
    print(f"Connecting {args.ip} ...")
    hand.connect()
    print(
        f"hardware side={hand.side}  (policy was trained on RIGHT). "
        "Air test: no coin. Ctrl+C stops."
    )
    hand.configure_motion(max_speed=args.max_speed, max_accel=args.max_accel, finger_torque_pct=args.torque_pct)
    hand.enable()
    q_cmd = q0.copy()
    safety.reset(q0)
    last_action = np.zeros(len(ACTUATED_LOGICAL), dtype=np.float64)
    period = 1.0 / args.hz
    t_end = time.monotonic() + args.seconds
    t0 = time.monotonic()
    n = 0
    pair_obs = None
    try:
        while _running and time.monotonic() < t_end:
            tick = time.monotonic()
            q, qd = hand.get_actuated_pv()
            q = np.asarray(q, dtype=np.float64)
            qd = np.asarray(qd, dtype=np.float64)
            if tracker is not None:
                ok, frame = cap.read()
                pair_obs = tracker.observe(frame, period) if ok else None
                if args.show and ok:
                    import cv2

                    cv2.imshow("balls", draw_overlay(frame, pair_obs, tracker.misses))
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
            # Freeze on a lost fix rather than acting on zeroed ball features:
            # the policy would read "pair centred, not rotating" and keep
            # squeezing at whatever it was last doing.
            have_fix = tracker is None or pair_obs is not None
            warmed = (tick - t0) >= args.warmup
            if warmed and have_fix:
                state = RuntimeState(
                    q=q,
                    qd=qd,
                    last_action=last_action,
                    pair=None if pair_obs is None else pair_obs.features,
                    spin=float(args.spin),
                )
                last_action = policy(obs_builder(state))
                target = q0 + last_action * scales * args.gain
            elif warmed:
                target = q_cmd
            else:
                target = q0
            q_cmd = args.ema * target + (1.0 - args.ema) * q_cmd
            q_send = safety.filter(q_cmd, finger_current=hand.get_finger_currents())
            hand.set_joint_positions(q_send, torque_nmm=args.torque_nmm)
            if n % int(args.hz) == 0:
                phase = "POLICY" if warmed and have_fix else ("HOLD" if warmed else "WARMUP")
                hit = f" CONTACT {safety.contact}" if safety.contact else ""
                track = "" if tracker is None else f" miss={tracker.misses}"
                print(
                    f"[{n}] {phase} a={np.round(last_action, 2)}  "
                    f"q_deg={np.round(np.rad2deg(q), 1)}{hit}{track}",
                    flush=True,
                )
            n += 1
            sleep = period - (time.monotonic() - tick)
            if sleep > 0:
                time.sleep(sleep)
    finally:
        hand.close()
        if cap is not None:
            cap.release()
        print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
