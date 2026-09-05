#!/usr/bin/env python3
"""Solve the camera -> palm-plane calibration by clicking the finger roots.

The baoding policy's ``pair`` observation is palm-relative metres, so the
tracker needs to know where the palm frame sits in the image. Click the four
finger metacarpal heads (index -> pinky); their palm-frame coordinates are known
from the hand's geometry, which is what ties the camera to the frame the policy
was trained in.

    python scripts/calibrate_palm.py --out configs/palm_calib.json

The hand must be in the cradle pose while calibrating, since the landmarks are
measured there. ``real/policy_runner.py`` holds exactly that pose during its
``--warmup`` window, so run it alongside with a long warmup. Two clicks are
enough mathematically, but four lets the residuals catch a mis-click.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from real.ball_tracker import BallTracker, PalmCalibration, draw_overlay, palm_landmarks

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate camera pixels to palm-plane metres")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--side", choices=["left", "right"], default="left")
    p.add_argument("--out", type=Path, default=Path("configs/palm_calib.json"))
    p.add_argument(
        "--preview",
        action="store_true",
        help="Skip clicking; load --out and show live pair features.",
    )
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    return p.parse_args()


def grab_frame(cap: cv2.VideoCapture) -> np.ndarray | None:
    """Live preview until SPACE freezes a frame. ESC aborts."""
    while True:
        ok, frame = cap.read()
        if not ok:
            return None
        shown = frame.copy()
        cv2.putText(
            shown,
            "SPACE = freeze and start clicking,  ESC = abort",
            (12, 30),
            _FONT,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.imshow("calibrate", shown)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            return frame
        if key == 27:
            return None


def collect_clicks(frame: np.ndarray, names: list[str]) -> list[tuple[float, float]] | None:
    """Click each named landmark in order. BACKSPACE undoes, ESC aborts."""
    picks: list[tuple[float, float]] = []

    def on_mouse(event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(picks) < len(names):
            picks.append((float(x), float(y)))

    cv2.setMouseCallback("calibrate", on_mouse)
    while True:
        shown = frame.copy()
        for i, (px, py) in enumerate(picks):
            cv2.circle(shown, (int(px), int(py)), 6, (0, 255, 0), -1)
            cv2.putText(shown, names[i], (int(px) + 9, int(py) - 6), _FONT, 0.55, (0, 255, 0), 2)
        if len(picks) < len(names):
            msg = f"click the {names[len(picks)].upper()} finger root"
            colour = (0, 200, 255)
        else:
            msg = "ENTER = solve,  BACKSPACE = undo"
            colour = (0, 255, 0)
        cv2.putText(shown, msg, (12, 30), _FONT, 0.7, colour, 2)
        cv2.imshow("calibrate", shown)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            return None
        if key == 8 and picks:
            picks.pop()
        if key in (13, 10) and len(picks) == len(names):
            return picks


def report(calib: PalmCalibration, picks: np.ndarray, truth: np.ndarray) -> float:
    """Print per-landmark reprojection error and return the worst, in mm."""
    got = calib.to_palm(picks)
    errors_mm = np.linalg.norm(got - truth, axis=1) * 1000.0
    print(f"scale = {calib.metres_per_pixel * 1000:.3f} mm/px")
    print(f"rotation = {np.degrees(calib.rotation):+.1f} deg")
    print(f"reflect = {calib.reflect}")
    print(f"palm origin at pixel ({calib.origin_px[0]:.1f}, {calib.origin_px[1]:.1f})")
    print("residuals:")
    for (gx, gy), (tx, ty), err in zip(got, truth, errors_mm):
        print(
            f"  got ({gx * 1000:+7.1f}, {gy * 1000:+7.1f}) mm  "
            f"want ({tx * 1000:+7.1f}, {ty * 1000:+7.1f})  err {err:5.1f} mm"
        )
    return float(errors_mm.max())


def verify(cap: cv2.VideoCapture, calib: PalmCalibration) -> None:
    """Live overlay of the palm axes plus whatever the ball detector finds.

    Reuses the tracker rather than re-deriving the drawing, so what you see here
    is literally what the policy will be fed.
    """
    tracker = BallTracker(calib)
    origin, x_end, y_end = calib.from_palm(
        np.array([[0.0, 0.0], [0.05, 0.0], [0.0, 0.05]], dtype=np.float64)
    )
    print("verify: +X should point at the fingers, +Y toward the index side. ESC exits.")
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        obs = tracker.observe(frame, 1.0 / 30.0)
        if n % 15 == 0:
            if obs is None:
                print(f"miss={tracker.misses}", flush=True)
            else:
                mid_x, mid_y, _, _, gap, rate = obs.features
                print(
                    f"mid=({mid_x*1000:+.0f},{mid_y*1000:+.0f})mm "
                    f"gap={gap*1000:.0f}mm rate={rate:+.2f}",
                    flush=True,
                )
        n += 1
        shown = draw_overlay(frame, obs, tracker.misses)
        for end, label, colour in (
            (x_end, "+X 50mm", (0, 0, 255)),
            (y_end, "+Y 50mm", (0, 255, 0)),
        ):
            cv2.arrowedLine(
                shown, tuple(origin.astype(int)), tuple(end.astype(int)), colour, 2, tipLength=0.2
            )
            cv2.putText(shown, label, tuple(end.astype(int) + 6), _FONT, 0.5, colour, 2)
        cv2.imshow("calibrate", shown)
        if cv2.waitKey(1) & 0xFF == 27:
            break


def main() -> int:
    args = _parse()
    landmarks = palm_landmarks(args.side)
    names = list(landmarks)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"cannot open camera {args.camera}", file=sys.stderr)
        return 1

    try:
        if args.preview:
            if not args.out.is_file():
                print(f"missing {args.out}; run without --preview first", file=sys.stderr)
                return 1
            calib = PalmCalibration.load(args.out)
            print(
                f"preview {args.out}  scale={calib.metres_per_pixel*1000:.3f} mm/px  "
                f"reflect={calib.reflect}"
            )
            verify(cap, calib)
            return 0

        print(f"{args.side} hand. Put it in the cradle pose, palm toward the camera.")
        frame = grab_frame(cap)
        if frame is None:
            print("aborted")
            return 1
        picks = collect_clicks(frame, names)
        if picks is None:
            print("aborted")
            return 1

        px = np.array(picks, dtype=np.float64)
        truth = np.array([landmarks[n] for n in names], dtype=np.float64)
        calib = PalmCalibration.solve(px, truth)
        worst_mm = report(calib, px, truth)
        # The balls are 30 mm across, so an error of that order means the policy
        # would see the pair in the wrong place entirely.
        if worst_mm > 8.0:
            print(
                f"\nworst residual {worst_mm:.1f} mm is too large to trust. "
                "A similarity fit cannot absorb perspective, so either the "
                "camera is off-axis or a click missed. Re-run before using this.",
            )
        else:
            print(f"\nworst residual {worst_mm:.1f} mm -- good.")

        args.out.parent.mkdir(parents=True, exist_ok=True)
        calib.save(args.out)
        print(f"saved -> {args.out}")
        verify(cap, calib)
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
