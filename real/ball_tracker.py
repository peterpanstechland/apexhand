"""Camera tracking for the two wooden baoding balls.

Produces exactly the six numbers in the policy's ``pair`` observation, in the
same order and units as :func:`coin_mdp.baoding_pair_obs`. That symmetry is the
whole point: the sim actor is trained only on features a camera can recover, so
deployment swaps this class in for the simulator's ground truth and nothing else
about the observation vector changes.

Two facts drive the design:

* **The balls are indistinguishable** -- same turned wood, same size. So the
  tracker never labels them. It reports the *pair* (midpoint, axis, gap), and
  the axis angle is doubled, which is invariant to whatever order the blobs came
  out of the detector.
* **The hand and camera are both fixed**, so a one-time similarity calibration
  is enough to map image pixels to palm-relative metres. Depth is not estimated
  at all -- see the observation docstring for why a single camera cannot.

Detection thresholds on brightness rather than hue: pale wood against the hand's
dark tactile shells separates far more reliably on value than on colour, and it
does not care about the lighting temperature.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

PAIR_OBS_DIM = 6

# Palm-frame (x forward, y lateral) positions of the four finger metacarpal
# heads, in metres, measured off the articulation in the palm-up cradle pose
# via ``scripts/debug_spawn.py``. These are the calibration landmarks: on a
# palm-up hand the finger roots are the most identifiable points in the image,
# and pinning them ties the camera to the exact frame the policy trained in.
#
# Abduction (``*_j0``) barely moves them, so they hold for any cradle-like pose.
# Left-hand values; the right hand mirrors about the sagittal plane.
_LEFT_MCP_LANDMARKS_M: dict[str, tuple[float, float]] = {
    "index": (0.119, 0.034),
    "middle": (0.122, 0.011),
    "ring": (0.119, -0.012),
    "pinky": (0.116, -0.035),
}


def palm_landmarks(side: str) -> dict[str, tuple[float, float]]:
    """Finger-root landmarks in palm-frame metres, index -> pinky."""
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    if side == "left":
        return dict(_LEFT_MCP_LANDMARKS_M)
    return {name: (x, -y) for name, (x, y) in _LEFT_MCP_LANDMARKS_M.items()}


@dataclass
class PalmCalibration:
    """Image pixels -> palm-plane metres.

    A similarity transform (uniform scale, rotation, translation), which assumes
    the camera looks roughly along the palm normal and the balls stay in the
    palm plane. That is exactly the palm-up cradle geometry, and it needs only
    two point correspondences to solve -- cheap enough to redo on the bench if
    the camera gets knocked.

    Axes must match the simulator's world frame, since the policy was trained
    there: palm +X toward the fingers, and +Y such that the index knuckle is on
    the positive side for a left hand (it mirrors for a right one -- see
    :func:`palm_landmarks`). Solve it with :mod:`scripts.calibrate_palm`, which
    supplies the correspondences from measured hand geometry.
    """

    metres_per_pixel: float
    rotation: float
    """Radians, rotating image axes onto palm axes."""
    origin_px: tuple[float, float]
    """Where the palm centre lands in the image."""
    reflect: bool = False
    """Flip image Y before rotating.

    A camera looking *down* onto the palm sees a mirrored handedness in
    OpenCV's y-down image. A rotation-only similarity cannot represent that
    and will send +X toward the wrist. Trying both orientations and keeping
    the lower residual is how the solver notices.
    """

    def _centred(self, points_px: np.ndarray) -> np.ndarray:
        centred = np.asarray(points_px, dtype=np.float64) - np.asarray(self.origin_px)
        if self.reflect:
            centred = centred * np.array([1.0, -1.0])
        return centred

    def to_palm(self, points_px: np.ndarray) -> np.ndarray:
        """(k, 2) pixels -> (k, 2) palm-relative metres."""
        c, s = math.cos(self.rotation), math.sin(self.rotation)
        rot = np.array([[c, -s], [s, c]])
        return (self._centred(points_px) @ rot.T) * self.metres_per_pixel

    def from_palm(self, points_m: np.ndarray) -> np.ndarray:
        """(k, 2) palm metres -> (k, 2) pixels. Inverse of :meth:`to_palm`."""
        c, s = math.cos(self.rotation), math.sin(self.rotation)
        rot = np.array([[c, -s], [s, c]])
        flipped = (np.asarray(points_m, dtype=np.float64) / self.metres_per_pixel) @ rot
        if self.reflect:
            flipped = flipped * np.array([1.0, -1.0])
        return flipped + np.asarray(self.origin_px)

    @classmethod
    def _fit_similarity(
        cls, z_src: np.ndarray, z_dst: np.ndarray
    ) -> tuple[complex, complex]:
        dz_src = z_src - z_src.mean()
        dz_dst = z_dst - z_dst.mean()
        denom = np.vdot(dz_src, dz_src).real
        if denom < 1e-12:
            raise ValueError("calibration points are coincident in the image")
        similarity = np.vdot(dz_src, dz_dst) / denom
        if abs(similarity) < 1e-12:
            raise ValueError("degenerate calibration: zero scale")
        origin = z_src.mean() - z_dst.mean() / similarity
        return similarity, origin

    @classmethod
    def solve(cls, points_px: np.ndarray, points_palm_m: np.ndarray) -> "PalmCalibration":
        """Least-squares fit from >=2 correspondences.

        Tries a rotation-only similarity and a reflected one (image Y flipped).
        A top-down camera is a reflection in OpenCV coordinates, so the
        reflected fit is the one that usually wins; the rotation-only fit is
        kept so a side-on camera still calibrates.
        """
        px = np.asarray(points_px, dtype=np.float64)
        palm = np.asarray(points_palm_m, dtype=np.float64)
        if px.shape != palm.shape or px.shape[0] < 2:
            raise ValueError(f"need >=2 matching point pairs, got {px.shape} / {palm.shape}")

        z_px = px[:, 0] + 1j * px[:, 1]
        z_palm = palm[:, 0] + 1j * palm[:, 1]
        candidates: list[PalmCalibration] = []
        for reflect in (False, True):
            src = np.conjugate(z_px) if reflect else z_px
            similarity, origin_src = cls._fit_similarity(src, z_palm)
            origin = np.conjugate(origin_src) if reflect else origin_src
            candidates.append(
                cls(
                    metres_per_pixel=abs(similarity),
                    rotation=math.atan2(similarity.imag, similarity.real),
                    origin_px=(float(origin.real), float(origin.imag)),
                    reflect=reflect,
                )
            )
        return min(
            candidates,
            key=lambda cal: float(np.linalg.norm(cal.to_palm(px) - palm, axis=1).max()),
        )

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "metres_per_pixel": self.metres_per_pixel,
                    "rotation": self.rotation,
                    "origin_px": list(self.origin_px),
                    "reflect": self.reflect,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: Path) -> "PalmCalibration":
        data = json.loads(Path(path).read_text())
        return cls(
            metres_per_pixel=float(data["metres_per_pixel"]),
            rotation=float(data["rotation"]),
            origin_px=tuple(float(v) for v in data["origin_px"]),
            reflect=bool(data.get("reflect", False)),
        )


@dataclass
class BallDetectorCfg:
    """Blob thresholds. Radii are in pixels and depend on the camera distance;
    print what :meth:`BallTracker.detect` finds and widen until both balls stick."""

    value_min: int = 120
    """Wood is bright. Raise to reject the hand, lower if the balls drop out."""
    saturation_max: int = 140
    """Rejects strongly coloured background without excluding warm wood."""
    min_radius_px: int = 8
    """Per-ball radius bounds, not blob bounds. A touching pair is one blob
    roughly twice this wide and is split, not rejected."""
    max_radius_px: int = 60
    blur_ksize: int = 5
    morph_ksize: int = 5


@dataclass
class PairObservation:
    """One tracker reading, plus enough context for the runner to judge it."""

    features: np.ndarray
    """(6,) matching ``baoding_pair_obs``."""
    centres_px: np.ndarray
    """(2, 2) detected blob centres, for overlay drawing."""
    centres_palm_m: np.ndarray
    """(2, 2) the same in palm metres."""


class BallTracker:
    """Stateful because the axis-angle rate is a finite difference."""

    def __init__(
        self,
        calibration: PalmCalibration,
        detector: BallDetectorCfg | None = None,
    ) -> None:
        self.calibration = calibration
        self.cfg = detector or BallDetectorCfg()
        self._prev_doubled: float | None = None
        self.misses = 0
        """Consecutive frames without a clean two-ball detection."""

    def reset(self) -> None:
        self._prev_doubled = None
        self.misses = 0

    # A touching pair's enclosing circle is about twice a ball's radius; a bit
    # of slack covers the pair being slightly apart or seen at a slant.
    _PAIR_ENCLOSING_FACTOR = 2.4

    def mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Bright, weakly-saturated pixels: pale wood against dark shells."""
        cfg = self.cfg
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        if cfg.blur_ksize > 1:
            hsv = cv2.GaussianBlur(hsv, (cfg.blur_ksize, cfg.blur_ksize), 0)
        out = cv2.inRange(
            hsv,
            np.array([0, 0, cfg.value_min], dtype=np.uint8),
            np.array([179, cfg.saturation_max, 255], dtype=np.uint8),
        )
        if cfg.morph_ksize > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.morph_ksize,) * 2)
            out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
            out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
        return out

    def _ball_peaks(self, contour: np.ndarray) -> list[tuple[float, tuple[float, float]]]:
        """(inscribed_radius, centre) for up to two balls inside one blob.

        The distance transform's value at a peak *is* the radius of the largest
        disk that fits there, which both locates a ball and measures it. That
        makes it a stronger test than contour circularity, and unlike
        circularity it still works when the two balls touch -- which for
        baoding is the normal case, not an edge case.
        """
        x, y, w, h = cv2.boundingRect(contour)
        pad = 2
        local = np.zeros((h + 2 * pad, w + 2 * pad), dtype=np.uint8)
        cv2.drawContours(local, [contour], -1, 255, -1, offset=(pad - x, pad - y))
        dist = cv2.distanceTransform(local, cv2.DIST_L2, 5)

        found: list[tuple[float, tuple[float, float]]] = []
        for _ in range(2):
            _, radius, _, loc = cv2.minMaxLoc(dist)
            if not self.cfg.min_radius_px <= radius <= self.cfg.max_radius_px:
                break
            found.append((radius, (float(loc[0] + x - pad), float(loc[1] + y - pad))))
            # Blank this ball out so the next peak has to land on the other one.
            # Touching balls sit 2r apart, so 1.6r clears one without eating
            # into its neighbour.
            cv2.circle(dist, loc, int(round(radius * 1.6)), 0.0, -1)
        return found

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """(2, 2) ball centres in pixels, or None if two were not found.

        No attempt is made to keep them in a consistent order -- the pair
        features are permutation invariant, so nothing downstream depends on it.
        """
        cfg = self.cfg
        mask = self.mask(frame_bgr)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: list[tuple[float, tuple[float, float]]] = []
        for contour in contours:
            _, enclosing = cv2.minEnclosingCircle(contour)
            # Big enough for one ball, small enough not to be the background.
            if not (
                cfg.min_radius_px
                <= enclosing
                <= self._PAIR_ENCLOSING_FACTOR * cfg.max_radius_px
            ):
                continue
            candidates.extend(self._ball_peaks(contour))

        if len(candidates) < 2:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return np.array([candidates[0][1], candidates[1][1]], dtype=np.float64)

    def observe(self, frame_bgr: np.ndarray, dt: float) -> PairObservation | None:
        """Detect and convert to the policy's ``pair`` features.

        ``None`` means the balls were not both visible this frame; the caller
        decides whether to reuse its last observation or bail out. Hiding a miss
        behind a stale value inside the tracker would let the policy run blind
        without anyone noticing.
        """
        centres_px = self.detect(frame_bgr)
        if centres_px is None:
            self.misses += 1
            return None
        self.misses = 0

        palm = self.calibration.to_palm(centres_px)
        delta = palm[1] - palm[0]
        gap = float(np.linalg.norm(delta))
        doubled = 2.0 * math.atan2(delta[1], delta[0])

        if self._prev_doubled is None or dt <= 0.0:
            rate = 0.0
        else:
            step = math.atan2(
                math.sin(doubled - self._prev_doubled), math.cos(doubled - self._prev_doubled)
            )
            rate = step / dt
        self._prev_doubled = doubled

        mid = 0.5 * (palm[0] + palm[1])
        features = np.array(
            [mid[0], mid[1], math.cos(doubled), math.sin(doubled), gap, rate],
            dtype=np.float32,
        )
        return PairObservation(features=features, centres_px=centres_px, centres_palm_m=palm)


def draw_overlay(frame_bgr: np.ndarray, obs: PairObservation | None, misses: int) -> np.ndarray:
    """Annotate a frame for the operator. Mutates and returns ``frame_bgr``."""
    if obs is None:
        cv2.putText(
            frame_bgr,
            f"NO PAIR ({misses} frames)",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        return frame_bgr
    a, b = obs.centres_px
    for centre in (a, b):
        cv2.circle(frame_bgr, (int(centre[0]), int(centre[1])), 6, (0, 255, 0), -1)
    cv2.line(frame_bgr, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (0, 255, 0), 2)
    mid_x, mid_y, cos_t, sin_t, gap, rate = obs.features
    cv2.putText(
        frame_bgr,
        f"mid=({mid_x*1000:+.0f},{mid_y*1000:+.0f})mm gap={gap*1000:.0f}mm rate={rate:+.2f}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    return frame_bgr
