"""Clamp retarget / policy targets to official limits and a per-tick Δq cap."""

from __future__ import annotations

import numpy as np

from real.joint_table import ACTUATED_LOGICAL, JOINT_LIMITS_DEG

# SDK firmware max is 1.7453 rad for 100°; raw deg2rad(100) overshoots and faults.
_LIMIT_MARGIN_RAD = 1e-3


def _limit_rad(name: str) -> tuple[float, float]:
    lo_deg, hi_deg = JOINT_LIMITS_DEG[name]
    return float(np.deg2rad(lo_deg) + _LIMIT_MARGIN_RAD), float(np.deg2rad(hi_deg) - _LIMIT_MARGIN_RAD)


def clamp_joint(name: str, q: float) -> float:
    lo, hi = _limit_rad(name)
    return float(np.clip(q, lo, hi))


class SafetyFilter:
    """Joint limits + max step. Overcurrent on a finger freezes it and opens flex a bit."""

    def __init__(
        self,
        max_step_rad: float,
        names: list[str] | None = None,
        current_limit_a: float = 0.0,
        backoff_rad: float = 0.04,
    ) -> None:
        self.names = list(names or ACTUATED_LOGICAL)
        self.max_step_rad = float(max_step_rad)
        self.current_limit_a = float(current_limit_a)
        self.backoff_rad = float(backoff_rad)
        self.lo = np.array([_limit_rad(n)[0] for n in self.names], dtype=np.float64)
        self.hi = np.array([_limit_rad(n)[1] for n in self.names], dtype=np.float64)
        self._last: np.ndarray | None = None
        self._idle: dict[str, list[float]] = {}
        self._thr: dict[str, float] = {}
        self.contact: list[str] = []

    def reset(self, q: np.ndarray | None = None) -> None:
        self._last = None if q is None else np.asarray(q, dtype=np.float64)
        self._idle.clear()
        self._thr.clear()
        self.contact = []

    def _threshold(self, finger: str, amp: float) -> float:
        if self.current_limit_a > 0:
            return self.current_limit_a
        samples = self._idle.setdefault(finger, [])
        if len(samples) < 20:
            samples.append(amp)
            peak = max(samples) if samples else 0.0
            self._thr[finger] = max(0.18, peak * 2.4 + 0.06)
        return self._thr.get(finger, 0.25)

    def filter(self, q_target, q_current=None, finger_current=None, pinch: float = 0.0) -> np.ndarray:
        q = np.clip(np.asarray(q_target, dtype=np.float64), self.lo, self.hi)
        ref = self._last
        if ref is None and q_current is not None:
            ref = np.asarray(q_current, dtype=np.float64)
        if ref is not None:
            q = ref + np.clip(q - ref, -self.max_step_rad, self.max_step_rad)
            q = np.clip(q, self.lo, self.hi)

        self.contact = []
        if finger_current and ref is not None:
            blocked = set()
            for finger, amp in finger_current.items():
                if amp >= self._threshold(finger, amp):
                    blocked.add(finger)
                    self.contact.append(f"{finger}:{amp:.2f}A")
            # Intentional thumb–index pinch looks like a collision; do not open those two.
            if pinch > 0.2:
                blocked.difference_update({"thumb", "index"})
            if blocked:
                for i, name in enumerate(self.names):
                    finger = name.split("_", 1)[0]
                    if finger not in blocked:
                        continue
                    # Hold abduction; ease flexion toward 0 (open).
                    if name.endswith(("_j1", "_j2", "_j3")):
                        q[i] = ref[i] + float(np.clip(0.0 - ref[i], -self.backoff_rad, self.backoff_rad))
                    else:
                        q[i] = ref[i]
                q = np.clip(q, self.lo, self.hi)

        self._last = q
        return q
