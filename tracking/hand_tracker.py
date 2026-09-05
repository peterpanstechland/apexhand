"""MediaPipe Hands: webcam frame → 21 landmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_MP_HANDS = None
_MP_CONNECTIONS = None


def _hands_api():
    global _MP_HANDS, _MP_CONNECTIONS
    if _MP_HANDS is None:
        import mediapipe as mp

        _MP_HANDS = mp.solutions.hands
        _MP_CONNECTIONS = _MP_HANDS.HAND_CONNECTIONS
    return _MP_HANDS, _MP_CONNECTIONS


@dataclass(frozen=True)
class TrackedHand:
    """One detected hand in MediaPipe image coordinates (x,y in [0,1], z relative)."""

    landmarks: np.ndarray  # (21, 3)
    handedness: str  # "Left" | "Right"
    score: float


class HandTracker:
    def __init__(
        self,
        *,
        max_hands: int = 1,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.6,
        model_complexity: int = 1,
    ) -> None:
        hands_api, connections = _hands_api()
        self.connections = connections
        self._impl = hands_api.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            model_complexity=model_complexity,
        )

    def process(self, bgr) -> TrackedHand | None:
        rgb = bgr[:, :, ::-1]
        result = self._impl.process(rgb)
        if not result.multi_hand_landmarks:
            return None
        lm = result.multi_hand_landmarks[0]
        pts = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float64)
        handed = "Right"
        score = 0.0
        if result.multi_handedness:
            cls = result.multi_handedness[0].classification[0]
            handed = cls.label
            score = float(cls.score)
        return TrackedHand(landmarks=pts, handedness=handed, score=score)

    def close(self) -> None:
        self._impl.close()
