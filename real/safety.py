"""Policy → hand safety filter. Must sit between ONNX and the SDK."""

from __future__ import annotations


class SafetyFilter:
    """Joint / velocity / acceleration limits + emergency open-hand.

    First-day air-test defaults from the project plan:
    - clip policy Δq
    - low-pass
    - consecutive-anomaly → OPEN HAND
    """

    def filter(self, q_target: list[float], q_current: list[float]) -> list[float]:
        raise NotImplementedError("No Apex Hand on this host.")
