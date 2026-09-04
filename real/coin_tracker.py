"""External-vision coin pose. Unimplemented: no camera in the one-day sprint."""

from __future__ import annotations


class CoinTracker:
    def estimate_pose(self) -> tuple[list[float], list[float]]:
        raise NotImplementedError("Use RealSense / USB cam + PΛN segmentation later.")
