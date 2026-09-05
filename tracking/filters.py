"""EMA smoother for retargeted joint vectors."""

from __future__ import annotations

import numpy as np


class EMA:
    def __init__(self, alpha: float) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"EMA alpha must be in (0, 1], got {alpha}")
        self.alpha = float(alpha)
        self.y: np.ndarray | None = None

    def reset(self, y: np.ndarray | None = None) -> None:
        self.y = None if y is None else np.asarray(y, dtype=np.float64)

    def update(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.y is None:
            self.y = x.copy()
        else:
            self.y = self.alpha * x + (1.0 - self.alpha) * self.y
        return self.y
