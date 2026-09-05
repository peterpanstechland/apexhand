"""Rebuild a policy's observation vector on hardware from its exported spec.

``scripts/export_onnx.py`` writes the actor's observation layout term by term
into ``joint_map.json``. This module fills each term from a named provider and
refuses to run if any term is unaccounted for.

The point is to make observation drift a startup error. The first hardware run
concatenated a hand-written 88-element layout with literal ``np.zeros(15)``
padding; when the policy changed shape the vector silently misaligned and the
hand barely moved. Nothing here knows a total dimension -- it all comes from
the spec, and a term this file cannot supply is a loud failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from real.joint_table import ACTUATED_LOGICAL, JOINT_LIMITS_DEG

# Must match ``joint_vel_obs_term``'s scale in the env cfg.
_JOINT_VEL_SCALE = 0.2


@dataclass
class RuntimeState:
    """Everything the providers can draw on for one control tick."""

    q: np.ndarray
    """Actuated joint angles, rad, in ``ACTUATED_LOGICAL`` order."""
    qd: np.ndarray
    """Actuated joint velocities, rad/s."""
    last_action: np.ndarray
    """Previous policy output, pre-scaling."""
    pair: np.ndarray | None = None
    """6-dim baoding pair features from ``real.ball_tracker``, or None if the
    tracker lost sight of the balls this tick."""
    spin: float = 1.0
    """Commanded rotation direction, +1 or -1."""


Provider = Callable[[RuntimeState], np.ndarray]


def joint_pos_limit_normalized(state: RuntimeState) -> np.ndarray:
    """Mirror of ``mdp.joint_pos_limit_normalized``: limits mapped to [-1, 1]."""
    out = np.empty(len(ACTUATED_LOGICAL), dtype=np.float32)
    for i, name in enumerate(ACTUATED_LOGICAL):
        lo, hi = np.deg2rad(JOINT_LIMITS_DEG[name])
        out[i] = np.clip(2.0 * (state.q[i] - lo) / (hi - lo) - 1.0, -1.0, 1.0)
    return out


def joint_vel_rel(state: RuntimeState) -> np.ndarray:
    return (state.qd * _JOINT_VEL_SCALE).astype(np.float32)


def last_action(state: RuntimeState) -> np.ndarray:
    return state.last_action.astype(np.float32)


def pair(state: RuntimeState) -> np.ndarray:
    """Baoding pair features. Zeros while the tracker has no fix.

    Zeros are the honest reading here: the observation is palm-relative, so an
    all-zero vector says "pair centred, no rotation", which is also the reset
    state the policy saw in sim. The caller is expected to stop commanding
    motion when the fix is lost -- see ``policy_runner``.
    """
    if state.pair is None:
        return np.zeros(6, dtype=np.float32)
    return state.pair.astype(np.float32)


def spin_command(state: RuntimeState) -> np.ndarray:
    return np.array([state.spin], dtype=np.float32)


PROVIDERS: dict[str, Provider] = {
    "joint_pos": joint_pos_limit_normalized,
    "joint_vel": joint_vel_rel,
    "last_action": last_action,
    "pair": pair,
    "spin_command": spin_command,
}


@dataclass
class ObsAssembler:
    """Builds observations in the exported term order.

    Construct via :meth:`from_spec` so the spec is validated before the hand is
    ever enabled.
    """

    terms: list[tuple[str, int, Provider]]
    dim: int
    needs_tracker: bool = field(default=False)

    @classmethod
    def from_spec(cls, spec: list[dict], onnx_dim: int | None = None) -> "ObsAssembler":
        missing = [t["name"] for t in spec if t["name"] not in PROVIDERS]
        if missing:
            raise RuntimeError(
                f"no hardware provider for observation term(s) {missing}. "
                f"This policy needs state the real hand cannot measure -- it was "
                f"probably trained with privileged terms in its actor group. "
                f"Known providers: {sorted(PROVIDERS)}"
            )
        terms = [(t["name"], int(t["dim"]), PROVIDERS[t["name"]]) for t in spec]
        dim = sum(d for _, d, _ in terms)
        if onnx_dim is not None and dim != onnx_dim:
            raise RuntimeError(
                f"spec sums to {dim} dims but the ONNX input wants {onnx_dim}. "
                f"joint_map.json and policy.onnx are out of sync."
            )
        return cls(terms=terms, dim=dim, needs_tracker=any(n == "pair" for n, _, _ in terms))

    def __call__(self, state: RuntimeState) -> np.ndarray:
        parts = []
        for name, want, provider in self.terms:
            value = np.asarray(provider(state), dtype=np.float32).reshape(-1)
            if value.size != want:
                raise RuntimeError(f"term {name!r} produced {value.size} dims, spec says {want}")
            parts.append(value)
        return np.concatenate(parts)

    def describe(self) -> str:
        body = "  ".join(f"{n}:{d}" for n, d, _ in self.terms)
        return f"obs {self.dim} dims = {body}"
