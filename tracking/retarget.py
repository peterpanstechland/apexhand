"""MediaPipe 21 landmarks → Apex 16 actuated joints (radians).

Uses ``joints.ACTUATED_LOGICAL`` / ``JOINT_LIMITS_DEG`` as the only joint table.
Coupled DIPs are not returned; the SDK layer copies them from the source joints.
"""

from __future__ import annotations

import numpy as np

from real.joint_table import ACTUATED_LOGICAL, JOINT_LIMITS_DEG

# MediaPipe Hands indices (official topology).
_WRIST = 0
_THUMB = (1, 2, 3, 4)  # CMC, MCP, IP, TIP
_FINGER = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.zeros_like(v)
    return v / n


_ABD_JOINTS = {
    "thumb_j0",
    "index_j0",
    "middle_j0",
    "ring_j0",
    "pinky_j0",
}


def _flex(prox: np.ndarray, mid: np.ndarray, dist: np.ndarray) -> float:
    """Unsigned fold in the image plane. Webcam z is too noisy for 3D angles."""
    u = _unit(prox[:2] - mid[:2])
    v = _unit(dist[:2] - mid[:2])
    if np.linalg.norm(u) < 1e-8 or np.linalg.norm(v) < 1e-8:
        return 0.0
    interior = float(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0)))
    return max(0.0, np.pi - interior)


def _palm_basis(lm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal (side, forward, normal). Side points index → pinky."""
    wrist = lm[_WRIST]
    index_mcp = lm[_FINGER["index"][0]]
    pinky_mcp = lm[_FINGER["pinky"][0]]
    side = pinky_mcp - index_mcp
    forward = 0.5 * (index_mcp + pinky_mcp) - wrist
    normal = np.cross(side, forward)
    normal = _unit(normal)
    forward = _unit(forward)
    side = _unit(np.cross(forward, normal))
    forward = _unit(np.cross(normal, side))
    return side, forward, normal


def _project_palm(v: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return v - np.dot(v, normal) * normal


def _finger_abd(lm: np.ndarray, mcp_i: int, pip_i: int, basis) -> float:
    side, forward, normal = basis
    finger = _project_palm(lm[pip_i] - lm[mcp_i], normal)
    rest = _project_palm(lm[mcp_i] - lm[_WRIST], normal)
    finger_u, rest_u = _unit(finger), _unit(rest)
    if np.linalg.norm(finger_u) < 1e-8 or np.linalg.norm(rest_u) < 1e-8:
        return 0.0
    # Relative yaw in the palm plane (atan2 side, forward).
    def yaw(u: np.ndarray) -> float:
        return float(np.arctan2(np.dot(u, side), np.dot(u, forward)))

    return yaw(finger_u) - yaw(rest_u)


def pinch_amount(landmarks: np.ndarray) -> float:
    """1 when thumb and index tips meet, 0 when far. Uses 2D and 3D distance."""
    lm = np.asarray(landmarks, dtype=np.float64)
    palm = float(np.linalg.norm(lm[_FINGER["index"][0]][:2] - lm[_FINGER["pinky"][0]][:2])) + 1e-6
    tip_t, tip_i = lm[_THUMB[3]], lm[_FINGER["index"][3]]
    p2 = float(np.clip(1.0 - np.linalg.norm(tip_t[:2] - tip_i[:2]) / (0.50 * palm), 0.0, 1.0))
    p3 = float(np.clip(1.0 - np.linalg.norm(tip_t - tip_i) / (0.70 * palm), 0.0, 1.0))
    return max(p2, p3)


def _clamp_logical(name: str, q: float) -> float:
    lo_deg, hi_deg = JOINT_LIMITS_DEG[name]
    lo, hi = np.deg2rad(lo_deg), np.deg2rad(hi_deg)
    if name in _ABD_JOINTS:
        q = lo + hi - q
    return float(np.clip(q, lo, hi))


def landmarks_to_actuated(landmarks: np.ndarray) -> np.ndarray:
    """Map (21, 3) MediaPipe landmarks to ``ACTUATED_LOGICAL`` radians."""
    lm = np.asarray(landmarks, dtype=np.float64)
    if lm.shape != (21, 3):
        raise ValueError(f"expected landmarks (21, 3), got {lm.shape}")

    basis = _palm_basis(lm)
    side, _forward, normal = basis
    q: dict[str, float] = {}

    for name, (mcp, pip, dip, _tip) in _FINGER.items():
        q[f"{name}_j0"] = _finger_abd(lm, mcp, pip, basis)
        q[f"{name}_j1"] = _flex(lm[_WRIST], lm[mcp], lm[pip])
        q[f"{name}_j2"] = _flex(lm[mcp], lm[pip], lm[dip])

    cmc, mcp, ip, _tip = _THUMB
    thumb_dir = lm[mcp] - lm[cmc]
    index_dir = lm[_FINGER["index"][0]] - lm[_WRIST]
    thumb_p = _unit(_project_palm(thumb_dir, normal))
    index_p = _unit(_project_palm(index_dir, normal))
    if np.linalg.norm(thumb_p) < 1e-8 or np.linalg.norm(index_p) < 1e-8:
        thumb_abd = 0.0
    else:
        thumb_abd = float(np.arccos(np.clip(np.dot(thumb_p, index_p), -1.0, 1.0)))
    q["thumb_j0"] = thumb_abd
    q["thumb_j1"] = float(np.arcsin(np.clip(np.dot(_unit(thumb_dir), normal), -1.0, 1.0)))
    q["thumb_j2"] = _flex(lm[_WRIST], lm[cmc], lm[mcp])
    q["thumb_j3"] = _flex(lm[cmc], lm[mcp], lm[ip])

    # Extra curl from tip distance — additive so the pose stays continuous.
    pinch = pinch_amount(lm)
    q["thumb_j1"] += pinch * np.deg2rad(25.0)
    q["thumb_j2"] += pinch * np.deg2rad(28.0)
    q["thumb_j3"] += pinch * np.deg2rad(40.0)
    q["index_j1"] += pinch * np.deg2rad(12.0)
    q["index_j2"] += pinch * np.deg2rad(22.0)

    return np.array([_clamp_logical(name, q[name]) for name in ACTUATED_LOGICAL], dtype=np.float64)
