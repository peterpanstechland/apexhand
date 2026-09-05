"""Load ``joints.py`` without importing Isaac via ``pan_dexterous_lab.assets``."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "source" / "pan_dexterous_lab" / "assets" / "joints.py"
_NAME = "apex_hand_joints"

if _NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(_NAME, _PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load joint table from {_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[_NAME] = module

from apex_hand_joints import (  # noqa: E402
    ACTUATED_LOGICAL,
    COUPLED_LOGICAL,
    COUPLED_SOURCE_LOGICAL,
    JOINT_LIMITS_DEG,
)

# Must match ``apex_cfg._KNUCKLE_ROLL_JOINT_POS`` (action 0).
DEFAULT_ACTUATED_POS = {
    "thumb_j0": 0.611,
    "thumb_j1": 0.262,
    "thumb_j2": 0.489,
    "thumb_j3": 0.349,
    "index_j0": 0.00,
    "index_j1": 0.262,
    "index_j2": 0.524,
    "middle_j0": 0.00,
    "middle_j1": 0.262,
    "middle_j2": 0.524,
    "ring_j0": 0.00,
    "ring_j1": 0.262,
    "ring_j2": 0.524,
    "pinky_j0": 0.00,
    "pinky_j1": 0.262,
    "pinky_j2": 0.524,
}

# Must match ``ActionsCfg.joint_pos.scale``.
ACTION_SCALE = {
    "thumb_j0": 0.5,
    "thumb_j1": 0.5,
    "thumb_j2": 0.5,
    "thumb_j3": 0.5,
    "index_j0": 0.04,
    "index_j1": 0.5,
    "index_j2": 0.5,
    "middle_j0": 0.04,
    "middle_j1": 0.5,
    "middle_j2": 0.5,
    "ring_j0": 0.04,
    "ring_j1": 0.5,
    "ring_j2": 0.5,
    "pinky_j0": 0.04,
    "pinky_j1": 0.5,
    "pinky_j2": 0.5,
}

__all__ = [
    "ACTUATED_LOGICAL",
    "COUPLED_LOGICAL",
    "COUPLED_SOURCE_LOGICAL",
    "JOINT_LIMITS_DEG",
    "DEFAULT_ACTUATED_POS",
    "ACTION_SCALE",
]
