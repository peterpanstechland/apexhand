"""File IPC and joint metadata for the interactive hand sandbox.

The Isaac process (``scripts/sandbox_hand.py``) and the FastAPI console share
``logs/webui/sandbox/{command.json,state.json,frame.jpg}``. No Isaac imports here.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from webui.runner import REPO_ROOT

_joints_path = REPO_ROOT / "source" / "pan_dexterous_lab" / "assets" / "joints.py"
_spec = importlib.util.spec_from_file_location("apex_joints_meta", _joints_path)
_joints = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_joints)
ACTUATED_LOGICAL = _joints.ACTUATED_LOGICAL
JOINT_LIMITS_DEG = _joints.JOINT_LIMITS_DEG

SANDBOX_DIR = REPO_ROOT / "logs" / "webui" / "sandbox"
COMMAND_PATH = SANDBOX_DIR / "command.json"
STATE_PATH = SANDBOX_DIR / "state.json"
FRAME_PATH = SANDBOX_DIR / "frame.jpg"

_FINGER_LABEL = {
    "thumb": "拇指",
    "index": "食指",
    "middle": "中指",
    "ring": "无名指",
    "pinky": "小指",
}
_JOINT_LABEL = {
    "thumb_j0": "对掌",
    "thumb_j1": "侧摆",
    "thumb_j2": "近端弯曲",
    "thumb_j3": "远端弯曲",
    "index_j0": "侧摆",
    "index_j1": "近端弯曲",
    "index_j2": "中段弯曲",
    "middle_j0": "侧摆",
    "middle_j1": "近端弯曲",
    "middle_j2": "中段弯曲",
    "ring_j0": "侧摆",
    "ring_j1": "近端弯曲",
    "ring_j2": "中段弯曲",
    "pinky_j0": "侧摆",
    "pinky_j1": "近端弯曲",
    "pinky_j2": "中段弯曲",
}

VIEW_LABELS = {
    "viewport": "自由相机（可拖）",
    "overhead": "俯视",
    "wrist": "腕部",
    "side": "侧面",
}


def _limits(name: str) -> tuple[float, float]:
    lo, hi = JOINT_LIMITS_DEG[name]
    return float(lo), float(hi)


def _clamp(name: str, value: float) -> float:
    lo, hi = _limits(name)
    return max(lo, min(hi, float(value)))


def joint_catalog() -> list[dict[str, Any]]:
    out = []
    for name in ACTUATED_LOGICAL:
        finger = name.split("_", 1)[0]
        lo, hi = _limits(name)
        out.append(
            {
                "id": name,
                "finger": finger,
                "finger_label": _FINGER_LABEL[finger],
                "label": _JOINT_LABEL[name],
                "min": lo,
                "max": hi,
                "step": 0.5,
            }
        )
    return out


def finger_groups() -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for joint in joint_catalog():
        key = joint["finger"]
        groups.setdefault(
            key,
            {"id": key, "label": joint["finger_label"], "joints": []},
        )
        groups[key]["joints"].append(joint)
    order = ["thumb", "index", "middle", "ring", "pinky"]
    return [groups[k] for k in order if k in groups]


def _open_pose() -> dict[str, float]:
    pose = {name: 0.0 for name in ACTUATED_LOGICAL}
    pose["thumb_j0"] = 15.0
    pose["thumb_j1"] = 8.0
    return {k: _clamp(k, v) for k, v in pose.items()}


def _fist_pose() -> dict[str, float]:
    pose = {name: 0.0 for name in ACTUATED_LOGICAL}
    pose.update(
        {
            "thumb_j0": 55.0,
            "thumb_j1": 28.0,
            "thumb_j2": 62.0,
            "thumb_j3": 58.0,
            "index_j1": 72.0,
            "index_j2": 82.0,
            "middle_j1": 74.0,
            "middle_j2": 84.0,
            "ring_j1": 74.0,
            "ring_j2": 84.0,
            "pinky_j1": 70.0,
            "pinky_j2": 80.0,
        }
    )
    return {k: _clamp(k, v) for k, v in pose.items()}


def _pinch_pose() -> dict[str, float]:
    pose = _open_pose()
    pose.update(
        {
            "thumb_j0": 42.0,
            "thumb_j1": 18.0,
            "thumb_j2": 38.0,
            "thumb_j3": 42.0,
            "index_j1": 48.0,
            "index_j2": 52.0,
        }
    )
    return {k: _clamp(k, v) for k, v in pose.items()}


def presets() -> list[dict[str, Any]]:
    return [
        {"id": "reset", "label": "复位", "hint": "重置场景，回到任务默认托姿", "targets": None},
        {"id": "open", "label": "张开", "hint": "手指伸直，方便看清关节", "targets": _open_pose()},
        {"id": "fist", "label": "握拳", "hint": "屈曲到接近上限，看耦合指尖", "targets": _fist_pose()},
        {"id": "pinch", "label": "捏合", "hint": "拇指和食指对捏，其余张开", "targets": _pinch_pose()},
    ]


def schema() -> dict[str, Any]:
    return {
        "joints": joint_catalog(),
        "fingers": finger_groups(),
        "presets": presets(),
        "views": [{"id": k, "label": v} for k, v in VIEW_LABELS.items()],
    }


def default_command() -> dict[str, Any]:
    return {
        "seq": 0,
        "reset": False,
        "pause": False,
        "stop": False,
        "reset_view": False,
        "view": "viewport",
        "targets_deg": {},
        "camera": None,
    }


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def write_command(payload: dict[str, Any]) -> dict[str, Any]:
    current = read_command()
    merged = default_command()
    merged.update(current)
    if payload.get("reset"):
        merged["targets_deg"] = {}
        for key, value in (payload.get("targets_deg") or {}).items():
            if key in JOINT_LIMITS_DEG:
                merged["targets_deg"][key] = _clamp(key, float(value))
    elif "targets_deg" in payload and payload["targets_deg"] is not None:
        targets = dict(merged.get("targets_deg") or {})
        for key, value in payload["targets_deg"].items():
            if key in JOINT_LIMITS_DEG:
                targets[key] = _clamp(key, float(value))
        merged["targets_deg"] = targets
    if payload.get("reset_view"):
        merged["camera"] = None
        merged["reset_view"] = True
        merged["view"] = "viewport"
    elif payload.get("camera") is not None:
        cam = dict(merged.get("camera") or {})
        incoming = payload["camera"]
        if isinstance(incoming, dict):
            for key in ("yaw", "pitch", "distance"):
                if incoming.get(key) is not None:
                    cam[key] = float(incoming[key])
            for key in ("target", "eye"):
                if incoming.get(key) is not None and len(incoming[key]) == 3:
                    cam[key] = [float(v) for v in incoming[key]]
        merged["camera"] = cam
        merged["reset_view"] = False
        merged["view"] = "viewport"
    for key in ("reset", "pause", "stop", "view"):
        if key in payload and payload[key] is not None:
            merged[key] = payload[key]
    merged["seq"] = int(merged.get("seq") or 0) + 1
    atomic_write_text(COMMAND_PATH, json.dumps(merged, indent=2))
    return merged


def read_command() -> dict[str, Any]:
    if not COMMAND_PATH.is_file():
        return default_command()
    try:
        data = json.loads(COMMAND_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return default_command()
    if not isinstance(data, dict):
        return default_command()
    return data


def read_state() -> dict[str, Any]:
    if not STATE_PATH.is_file():
        return {"ready": False, "message": "还没有启动交互仿真。"}
    try:
        data = json.loads(STATE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {"ready": False, "message": "状态文件还在写。"}
    if not isinstance(data, dict):
        return {"ready": False, "message": "状态文件损坏。"}
    return data
