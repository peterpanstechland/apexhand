#!/usr/bin/env python3
"""1-env interactive sandbox: stream RGB and follow web-UI joint sliders."""

from __future__ import annotations

import argparse
import io
import json
import math
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Interactive Apex-hand sandbox for the web console.")
parser.add_argument("--task", type=str, default="PAN-CoinHold-Apex-Play-v0")
parser.add_argument("--control-dir", type=str, default="logs/webui/sandbox")
parser.add_argument("--physics", type=str, default="physx")
parser.add_argument("--object", type=str, default="pan_coin_32mm")
parser.add_argument("--object2", type=str, default="")
parser.add_argument("--hand-pose", type=str, default="palm_down_knuckle")
parser.add_argument("--cameras", type=str, default="none")
parser.add_argument("--width", type=int, default=800)
parser.add_argument("--height", type=int, default=450)
AppLauncher.add_app_launcher_args(parser)
args_cli, _hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

import pan_dexterous_lab.tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import resolve_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from pan_dexterous_lab.assets.cameras import OVERHEAD_RGB256, SIDE_RGB128, WRIST_RGB128
from pan_dexterous_lab.assets.joints import ACTUATED_LOGICAL, JOINT_LIMITS_DEG, actuated_joint_names


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def _read_command(path: Path, fallback: dict) -> dict:
    if not path.is_file():
        return fallback
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return fallback
    return data if isinstance(data, dict) else fallback


def _to_uint8_rgb(frame) -> np.ndarray | None:
    if frame is None:
        return None
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    if not isinstance(frame, np.ndarray):
        return None
    if frame.ndim == 4:
        frame = frame[0]
    if frame.ndim != 3:
        return None
    pix = frame[..., :3]
    if pix.dtype != np.uint8:
        pix = np.clip(pix, 0, 255) if pix.max() > 1.5 else np.clip(pix * 255.0, 0, 255)
        pix = pix.astype(np.uint8)
    return pix


def _jpeg_bytes(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=78)
    return buf.getvalue()


def _disable_terminations(env_cfg) -> None:
    terms = getattr(env_cfg, "terminations", None)
    if terms is None:
        return
    fields = getattr(type(terms), "__dataclass_fields__", {})
    for name in fields:
        if name in {"class_type"}:
            continue
        try:
            setattr(terms, name, None)
        except Exception:
            pass


def _apply_cameras(env_cfg, layout: str) -> None:
    # Always keep a close overhead shot so the UI has a second view.
    env_cfg.scene.overhead_camera = OVERHEAD_RGB256
    if layout == "wrist_only":
        env_cfg.scene.wrist_camera = WRIST_RGB128
        env_cfg.scene.side_camera = None
    elif layout == "overhead_only":
        env_cfg.scene.wrist_camera = None
        env_cfg.scene.side_camera = None
    elif layout == "tri_view":
        env_cfg.scene.wrist_camera = WRIST_RGB128
        env_cfg.scene.side_camera = SIDE_RGB128


def _sensor_rgb(scene, name: str) -> np.ndarray | None:
    try:
        sensor = scene[name]
    except Exception:
        return None
    if sensor is None:
        return None
    try:
        output = sensor.data.output
        rgb = output["rgb"] if "rgb" in output else None
    except Exception:
        return None
    return _to_uint8_rgb(rgb)


def _logical_from_joint(name: str) -> str:
    if name.startswith("right_"):
        return name[6:]
    if name.startswith("left_"):
        return name[5:]
    return name


def _as_xyz(value) -> tuple[float, float, float] | None:
    if value is None:
        return None
    try:
        nums = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    if len(nums) != 3:
        return None
    return nums[0], nums[1], nums[2]


def spherical_from_eye(eye: tuple[float, float, float], lookat: tuple[float, float, float]) -> dict:
    dx = eye[0] - lookat[0]
    dy = eye[1] - lookat[1]
    dz = eye[2] - lookat[2]
    dist = math.sqrt(dx * dx + dy * dy + dz * dz) or 0.16
    pitch = math.asin(max(-1.0, min(1.0, dz / dist)))
    yaw = math.atan2(dy, dx)
    return {
        "yaw": yaw,
        "pitch": pitch,
        "distance": dist,
        "target": [lookat[0], lookat[1], lookat[2]],
        "eye": [eye[0], eye[1], eye[2]],
    }


def eye_from_spherical(cam: dict) -> tuple[float, float, float]:
    target = _as_xyz(cam.get("target")) or (0.0, 0.0, 0.5)
    yaw = float(cam.get("yaw") or 0.0)
    pitch = float(cam.get("pitch") or 0.4)
    dist = max(0.03, float(cam.get("distance") or 0.16))
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (target[0] + dist * cp * cy, target[1] + dist * cp * sy, target[2] + dist * sp)


def resolve_camera(cam: dict | None, fallback: dict) -> dict:
    if not cam:
        return dict(fallback)
    out = dict(fallback)
    for key in ("yaw", "pitch", "distance"):
        if cam.get(key) is not None:
            out[key] = float(cam[key])
    target = _as_xyz(cam.get("target"))
    if target is not None:
        out["target"] = list(target)
    eye = _as_xyz(cam.get("eye"))
    if eye is not None and cam.get("yaw") is None:
        lookat = tuple(out["target"])
        out.update(spherical_from_eye(eye, lookat))
    else:
        computed = eye_from_spherical(out)
        out["eye"] = [computed[0], computed[1], computed[2]]
    out["pitch"] = max(-1.35, min(1.35, float(out["pitch"])))
    out["distance"] = max(0.03, min(1.2, float(out["distance"])))
    return out


def apply_viewport_camera(uw, eye: tuple[float, float, float], lookat: tuple[float, float, float]) -> None:
    uw.cfg.viewer.eye = eye
    uw.cfg.viewer.lookat = lookat
    try:
        uw.sim.set_camera_view(eye, lookat)
    except Exception:
        pass
    rec = getattr(uw, "video_recorder", None)
    if rec is None:
        return
    rec.cfg.eye = eye
    rec.cfg.lookat = lookat
    capture = getattr(rec, "_capture", None)
    if capture is None:
        return
    if hasattr(capture, "update_camera"):
        try:
            capture.update_camera(eye, lookat)
            return
        except Exception:
            pass
    cap_cfg = getattr(capture, "cfg", None)
    if cap_cfg is not None:
        cap_cfg.eye = eye
        cap_cfg.lookat = lookat
        prim = getattr(cap_cfg, "camera_prim_path", "/OmniverseKit_Persp")
    else:
        prim = "/OmniverseKit_Persp"
    try:
        from isaacsim.core.rendering_manager import ViewportManager

        ViewportManager.set_camera_view(prim, eye=list(eye), target=list(lookat))
    except Exception:
        pass


def main() -> None:
    control = Path(args_cli.control_dir)
    control.mkdir(parents=True, exist_ok=True)
    command_path = control / "command.json"
    state_path = control / "state.json"
    frame_path = control / "frame.jpg"

    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    selected = [args_cli.physics, args_cli.object, args_cli.hand_pose]
    if args_cli.object2:
        selected.append(args_cli.object2)
    env_cfg = resolve_presets(env_cfg, selected=tuple(selected))
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.num_envs = 1
    if args_cli.physics == "newton_mjwarp":
        env_cfg.scene.clone_in_fabric = False
    env_cfg.episode_length_s = 1.0e6
    env_cfg.video_recorder.window_width = int(args_cli.width)
    env_cfg.video_recorder.window_height = int(args_cli.height)
    env_cfg.viewer.resolution = (int(args_cli.width), int(args_cli.height))
    _disable_terminations(env_cfg)
    _apply_cameras(env_cfg, args_cli.cameras)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env.reset()
    uw = env.unwrapped
    robot = uw.scene["robot"]
    side = "left" if any(n.startswith("left_") for n in robot.joint_names) else "right"
    act_names = actuated_joint_names(side)
    act_ids, _ = robot.find_joints(act_names, preserve_order=True)
    act_ids_t = torch.tensor(act_ids, device=uw.device, dtype=torch.int32)
    term = uw.action_manager.get_term("joint_pos")

    default_q = robot.data.default_joint_pos.torch[0, act_ids].detach().clone()
    user_q = default_q.unsqueeze(0).to(uw.device)

    def process_actions(actions: torch.Tensor) -> None:
        term._raw_actions[:] = actions
        term._processed_actions[:] = user_q
        term._prev_applied_actions[:] = user_q

    orig_apply = term.apply_actions

    def apply_actions() -> None:
        orig_apply()
        robot.write_joint_position_to_sim_index(position=user_q, joint_ids=act_ids_t)

    term.process_actions = process_actions
    term.apply_actions = apply_actions

    views = ["viewport"]
    for key, label in (("overhead_camera", "overhead"), ("wrist_camera", "wrist"), ("side_camera", "side")):
        try:
            if uw.scene[key] is not None:
                views.append(label)
        except Exception:
            pass

    zeros = torch.zeros(1, env.action_space.shape[-1], device=uw.device)
    cmd = {"targets_deg": {}, "reset": False, "pause": False, "stop": False, "view": "viewport"}
    last_reset_seq = -1
    last_view_seq = -1
    last_cam_key = None
    default_eye = tuple(float(v) for v in uw.cfg.viewer.eye)
    default_lookat = tuple(float(v) for v in uw.cfg.viewer.lookat)
    default_cam = spherical_from_eye(default_eye, default_lookat)
    current_cam = dict(default_cam)
    apply_viewport_camera(uw, default_eye, default_lookat)
    step = 0
    t0 = time.time()
    fps = 0.0

    def targets_from_cmd(payload: dict) -> torch.Tensor:
        deg = payload.get("targets_deg") or {}
        out = user_q.clone()
        for i, logical in enumerate(ACTUATED_LOGICAL):
            if logical not in deg:
                continue
            lo, hi = JOINT_LIMITS_DEG[logical]
            out[0, i] = math.radians(max(lo, min(hi, float(deg[logical]))))
        return out

    def current_deg() -> dict[str, float]:
        q = robot.data.joint_pos.torch[0, act_ids].detach().cpu().tolist()
        return {logical: round(math.degrees(float(v)), 2) for logical, v in zip(ACTUATED_LOGICAL, q)}

    def default_deg() -> dict[str, float]:
        return {logical: round(math.degrees(float(v)), 2) for logical, v in zip(ACTUATED_LOGICAL, default_q.tolist())}

    def object_xyz() -> list[float] | None:
        try:
            pos = uw.scene["object"].data.root_pos_w[0].detach().cpu().tolist()
            return [round(float(x), 4) for x in pos]
        except Exception:
            return None

    def grab_frame(view: str) -> np.ndarray | None:
        if view == "overhead":
            rgb = _sensor_rgb(uw.scene, "overhead_camera")
            if rgb is not None:
                return rgb
        elif view == "wrist":
            rgb = _sensor_rgb(uw.scene, "wrist_camera")
            if rgb is not None:
                return rgb
        elif view == "side":
            rgb = _sensor_rgb(uw.scene, "side_camera")
            if rgb is not None:
                return rgb
        return _to_uint8_rgb(env.render())

    def write_state(ready: bool, message: str, view: str) -> None:
        payload = {
            "ready": ready,
            "message": message,
            "task": args_cli.task,
            "side": side,
            "view": view,
            "views": views,
            "fps": round(fps, 1),
            "step": step,
            "joints": current_deg(),
            "default_deg": default_deg(),
            "object_xyz": object_xyz(),
            "camera": current_cam,
            "updated_at": time.time(),
        }
        _atomic_write_text(state_path, json.dumps(payload))

    write_state(False, "仿真已起来，正在出第一帧…", "viewport")
    print("[sandbox] env ready; streaming to", control, flush=True)

    while simulation_app.is_running():
        cmd = _read_command(command_path, cmd)
        if cmd.get("stop"):
            print("[sandbox] stop requested", flush=True)
            break
        seq = int(cmd.get("seq") or 0)
        if cmd.get("reset") and seq != last_reset_seq:
            env.reset()
            default_q = robot.data.default_joint_pos.torch[0, act_ids].detach().clone()
            user_q = default_q.unsqueeze(0).to(uw.device)
            last_reset_seq = seq
        if cmd.get("targets_deg"):
            user_q = targets_from_cmd(cmd)
        if cmd.get("reset_view") and seq != last_view_seq:
            current_cam = dict(default_cam)
            last_view_seq = seq
        elif cmd.get("camera"):
            current_cam = resolve_camera(cmd.get("camera"), current_cam)
        eye = tuple(current_cam["eye"])
        lookat = tuple(current_cam["target"])
        cam_key = (round(eye[0], 4), round(eye[1], 4), round(eye[2], 4), round(lookat[0], 4), round(lookat[1], 4), round(lookat[2], 4))
        cam_changed = cam_key != last_cam_key
        if cam_changed:
            apply_viewport_camera(uw, eye, lookat)
            last_cam_key = cam_key
        view = str(cmd.get("view") or "viewport")
        if not cmd.get("pause"):
            env.step(zeros)
        elif cam_changed or step % 4 == 0:
            uw.sim.render()
        step += 1
        if cam_changed or step % 2 == 0:
            elapsed = time.time() - t0
            fps = step / max(elapsed, 1e-3)
            rgb = grab_frame(view)
            if rgb is not None:
                _atomic_write_bytes(frame_path, _jpeg_bytes(rgb))
            write_state(True, "拖画面转视角，拖滑条玩手。", view)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
