"""Turn a recipe dict into an argv and manage the Isaac subprocess."""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_SH = REPO_ROOT / "env.sh"
RUNS_ROOT = REPO_ROOT / "logs" / "webui" / "runs"
RECIPES_DIR = REPO_ROOT / "configs" / "recipes"

_LOG_DIR_RE = re.compile(r"Logging experiment in directory:\s+(\S+)")
_RUN_NAME_RE = re.compile(r"Exact experiment name requested from command line:\s+(\S+)")

_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}


def _yaml_load(text: str) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError("recipe must be a mapping")
    return data


def _yaml_dump(data: dict[str, Any]) -> str:
    import yaml

    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False)


def list_recipes() -> list[dict[str, Any]]:
    RECIPES_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for path in sorted(RECIPES_DIR.glob("*.yaml")):
        try:
            data = _yaml_load(path.read_text())
        except Exception as exc:
            data = {"name": path.stem, "error": str(exc)}
        data.setdefault("name", path.stem)
        data["file"] = str(path.relative_to(REPO_ROOT))
        out.append(data)
    return out


def save_recipe(name: str, data: dict[str, Any]) -> Path:
    RECIPES_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "recipe"
    path = RECIPES_DIR / f"{safe}.yaml"
    payload = dict(data)
    payload["name"] = safe
    path.write_text(_yaml_dump(payload))
    return path


def delete_recipe(name: str) -> bool:
    path = RECIPES_DIR / f"{name}.yaml"
    if path.is_file():
        path.unlink()
        return True
    return False


def format_override(path: str, value: Any) -> str:
    if isinstance(value, bool):
        lit = "true" if value else "false"
    elif isinstance(value, (list, tuple)):
        inner = ",".join(str(v) for v in value)
        lit = f"[{inner}]"
    elif value is None:
        lit = "null"
    else:
        lit = str(value)
    return f"{path}={lit}"


def build_argv(recipe: dict[str, Any], mode: str | None = None) -> list[str]:
    """Build the python argv (no ``env.sh`` wrapper) for train/play/helpers."""
    mode = mode or recipe.get("mode") or "train"
    task = recipe.get("task") or "PAN-CoinHold-Apex-v0"
    cameras = recipe.get("cameras") or recipe.get("camera_layout") or "none"
    if cameras != "none" and "Vision" not in task and "Baoding" not in task and mode in {
        "train",
        "play",
        "preview",
        "probe",
        "sandbox",
    }:
        task = task.replace("PAN-CoinHold-Apex-v0", "PAN-CoinHold-Apex-Vision-v0")
        task = task.replace("PAN-CoinHold-Apex-Play-v0", "PAN-CoinHold-Apex-Vision-Play-v0")
    cli = dict(recipe.get("cli") or {})
    overrides = dict(recipe.get("overrides") or {})
    extras = list(recipe.get("hydra") or [])

    if mode == "play":
        script = REPO_ROOT / "scripts" / "play.py"
        play_task = recipe.get("play_task") or task.replace("-v0", "-Play-v0")
        if "Play" not in play_task and task.endswith("-v0"):
            play_task = task[:-3] + "-Play-v0"
        if cameras != "none" and "Vision" not in play_task:
            play_task = play_task.replace("PAN-CoinHold-Apex-Play-v0", "PAN-CoinHold-Apex-Vision-Play-v0")
        task = play_task
    elif mode == "preview":
        script = REPO_ROOT / "scripts" / "preview_scene.py"
    elif mode == "probe":
        script = REPO_ROOT / "scripts" / "reward_probe.py"
    elif mode == "sim2sim":
        script = REPO_ROOT / "scripts" / "sim2sim_check.py"
    elif mode == "export_mjcf":
        script = REPO_ROOT / "scripts" / "export_mjcf.py"
    elif mode == "export_onnx":
        script = REPO_ROOT / "scripts" / "export_onnx.py"
    elif mode == "check_reward":
        script = REPO_ROOT / "scripts" / "check_user_reward.py"
    elif mode == "sandbox":
        script = REPO_ROOT / "scripts" / "sandbox_hand.py"
        play_task = recipe.get("play_task") or task.replace("-v0", "-Play-v0")
        if "Play" not in play_task and task.endswith("-v0"):
            play_task = task[:-3] + "-Play-v0"
        if cameras != "none" and "Vision" not in play_task:
            play_task = play_task.replace("PAN-CoinHold-Apex-Play-v0", "PAN-CoinHold-Apex-Vision-Play-v0")
        task = play_task
    else:
        script = REPO_ROOT / "scripts" / "train.py"

    argv = ["python", str(script), "--task", task, "--headless"]

    if mode in {"train", "play", "preview"}:
        num_envs = cli.get("num_envs")
        if num_envs is not None:
            argv += ["--num_envs", str(int(num_envs))]
    if mode == "train" and cli.get("max_iterations") is not None:
        argv += ["--max_iterations", str(int(cli["max_iterations"]))]
    if cli.get("seed") is not None:
        argv += ["--seed", str(int(cli["seed"]))]
    if cli.get("video") or mode == "play":
        argv += ["--video"]
        if cli.get("video_length"):
            argv += ["--video_length", str(int(cli["video_length"]))]
    if cli.get("run_name"):
        argv += ["--run_name", str(cli["run_name"])]
    if cli.get("experiment_name"):
        argv += ["--experiment_name", str(cli["experiment_name"])]
    if cli.get("resume"):
        argv += ["--resume"]
    if cli.get("load_run"):
        argv += ["--load_run", str(cli["load_run"])]
    if cli.get("checkpoint"):
        argv += ["--checkpoint", str(cli["checkpoint"])]
    if mode == "export_mjcf" and cli.get("mjcf_out"):
        argv += ["--out", str(cli["mjcf_out"])]
    if mode == "sim2sim" and cli.get("checkpoint"):
        pass  # already added
    if mode == "preview":
        argv += ["--steps", str(int(cli.get("preview_steps", 8)))]
        if cli.get("preview_out"):
            argv += ["--out", str(cli["preview_out"])]
    if mode == "sandbox":
        argv += [
            "--control-dir",
            "logs/webui/sandbox",
            "--physics",
            str(recipe.get("physics") or "physx"),
            "--object",
            str(recipe.get("object") or "pan_coin_32mm"),
            "--hand-pose",
            str(recipe.get("hand_pose") or "palm_down_knuckle"),
            "--cameras",
            str(cameras),
        ]
        if recipe.get("object2"):
            argv += ["--object2", str(recipe["object2"])]

    physics = recipe.get("physics") or "physx"
    argv.append(f"physics={physics}")
    if physics == "newton_mjwarp":
        extras.append("env.scene.clone_in_fabric=false")

    if recipe.get("object"):
        extras.append(f"env.scene.object={recipe['object']}")
    if recipe.get("hand_pose"):
        extras.append(f"env.scene.robot={recipe['hand_pose']}")
    if recipe.get("object2"):
        extras.append(f"env.scene.object2={recipe['object2']}")

    if cameras == "wrist_only":
        extras += ["env.scene.overhead_camera=off", "env.scene.side_camera=off", "env.scene.wrist_camera=rgb128"]
    elif cameras == "overhead_only":
        extras += ["env.scene.wrist_camera=off", "env.scene.side_camera=off", "env.scene.overhead_camera=rgb128"]
    elif cameras == "tri_view":
        extras += [
            "env.scene.wrist_camera=rgb128",
            "env.scene.overhead_camera=rgb128",
            "env.scene.side_camera=rgb128",
        ]

    for path, value in overrides.items():
        if path.startswith("--"):
            continue
        extras.append(format_override(path, value))

    argv.extend(extras)
    return argv


def wrap_with_env(argv: list[str], extra_env: dict[str, str] | None = None) -> list[str]:
    inner = "source " + shlex.quote(str(ENV_SH)) + " && " + " ".join(shlex.quote(a) for a in argv)
    return ["bash", "-lc", inner]


def _read_tail(path: Path, n: int = 200) -> str:
    if not path.is_file():
        return ""
    data = path.read_bytes()
    if len(data) > 512_000:
        data = data[-512_000:]
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    return "\n".join(lines[-n:])


def _watch(job_id: str) -> None:
    job = _jobs[job_id]
    proc: subprocess.Popen = job["proc"]
    log_path = Path(job["log_path"])
    isaac_dir = None
    with log_path.open("a", encoding="utf-8") as log:
        assert proc.stdout is not None
        for raw in proc.stdout:
            log.write(raw)
            log.flush()
            job["log_bytes"] = log_path.stat().st_size
            m = _LOG_DIR_RE.search(raw)
            if m:
                isaac_dir = m.group(1)
                job["isaac_log_dir"] = isaac_dir
            m2 = _RUN_NAME_RE.search(raw)
            if m2:
                job["isaac_run_name"] = m2.group(1)
    rc = proc.wait()
    job["returncode"] = rc
    job["ended_at"] = time.time()
    job["status"] = "succeeded" if rc == 0 else "failed"
    meta_path = Path(job["dir"]) / "job.json"
    _write_meta(job)
    chain = job.get("recipe", {}).get("chain")
    if rc == 0 and chain and job.get("mode") == "train":
        nxt = dict(chain)
        nxt.setdefault("cli", {})
        if job.get("isaac_run_name"):
            nxt["cli"]["resume"] = True
            nxt["cli"]["load_run"] = job["isaac_run_name"]
        if isinstance(chain, dict) and chain.get("task"):
            nxt["task"] = chain["task"]
        try:
            start_job(nxt, mode="train", parent=job_id)
        except Exception as exc:
            job["chain_error"] = str(exc)
    _write_meta(job)
    _ = meta_path


def _write_meta(job: dict[str, Any]) -> None:
    skip = {"proc", "thread"}
    payload = {k: v for k, v in job.items() if k not in skip}
    Path(job["dir"]).mkdir(parents=True, exist_ok=True)
    (Path(job["dir"]) / "job.json").write_text(json.dumps(payload, indent=2, default=str))


def _running_jobs() -> list[dict[str, Any]]:
    with _lock:
        return [job for job in _jobs.values() if job.get("status") == "running"]


def stop_jobs_by_mode(*modes: str) -> None:
    for job in _running_jobs():
        if job.get("mode") in modes:
            try:
                stop_job(job["id"])
            except KeyError:
                pass


def start_job(recipe: dict[str, Any], mode: str | None = None, parent: str | None = None) -> dict[str, Any]:
    mode = mode or recipe.get("mode") or "train"
    if mode == "sandbox":
        stop_jobs_by_mode("sandbox")
        busy = [job["mode"] for job in _running_jobs()]
        if busy:
            raise RuntimeError("先停掉正在跑的 Isaac 任务再玩手。12GB 显存撑不住两个 Kit。")
        from webui.sandbox import COMMAND_PATH, STATE_PATH, atomic_write_text, default_command

        atomic_write_text(COMMAND_PATH, json.dumps(default_command()))
        atomic_write_text(STATE_PATH, json.dumps({"ready": False, "message": "正在启动 Kit…"}))
    elif mode in {"train", "play", "preview", "probe", "export_onnx", "export_mjcf", "sim2sim", "check_reward"}:
        stop_jobs_by_mode("sandbox")
    argv = build_argv(recipe, mode=mode)
    wrapped = wrap_with_env(argv)
    job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    job_dir = RUNS_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    log_path = job_dir / "stdout.log"
    (job_dir / "recipe.yaml").write_text(_yaml_dump(recipe))
    (job_dir / "argv.txt").write_text(" ".join(shlex.quote(a) for a in argv) + "\n")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        wrapped,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
        env=env,
    )
    job = {
        "id": job_id,
        "mode": mode,
        "status": "running",
        "pid": proc.pid,
        "argv": argv,
        "recipe": recipe,
        "dir": str(job_dir),
        "log_path": str(log_path),
        "started_at": time.time(),
        "ended_at": None,
        "returncode": None,
        "isaac_log_dir": None,
        "isaac_run_name": None,
        "parent": parent,
        "log_bytes": 0,
        "proc": proc,
    }
    thread = threading.Thread(target=_watch, args=(job_id,), daemon=True)
    job["thread"] = thread
    with _lock:
        _jobs[job_id] = job
    _write_meta(job)
    thread.start()
    return public_job(job)


def stop_job(job_id: str) -> dict[str, Any]:
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        raise KeyError(job_id)
    proc: subprocess.Popen | None = job.get("proc")
    if proc and proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        job["status"] = "stopped"
        job["ended_at"] = time.time()
        job["returncode"] = proc.returncode
        _write_meta(job)
    return public_job(job)


def public_job(job: dict[str, Any]) -> dict[str, Any]:
    skip = {"proc", "thread", "recipe"}
    out = {}
    for key, value in job.items():
        if key in skip:
            continue
        if type(value).__module__.startswith("subprocess") or type(value).__name__ in {"Thread", "lock"}:
            continue
        out[key] = value
    recipe = job.get("recipe") or {}
    out["task"] = recipe.get("task") or job.get("task")
    out["recipe_name"] = recipe.get("name") or job.get("recipe_name")
    return out


def get_job(job_id: str) -> dict[str, Any] | None:
    with _lock:
        job = _jobs.get(job_id)
    if job:
        return job
    meta = RUNS_ROOT / job_id / "job.json"
    if meta.is_file():
        data = json.loads(meta.read_text())
        if data.get("status") == "running":
            data["status"] = "orphaned"
        return data
    return None


def list_jobs() -> list[dict[str, Any]]:
    seen = set()
    out = []
    with _lock:
        for job in _jobs.values():
            out.append(public_job(job))
            seen.add(job["id"])
    if RUNS_ROOT.is_dir():
        for meta in sorted(RUNS_ROOT.glob("*/job.json"), reverse=True):
            jid = meta.parent.name
            if jid in seen:
                continue
            try:
                data = json.loads(meta.read_text())
            except Exception:
                continue
            if data.get("status") == "running":
                data["status"] = "orphaned"
            out.append(public_job(data))
    out.sort(key=lambda j: j.get("started_at") or 0, reverse=True)
    return out[:80]


def job_log(job_id: str, tail: int = 200) -> str:
    job = get_job(job_id)
    if not job:
        return ""
    return _read_tail(Path(job["log_path"]), tail)
