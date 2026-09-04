"""Catch beginner PPO / reward mistakes before a long Isaac run starts."""

from __future__ import annotations

from typing import Any

from webui.schema import PARAMS


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _pair(value: Any, default: list[float]) -> list[float]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return [float(value[0]), float(value[1])]
    return list(default)


def check_recipe(recipe: dict[str, Any]) -> list[dict[str, str]]:
    """Return alerts: each item is {level, code, message} with level info|warn|error."""
    cli = recipe.get("cli") or {}
    overrides = recipe.get("overrides") or {}
    defaults = {p.id: p.default for p in PARAMS}
    path_to_id = {p.path: p.id for p in PARAMS}

    def val(param_id: str, path: str | None = None) -> Any:
        if path and path in overrides:
            return overrides[path]
        # allow overrides keyed by param id
        if param_id in overrides:
            return overrides[param_id]
        if param_id in cli:
            return cli[param_id]
        return defaults.get(param_id)

    alerts: list[dict[str, str]] = []

    num_envs = _as_int(val("num_envs", "--num_envs"), 512)
    steps = _as_int(val("num_steps_per_env", "agent.num_steps_per_env"), 24)
    mini = _as_int(val("num_mini_batches", "agent.algorithm.num_mini_batches"), 4)
    total = num_envs * steps
    if mini <= 0:
        alerts.append({"level": "error", "code": "mini_batches", "message": "小批次数必须 ≥ 1。"})
    elif total % mini != 0:
        alerts.append(
            {
                "level": "error",
                "code": "mini_batches",
                "message": (
                    f"num_envs × num_steps_per_env = {num_envs}×{steps}={total}，"
                    f"不能被小批次数 {mini} 整除。PPO 会直接报错。"
                ),
            }
        )

    gamma = _as_float(val("gamma", "agent.algorithm.gamma"), 0.998)
    episode_s = _as_float(val("episode_length_s", "env.episode_length_s"), 3.0)
    decimation = _as_int(val("decimation", "env.decimation"), 4)
    dt = _as_float(val("sim_dt", "env.sim.dt"), 1.0 / 240.0)
    ep_steps = max(1, int(round(episode_s / (dt * decimation))))
    horizon = 1.0 / max(1e-6, 1.0 - gamma)
    if horizon > ep_steps * 1.8:
        alerts.append(
            {
                "level": "warn",
                "code": "gamma_horizon",
                "message": (
                    f"γ={gamma} 的有效视野约 {horizon:.0f} 步，但一局只有 {ep_steps} 步"
                    f"（{episode_s}s / (dt×decimation)）。远处奖励几乎用不上，"
                    f"Hold 可以试 γ=0.99，或把一局加长。"
                ),
            }
        )

    weights: list[tuple[str, float]] = []
    for p in PARAMS:
        if ".weight" in p.path and p.kind == "hydra":
            w = _as_float(val(p.id, p.path), _as_float(p.default, 0.0))
            if abs(w) > 0:
                weights.append((p.label, w))
    if weights:
        mags = [abs(w) for _, w in weights]
        lo, hi = min(mags), max(mags)
        if lo > 0 and hi / lo >= 1000:
            big = max(weights, key=lambda t: abs(t[1]))
            small = min(weights, key=lambda t: abs(t[1]))
            alerts.append(
                {
                    "level": "info",
                    "code": "reward_scale",
                    "message": (
                        f"奖励权重量级相差 ≥1000 倍：{big[0]}={big[1]} vs {small[0]}={small[1]}。"
                        f"小项几乎只是微调；若你指望它主导行为，请加大。"
                    ),
                }
            )

    cameras = (recipe.get("cameras") or recipe.get("camera_layout") or "none")
    task = recipe.get("task") or ""
    if cameras != "none" or "Vision" in task:
        if num_envs > 256:
            alerts.append(
                {
                    "level": "warn",
                    "code": "vram_cameras",
                    "message": (
                        f"开了相机还把并行环境设成 {num_envs}。"
                        f"RTX 4080 Laptop 只有 12 GB，建议 128，最多 256。"
                    ),
                }
            )

    physics = recipe.get("physics") or "physx"
    if physics == "newton_mjwarp" and (recipe.get("cli") or {}).get("clone_in_fabric", True):
        alerts.append(
            {
                "level": "info",
                "code": "newton_fabric",
                "message": "Newton 不支持 Fabric clone。开训时会自动加上 env.scene.clone_in_fabric=false。",
            }
        )

    mass = _pair(val("dr_object_mass", "env.events.object_scale_mass.params.mass_distribution_params"), [0.85, 1.15])
    if mass[0] > mass[1]:
        alerts.append({"level": "error", "code": "mass_range", "message": "质量缩放区间下限大于上限。"})
    if mass[0] < 0.2 or mass[1] > 3.0:
        alerts.append(
            {
                "level": "warn",
                "code": "mass_range",
                "message": f"质量缩放到 [{mass[0]}, {mass[1]}]，区间很极端，早期会很难学。",
            }
        )

    pose = recipe.get("hand_pose") or "palm_down_knuckle"
    obj = recipe.get("object") or "pan_coin_32mm"
    if obj.startswith("baoding") and pose != "palm_up_cradle":
        alerts.append(
            {
                "level": "warn",
                "code": "pose_object",
                "message": "保健球配掌心朝下时，球会从指背上滚掉。建议手部姿态改成「掌心朝上 · 托举」。",
            }
        )
    if obj == "pan_coin_32mm" and pose == "palm_up_cradle" and "Baoding" not in task:
        alerts.append(
            {
                "level": "warn",
                "code": "pose_object",
                "message": "掌心朝上 + 硬币：当前 Hold/Transfer 奖励是按指背座位写的，会变得没有意义。",
            }
        )

    if not alerts:
        alerts.append({"level": "info", "code": "ok", "message": "参数体检未发现硬错误。"})
    return alerts


def sentry_alerts(series: dict[str, list], desired_kl: float = 0.01) -> list[dict[str, str]]:
    """Look at recent tensorboard series and flag stuck / dead-explore / KL blowup."""
    alerts: list[dict[str, str]] = []

    def last_n(name_sub: str, n: int = 50) -> list[float]:
        for key, pts in series.items():
            if name_sub.lower() in key.lower():
                return [float(p["y"]) for p in pts[-n:]]
        return []

    succ = last_n("success", 300)
    if len(succ) >= 50:
        window = succ[-300:] if len(succ) >= 300 else succ
        if max(window) - min(window) < 1e-4 and max(window) < 0.05:
            alerts.append(
                {
                    "level": "warn",
                    "code": "success_stalled",
                    "message": "成功率连续很久几乎为 0 且没有抬头。检查奖励是否互相打架，或先把 Stage A 训到会托。",
                }
            )

    ent = last_n("entropy", 40)
    if len(ent) >= 10 and sum(ent[-10:]) / 10.0 < 0.01:
        alerts.append(
            {
                "level": "warn",
                "code": "entropy_collapse",
                "message": "熵掉到 0.01 以下，探索基本停了。略增熵系数，或从更早的 checkpoint 重开。",
            }
        )

    kl = last_n("/kl", 20) or last_n("kl_mean", 20) or last_n("KL", 20)
    if len(kl) >= 8 and sum(kl[-8:]) / 8.0 > desired_kl * 2.0:
        alerts.append(
            {
                "level": "warn",
                "code": "kl_high",
                "message": f"近期 KL 持续高于目标 {desired_kl} 的两倍。学习率偏大，或一批数据回扫太多次。",
            }
        )
    return alerts


def vram_estimate(num_envs: int, n_cameras: int, width: int, height: int) -> dict[str, Any]:
    """Very rough 12 GB-aware estimate. Not a profiler."""
    cam_bytes = num_envs * max(n_cameras, 0) * width * height * 3 * 4
    # physics + network + autograd scratch: ~12–20 MB/env without cameras on this hand.
    phys_bytes = num_envs * 16 * 1024 * 1024
    total = cam_bytes * 2.5 + phys_bytes + 1.5 * 1024**3
    gb = total / (1024**3)
    return {
        "gb": round(gb, 2),
        "limit_gb": 12.0,
        "ok": gb < 10.5,
        "message": (
            f"粗估显存 {gb:.1f} GB / 12 GB。"
            + (" 余量还行。" if gb < 10.5 else " 很可能 OOM，减并行环境或分辨率。")
        ),
    }
