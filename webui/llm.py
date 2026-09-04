"""OpenAI-compatible chat client (requests only — no openai SDK)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).resolve().parent / ".llm.json"
USER_REWARDS = REPO_ROOT / "source" / "pan_dexterous_lab" / "tasks" / "coin_roll" / "mdp" / "user_rewards.py"

DEFAULT_CONFIG = {
    "base_url": "https://api.deepseek.com/v1",
    "api_key": "",
    "model": "deepseek-chat",
    "timeout_s": 120,
}

SYSTEM_PROMPT = """你是 PΛN Dexterous Lab 的奖励函数作者。用户用自然语言描述想要的行为，你只输出一个 Python 函数。

硬性约定：
1. 函数签名必须是 (env, ...) -> torch.Tensor，返回形状 (num_envs,) 的 float 张量。
2. 禁止对 env 维写 Python for 循环；用批量 torch 运算。
3. 跨步状态必须用 maybe_reset_buf(env, "名字")，不要用全局变量。
4. 只允许 import：torch、typing、isaaclab.managers.SceneEntityCfg、以及
   pan_dexterous_lab.tasks.coin_roll.mdp._geom 里的 helper。
5. 不要写 if __name__、不要改文件、不要打印。只给一个函数。
6. 奖励是「每步的标量」，不是终止条件。成功类奖励应只在条件首次满足时给 1，用 maybe_reset_buf 做已发放标记。

坐标系（默认掌心朝下指背任务）：
- 手固定在世界系，掌心朝下，指背朝世界 +Z。
- 手指指向世界 -X；食指→小指沿世界 +Y。
- 硬币坐在近节指骨背面。knuckle_surface_pos 给出座位。
- palm_axes 返回 (origin, x_hat, y_hat, z_hat)：origin=食指座位，y_hat=食指→小指，z_hat≈世界+Z，x_hat=翻滚轴。

保健球任务（掌心朝上）请用 object / object2 两个刚体，掌心法向仍近似世界 +Z。

可用 helper（from pan_dexterous_lab.tasks.coin_roll.mdp._geom import ...）：
- _as_torch(x)
- coin_and_robot(env, object_cfg, robot_cfg) -> (coin, robot)
- knuckle_surface_pos(env, robot, finger_indices=(0,1,2,3), side="right") -> (N,k,3)
- knuckle_distances(env, coin, robot) -> (N,4)
- palm_axes(env, robot) -> origin, x_hat, y_hat, z_hat
- maybe_reset_buf(env, name, dim=1) -> (N,) 或 (N,dim)
- cached_knuckle_ids(env, robot) -> (prox_ids, dist_ids)
- quat_apply(q, v)

刚体数据：
- coin.data.root_pos_w, root_quat_w (wxyz), root_lin_vel_w, root_ang_vel_w
- robot.data.body_pos_w, joint_pos, joint_vel
- env.scene["object2"] 在双球任务里存在
- env.num_envs, env.device, env.step_dt, env.episode_length_buf

已有奖励函数（不要重复发明同名）：
{existing}

只输出一个 markdown 代码块，语言标记 python，里面是完整函数定义。函数名用英文蛇形，例如 keep_coin_flat。
"""


def load_config() -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.is_file():
        try:
            cfg.update(json.loads(CONFIG_PATH.read_text()))
        except json.JSONDecodeError:
            pass
    return cfg


def save_config(update: dict[str, Any]) -> dict[str, Any]:
    cfg = load_config()
    for key in ("base_url", "api_key", "model", "timeout_s"):
        if key in update and update[key] is not None:
            cfg[key] = update[key]
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    public = dict(cfg)
    if public.get("api_key"):
        public["api_key"] = public["api_key"][:4] + "…" + public["api_key"][-2:]
        public["has_key"] = True
    else:
        public["has_key"] = False
    return public


def public_config() -> dict[str, Any]:
    cfg = load_config()
    key = cfg.get("api_key") or ""
    return {
        "base_url": cfg.get("base_url"),
        "model": cfg.get("model"),
        "timeout_s": cfg.get("timeout_s"),
        "has_key": bool(key),
        "api_key": (key[:4] + "…" + key[-2:]) if len(key) > 8 else "",
    }


def _existing_reward_docs() -> str:
    rewards = REPO_ROOT / "source" / "pan_dexterous_lab" / "tasks" / "coin_roll" / "mdp" / "rewards.py"
    baoding = rewards.with_name("rewards_baoding.py")
    chunks = []
    for path in (rewards, baoding):
        if not path.is_file():
            continue
        text = path.read_text()
        chunks.append(f"# {path.name}\n" + text[:8000])
    return "\n\n".join(chunks)


def _extract_code(content: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*([\s\S]*?)```", content)
    if blocks:
        return blocks[0].strip() + "\n"
    return content.strip() + "\n"


def chat(user_prompt: str, extra_messages: list[dict[str, str]] | None = None) -> dict[str, Any]:
    cfg = load_config()
    if not cfg.get("api_key"):
        raise RuntimeError("还没有配置 API Key。在界面「大模型」一栏填 OpenAI 兼容的 endpoint / key / model。")
    base = cfg["base_url"].rstrip("/")
    if not base.endswith("/v1") and not base.endswith("/v1/chat/completions"):
        url = base + "/v1/chat/completions"
    elif base.endswith("/v1"):
        url = base + "/chat/completions"
    else:
        url = base
    system = SYSTEM_PROMPT.format(existing=_existing_reward_docs())
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}]
    if extra_messages:
        messages.extend(extra_messages)
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"},
        json={
            "model": cfg["model"],
            "messages": messages,
            "temperature": 0.2,
        },
        timeout=float(cfg.get("timeout_s") or 120),
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"LLM HTTP {resp.status_code}: {resp.text[:800]}")
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    code = _extract_code(content)
    return {"content": content, "code": code, "raw": {"usage": data.get("usage")}}


def generate_reward(prompt: str, feedback: str | None = None) -> dict[str, Any]:
    extra = None
    if feedback:
        extra = [
            {
                "role": "user",
                "content": "上一次生成的函数自检失败，请按下面的报错整段重写：\n" + feedback,
            }
        ]
    return chat(prompt, extra_messages=extra)
