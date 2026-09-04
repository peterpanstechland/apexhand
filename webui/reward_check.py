"""Static + optional live checks for LLM-generated reward functions."""

from __future__ import annotations

import ast
import difflib
from pathlib import Path
from typing import Any

from webui.llm import USER_REWARDS
from webui.runner import start_job

ALLOWED_IMPORT_ROOTS = (
    "torch",
    "typing",
    "isaaclab.managers",
    "pan_dexterous_lab",
)

FORBIDDEN = {"os", "subprocess", "socket", "pathlib", "shutil", "sys", "requests", "http"}


def static_check(code: str) -> dict[str, Any]:
    report: dict[str, Any] = {"ok": False, "errors": [], "functions": []}
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        report["errors"].append(f"语法错误: {exc}")
        return report
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN:
                    report["errors"].append(f"禁止 import {alias.name}")
                elif not any(alias.name == a or alias.name.startswith(a + ".") for a in ALLOWED_IMPORT_ROOTS):
                    if root not in {"torch", "typing"}:
                        report["errors"].append(f"不允许的 import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            root = mod.split(".")[0]
            if root in FORBIDDEN:
                report["errors"].append(f"禁止 from {mod} import ...")
            elif not any(mod == a or mod.startswith(a + ".") for a in ALLOWED_IMPORT_ROOTS):
                if root not in {"torch", "typing"}:
                    report["errors"].append(f"不允许的 from {mod}")
        elif isinstance(node, ast.FunctionDef):
            report["functions"].append(node.name)
    if not report["functions"]:
        report["errors"].append("没有找到函数定义。")
    report["ok"] = not report["errors"]
    return report


def compose_file(code: str, entry: str | None = None) -> str:
    """Keep the module header and append / replace generated functions."""
    header = '''"""User-authored reward terms. The Web UI writes new functions here after review."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

'''
    body = code.strip() + "\n"
    fns = static_check(code).get("functions") or ["user_reward"]
    target = entry or fns[0]
    if target != "user_reward":
        body += f"\nuser_reward = {target}\n"
    elif "def user_reward" not in body:
        body += "\nuser_reward = user_reward\n"
    return header + body


def diff_against_current(new_text: str) -> str:
    old = USER_REWARDS.read_text() if USER_REWARDS.is_file() else ""
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile="user_rewards.py",
            tofile="user_rewards.py (proposed)",
        )
    )


def apply_code(new_text: str) -> None:
    USER_REWARDS.parent.mkdir(parents=True, exist_ok=True)
    USER_REWARDS.write_text(new_text)


def start_live_check(task: str = "PAN-CoinHold-Apex-Play-v0") -> dict[str, Any]:
    return start_job({"task": task, "mode": "check_reward", "name": "check_reward"}, mode="check_reward")
