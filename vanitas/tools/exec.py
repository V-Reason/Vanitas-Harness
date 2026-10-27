"""人类用户的 /exec / !cmd：原生 shell，无白名单与目录限制。

这是面向工程师的完全信任通道，命令由用户输入、不经过模型；
AgentTools 的沙箱边界不在此模块的职责内。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from ..config import AppConfig
from ..errors import ToolDenied
from . import runner
from .runner import ExecResult

CANCEL_MESSAGE = "已取消"
_CANCEL_MESSAGE = CANCEL_MESSAGE


def _ask(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError as exc:
        raise ToolDenied(_CANCEL_MESSAGE) from exc


def run_exec(
    config: AppConfig,
    command: str,
    *,
    confirm: Callable[[str], str] | None = None,
) -> ExecResult:
    """人类用户的 `/exec`：原生 shell，无白名单/目录限制；AgentTools 不受影响。"""
    if not command.strip():
        raise ToolDenied("用法：/exec <命令>")
    settings = config.settings.tools
    if not settings.exec_enabled:
        raise ToolDenied(
            "本地命令执行已关闭；如需启用，把 config\\settings.json 的 "
            "tools.exec_enabled 改为 true 后重试"
        )
    if settings.exec_require_confirm:
        answer = (confirm or _ask)(f"执行本地命令？\n  {command}\n[y/N] ")
        if answer.strip().lower() not in {"y", "yes", "是"}:
            raise ToolDenied(_CANCEL_MESSAGE)

    original = os.environ.get("VANITAS_ORIG_CWD") or os.getcwd()
    cwd = Path(original).expanduser()
    if not cwd.is_dir():
        cwd = Path.cwd()
    return runner.run(config, command, cwd, shell=True)


__all__ = ["CANCEL_MESSAGE", "_CANCEL_MESSAGE", "ExecResult", "_ask", "run_exec"]
