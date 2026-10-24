"""手动 /exec（范式笔记 D4）：白名单 + 逐次确认。

项目边界（D1=B）明确不实现工具调用自动循环：这里只提供「人按一次、命令跑一次」的最小面。
普通程序 shell=False：& | > 只能作为字面参数，无法串联第二条命令；
cmd 内置命令（echo/dir/type…）没有 .exe，改走受控 `cmd /c`，并先拒绝元字符。
"""

from __future__ import annotations

import locale
import os
import shlex
import subprocess
from dataclasses import dataclass
from typing import Callable

from .config import AppConfig
from .errors import ToolDenied

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
WINDOWS_BUILTINS = frozenset(
    {"echo", "dir", "type", "set", "ver", "cls", "copy", "del", "md", "rd", "cd", "title", "vol"}
)
SHELL_METACHARS = frozenset("&|<>^%")


@dataclass
class ExecResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str


def _decode(raw: bytes) -> str:
    for encoding in ("utf-8", locale.getpreferredencoding(False), "mbcs"):
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", "replace")


def normalize(command: str) -> str:
    return " ".join(command.split()).lower()


def is_whitelisted(command: str, whitelist: tuple[str, ...] | list[str]) -> bool:
    normalized = normalize(command)
    for entry in whitelist:
        candidate = normalize(entry)
        if normalized == candidate or normalized.startswith(candidate + " "):
            return True
    return False


def split_command(command: str) -> list[str]:
    """按 Windows 习惯分词：保留引号内空格，再剥掉成对引号；反斜杠不当转义符。"""
    try:
        tokens = shlex.split(command, posix=False)
    except ValueError as exc:
        raise ToolDenied(f"命令无法解析：{exc}") from exc
    cleaned: list[str] = []
    for token in tokens:
        if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
            token = token[1:-1]
        cleaned.append(token)
    if not cleaned:
        raise ToolDenied("命令为空")
    return cleaned


def resolve_argv(command: str) -> list[str]:
    """决定实际启动参数：普通程序 shell=False；cmd 内置命令走受控的 `cmd /c`。"""
    argv = split_command(command)
    if os.name == "nt" and argv[0].lower() in WINDOWS_BUILTINS:
        if any(char in command for char in SHELL_METACHARS):
            raise ToolDenied(
                "内置命令不允许包含 & | < > ^ %（避免串联第二条命令）；"
                "需要组合命令请拆成多次 /exec"
            )
        return ["cmd", "/d", "/s", "/c", command]
    return argv


def _ask(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError as exc:
        raise ToolDenied("未收到确认输入，已取消执行") from exc


def run_exec(
    config: AppConfig,
    command: str,
    *,
    confirm: Callable[[str], str] | None = None,
) -> ExecResult:
    if not command.strip():
        raise ToolDenied("用法：/exec <命令>")
    settings = config.settings.tools
    if not settings.exec_enabled:
        raise ToolDenied(
            "本地命令执行已关闭；如需启用，把 config\\settings.json 的 "
            "tools.exec_enabled 改为 true 后重试"
        )
    if not is_whitelisted(command, settings.exec_whitelist):
        raise ToolDenied(
            f"命令不在白名单内：{command}\n"
            f"允许的前缀：{'；'.join(settings.exec_whitelist)}"
        )
    if settings.exec_require_confirm:
        answer = (confirm or _ask)(f"执行本地命令？\n  {command}\n[y/N] ")
        if answer.strip().lower() not in {"y", "yes", "是"}:
            raise ToolDenied("用户取消执行（未确认）")

    argv = resolve_argv(command)
    timeout = config.settings.request.timeout_sec
    try:
        completed = subprocess.run(
            argv,
            cwd=str(config.base_dir),
            capture_output=True,
            timeout=timeout,
            shell=False,
            creationflags=_NO_WINDOW,
        )
    except FileNotFoundError as exc:
        raise ToolDenied(f"命令不存在：{argv[0]}（PATH 中找不到）") from exc
    except subprocess.TimeoutExpired as exc:
        raise ToolDenied(f"命令超过 {timeout:.0f}s 未结束，已终止：{command}") from exc
    except OSError as exc:
        raise ToolDenied(f"命令启动失败：{exc.strerror or exc}") from exc
    return ExecResult(
        command=command,
        exit_code=completed.returncode,
        stdout=_decode(completed.stdout or b""),
        stderr=_decode(completed.stderr or b""),
    )
