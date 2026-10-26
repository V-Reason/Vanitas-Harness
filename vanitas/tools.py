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
from pathlib import Path
from typing import Callable

from .config import AppConfig
from .errors import ToolDenied

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
WINDOWS_BUILTINS = frozenset(
    {"echo", "dir", "type", "set", "ver", "cls", "copy", "del", "md", "rd", "cd", "title", "vol"}
)
SHELL_METACHARS = frozenset("&|<>^%")
DESTRUCTIVE_COMMANDS = frozenset(
    {
        "del",
        "erase",
        "rd",
        "rmdir",
        "rm",
        "mv",
        "move",
        "ren",
        "rename",
        "copy",
        "xcopy",
        "robocopy",
        "mkdir",
        "md",
    }
)
_COMMAND_SUFFIXES = (".exe", ".com", ".bat", ".cmd")
_HELP_SWITCHES = frozenset({"/?", "-?", "-h", "--help"})


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


def _command_name(token: str) -> str:
    """把 `C:\\Windows\\System32\\cmd.exe` 归一为 `cmd`。"""
    if not token:
        return ""
    cleaned = token.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in "\"'":
        cleaned = cleaned[1:-1]
    try:
        name = Path(cleaned).name.strip().lower()
    except (TypeError, ValueError):
        return ""
    for suffix in _COMMAND_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _unwrap_cmd(argv: list[str]) -> list[str]:
    """展开 `cmd /c "del ..."`：让内层命令重新变成可识别的 argv。"""
    if not argv or _command_name(argv[0]) != "cmd":
        return list(argv)
    for index, token in enumerate(argv[1:], start=1):
        if token.lower() not in {"/c", "/k"}:
            continue
        rest = argv[index + 1:]
        if not rest:
            return list(argv)
        if len(rest) == 1 and any(char.isspace() for char in rest[0]):
            try:
                inner = split_command(rest[0])
                # 直接传入的 argv 可能保留一整段带引号的内层命令；
                # 再拆一层，确保 `cmd /c "del C:\\..."` 也能识别动词。
                if len(inner) == 1 and any(char.isspace() for char in inner[0]):
                    inner = split_command(inner[0])
            except ToolDenied:
                return list(argv)
            return [*argv[: index + 1], *inner]
        return list(argv)
    return list(argv)


def _destructive_index(argv: list[str]) -> tuple[int | None, list[str]]:
    """返回（破坏性动词下标，展开后的 argv）；非破坏性命令下标为 None。"""
    tokens = _unwrap_cmd(argv)
    if not tokens:
        return None, tokens
    index = 0
    if _command_name(tokens[0]) == "cmd":
        for position, token in enumerate(tokens[1:], start=1):
            if token.lower() in {"/c", "/k"}:
                index = position + 1
                break
        else:
            return None, tokens
    if index >= len(tokens):
        return None, tokens
    verb = _command_name(tokens[index])
    if verb not in DESTRUCTIVE_COMMANDS:
        return None, tokens
    return index, tokens


def _destructive_verb(argv: list[str]) -> str | None:
    index, tokens = _destructive_index(argv)
    if index is None:
        return None
    return _command_name(tokens[index])


def _looks_like_switch(token: str) -> bool:
    """区分 `/f`、`-rf` 这类开关与 `C:\\...`、`/etc/...` 这类路径。"""
    if not token or token in _HELP_SWITCHES:
        return True
    if token.startswith("/"):
        body = token[1:]
        if not body:
            return True
        # /etc/passwd 含第二个分隔符，按路径处理（宁可多校验）。
        return "/" not in body and "\\" not in body
    if token.startswith("-"):
        if "/" in token or "\\" in token:
            return False
        # -file.txt 更像路径；--help/-rf 仍按开关处理。
        if "." in token and not token.startswith("--"):
            return False
        return True
    return False


def _is_destructive_command(argv: list[str]) -> list[str]:
    """返回破坏性/写命令涉及的全部路径参数（引用原样，未解析为绝对路径）。"""
    index, tokens = _destructive_index(argv)
    if index is None:
        return []
    paths: list[str] = []
    for token in tokens[index + 1:]:
        if not token or _looks_like_switch(token):
            continue
        paths.append(token)
    return paths


def _is_within_cwd(path: str, cwd: Path) -> bool:
    """路径解析后是否落在 cwd 内；解析失败按不安全处理。"""
    try:
        base = Path(cwd).expanduser().resolve()
        expanded = os.path.expandvars(os.path.expanduser(str(path)))
        candidate = Path(expanded)
        if not candidate.is_absolute():
            candidate = base / candidate
        resolved = candidate.resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
    return resolved == base or base in resolved.parents


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
    tokens = split_command(command)
    if _destructive_verb(tokens) is not None:
        targets = _is_destructive_command(tokens)
        if not targets:
            if not any(token in _HELP_SWITCHES for token in tokens):
                raise ToolDenied("安全限制：破坏性命令必须显式指定目标路径")
        else:
            outside = [
                path for path in targets if not _is_within_cwd(path, config.base_dir)
            ]
            if outside:
                raise ToolDenied(
                    "安全限制：禁止修改或删除工作目录之外的文件/目录\n"
                    f"越界路径：{'；'.join(outside)}"
                )
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
