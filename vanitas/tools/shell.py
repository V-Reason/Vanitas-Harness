"""Agent 只读 Shell：分词、白名单、写参数拦截、cmd 内建命令。

- execute_readonly_shell 是唯一对外入口；
- 使用 shell=False，不启动命令解释器；
- 白名单 = settings.readonly_shell_prefixes | settings.project_prefixes。
"""
from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Iterable

from ..config import AppConfig, ToolsSettings
from ..errors import ToolDenied
from . import fs, runner, sandbox
from .runner import ExecResult


WINDOWS_BUILTINS = frozenset(
    {
        "echo",
        "dir",
        "type",
        "set",
        "ver",
        "cls",
        "copy",
        "del",
        "erase",
        "md",
        "rd",
        "ren",
        "rename",
        "move",
        "cd",
        "title",
        "vol",
    }
)
# 仅用于 /exec 的 cmd 内置命令；Agent 不经过这里。
_CMD_CONTROL_CHARS = frozenset("&|<>^%")
_READONLY_BUILTINS = frozenset({"echo", "dir", "ls", "type", "cat", "pwd"})


_COMMAND_SUFFIXES = (".exe", ".com", ".bat", ".cmd")
_HELP_SWITCHES = frozenset({"/?", "-?", "-h", "--help"})


def normalize(command: str) -> str:
    return " ".join(command.split()).lower()


def is_whitelisted(command: str, whitelist: Iterable[str]) -> bool:
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
    """为 `/exec` 决定启动参数：普通程序 shell=False；cmd 内置命令走受控 cmd /c。"""
    argv = split_command(command)
    if os.name == "nt" and argv[0].lower() in WINDOWS_BUILTINS:
        if any(char in command for char in _CMD_CONTROL_CHARS):
            raise ToolDenied(
                "内置命令不允许包含 & | < > ^ %，避免串联第二条命令；"
                "需要组合命令请拆成多次执行"
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
                if len(inner) == 1 and any(char.isspace() for char in inner[0]):
                    inner = split_command(inner[0])
            except ToolDenied:
                return list(argv)
            return [*argv[: index + 1], *inner]
        return list(argv)
    return list(argv)


def _destructive_index(
    argv: list[str], destructive_commands: frozenset[str]
) -> tuple[int | None, list[str]]:
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
    if verb not in destructive_commands:
        return None, tokens
    return index, tokens


def _destructive_verb(
    argv: list[str], destructive_commands: frozenset[str]
) -> str | None:
    index, tokens = _destructive_index(argv, destructive_commands)
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


def _is_destructive_command(
    argv: list[str], destructive_commands: frozenset[str]
) -> list[str]:
    """返回破坏性/写命令涉及的全部路径参数（引用原样，未解析为绝对路径）。"""
    index, tokens = _destructive_index(argv, destructive_commands)
    if index is None:
        return []
    paths: list[str] = []
    for token in tokens[index + 1:]:
        if not token or _looks_like_switch(token):
            continue
        paths.append(token)
    return paths




def _readonly_prefix(tokens: list[str]) -> str:
    name = _command_name(tokens[0])
    if name == "git":
        if len(tokens) < 2:
            return "git"
        return f"git {tokens[1].lower()}"
    if name == "python":
        if len(tokens) >= 2 and Path(tokens[1]).name.lower() == "verify.py":
            return "python verify.py"
        if (
            len(tokens) >= 3
            and tokens[1].lower() == "-m"
            and tokens[2].lower() == "unittest"
        ):
            return "python -m unittest"
        return "python"
    return name


def _reject_write_flags(
    tokens: list[str],
    blocked_exact: frozenset[str],
    blocked_prefixes: tuple[str, ...],
) -> None:
    for token in tokens:
        low = token.lower()
        if low in blocked_exact or low.startswith(blocked_prefixes):
            raise ToolDenied(
                f"安全限制：只读 Shell 不允许写操作参数 {token!r}"
            )


def _check_readonly_paths(
    root: Path, tokens: list[str], prefix: str, settings: ToolsSettings
) -> None:
    if prefix == "git" or prefix.startswith("git "):
        _check_git(
            root,
            tokens,
            settings.git_blocked_options,
            settings.git_blocked_prefixes,
        )
        return
    if prefix == "echo":
        return
    if prefix.startswith("python"):
        for candidate in tokens[1:]:
            if not _looks_like_switch(candidate):
                sandbox.is_inside_workspace(candidate, root)
        return
    name = prefix
    args = tokens[1:]
    if name in {"dir", "ls", "type", "cat", "head", "tail", "wc", "grep"}:
        candidates = [token for token in args if not _looks_like_switch(token)]
    elif name == "find":
        # 保守策略：find 的非开关参数全部按工作区路径校验；通配符
        # 如 *.md 会解析为工作区内路径，不会误放行绝对外部路径。
        candidates = [token for token in args if not _looks_like_switch(token)]
    elif name == "findstr":
        non_switches = [token for token in args if not _looks_like_switch(token)]
        candidates = non_switches[1:]
    else:
        candidates = []
    for candidate in candidates:
        sandbox.is_inside_workspace(candidate, root)


def _check_git(
    root: Path,
    tokens: list[str],
    blocked_options: frozenset[str],
    blocked_prefixes: tuple[str, ...],
) -> None:
    for token in tokens[1:]:
        low = token.lower()
        if low in blocked_options or low.startswith(blocked_prefixes):
            raise ToolDenied(
                f"安全限制：git 只读子命令不允许参数 {token!r}"
            )
    if "--" in tokens:
        for candidate in tokens[tokens.index("--") + 1:]:
            sandbox.is_inside_workspace(candidate, root)


def _run_builtin(
    config: AppConfig, root: Path, command: str, prefix: str, tokens: list[str]
) -> ExecResult:
    try:
        if prefix == "echo":
            text = " ".join(tokens[1:])
            if not text:
                text = "ECHO is on."
            return ExecResult(command, 0, text + "\n", "")
        if prefix == "pwd":
            return ExecResult(command, 0, str(root) + "\n", "")
        if prefix in {"dir", "ls"}:
            args = [token for token in tokens[1:] if not _looks_like_switch(token)]
            target = args[0] if args else "."
            names = fs.list_dir(
                target,
                root=root,
                max_entries=config.settings.tools.max_list_entries,
            )
            return ExecResult(command, 0, "\n".join(names) + "\n", "")
        # type / cat
        args = [token for token in tokens[1:] if not _looks_like_switch(token)]
        if not args:
            raise ToolDenied(f"{prefix} 需要至少一个文件")
        chunks: list[str] = []
        for path in args:
            target = sandbox._require_path(path, root)
            if not target.is_file():
                raise ToolDenied(f"不是可读取的文件：{path}")
            max_bytes = config.settings.tools.max_read_bytes
            if target.stat().st_size > max_bytes:
                raise ToolDenied(
                    f"文件超过 {max_bytes // 1024} KiB，已拒绝读取：{path}"
                )
            chunks.append(runner._decode(target.read_bytes()))
        return ExecResult(command, 0, "".join(chunks), "")
    except ToolDenied as exc:
        raise
    except OSError as exc:
        raise ToolDenied(
            f"只读命令执行失败：{command}；{exc.strerror or exc}"
        ) from exc


def execute_readonly_shell(
    config: AppConfig,
    root: Path,
    command: str,
) -> ExecResult:
    """只读命令执行：前缀白名单 + 写操作参数黑名单，不启动 shell。"""
    if not isinstance(command, str) or not command.strip():
        raise ToolDenied("shell 命令不能为空")
    command = command.strip()
    tokens = split_command(command)
    settings = config.settings.tools
    prefix = _readonly_prefix(tokens)
    allowed_prefixes = settings.readonly_shell_prefixes | frozenset(
        settings.project_prefixes
    )
    if prefix not in allowed_prefixes:
        raise ToolDenied(
            f"安全限制：只允许只读命令；收到 {prefix!r}。"
            f"允许：{'、'.join(sorted(allowed_prefixes))}"
        )
    _reject_write_flags(
        tokens,
        settings.readonly_blocked_flags,
        settings.readonly_blocked_prefixes,
    )
    _check_readonly_paths(root, tokens, prefix, settings)
    if prefix in _READONLY_BUILTINS:
        return _run_builtin(config, root, command, prefix, tokens)
    env = os.environ.copy()
    env.update(
        {
            "GIT_PAGER": "cat",
            "PAGER": "cat",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_OPTIONAL_LOCKS": "0",
        }
    )
    return runner.run(
        config, command, root, argv=tokens, shell=False, env=env
    )


__all__ = [
    "WINDOWS_BUILTINS", "_CMD_CONTROL_CHARS", "_READONLY_BUILTINS",
    "_COMMAND_SUFFIXES", "_HELP_SWITCHES",
    "normalize", "is_whitelisted", "split_command", "resolve_argv",
    "_command_name", "_unwrap_cmd", "_destructive_index", "_destructive_verb",
    "_looks_like_switch", "_is_destructive_command",
    "_readonly_prefix", "_reject_write_flags", "_check_readonly_paths",
    "_check_git", "_run_builtin", "execute_readonly_shell",
]
