"""受控工具执行：手动 /exec + Agent 内置工具 + CWD 沙盒。

两条路径刻意分开：
- `run_exec` 保留旧 `/exec` 契约：配置白名单、逐次确认、以仓库根为工作目录；
- `AgentTools` 供 Agent 闭环使用：只读命令白名单、当前目录内的破坏性操作、
  所有文件参数经过 `_is_safe_path`。Agent 的每一步确认由 agent.py 负责。

普通程序 `shell=False`：`& | >` 只能作为字面参数，无法串联第二条命令；
cmd 内置命令（echo/dir/type/ren…）没有 .exe，改走受控 `cmd /c`，并先拒绝元字符。
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
        "format",
    }
)
# Agent 的 shell 不接受解释器/脚本宿主：路径无法证明安全，直接拒绝。
AGENT_BLOCKED_COMMANDS = frozenset(
    {
        "python",
        "python3",
        "pythonw",
        "py",
        "powershell",
        "pwsh",
        "cmd",
        "wscript",
        "cscript",
        "mshta",
        "rundll32",
        "regsvr32",
        "format",
        "diskpart",
    }
)
AGENT_READONLY_COMMANDS = frozenset({"echo", "dir", "type"})
AGENT_GIT_SUBCOMMANDS = frozenset({"status"})
CANCEL_MESSAGE = "已取消"
MAX_READ_BYTES = 256 * 1024
MAX_LIST_ENTRIES = 200
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


def _resolve_candidate(path_str: str, root: Path | None = None) -> Path:
    base = Path(root).expanduser().resolve() if root is not None else Path.cwd().resolve()
    expanded = os.path.expandvars(os.path.expanduser(str(path_str)))
    candidate = Path(expanded)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def _is_safe_path(path_str: str, root: Path | None = None) -> bool:
    """路径解析后是否落在沙盒根内；空值、NUL 与解析失败按不安全处理。"""
    if not isinstance(path_str, str):
        return False
    raw = path_str.strip()
    if not raw or "\x00" in raw:
        return False
    try:
        base = Path(root).expanduser().resolve() if root is not None else Path.cwd().resolve()
        resolved = _resolve_candidate(raw, base)
        base_key = os.path.normcase(str(base))
        target_key = os.path.normcase(str(resolved))
        return os.path.commonpath([base_key, target_key]) == base_key
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def _require_safe_path(path_str: str, root: Path | None = None) -> Path:
    if not _is_safe_path(path_str, root):
        raise ToolDenied("安全限制：越界操作已被拦截。")
    base = Path(root).expanduser().resolve() if root is not None else Path.cwd().resolve()
    return _resolve_candidate(path_str, base)


def _is_within_cwd(path: str, cwd: Path) -> bool:
    """兼容旧调用：路径解析后是否落在 cwd 内。"""
    return _is_safe_path(path, cwd)


def _ask(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError as exc:
        raise ToolDenied(CANCEL_MESSAGE) from exc


def _execute(
    config: AppConfig,
    command: str,
    argv: list[str],
    cwd: Path,
) -> ExecResult:
    timeout = config.settings.request.timeout_sec
    try:
        completed = subprocess.run(
            argv,
            cwd=str(cwd),
            capture_output=True,
            timeout=timeout,
            shell=False,
            creationflags=_NO_WINDOW,
        )
    except FileNotFoundError as exc:
        raise ToolDenied(f"命令不存在：{argv[0]}，PATH 中找不到") from exc
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
                path for path in targets if not _is_safe_path(path, config.base_dir)
            ]
            if outside:
                raise ToolDenied(
                    "安全限制：禁止修改或删除工作目录之外的文件/目录；"
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
            f"命令不在白名单内：{command}；"
            f"允许的前缀：{'；'.join(settings.exec_whitelist)}"
        )
    if settings.exec_require_confirm:
        answer = (confirm or _ask)(f"执行本地命令？\n  {command}\n[y/N] ")
        if answer.strip().lower() not in {"y", "yes", "是"}:
            raise ToolDenied(CANCEL_MESSAGE)

    argv = resolve_argv(command)
    return _execute(config, command, argv, config.base_dir)


class AgentTools:
    """Agent 闭环的内置工具；所有路径都限制在构造时确定的根目录内。"""

    def __init__(self, config: AppConfig, root: Path | None = None) -> None:
        self.config = config
        if root is not None:
            self.root = Path(root).expanduser().resolve()
        else:
            # bat 为导入包会 cd 到仓库根；Agent 沙盒必须回到用户原 CWD。
            original = os.environ.get("VANITAS_ORIG_CWD")
            self.root = (
                Path(original).expanduser().resolve()
                if original
                else Path.cwd().resolve()
            )
        if self.root.parent == self.root:
            raise ToolDenied("安全限制：工作目录不能是磁盘根目录")

    def _require(self, path: str) -> Path:
        return _require_safe_path(path, self.root)

    def read_file(self, path: str) -> str:
        target = self._require(path)
        try:
            if not target.is_file():
                raise ToolDenied(f"不是可读取的文件：{path}")
            size = target.stat().st_size
            if size > MAX_READ_BYTES:
                raise ToolDenied(
                    f"文件超过 {MAX_READ_BYTES // 1024} KiB，已拒绝读取：{path}"
                )
            raw = target.read_bytes()
        except ToolDenied:
            raise
        except OSError as exc:
            raise ToolDenied(f"读取失败：{path}；{exc.strerror or exc}") from exc
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ToolDenied(f"文件不是 UTF-8：{path}") from exc

    def write_file(self, path: str, content: str) -> int:
        if not isinstance(content, str):
            raise ToolDenied("write_file 的 content 必须是字符串")
        target = self._require(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            data = content.encode("utf-8")
            target.write_bytes(data)
        except OSError as exc:
            raise ToolDenied(f"写入失败：{path}；{exc.strerror or exc}") from exc
        return len(data)

    def list_dir(self, path: str = ".") -> list[str]:
        target = self._require(path)
        if not target.is_dir():
            raise ToolDenied(f"不是目录：{path}")
        entries: list[str] = []
        truncated = False
        try:
            with os.scandir(target) as iterator:
                for entry in iterator:
                    if len(entries) >= MAX_LIST_ENTRIES:
                        truncated = True
                        break
                    suffix = "/" if entry.is_dir(follow_symlinks=False) else ""
                    entries.append(entry.name + suffix)
        except OSError as exc:
            raise ToolDenied(f"目录读取失败：{path}；{exc.strerror or exc}") from exc
        entries.sort(key=lambda item: (not item.endswith("/"), item.casefold()))
        if truncated:
            entries.append("…")
        return entries

    def run_shell(self, command: str) -> ExecResult:
        if not isinstance(command, str) or not command.strip():
            raise ToolDenied("shell 命令不能为空")
        command = command.strip()
        if any(char in command for char in SHELL_METACHARS):
            raise ToolDenied("安全限制：不允许管道、重定向或命令串联。")
        tokens = split_command(command)
        name = _command_name(tokens[0])
        if name in AGENT_BLOCKED_COMMANDS:
            raise ToolDenied(f"安全限制：不允许执行解释器或脚本宿主命令：{name}")
        self._check_command(tokens, name)
        if not self.config.settings.tools.exec_enabled:
            raise ToolDenied(
                "本地命令执行已关闭；如需启用，把 config\\settings.json 的 "
                "tools.exec_enabled 改为 true 后重试"
            )
        argv = resolve_argv(command)
        return _execute(self.config, command, argv, self.root)

    def _check_command(self, tokens: list[str], name: str) -> None:
        if _destructive_verb(tokens) is not None:
            targets = _is_destructive_command(tokens)
            if not targets:
                if not any(token in _HELP_SWITCHES for token in tokens):
                    raise ToolDenied("安全限制：破坏性命令必须显式指定目标路径")
            else:
                outside = [
                    path for path in targets if not _is_safe_path(path, self.root)
                ]
                if outside:
                    raise ToolDenied("安全限制：越界操作已被拦截。")
            return

        if name == "git":
            if len(tokens) < 2 or tokens[1].lower() not in AGENT_GIT_SUBCOMMANDS:
                raise ToolDenied("安全限制：Agent 只允许 git status。")
        elif name not in AGENT_READONLY_COMMANDS:
            raise ToolDenied(
                "安全限制：Agent 只允许只读命令或当前目录内的破坏性操作。"
            )

        outside = [
            token
            for token in tokens[1:]
            if token and not _looks_like_switch(token) and not _is_safe_path(token, self.root)
        ]
        if outside:
            raise ToolDenied("安全限制：越界操作已被拦截。")
