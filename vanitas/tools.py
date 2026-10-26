"""受控工具执行：手动 /exec + Agent 内置工具 + 工作区沙盒。

两条路径刻意分开：
- `run_exec` 保留 `/exec` 契约：手动确认、以仓库根为工作目录、硬编码的兼容前缀；
- `AgentTools` 供 Agent 闭环使用：强类型文件工具、只读 Shell、工作区内路径校验。

安全边界是参数级而不是 OS 级：所有文件路径在进入 `Path` 操作前都必须经过
`is_inside_workspace`；Shell 使用 `shell=False`，不启动命令解释器。
"""

from __future__ import annotations

import locale
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .config import AppConfig, ToolsSettings
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
# 仅用于 /exec 的 cmd 内置命令；Agent 不经过这里。
_CMD_CONTROL_CHARS = frozenset("&|<>^%")
_READONLY_BUILTINS = frozenset({"echo", "dir", "ls", "type", "cat", "pwd"})
CANCEL_MESSAGE = "已取消"
_CANCEL_MESSAGE = CANCEL_MESSAGE
_COMMAND_SUFFIXES = (".exe", ".com", ".bat", ".cmd")
_HELP_SWITCHES = frozenset({"/?", "-?", "-h", "--help"})
MAX_CAPTURE_BYTES = 512 * 1024  # 512 KiB 底层流物理硬截断上限


@dataclass
class ExecResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str


def _decode(raw: bytes) -> str:
    for encoding in ("utf-8-sig", locale.getpreferredencoding(False), "mbcs"):
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", "replace")


def _truncate_bytes(
    raw: bytes, max_bytes: int = MAX_CAPTURE_BYTES
) -> tuple[bytes, bool, int]:
    """在内存解码前对底层子进程输出实施字节级硬折叠，杜绝 OOM。"""
    if len(raw) <= max_bytes:
        return raw, False, 0
    omitted = len(raw) - max_bytes
    head_size = max_bytes // 4
    tail_size = max_bytes - head_size
    folded = raw[:head_size] + b"\n... [stdout truncated] ...\n" + raw[-tail_size:]
    return folded, True, omitted


def _append_capture_warning(text: str, label: str, omitted: int) -> str:
    note = (
        f"[... 底层{label}超出 {MAX_CAPTURE_BYTES // 1024} KiB，"
        f"已在字节流阶段硬截断，丢弃 {omitted} 字节 ...]"
    )
    return f"{text}\n{note}" if text else note


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


def _workspace_root(root: Path | str | None) -> Path:
    if root is not None:
        return Path(root).expanduser().resolve()
    # bat 会先 cd 到仓库根；Agent 的“当前工作区”必须回到用户原 CWD。
    original = os.environ.get("VANITAS_ORIG_CWD")
    if original:
        return Path(original).expanduser().resolve()
    return Path.cwd().resolve()


def _path_text(path: str | os.PathLike[str]) -> str:
    try:
        raw = os.fspath(path)
    except TypeError as exc:
        raise ToolDenied("路径必须是字符串") from exc
    if not isinstance(raw, str):
        raise ToolDenied("路径必须是字符串")
    raw = raw.strip()
    if not raw or "\x00" in raw:
        raise ToolDenied("安全限制：路径不能为空")
    return raw


def _resolve_candidate(path_str: str, root: Path | str | None = None) -> Path:
    base = _workspace_root(root)
    expanded = os.path.expandvars(os.path.expanduser(str(path_str)))
    candidate = Path(expanded)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def is_inside_workspace(
    path: str | os.PathLike[str],
    root: Path | str | None = None,
    *,
    raise_on_error: bool = True,
) -> bool:
    """规范化路径并判断是否位于工作区内。

    默认在越界、空路径或路径无法解析时抛出 `ToolDenied`；需要布尔查询时传
    `raise_on_error=False`。Windows 通过 `normcase` + `commonpath` 做大小写
    不敏感比较，跨盘符同样判为越界。
    """
    try:
        raw = _path_text(path)
        # 独立调用严格以进程 CWD 为根；AgentTools 会显式传入原 CWD。
        base = (
            Path.cwd().resolve()
            if root is None
            else Path(root).expanduser().resolve()
        )
        resolved = _resolve_candidate(raw, base)
        base_key = os.path.normcase(str(base))
        target_key = os.path.normcase(str(resolved))
        inside = os.path.commonpath([base_key, target_key]) == base_key
    except (OSError, RuntimeError, TypeError, ValueError, ToolDenied) as exc:
        inside = False
        raw = str(path) if "raw" not in locals() else raw
        detail = str(exc) or "无法解析路径"
    else:
        detail = "路径不在工作区内"
    if inside:
        return True
    if raise_on_error:
        raise ToolDenied(f"安全限制：越界操作已被拦截：{raw}（{detail}）")
    return False


def _require_path(
    path: str | os.PathLike[str], root: Path | str | None = None
) -> Path:
    raw = _path_text(path)
    # 与 is_inside_workspace 一致：未显式给根时使用进程 CWD；
    # AgentTools 永远显式传入原 CWD，避免 bat 的仓库根劫持。
    base = (
        Path.cwd().resolve()
        if root is None
        else Path(root).expanduser().resolve()
    )
    is_inside_workspace(raw, base)
    return _resolve_candidate(raw, base)


def _is_safe_path(path_str: str, root: Path | str | None = None) -> bool:
    """兼容旧调用：返回路径是否落在工作区内，不抛异常。"""
    return is_inside_workspace(path_str, root, raise_on_error=False)


def _is_within_cwd(path: str, cwd: Path) -> bool:
    """兼容旧调用：路径解析后是否落在 cwd 内。"""
    return _is_safe_path(path, cwd)


def _ask(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError as exc:
        raise ToolDenied(_CANCEL_MESSAGE) from exc


def _execute(
    config: AppConfig,
    command: str,
    argv: list[str],
    cwd: Path,
    *,
    env: dict[str, str] | None = None,
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
            env=env,
        )
    except FileNotFoundError as exc:
        raise ToolDenied(f"命令不存在：{argv[0]}，PATH 中找不到") from exc
    except subprocess.TimeoutExpired as exc:
        raise ToolDenied(f"命令超过 {timeout:.0f}s 未结束，已终止：{command}") from exc
    except OSError as exc:
        raise ToolDenied(f"命令启动失败：{exc.strerror or exc}") from exc
    raw_stdout = completed.stdout or b""
    raw_stderr = completed.stderr or b""
    stdout_bytes, stdout_truncated, stdout_omitted = _truncate_bytes(raw_stdout)
    stderr_bytes, stderr_truncated, stderr_omitted = _truncate_bytes(raw_stderr)
    stdout = _decode(stdout_bytes)
    stderr = _decode(stderr_bytes)
    if stdout_truncated:
        stdout = _append_capture_warning(stdout, "标准输出", stdout_omitted)
    if stderr_truncated:
        stderr = _append_capture_warning(stderr, "标准错误", stderr_omitted)
    return ExecResult(
        command=command,
        exit_code=completed.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def run_exec(
    config: AppConfig,
    command: str,
    *,
    confirm: Callable[[str], str] | None = None,
) -> ExecResult:
    """手动 `/exec`：保持逐次确认与仓库根边界，不复用 Agent 的只读策略。"""
    if not command.strip():
        raise ToolDenied("用法：/exec <命令>")
    tokens = split_command(command)
    settings = config.settings.tools
    if _destructive_verb(tokens, settings.destructive_commands) is not None:
        targets = _is_destructive_command(tokens, settings.destructive_commands)
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
    if not settings.exec_enabled:
        raise ToolDenied(
            "本地命令执行已关闭；如需启用，把 config\\settings.json 的 "
            "tools.exec_enabled 改为 true 后重试"
        )
    if not is_whitelisted(command, settings.manual_exec_prefixes):
        raise ToolDenied(
            f"命令不在兼容白名单内：{command}；"
            f"允许的前缀：{'；'.join(settings.manual_exec_prefixes)}"
        )
    if settings.exec_require_confirm:
        answer = (confirm or _ask)(f"执行本地命令？\n  {command}\n[y/N] ")
        if answer.strip().lower() not in {"y", "yes", "是"}:
            raise ToolDenied(_CANCEL_MESSAGE)

    argv = resolve_argv(command)
    return _execute(config, command, argv, config.base_dir)


# --------------------------------------------------------------------------- #
# 强类型文件工具：既可直接调用，也可由 AgentTools 绑定工作区后调用
# --------------------------------------------------------------------------- #


def read_file(
    path: str, *, root: Path | str | None = None, max_bytes: int
) -> str:
    target = _require_path(path, root)
    try:
        if not target.is_file():
            raise ToolDenied(f"不是可读取的文件：{path}")
        size = target.stat().st_size
        if size > max_bytes:
            raise ToolDenied(
                f"文件超过 {max_bytes // 1024} KiB，已拒绝读取：{path}"
            )
        raw = target.read_bytes()
    except ToolDenied:
        raise
    except OSError as exc:
        raise ToolDenied(f"读取失败：{path}；{exc.strerror or exc}") from exc
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ToolDenied(f"文件不是 UTF-8：{path}") from exc


def write_file(path: str, content: str, *, root: Path | str | None = None) -> int:
    if not isinstance(content, str):
        raise ToolDenied("write_file 的 content 必须是字符串")
    target = _require_path(path, root)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        data = content.encode("utf-8")
        target.write_bytes(data)
    except OSError as exc:
        raise ToolDenied(f"写入失败：{path}；{exc.strerror or exc}") from exc
    return len(data)


def _reject_illegal_move_text(value: str, illegal_chars: str) -> None:
    """模型常把 `#`、通配符等路径污染带进 move_file；提前给出可读拒绝。"""
    if any(char in illegal_chars or ord(char) < 32 for char in value):
        raise ToolDenied("源文件不存在或路径非法")


def move_file(
    src: str,
    dst: str,
    *,
    root: Path | str | None = None,
    illegal_chars: str,
) -> str:
    source = _require_path(src, root)
    target = _require_path(dst, root)
    _reject_illegal_move_text(src, illegal_chars)
    _reject_illegal_move_text(dst, illegal_chars)
    if source == target:
        raise ToolDenied(f"源和目标相同：{src}")
    try:
        if not source.is_file():
            raise ToolDenied("源文件不存在或路径非法")
        if target.exists():
            raise ToolDenied(f"目标已存在：{dst}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
    except ToolDenied:
        raise
    except OSError as exc:
        raise ToolDenied(f"移动失败：{src} -> {dst}；{exc.strerror or exc}") from exc
    return f"重命名 {src} -> {dst}"


def delete_file(path: str, *, root: Path | str | None = None) -> str:
    target = _require_path(path, root)
    try:
        if not target.is_file():
            raise ToolDenied(f"不是可删除的文件：{path}")
        target.unlink()
    except ToolDenied:
        raise
    except OSError as exc:
        raise ToolDenied(f"删除失败：{path}；{exc.strerror or exc}") from exc
    return f"删除 {path}"


def list_dir(
    path: str = ".",
    *,
    root: Path | str | None = None,
    max_entries: int,
) -> list[str]:
    target = _require_path(path, root)
    if not target.is_dir():
        raise ToolDenied(f"不是目录：{path}")
    entries: list[str] = []
    truncated = False
    try:
        with os.scandir(target) as iterator:
            for entry in iterator:
                if len(entries) >= max_entries:
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


class AgentTools:
    """Agent 闭环的内置工具；所有路径都限制在构造时确定的根目录内。"""

    def __init__(self, config: AppConfig, root: Path | None = None) -> None:
        self.config = config
        self.root = _workspace_root(root)
        if self.root.parent == self.root:
            raise ToolDenied("安全限制：工作目录不能是磁盘根目录")

    def _require(self, path: str) -> Path:
        return _require_path(path, self.root)

    def read_file(self, path: str) -> str:
        return read_file(
            path,
            root=self.root,
            max_bytes=self.config.settings.tools.max_read_bytes,
        )

    def write_file(self, path: str, content: str) -> int:
        return write_file(path, content, root=self.root)

    def move_file(self, src: str, dst: str) -> str:
        return move_file(
            src,
            dst,
            root=self.root,
            illegal_chars=self.config.settings.tools.move_illegal_chars,
        )

    def delete_file(self, path: str) -> str:
        return delete_file(path, root=self.root)

    def list_dir(self, path: str = ".") -> list[str]:
        return list_dir(
            path,
            root=self.root,
            max_entries=self.config.settings.tools.max_list_entries,
        )

    # ------------------------------------------------------------------ #
    # 只读 Shell
    # ------------------------------------------------------------------ #

    def run_shell_ro(self, command: str) -> ExecResult:
        """只读命令执行：前缀白名单 + 写操作参数黑名单，不启动 shell。"""
        if not isinstance(command, str) or not command.strip():
            raise ToolDenied("shell 命令不能为空")
        command = command.strip()
        tokens = split_command(command)
        settings = self.config.settings.tools
        prefix = self._readonly_prefix(tokens)
        if prefix not in settings.readonly_shell_prefixes:
            raise ToolDenied(
                f"安全限制：只允许只读命令；收到 {prefix!r}。"
                f"允许：{'、'.join(sorted(settings.readonly_shell_prefixes))}"
            )
        self._reject_write_flags(
            tokens,
            settings.readonly_blocked_flags,
            settings.readonly_blocked_prefixes,
        )
        self._check_readonly_paths(tokens, prefix, settings)
        if prefix in _READONLY_BUILTINS:
            return self._run_builtin(command, prefix, tokens)
        env = os.environ.copy()
        env.update(
            {
                "GIT_PAGER": "cat",
                "PAGER": "cat",
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_OPTIONAL_LOCKS": "0",
            }
        )
        return _execute(self.config, command, tokens, self.root, env=env)

    # 兼容旧测试/调用；语义已改为只读，不再接受 ren/del 等修改命令。
    def run_shell(self, command: str) -> ExecResult:
        return self.run_shell_ro(command)

    @staticmethod
    def _readonly_prefix(tokens: list[str]) -> str:
        name = _command_name(tokens[0])
        if name == "git":
            if len(tokens) < 2:
                return "git"
            return f"git {tokens[1].lower()}"
        if name == "python":
            if len(tokens) >= 2 and tokens[1].lower() == "verify.py":
                return "python verify.py"
            if (
                len(tokens) >= 3
                and tokens[1].lower() == "-m"
                and tokens[2].lower() == "unittest"
            ):
                return "python -m unittest"
            return "python"
        return name

    @staticmethod
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
        self, tokens: list[str], prefix: str, settings: ToolsSettings
    ) -> None:
        if prefix == "git" or prefix.startswith("git "):
            self._check_git(
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
                    is_inside_workspace(candidate, self.root)
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
            is_inside_workspace(candidate, self.root)

    def _check_git(
        self,
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
                is_inside_workspace(candidate, self.root)

    def _run_builtin(
        self, command: str, prefix: str, tokens: list[str]
    ) -> ExecResult:
        try:
            if prefix == "echo":
                text = " ".join(tokens[1:])
                if not text:
                    text = "ECHO is on."
                return ExecResult(command, 0, text + "\n", "")
            if prefix == "pwd":
                return ExecResult(command, 0, str(self.root) + "\n", "")
            if prefix in {"dir", "ls"}:
                args = [token for token in tokens[1:] if not _looks_like_switch(token)]
                target = args[0] if args else "."
                names = self.list_dir(target)
                return ExecResult(command, 0, "\n".join(names) + "\n", "")
            # type / cat
            args = [token for token in tokens[1:] if not _looks_like_switch(token)]
            if not args:
                raise ToolDenied(f"{prefix} 需要至少一个文件")
            chunks: list[str] = []
            for path in args:
                target = self._require(path)
                if not target.is_file():
                    raise ToolDenied(f"不是可读取的文件：{path}")
                max_bytes = self.config.settings.tools.max_read_bytes
                if target.stat().st_size > max_bytes:
                    raise ToolDenied(
                        f"文件超过 {max_bytes // 1024} KiB，已拒绝读取：{path}"
                    )
                chunks.append(_decode(target.read_bytes()))
            return ExecResult(command, 0, "".join(chunks), "")
        except ToolDenied as exc:
            raise
        except OSError as exc:
            raise ToolDenied(f"只读命令执行失败：{command}；{exc.strerror or exc}") from exc
