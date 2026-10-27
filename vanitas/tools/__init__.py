"""工具层公开门面：沙箱、进程、文件、只读 Shell、人类 /exec 与 AgentTools。

所有旧的 `from vanitas.tools import X` 保持可用；实现分别位于 sandbox / runner /
fs / shell / exec，本模块只做 re-export 与 AgentTools 参数绑定。
"""

from __future__ import annotations

import subprocess  # noqa: F401 - 兼容旧测试对 vanitas.tools.subprocess 的补丁
from pathlib import Path

from ..config import AppConfig
from ..errors import ToolDenied
from . import fs, runner, sandbox, shell
from .exec import CANCEL_MESSAGE, _CANCEL_MESSAGE, _ask, run_exec
from .fs import (
    _reject_illegal_move_text,
    delete_file,
    delete_path,
    grep_workspace,
    list_dir,
    move_file,
    patch_file,
    read_file,
    read_file_range,
    write_file,
)
from .runner import (
    MAX_CAPTURE_BYTES,
    ExecResult,
    _NO_WINDOW,
    _append_capture_warning,
    _decode,
    _truncate_bytes,
    run,
)
from .sandbox import (
    _is_safe_path,
    _is_within_cwd,
    _path_text,
    _require_path,
    _resolve_candidate,
    _workspace_root,
    is_inside_workspace,
)
from .shell import (
    WINDOWS_BUILTINS,
    _CMD_CONTROL_CHARS,
    _COMMAND_SUFFIXES,
    _HELP_SWITCHES,
    _READONLY_BUILTINS,
    _check_git,
    _check_readonly_paths,
    _command_name,
    _destructive_index,
    _destructive_verb,
    _is_destructive_command,
    _looks_like_switch,
    _readonly_prefix,
    _reject_write_flags,
    _run_builtin,
    _unwrap_cmd,
    execute_readonly_shell,
    is_whitelisted,
    normalize,
    resolve_argv,
    split_command,
)


def _execute(
    config: AppConfig,
    command: str,
    argv: list[str],
    cwd: Path,
    *,
    env: dict[str, str] | None = None,
) -> ExecResult:
    """兼容旧测试：shell=False 的受控执行；实际逻辑在 runner.run。"""
    return runner.run(config, command, cwd, argv=argv, shell=False, env=env)


def _execute_shell(
    config: AppConfig, command: str, cwd: Path
) -> ExecResult:
    """兼容旧测试：shell=True 的人类 /exec；实际逻辑在 runner.run。"""
    return runner.run(config, command, cwd, shell=True)


class AgentTools:
    """Agent 闭环的内置工具；所有路径都限制在构造时确定的根目录内。

    本类只做「方法签名 + 参数取值」，实现全部下沉到 fs / shell / sandbox。
    """

    def __init__(self, config: AppConfig, root: Path | None = None) -> None:
        self.config = config
        self.root = sandbox._workspace_root(root)
        if self.root.parent == self.root:
            raise ToolDenied("安全限制：工作目录不能是磁盘根目录")

    def _require(self, path: str) -> Path:
        return sandbox._require_path(path, self.root)

    def read_file(self, path: str) -> str:
        return fs.read_file(
            path,
            root=self.root,
            max_bytes=self.config.settings.tools.max_read_bytes,
        )

    def write_file(self, path: str, content: str) -> int:
        return fs.write_file(path, content, root=self.root)

    def move_file(self, src: str, dst: str) -> str:
        return fs.move_file(
            src,
            dst,
            root=self.root,
            illegal_chars=self.config.settings.tools.move_illegal_chars,
        )

    def delete_path(self, path: str) -> str:
        return fs.delete_path(path, root=self.root)

    def delete_file(self, path: str) -> str:
        return self.delete_path(path)

    def list_dir(self, path: str = ".") -> list[str]:
        return fs.list_dir(
            path,
            root=self.root,
            max_entries=self.config.settings.tools.max_list_entries,
        )

    def patch_file(self, path: str, search: str, replace: str) -> str:
        settings = self.config.settings.tools
        return fs.patch_file(
            path,
            search,
            replace,
            root=self.root,
            max_bytes=settings.max_patch_bytes,
        )

    def read_file_range(
        self, path: str, start_line: int = 1, end_line: int = 100
    ) -> str:
        settings = self.config.settings.tools
        return fs.read_file_range(
            path,
            start_line,
            end_line,
            root=self.root,
            max_bytes=settings.max_read_bytes,
            max_lines=settings.max_range_lines,
        )

    def grep_workspace(
        self,
        pattern: str,
        path: str = ".",
        max_results: int | None = None,
    ) -> list[str]:
        settings = self.config.settings.tools
        return fs.grep_workspace(
            pattern,
            path,
            root=self.root,
            max_results=(
                settings.max_grep_results
                if max_results is None
                else max_results
            ),
            max_file_bytes=settings.max_grep_file_bytes,
            ignore_dirs=settings.grep_ignore_dirs,
        )

    def run_shell_ro(self, command: str) -> ExecResult:
        return shell.execute_readonly_shell(self.config, self.root, command)

    # 兼容旧调用；语义与 run_shell_ro 相同，不再接受修改命令。
    def run_shell(self, command: str) -> ExecResult:
        return self.run_shell_ro(command)


# 兼容旧测试对 AgentTools._readonly_prefix 的直接调用；只读绑定，不改变公开方法。
AgentTools._readonly_prefix = staticmethod(shell._readonly_prefix)  # type: ignore[attr-defined]


__all__ = [
    "ExecResult",
    "CANCEL_MESSAGE",
    "MAX_CAPTURE_BYTES",
    "read_file",
    "write_file",
    "move_file",
    "delete_path",
    "delete_file",
    "list_dir",
    "patch_file",
    "read_file_range",
    "grep_workspace",
    "is_inside_workspace",
    "_require_path",
    "_is_safe_path",
    "_is_within_cwd",
    "_workspace_root",
    "_path_text",
    "_resolve_candidate",
    "run",
    "_execute",
    "_execute_shell",
    "_decode",
    "_truncate_bytes",
    "_append_capture_warning",
    "normalize",
    "is_whitelisted",
    "split_command",
    "resolve_argv",
    "execute_readonly_shell",
    "run_exec",
    "AgentTools",
]
