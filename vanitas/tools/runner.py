"""进程执行唯一入口：超时、截断、解码、错误映射集中在此。

- shell=False：受控 argv 执行，供 AgentTools.run_shell_ro 使用
- shell=True：人类 /exec 的原生 shell，供 tools.exec.run_exec 使用

两条路径共享同一份超时/截断/解码逻辑，杜绝 _execute 与 _execute_shell 的代码漂移。
"""
from __future__ import annotations

import locale
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..config import AppConfig
from ..errors import ToolDenied

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
MAX_CAPTURE_BYTES = 512 * 1024


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


def _append_capture_warning(
    text: str,
    label: str,
    omitted: int,
    max_bytes: int = MAX_CAPTURE_BYTES,
) -> str:
    note = (
        f"[... 底层{label}超出 {max_bytes // 1024} KiB，"
        f"已在字节流阶段硬截断，丢弃 {omitted} 字节 ...]"
    )
    return f"{text}\n{note}" if text else note



def run(
    config: AppConfig,
    command: str,
    cwd: Path,
    *,
    argv: list[str] | None = None,
    shell: bool = False,
    env: dict[str, str] | None = None,
) -> ExecResult:
    """统一进程执行入口。"""
    if shell:
        if argv is not None:
            raise ValueError("shell=True 时不允许传 argv")
        invoke: str | list[str] = command
        process_env = None
    else:
        if not argv:
            raise ValueError("shell=False 时必须提供非空 argv")
        invoke = argv
        process_env = env

    timeout = config.settings.request.timeout_sec
    try:
        completed = subprocess.run(
            invoke,
            cwd=str(cwd),
            capture_output=True,
            timeout=timeout,
            shell=shell,
            creationflags=_NO_WINDOW,
            env=process_env,
        )
    except FileNotFoundError as exc:
        if shell:
            raise ToolDenied(
                f"命令启动失败：{exc.strerror or exc}"
            ) from exc
        missing = argv[0] if argv else command
        raise ToolDenied(f"命令不存在：{missing}，PATH 中找不到") from exc
    except subprocess.TimeoutExpired as exc:
        raise ToolDenied(
            f"命令超过 {timeout:.0f}s 未结束，已终止：{command}"
        ) from exc
    except OSError as exc:
        raise ToolDenied(f"命令启动失败：{exc.strerror or exc}") from exc

    raw_stdout = completed.stdout or b""
    raw_stderr = completed.stderr or b""
    capture_limit = config.settings.tools.max_capture_bytes
    stdout_bytes, stdout_truncated, stdout_omitted = _truncate_bytes(
        raw_stdout, capture_limit
    )
    stderr_bytes, stderr_truncated, stderr_omitted = _truncate_bytes(
        raw_stderr, capture_limit
    )
    stdout = _decode(stdout_bytes)
    stderr = _decode(stderr_bytes)
    if stdout_truncated:
        stdout = _append_capture_warning(
            stdout, "标准输出", stdout_omitted, capture_limit
        )
    if stderr_truncated:
        stderr = _append_capture_warning(
            stderr, "标准错误", stderr_omitted, capture_limit
        )
    return ExecResult(
        command=command,
        exit_code=completed.returncode,
        stdout=stdout,
        stderr=stderr,
    )


__all__ = [
    "ExecResult", "MAX_CAPTURE_BYTES", "_NO_WINDOW", "run",
    "_decode", "_truncate_bytes", "_append_capture_warning",
]
