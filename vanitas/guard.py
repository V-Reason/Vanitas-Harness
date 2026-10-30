"""单实例守卫：运行时 / 模型完整性 / 旧实例 / 端口。

退出码契约（沿用旧守卫，并由 tests/test_guard.py 覆盖）：
  0 通过 | 1 端口被占 | 2 模型缺失/分片不全/非法 | 3 运行时缺失 | 4 旧实例未退出 | 9 参数错误

顺序刻意保持「先验模型、再杀旧实例、最后查端口」：
目标模型本身不可用时，不应连累正在服务的实例；杀完仍占用端口说明持有者是外部进程，
必须报出 PID 让操作者决策，而不是静默换端口。

分片模型（<前缀>-00001-of-0000N.gguf）：只接受第 1 片作为入口，逐片校验存在性与总体积；
第 2..N 片由 llama.cpp 依据文件名自动续载，本模块不拼路径、不合并权重。
"""

from __future__ import annotations

import argparse
import locale
import re
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import BASE_DIR, SHARD_RE
from .ui import FALLBACK_THEME, Palette, resolve_enabled

RUNTIME_FILES = ("llama-server.exe", "llama.dll", "ggml-base.dll", "ggml-cuda.dll")
MIN_MODEL_BYTES = 100 * 1024 * 1024
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_LISTEN_STATES = frozenset({"LISTENING", "侦听", "监听"})


@dataclass
class GuardResult:
    code: int
    stdout: list[str] = field(default_factory=list)
    stderr: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# 子进程工具：一律带超时，失败降级为空结果，不让守卫自身卡死
# --------------------------------------------------------------------------- #


def _decode(raw: bytes) -> str:
    for encoding in ("utf-8", locale.getpreferredencoding(False), "mbcs"):
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", "replace")


def _run(command: list[str], timeout: float = 5.0) -> str:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            timeout=timeout,
            creationflags=_NO_WINDOW,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return _decode(completed.stdout or b"")


def _csv_first_field(line: str) -> str:
    stripped = line.strip()
    if not stripped.startswith('"'):
        return ""
    return stripped.split('","')[0].strip('"')


def _process_name(pid: int) -> str:
    output = _run(["tasklist", "/fi", f"pid eq {pid}", "/fo", "csv", "/nh"])
    for line in output.splitlines():
        name = _csv_first_field(line)
        if name:
            return name
    return "未知进程"


def _llama_pids() -> list[int]:
    output = _run(
        ["tasklist", "/fi", "imagename eq llama-server.exe", "/fo", "csv", "/nh"]
    )
    pids: list[int] = []
    for line in output.splitlines():
        if not line.strip().startswith('"'):
            continue
        parts = line.strip().split('","')
        if len(parts) >= 2 and parts[1].isdigit():
            pids.append(int(parts[1]))
    return pids


def _listener_pid(port: int) -> int | None:
    for line in _run(["netstat", "-ano", "-p", "tcp"]).splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        local, state, pid = fields[1], fields[-2], fields[-1]
        if (
            local.endswith(f":{port}")
            and state.upper() in _LISTEN_STATES
            and pid.isdigit()
        ):
            return int(pid)
    return None


def _port_owner(port: int, host: str) -> tuple[int, str] | None:
    """返回 (pid, 进程名)；端口空闲返回 None。pid=0 表示占用者不可识别。"""
    pid = _listener_pid(port)
    if pid:
        return pid, _process_name(pid)
    # netstat 可能看不到（权限/时序），再用独占 bind 复核；不设 SO_REUSEADDR，
    # 否则 Windows 允许绑定到已被监听的端口，会把「占用」误判成「空闲」。
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((host, port))
    except OSError:
        return 0, "未知进程"
    return None


def gpu_memory() -> tuple[int, int] | None:
    """返回 (已用 MiB, 总量 MiB)；nvidia-smi 不可用/输出异常时 None。"""
    output = _run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    line = output.strip().splitlines()[0] if output.strip() else ""
    fields = [item.strip() for item in line.split(",")]
    if len(fields) == 2 and all(item.isdigit() for item in fields):
        return int(fields[0]), int(fields[1])
    return None


def _gpu_snapshot() -> str | None:
    memory = gpu_memory()
    if memory is None:
        return None
    used, total = memory
    return f"GPU 已用 {used} MiB / 总量 {total} MiB"


def process_memory_kb(pid: int) -> int | None:
    """通过 tasklist 读取进程工作集（KiB）；查询失败时 None。"""
    if pid <= 0:
        return None
    output = _run(["tasklist", "/fi", f"pid eq {pid}", "/fo", "csv", "/nh"])
    for line in output.splitlines():
        if not line.strip().startswith('"'):
            continue
        fields = [field.strip('"') for field in line.strip().split('","')]
        if len(fields) < 5 or not fields[1].isdigit():
            continue
        raw = fields[4].replace(",", "").replace("K", "").strip()
        if raw.isdigit():
            return int(raw)
    return None


def _split_shards(path: Path) -> tuple[Path, int, int, str] | str:
    """按文件名推断分片；返回 (目录, 下标, 总数, 前缀) 或状态码字符串。"""
    match = SHARD_RE.match(path.stem)
    if match is None:
        return "SINGLE"
    index = int(match.group("index"))
    total = int(match.group("total"))
    if index != 1:
        return "SHARD_ORDER"
    if total < 1:
        return "BAD_SHARD_COUNT"
    prefix = match.group("prefix")
    return path.parent, index, total, prefix  # type: ignore[return-value]


def _shard_paths(directory: Path, prefix: str, total: int) -> list[Path]:
    return [
        directory / f"{prefix}-{index:05d}-of-{total:05d}.gguf"
        for index in range(1, total + 1)
    ]


def evaluate(
    port: int,
    model_path: str | Path,
    *,
    base_dir: Path | str | None = None,
    host: str = "127.0.0.1",
    allow_kill: bool = True,
    min_bytes: int = MIN_MODEL_BYTES,
    kill_wait_sec: float = 5.0,
) -> GuardResult:
    """执行全部检查，返回 GuardResult；不抛异常，调用方只看 code 与消息。"""
    out: list[str] = []
    err: list[str] = []
    root = Path(base_dir).resolve() if base_dir is not None else BASE_DIR

    try:
        port_number = int(str(port).strip())
    except (TypeError, ValueError):
        return GuardResult(9, out, [f"✗ 端口必须是整数：{port!r}"])
    if not 1 <= port_number <= 65535:
        return GuardResult(9, out, [f"✗ 端口必须在 1–65535 之间：{port_number}"])
    if not str(model_path).strip():
        return GuardResult(9, out, ["✗ 缺少模型路径"])

    # 1) 运行时完整性：缺 DLL 时 llama-server 只会抛 0xc0000135，先替操作者挡住
    missing = [name for name in RUNTIME_FILES if not (root / "bin" / name).is_file()]
    if missing:
        for name in missing:
            err.append(f"✗ 缺少运行组件：\"{root / 'bin' / name}\"")
        err.append("· 请将 llama.cpp CUDA 包完整解压到 bin\\ 目录")
        return GuardResult(3, out, err)

    target = Path(model_path).expanduser()
    if not target.is_absolute():
        target = root / target
    target = target.resolve()

    # 2) 模型校验：存在性 -> 分片完整性 -> 体积下限 -> GGUF magic
    if not target.is_file():
        err.append(f"✗ 找不到模型文件：\"{target}\"")
        err.append("· 核对 models\\ 下的文件名，或先完成下载")
        return GuardResult(2, out, err)

    split = _split_shards(target)
    if split == "SHARD_ORDER":
        err.append(f"✗ 启动入口错误：必须使用第 1 个分片，收到 \"{target.name}\"")
        err.append("· 正确入口形如 models\\<前缀>-00001-of-0000N.gguf")
        return GuardResult(2, out, err)
    if split == "BAD_SHARD_COUNT":
        err.append(f"✗ 分片编号不合法：\"{target}\"")
        return GuardResult(2, out, err)

    if split == "SINGLE":
        files = [target]
        shard_count = 1
    else:
        directory, _index, total, prefix = split  # type: ignore[misc]
        files = _shard_paths(directory, prefix, total)
        shard_count = total
        missing_shards = [item for item in files if not item.is_file()]
        if missing_shards:
            for item in missing_shards:
                err.append(f"✗ 分片缺失：\"{item}\"")
            err.append("· 分片必须同目录、同前缀、编号连续")
            return GuardResult(2, out, err)

    total_bytes = 0
    for item in files:
        try:
            total_bytes += item.stat().st_size
        except OSError as exc:
            err.append(f"✗ 无法读取分片：\"{item}\"；{exc.strerror or exc}")
            return GuardResult(2, out, err)
    if total_bytes < min_bytes:
        err.append(
            f"✗ 模型体积不足 {min_bytes // (1024 * 1024)} MB，"
            f"疑似下载未完成：\"{target}\""
        )
        return GuardResult(2, out, err)

    try:
        with target.open("rb") as handle:
            magic = handle.read(4)
    except OSError as exc:
        err.append(f"✗ 无法读取文件头：\"{target}\"；{exc.strerror or exc}")
        return GuardResult(2, out, err)
    if magic != b"GGUF":
        err.append(f"✗ 文件不是有效的 GGUF 模型：\"{target}\"")
        return GuardResult(2, out, err)

    # 3) 清理存量实例：8 GB 显存禁不起双实例，宁可杀错不可放过
    if allow_kill:
        stale = _llama_pids()
        if stale:
            out.append(
                "⚠ 发现遗留服务，正在清理 · PID "
                + "、".join(str(pid) for pid in stale)
            )
            _run(["taskkill", "/f", "/im", "llama-server.exe"])
            deadline = time.monotonic() + max(0.0, kill_wait_sec)
            while time.monotonic() < deadline:
                if not _llama_pids():
                    break
                time.sleep(0.2)
            if _llama_pids():
                err.append(
                    f"✗ 旧服务 {kill_wait_sec:.0f} 秒内未退出，请手动结束进程后重试"
                )
                return GuardResult(4, out, err)

    # 4) 端口占用：必须报出 PID 与映像名，否则操作者无从下手
    owner = _port_owner(port_number, host)
    if owner is not None:
        pid, name = owner
        err.append(f"✗ 端口 {port_number} 被 PID {pid} ({name}) 占用")
        err.append("· 结束该进程，或设置环境变量 VANITAS_PORT=<空闲端口>")
        return GuardResult(1, out, err)

    # 5) 显存快照：只展示不拦截，阈值随档位变化，不替操作者拍板
    snapshot = _gpu_snapshot()
    if snapshot:
        out.append(f"· {snapshot}")

    size_mb = round(total_bytes / (1024 * 1024))
    shard_desc = "单文件" if shard_count == 1 else f"{shard_count} 个分片，校验完整"
    out.append(
        f"✓ 环境核验通过 · 模型 {size_mb} MB · {shard_desc}"
        f" · 端口 {port_number} 空闲 · 显存通道已独占"
    )
    return GuardResult(0, out, err)


_PREFIX_METHODS = (
    ("✓ ", "ok"),
    ("✗ ", "err"),
    ("⚠ ", "warn"),
    ("⋯ ", "spin"),
    ("· ", "meta"),
)


def _colorize_line(line: str, palette: Palette) -> str:
    """按首字符选语义色；符号由 Palette 重新拼回，不改变文本内容。"""
    for prefix, method in _PREFIX_METHODS:
        if line.startswith(prefix):
            return getattr(palette, method)(line[len(prefix):])
    return line


def format_for_cli(result: GuardResult, palette: Palette | None = None) -> tuple[list[str], list[str]]:
    """只做着色，不加任何前缀：退出码本身就是机读契约，消息只负责人读。"""
    painter = palette or Palette({}, enabled=False)
    stdout_lines = [_colorize_line(line, painter) for line in result.stdout]
    stderr_lines = [_colorize_line(line, painter) for line in result.stderr]
    return stdout_lines, stderr_lines


def emit(result: GuardResult, *, palette: Palette | None = None) -> None:
    stdout_lines, stderr_lines = format_for_cli(result, palette)
    for line in stdout_lines:
        print(line)
    for line in stderr_lines:
        print(line, file=sys.stderr)


def process_name(pid: int) -> str:
    """供 launcher / status 复用：PID 对应的映像名，查不到返回「未知进程」。"""
    return _process_name(pid)


def llama_pids() -> list[int]:
    """当前存活的 llama-server.exe PID 列表（只读，不杀进程）。"""
    return _llama_pids()


def gpu_status() -> str | None:
    """显存快照（只展示不拦截）；nvidia-smi 不可用时返回 None。"""
    return _gpu_snapshot()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m vanitas.guard",
        description="Vanitas-Harness 守卫：运行时 / 模型完整性 / 旧实例 / 端口",
    )
    parser.add_argument("--port", required=False, help="目标端口")
    parser.add_argument("--model", required=False, help="模型路径，分片只传首片")
    parser.add_argument("--base-dir", default=None, help="仓库根，默认按包位置推断")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--no-kill",
        action="store_true",
        help="不回收存量 llama-server，测试用；生产必须允许回收",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    args = parse_args(argv)
    result = evaluate(
        args.port,
        args.model,
        base_dir=args.base_dir,
        host=args.host,
        allow_kill=not args.no_kill,
    )
    # 独立入口没有 settings 可读，固定用默认主题，仅 auto 判断终端能力
    palette = Palette(
        FALLBACK_THEME,
        resolve_enabled("auto", sys.stdout) and resolve_enabled("auto", sys.stderr),
    )
    emit(result, palette=palette)
    return result.code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
