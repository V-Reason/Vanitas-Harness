"""llama-server 生命周期管理。

参数与旧启动器逐字一致：-fa on / -fit off / -a / --metrics / -np 1 / 分片只传首片。
启动前必须过守卫；启动后必须以 /health 确认就绪，不能只看进程活着。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from . import guard
from .config import AppConfig, ModelSpec, ProfileSpec
from .engine import EngineClient
from .errors import GuardError, LaunchError
from .ui import Palette

SERVER_EXE = "llama-server.exe"
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass
class RunRecord:
    model_id: str
    pid: int
    port: int
    profile: str
    alias: str
    started_at: str
    log_path: str


@dataclass
class RunStatus:
    record: RunRecord
    alive: bool
    healthy: bool


def run_dir() -> Path:
    return Path.home() / ".vanitas_run"


def log_dir() -> Path:
    return Path.home() / ".vanitas_logs"


def record_path(model_id: str) -> Path:
    return run_dir() / f"{model_id}.json"


def _ensure_dirs() -> None:
    run_dir().mkdir(parents=True, exist_ok=True)
    log_dir().mkdir(parents=True, exist_ok=True)


def _write_record(record: RunRecord) -> None:
    _ensure_dirs()
    record_path(record.model_id).write_text(
        json.dumps(asdict(record), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_record(path: Path) -> RunRecord | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        return RunRecord(
            model_id=str(data["model_id"]),
            pid=int(data["pid"]),
            port=int(data["port"]),
            profile=str(data["profile"]),
            alias=str(data["alias"]),
            started_at=str(data["started_at"]),
            log_path=str(data["log_path"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def read_records() -> list[RunRecord]:
    directory = run_dir()
    if not directory.is_dir():
        return []
    records: list[RunRecord] = []
    for path in sorted(directory.glob("*.json")):
        record = _read_record(path)
        if record is not None:
            records.append(record)
    return records


def clear_record(model_id: str) -> None:
    try:
        record_path(model_id).unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        print(f"⚠ 无法删除运行记录 {record_path(model_id)}：{exc}", file=sys.stderr)


def clear_records() -> None:
    for record in read_records():
        clear_record(record.model_id)


def pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    return guard.process_name(pid) != "未知进程"


def build_args(
    model: ModelSpec, profile: ProfileSpec, port: int, host: str, base_dir: Path
) -> list[str]:
    """逐字迁入旧 bat 的参数；顺序固定，便于对照与排障。"""
    return [
        str(Path(base_dir) / "bin" / SERVER_EXE),
        "-m",
        str(model.path),
        "--host",
        host,
        "--port",
        str(port),
        "-c",
        str(profile.ctx),
        "-ngl",
        "999",
        "-np",
        "1",
        "-fa",
        "on",
        "-ctk",
        profile.kv_k,
        "-ctv",
        profile.kv_v,
        "-a",
        model.alias,
        "-fit",
        "off",
        "--metrics",
    ]


def read_log_tail(path: str | Path, limit: int = 15) -> str:
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"（日志不可读：{exc.strerror or exc}）"
    tail = lines[-limit:]
    return "\n".join(tail) if tail else "（日志为空）"


def _log_path(model: ModelSpec) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return log_dir() / f"{model.id}-{stamp}.log"


def _guard_or_raise(
    config: AppConfig, model: ModelSpec, port: int, palette: Palette | None
) -> None:
    result = guard.evaluate(
        port,
        model.path,
        base_dir=config.base_dir,
        host=config.settings.host,
        allow_kill=True,
        kill_wait_sec=config.settings.launcher.kill_wait_sec,
    )
    if result.code != 0:
        # 失败行交给调用方统一上色打印，避免「守卫打一遍、CLI 再打一遍」
        raise GuardError(result.code, result.stderr or result.stdout)
    guard.emit(result, palette=palette)


def _spawn(
    config: AppConfig,
    model: ModelSpec,
    profile: ProfileSpec,
    port: int,
    log_path: Path,
) -> subprocess.Popen:
    args = build_args(model, profile, port, config.settings.host, config.base_dir)
    bin_dir = Path(config.base_dir) / "bin"
    if not bin_dir.is_dir():
        raise LaunchError(f"运行时目录不存在：{bin_dir}（守卫应已拦截，请检查 bin\\）")
    try:
        if os.name == "nt":
            handle = log_path.open("ab")
            try:
                process = subprocess.Popen(
                    args,
                    cwd=str(bin_dir),
                    stdin=subprocess.DEVNULL,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    creationflags=_NO_WINDOW,
                )
            finally:
                handle.close()
        else:  # 仅为可读性保留；本机只跑 Windows
            with log_path.open("ab") as handle:
                process = subprocess.Popen(
                    args,
                    cwd=str(bin_dir),
                    stdin=subprocess.DEVNULL,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
    except OSError as exc:
        raise LaunchError(
            f"无法启动 {bin_dir / SERVER_EXE}：{exc.strerror or exc}；"
            "请确认 bin\\ 下 DLL 完整"
        ) from exc
    return process


def start(
    config: AppConfig,
    model: ModelSpec,
    profile: ProfileSpec,
    *,
    background: bool = True,
    palette: Palette | None = None,
    verbose: bool = False,
) -> RunRecord:
    paint = palette or Palette({}, enabled=False)
    _ensure_dirs()
    port = config.effective_port(model)
    _guard_or_raise(config, model, port, palette)
    log_path = _log_path(model)
    process = _spawn(config, model, profile, port, log_path)
    record = RunRecord(
        model_id=model.id,
        pid=process.pid,
        port=port,
        profile=profile.key,
        alias=model.alias,
        started_at=datetime.now().isoformat(timespec="seconds"),
        log_path=str(log_path),
    )
    _write_record(record)
    print(paint.meta(f"{model.display} · 档位 {profile.key}"))
    if verbose:
        print(
            paint.meta(
                f"ctx {profile.ctx} · KV {profile.kv_k}/{profile.kv_v}"
                f" · {endpoint(config, port)} · PID {record.pid}"
            )
        )
        print(paint.meta(f"日志 {log_path}"))
    return record


def endpoint(config: AppConfig, port: int) -> str:
    return f"http://{config.settings.host}:{port}"


def wait_ready(config: AppConfig, record: RunRecord) -> float:
    """轮询 /health；进程提前退出立刻失败并附日志尾部，避免「启动即退出、无日志」。"""
    client = EngineClient(
        config.settings.request, config.settings.host, record.port, template_kwargs={}
    )
    deadline = time.monotonic() + config.settings.launcher.health_timeout_sec
    interval = config.settings.launcher.health_poll_interval_ms / 1000.0
    started = time.monotonic()
    while time.monotonic() < deadline:
        if not pid_running(record.pid):
            tail = read_log_tail(record.log_path)
            clear_record(record.model_id)
            raise LaunchError(
                f"llama-server（PID {record.pid}）在就绪前退出。日志尾部：\n{tail}"
            )
        if client.health(timeout=3.0):
            ids = client.model_ids()
            if record.alias not in ids:
                raise LaunchError(
                    f"端口 {record.port} 上运行的模型标识为 {ids}，"
                    f"与期望 {record.alias!r} 不一致（端口被别的实例占用？）"
                )
            return time.monotonic() - started
        time.sleep(interval)
    tail = read_log_tail(record.log_path)
    raise LaunchError(
        f"等待 {config.settings.launcher.health_timeout_sec:.0f}s 后服务仍未就绪"
        f"（PID {record.pid}，端口 {record.port}）。日志尾部：\n{tail}"
    )


def run_foreground(
    config: AppConfig,
    model: ModelSpec,
    profile: ProfileSpec,
    *,
    palette: Palette | None = None,
    verbose: bool = False,
) -> int:
    paint = palette or Palette({}, enabled=False)
    port = config.effective_port(model)
    _guard_or_raise(config, model, port, palette)
    args = build_args(model, profile, port, config.settings.host, config.base_dir)
    bin_dir = Path(config.base_dir) / "bin"
    print(paint.meta(f"{model.display} · 档位 {profile.key}（前台）"))
    if verbose:
        print(paint.meta(f"地址 {endpoint(config, port)}  参数：{' '.join(args[1:])}"))
    try:
        process = subprocess.Popen(args, cwd=str(bin_dir))
    except OSError as exc:
        raise LaunchError(
            f"无法启动 {bin_dir / SERVER_EXE}：{exc.strerror or exc}；请确认 bin\\ 下 DLL 完整"
        ) from exc
    try:
        code = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        try:
            code = process.wait(timeout=config.settings.launcher.kill_wait_sec)
        except subprocess.TimeoutExpired:
            process.kill()
            code = process.wait()
        raise
    print(paint.meta(f"llama-server 已退出，退出码 {code}"))
    return code


def stop_all(config: AppConfig) -> list[int]:
    pids = guard.llama_pids()
    if not pids:
        clear_records()
        return []
    for pid in pids:
        guard._run(["taskkill", "/f", "/pid", str(pid)])
    deadline = time.monotonic() + max(0.0, config.settings.launcher.kill_wait_sec)
    while time.monotonic() < deadline:
        if not guard.llama_pids():
            break
        time.sleep(0.2)
    remaining = guard.llama_pids()
    if remaining:
        raise LaunchError(
            f"仍有 llama-server 进程未退出：{remaining}；请手动结束后再重试"
        )
    clear_records()
    return pids


def inspect(config: AppConfig) -> list[RunStatus]:
    """D6：端点可见性——pid 记录 + /health + 当前模型，供 /status 与 /model 使用。"""
    statuses: list[RunStatus] = []
    for record in read_records():
        alive = pid_running(record.pid)
        healthy = False
        if alive:
            client = EngineClient(
                config.settings.request,
                config.settings.host,
                record.port,
                template_kwargs={},
            )
            healthy = client.health(timeout=2.0)
        statuses.append(RunStatus(record=record, alive=alive, healthy=healthy))
    return statuses
