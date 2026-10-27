"""REPL 状态栏：提示符格式化 + /status 紧凑/展开渲染。

本模块不 import Repl 类型；依赖 Repl 的片段通过参数传入，纯函数式便于单测。
"""
from __future__ import annotations

import contextlib
import io
from datetime import datetime
from typing import TYPE_CHECKING

from .. import guard, launcher
from ..ui import SYMBOLS, Palette

if TYPE_CHECKING:
    from .loop import Repl


def context_bar(pct: float, width: int = 20) -> str:
    filled = max(0, min(width, int(round(pct / 100 * width))))
    return "█" * filled + "░" * (width - filled)


def format_uptime(started_at: str) -> str:
    try:
        started = datetime.fromisoformat(started_at)
    except ValueError:
        return "-"
    seconds = max(0, int((datetime.now() - started).total_seconds()))
    return f"{seconds // 60}分{seconds % 60:02d}秒"


def format_prompt(
    *,
    model_id: str,
    profile_key: str,
    used_tokens: int,
    total_ctx: int,
    sleeping: bool,
    palette: Palette,
    warn_pct: int,
    danger_pct: int,
) -> str:
    """提示符：<model>@<profile> [ctx: N%] › 。"""
    pct = int((used_tokens / total_ctx) * 100) if total_ctx > 0 else 0
    tag_text = f"ctx: {pct}%"
    if pct >= danger_pct:
        tag = palette.error(f"[{tag_text}]")
    elif pct >= warn_pct:
        tag = palette.warning(f"[{tag_text}]")
    else:
        tag = palette.muted(f"[{tag_text}]")
    head = palette.accent(f"{model_id}@{profile_key}")
    if sleeping:
        head += " " + palette.muted("[休眠]")
    return f"{head} {tag} {SYMBOLS['prompt']} "


def format_status_compact(repl: "Repl") -> list[str]:
    """默认 /status 输出：4 行紧凑视图。"""
    used = repl.session.message_tokens()
    total = repl.profile.ctx
    pct = (used / total * 100) if total > 0 else 0.0
    ready = bool(repl.client and repl.client.health())
    lines: list[str] = []
    lines.append(
        repl.palette.meta(
            f"模型 {repl.model.id} · 档位 {repl.profile.key} · "
            f"{repl.profile.desc}"
        )
    )
    memory = guard.gpu_memory()
    if memory is not None:
        gpu_used, gpu_total = memory
        gpu_text = f"显存 {gpu_used:,} / {gpu_total:,} MiB"
    else:
        gpu_text = "显存 -"
    state = repl.palette.ok("就绪") if ready else repl.palette.err("停止")
    lines.append(
        f"{repl.palette.meta(f'端口 {repl.port}')}  {state} · {gpu_text}"
    )
    repl_settings = repl.config.settings.repl
    bar = context_bar(pct, repl_settings.status_bar_width)
    if pct >= repl_settings.ctx_danger_pct:
        bar_text = repl.palette.error(bar)
    elif pct >= repl_settings.ctx_warn_pct:
        bar_text = repl.palette.warning(bar)
    else:
        bar_text = repl.palette.muted(bar)
    lines.append(
        repl.palette.meta(
            f"视窗 {bar_text} {pct:.1f}% · 已用 {used:,} / "
            f"上限 {total:,} tokens"
        )
    )
    loop = getattr(repl, "loop", None)
    steps_text = (
        f"{loop.steps}/{loop.max_steps}" if loop is not None
        else f"-/{repl.config.settings.agent.max_steps}"
    )
    if repl.last_turn is not None:
        ttft = (
            f"{repl.last_turn.ttft_ms:.0f} ms"
            if repl.last_turn.ttft_ms is not None
            else "-"
        )
        tps = f"{repl.last_turn.tps:.1f} t/s"
    else:
        ttft = "-"
        tps = "-"
    lines.append(
        repl.palette.meta(
            f"Agent 步数 {steps_text} · 最近 TTFT {ttft} · 生成 {tps}"
        )
    )
    return lines


def _print_status_verbose(repl: "Repl") -> None:
    detail = True
    print(repl.palette.meta(f"模型 {repl.model.id} · {repl.model.display}"))
    print(
        repl.palette.meta(
            f"档位 {repl.profile.key} · {repl.profile.desc}"
            f" · ctx {repl.profile.ctx} · KV "
            f"{repl.profile.kv_k}/{repl.profile.kv_v}"
        )
    )
    if repl.pending_profile is not None:
        print(
            repl.palette.meta(
                f"待生效档位 {repl.pending_profile.key} · "
                f"{repl.pending_profile.desc}"
            )
        )
    ready = bool(repl.client and repl.client.health())
    state = repl.palette.ok("就绪") if ready else repl.palette.err("停止")
    head = repl.palette.meta(f"端口 {repl.port}")
    print(f"{head}  {state}")

    used = repl.session.message_tokens()
    total = repl.profile.ctx
    pct = (used / total * 100) if total > 0 else 0.0
    repl_settings = repl.config.settings.repl
    bar = context_bar(pct, repl_settings.status_bar_width)
    if pct >= repl_settings.ctx_danger_pct:
        bar_text = repl.palette.error(bar)
    elif pct >= repl_settings.ctx_warn_pct:
        bar_text = repl.palette.warning(bar)
    else:
        bar_text = repl.palette.muted(bar)
    print(
        repl.palette.meta(
            f"视窗监控 {bar_text} {pct:.1f}% · 已用 {used:,} / "
            f"上限 {total:,} tokens · 剩余 {max(0, total - used):,}"
        )
    )
    if repl._sleeping:
        print(repl.palette.meta("空闲休眠中；输入任意内容自动唤醒"))
    else:
        remaining = repl._watchdog.remaining()
        if remaining is not None and repl.client is not None:
            print(repl.palette.meta(f"空闲倒计时 {remaining:.0f}s"))

    memory = guard.gpu_memory()
    if memory is not None:
        gpu_used, gpu_total = memory
        free = max(0, gpu_total - gpu_used)
        print(
            repl.palette.meta(
                f"显存物理 GPU 已用 {gpu_used:,} MiB / "
                f"总量 {gpu_total:,} MiB · 余量 {free:,} MiB"
            )
        )
    else:
        gpu = guard.gpu_status()
        print(
            repl.palette.meta(gpu)
            if gpu
            else repl.palette.meta("GPU 信息不可用，nvidia-smi 未找到")
        )

    records = launcher.read_records()
    current = next(
        (
            record
            for record in records
            if record.model_id == repl.model.id
            and launcher.pid_running(record.pid)
        ),
        None,
    )
    if current is not None:
        rss = guard.process_memory_kb(current.pid)
        rss_text = f"{rss / 1024:.0f} MiB" if rss is not None else "-"
        print(
            repl.palette.meta(
                f"系统进程 PID {current.pid} · RSS {rss_text} · "
                f"存活 {format_uptime(current.started_at)}"
            )
        )
    else:
        print(repl.palette.meta("系统进程 - · 无当前模型运行记录"))
    if detail:
        for record in records:
            alive = launcher.pid_running(record.pid)
            print(
                repl.palette.meta(
                    f"{record.model_id} PID {record.pid} 端口 {record.port}"
                    f" 档位 {record.profile} {'运行' if alive else '停止'}"
                )
            )

    if repl.last_turn is not None:
        ttft = (
            f"{repl.last_turn.ttft_ms:.0f} ms"
            if repl.last_turn.ttft_ms is not None
            else "-"
        )
        print(
            repl.palette.meta(
                f"最近推理 TTFT {ttft} · 生成 {repl.last_turn.tps:.1f} t/s"
            )
        )
    else:
        print(repl.palette.meta("最近推理 TTFT - · 生成 -"))

    loop = getattr(repl, "loop", None)
    if loop is not None:
        print(
            repl.palette.meta(
                f"Agent 步数配额 {loop.steps}/{loop.max_steps}"
                f" · 熔断停滞计数 {loop.stagnation_count}/{loop.stagnation_limit}"
            )
        )
    else:
        print(
            repl.palette.meta(
                f"Agent 步数配额 -/{repl.config.settings.agent.max_steps}"
                " · 熔断停滞计数 -/3"
            )
        )
    rounds = sum(
        1 for item in repl.session.messages if item.get("role") == "assistant"
    )
    print(
        repl.palette.meta(
            f"会话 {len(repl.session.messages)} 条消息 "
            f"({rounds} 轮交互) · 约 {used} tokens"
        )
    )


def format_status_verbose(repl: "Repl") -> list[str]:
    """保留旧版完整的 /status -v 展开输出。"""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        _print_status_verbose(repl)
    return buffer.getvalue().splitlines()


__all__ = [
    "context_bar",
    "format_uptime",
    "format_prompt",
    "format_status_compact",
    "format_status_verbose",
]
