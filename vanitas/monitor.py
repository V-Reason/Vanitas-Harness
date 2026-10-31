"""翻译实况看板与终态统计（UI 层）。

`vanitas.translate` 只发 `SegmentStat` 事件，本模块负责环形缓冲、ANSI 局部重绘
与最终汇总；业务层不反向依赖终端渲染细节。非 TTY/未启用着色时只保留
终态统计，避免污染管道与 CI 日志。
"""

from __future__ import annotations

import shutil
import sys
import threading
import time
import unicodedata
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from .metrics import SegmentStat, compute_rate, count_cjk, format_duration
from .ui import Palette, strip_ansi

FALLBACK_SIZE = (80, 24)
FULL_REPORT_LIMIT = 12
FAILURE_DETAIL_LIMIT = 5


def _clip(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)] + "…"


def _char_width(char: str) -> int:
    """终端显示宽度：CJK 全角按 2 列，组合字符不占列。"""
    if unicodedata.combining(char):
        return 0
    return 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1


def _display_width(text: str) -> int:
    return sum(_char_width(char) for char in text)


def _clip_display(text: str, limit: int) -> str:
    """按终端显示宽度裁剪，保证单行不因 CJK 双宽而折行。"""
    if limit <= 0:
        return ""
    if _display_width(text) <= limit:
        return text
    target = max(0, limit - 1)
    width = 0
    pieces: list[str] = []
    for char in text:
        step = _char_width(char)
        if width + step > target:
            break
        pieces.append(char)
        width += step
    return "".join(pieces) + "…"


def _pad_display(text: str, width: int) -> str:
    """按显示宽度右侧补空格；调用方保证 text 已被裁剪。"""
    return text + " " * max(0, width - _display_width(text))


class TranslationMonitor:
    """最近 N 段实时对比 + 任务终态统计。"""

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        enabled: bool | None = None,
        capacity: int = 5,
        palette: Palette | None = None,
    ) -> None:
        self._stream = stream if stream is not None else sys.stderr
        self._palette = palette or Palette({}, enabled=False)
        if enabled is None:
            try:
                enabled = bool(self._stream.isatty())
            except (AttributeError, ValueError, OSError):
                enabled = False
        self._enabled = bool(enabled)
        self._records: deque[tuple[int, SegmentStat]] = deque(
            maxlen=max(1, int(capacity))
        )
        self._lock = threading.Lock()
        self._started = time.perf_counter()
        self._segments = 0
        self._tokens = 0
        self._zh_chars = 0
        self._drawn = 0
        self._size: tuple[int, int] | None = None
        self._task_index = 0
        self._task_total = 0
        self._task_label = ""
        self._segment_done = 0
        self._segment_total = 0

    # ------------------------------------------------------------------ #
    # 事件
    # ------------------------------------------------------------------ #

    def record(self, stat: SegmentStat) -> None:
        with self._lock:
            self._segments += 1
            self._tokens += max(0, int(stat.tokens))
            self._zh_chars += count_cjk(stat.translation)
            self._records.append((self._segments, stat))
            self._draw_locked()

    def print_line(self, text: str) -> None:
        with self._lock:
            self._clear_locked()
            if not self._enabled:
                text = strip_ansi(text)
            self._write(text + "\n")

    def begin_file(self, index: int, total: int, label: str) -> None:
        """开始一个新文件：重置段统计，记录文件上下文并立即绘制头部。

        非 TTY 不画面板，但输出一行“正在翻译”，让管道日志也知道当前文件。
        """
        with self._lock:
            self._clear_locked()
            self._reset_stats_locked()
            self._task_index = max(0, int(index))
            self._task_total = max(0, int(total))
            self._task_label = label
            self._segment_done = 0
            self._segment_total = 0
            if self._enabled:
                self._draw_locked()
            else:
                self._write(
                    strip_ansi(
                        self._palette.meta(
                            f"[{self._task_index}/{self._task_total}] "
                            f"正在翻译 {label}"
                        )
                    )
                    + "\n"
                )

    def set_segment_progress(self, done: int, total: int) -> None:
        """更新当前文件的段进度；由 translate_file 的 on_segment 回调驱动。"""
        with self._lock:
            self._segment_done = max(0, int(done))
            self._segment_total = max(0, int(total))
            self._draw_locked()

    def reset(self) -> None:
        """开始一个新文件的统计：清面板、清环形缓冲并重置计时。"""
        with self._lock:
            self._clear_locked()
            self._reset_stats_locked()
            self._task_index = 0
            self._task_total = 0
            self._task_label = ""
            self._segment_done = 0
            self._segment_total = 0

    def clear(self) -> None:
        """只擦除看板、不输出统计；用于异常退出路径。"""
        with self._lock:
            self._clear_locked()

    @property
    def segments(self) -> int:
        with self._lock:
            return self._segments

    @property
    def tokens(self) -> int:
        with self._lock:
            return self._tokens

    @property
    def zh_chars(self) -> int:
        with self._lock:
            return self._zh_chars

    def finish(self) -> str:
        with self._lock:
            self._clear_locked()
            summary = "\n".join(self._summary_lines())
            self._write(summary + "\n")
            return summary

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #

    def _elapsed(self) -> float:
        return max(0.0, time.perf_counter() - self._started)

    def _reset_stats_locked(self) -> None:
        self._records.clear()
        self._started = time.perf_counter()
        self._segments = 0
        self._tokens = 0
        self._zh_chars = 0
        self._size = None

    def _summary_lines(self) -> list[str]:
        elapsed = self._elapsed()
        summary = (
            f"{self._segments} 段 · {self._zh_chars} 字 · "
            f"{format_duration(elapsed)} · "
            f"{compute_rate(self._tokens, elapsed):.2f} tok/s · "
            f"{compute_rate(self._zh_chars, elapsed):.2f} 字/秒"
        )
        # enabled 是「终端可安全输出控制序列」的唯一开关；即使调用方传了
        # enabled=True 的 Palette，非 TTY 下也不能把 ANSI 写进管道。
        return [self._palette.muted(summary) if self._enabled else summary]

    def _header_line(self) -> str:
        if self._task_total <= 0:
            return ""
        if self._segment_total > 0:
            done = min(self._segment_done, self._segment_total)
            percent = min(100, int(done * 100 / self._segment_total))
            progress = f"{done}/{self._segment_total} 段 · {percent}%"
        else:
            progress = "准备中"
        return (
            f"[{self._task_index}/{self._task_total}] {self._task_label} · "
            f"{progress} · {format_duration(self._elapsed())}"
        )

    def _rows(self) -> list[str]:
        return [
            f"[{index:02d}] {_clip(stat.source, 24)} → "
            f"{_clip(stat.translation, 30)} | "
            f"{stat.elapsed_s:.1f}s | {stat.tps:.1f} tok/s"
            for index, stat in self._records
        ]

    def _terminal_size(self) -> tuple[int, int]:
        try:
            size = shutil.get_terminal_size(fallback=FALLBACK_SIZE)
            return max(20, int(size.columns)), max(3, int(size.lines))
        except (OSError, TypeError, ValueError):
            return FALLBACK_SIZE

    def _max_rows(self, lines: int) -> int:
        capacity = self._records.maxlen or 1
        # 头部、完成行/摘要/提示各留空间，避免面板把终端顶出可视区。
        reserved = 3 if self._task_total > 0 else 2
        return max(1, min(int(capacity), int(lines) - reserved))

    def _draw_locked(self) -> None:
        if not self._enabled:
            return
        columns, lines = self._terminal_size()
        if self._size is not None and self._size != (columns, lines):
            # 尺寸变化后旧行的实际折行数已不可靠，先清掉再按新宽度重画。
            self._clear_locked()
        self._size = (columns, lines)
        header = self._header_line()
        rows = self._rows()[-self._max_rows(lines):]
        panel = ([header] if header else []) + rows
        if not panel:
            return
        if self._drawn:
            self._stream.write(f"\x1b[{self._drawn}A")
        height = max(self._drawn, len(panel))
        limit = max(1, columns - 1)
        for index in range(height):
            self._stream.write("\r\x1b[2K")
            if index < len(panel):
                self._stream.write(_clip_display(panel[index], limit))
            self._stream.write("\n")
        self._drawn = height
        self._flush()

    def _clear_locked(self) -> None:
        if not self._enabled or self._drawn <= 0:
            self._drawn = 0
            return
        self._stream.write(f"\x1b[{self._drawn}A")
        for _ in range(self._drawn):
            self._stream.write("\r\x1b[2K\n")
        # 擦除后面板区仍是 N 行空白；回到顶部并清到屏幕末尾，避免完成行前出现大片空行。
        self._stream.write(f"\x1b[{self._drawn}A\x1b[J")
        self._drawn = 0
        self._flush()

    def _write(self, text: str) -> None:
        self._stream.write(text)
        self._flush()

    def _flush(self) -> None:
        try:
            self._stream.flush()
        except (AttributeError, ValueError):
            pass


@dataclass(frozen=True)
class FileReport:
    """一个文件的终态；只含 UI 需要的字段，不反向依赖 translate 业务对象。"""

    index: int
    total: int
    label: str
    output: str
    segments: int
    elapsed_s: float
    tokens: int = 0
    zh_chars: int = 0
    failures: tuple[str, ...] = ()
    error: str | None = None
    interrupted: bool = False
    overwrote: bool = False

    @property
    def status(self) -> str:
        if self.interrupted:
            return "interrupted"
        if self.error:
            return "failed"
        if self.failures:
            return "partial"
        return "ok"


class TranslationReport:
    """批处理汇总：统计、逐文件行、失败明细；render() 只返回文本，不打印。"""

    def __init__(self) -> None:
        self._items: list[FileReport] = []

    @property
    def items(self) -> tuple[FileReport, ...]:
        return tuple(self._items)

    def add(self, item: FileReport) -> None:
        self._items.append(item)

    def render(
        self,
        palette: Palette,
        *,
        verbose: bool = False,
        width: int | None = None,
        elapsed_s: float | None = None,
        out_dir: Path | None = None,
    ) -> list[str]:
        if not self._items:
            return [palette.muted("翻译未执行")]
        if width is None:
            try:
                width = int(shutil.get_terminal_size(FALLBACK_SIZE).columns)
            except (OSError, TypeError, ValueError):
                width = FALLBACK_SIZE[0]
        width = max(40, width)

        ok = sum(1 for item in self._items if item.status == "ok")
        partial = sum(1 for item in self._items if item.status == "partial")
        failed = sum(1 for item in self._items if item.status == "failed")
        interrupted = sum(
            1 for item in self._items if item.status == "interrupted"
        )
        overwrote = sum(1 for item in self._items if item.overwrote)
        segments = sum(item.segments for item in self._items)
        chars = sum(item.zh_chars for item in self._items)
        tokens = sum(item.tokens for item in self._items)
        elapsed = (
            max(0.0, float(elapsed_s))
            if elapsed_s is not None
            else sum(item.elapsed_s for item in self._items)
        )

        title = "翻译中断" if interrupted else "翻译完成"
        lines = [
            palette.bold(
                f"{title} · {len(self._items)} 个文件 · "
                f"{segments} 段 · {chars:,} 字"
            ),
            palette.muted(
                f"成功 {ok} · 部分失败 {partial} · 失败 {failed}"
                f" · 中断 {interrupted} · 覆盖 {overwrote}"
            ),
            palette.muted(
                f"耗时 {format_duration(elapsed)} · "
                f"{compute_rate(tokens, elapsed):.2f} tok/s"
            ),
        ]

        show_all = verbose or len(self._items) <= FULL_REPORT_LIMIT
        hidden = 0
        for item in self._items:
            if not show_all and item.status == "ok":
                hidden += 1
                continue
            lines.append(self._file_line(item, palette, width))
        if hidden:
            lines.append(
                palette.muted(
                    f"· 仅显示异常文件；-v 查看全部 {len(self._items)} 个"
                )
            )

        if any(item.failures or item.error or item.interrupted for item in self._items):
            lines.append("")
            lines.append(palette.warning("失败明细（失败段保留原文）"))
            for item in self._items:
                if not (item.failures or item.error or item.interrupted):
                    continue
                lines.append(
                    f"  {self._mark(item)} [{item.index}/{item.total}] {item.label}"
                )
                for message in item.failures[:FAILURE_DETAIL_LIMIT]:
                    lines.append(f"    {message}")
                if len(item.failures) > FAILURE_DETAIL_LIMIT:
                    lines.append(
                        f"    … 另有 {len(item.failures) - FAILURE_DETAIL_LIMIT} 处"
                    )
                if item.error:
                    lines.append(f"    {item.error}")
                if item.interrupted:
                    lines.append("    用户中断，当前文件未写出")

        lines.append("")
        lines.append(
            palette.muted(
                f"输出：{out_dir}" if out_dir is not None else "输出：同源目录"
            )
        )
        return lines

    def _file_line(
        self, item: FileReport, palette: Palette, width: int
    ) -> str:
        mark = self._mark(item)
        suffix: list[str] = []
        if item.failures:
            suffix.append(f"{len(item.failures)} 段失败")
        if item.status == "failed":
            suffix.append("失败")
        if item.status == "interrupted":
            suffix.append("中断")
        if item.overwrote:
            suffix.append("覆盖")
        suffix_text = ("  " + " · ".join(suffix)) if suffix else ""
        # 先给状态后缀留位，避免长文件名把“N 段失败/覆盖”裁掉。
        label_width = max(12, width - 32 - _display_width(suffix_text))
        label = _clip_display(item.label, label_width)
        segments = "—" if item.status == "failed" else str(item.segments)
        plain = (
            f"  {mark} [{item.index}/{item.total}] "
            f"{_pad_display(label, label_width)}  "
            f"{segments:>4} 段  {format_duration(item.elapsed_s):>9}"
        )
        plain += suffix_text
        if _display_width(plain) > width:
            return _clip_display(plain, width)
        colored_mark = self._colored_mark(item, palette)
        return f"  {colored_mark}" + plain[len(f"  {mark}"):]

    @staticmethod
    def _mark(item: FileReport) -> str:
        return {
            "ok": "✓",
            "partial": "⚠",
            "failed": "✗",
            "interrupted": "…",
        }[item.status]

    def _colored_mark(self, item: FileReport, palette: Palette) -> str:
        if item.status == "ok":
            return palette.success("✓")
        if item.status == "partial":
            return palette.warning("⚠")
        if item.status == "interrupted":
            return palette.warning("…")
        return palette.error("✗")
