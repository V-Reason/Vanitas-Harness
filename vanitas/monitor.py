"""翻译实况看板与终态统计（UI 层）。

`translate.py` 只发 `SegmentStat` 事件，本模块负责环形缓冲、ANSI 局部重绘
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
from typing import TextIO

from .metrics import SegmentStat, compute_rate, count_cjk, format_duration
from .ui import Palette, strip_ansi

FALLBACK_SIZE = (80, 24)


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

    def reset(self) -> None:
        """开始一个新文件的统计：清面板、清环形缓冲并重置计时。"""
        with self._lock:
            self._clear_locked()
            self._records.clear()
            self._started = time.perf_counter()
            self._segments = 0
            self._tokens = 0
            self._zh_chars = 0
            self._size = None

    def clear(self) -> None:
        """只擦除看板、不输出统计；用于异常退出路径。"""
        with self._lock:
            self._clear_locked()

    @property
    def segments(self) -> int:
        with self._lock:
            return self._segments

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
        # 给完成行/摘要/提示留出空间，避免面板把终端顶出可视区。
        return max(1, min(int(capacity), int(lines) - 2))

    def _draw_locked(self) -> None:
        if not self._enabled:
            return
        columns, lines = self._terminal_size()
        if self._size is not None and self._size != (columns, lines):
            # 尺寸变化后旧行的实际折行数已不可靠，先清掉再按新宽度重画。
            self._clear_locked()
        self._size = (columns, lines)
        rows = self._rows()[-self._max_rows(lines):]
        if not rows:
            return
        if self._drawn:
            self._stream.write(f"\x1b[{self._drawn}A")
        height = max(self._drawn, len(rows))
        limit = max(1, columns - 1)
        for index in range(height):
            self._stream.write("\r\x1b[2K")
            if index < len(rows):
                self._stream.write(_clip_display(rows[index], limit))
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
