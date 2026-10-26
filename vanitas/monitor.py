"""翻译实况看板与终态统计（UI 层）。

`translate.py` 只发 `SegmentStat` 事件，本模块负责环形缓冲、ANSI 局部重绘
与最终汇总；业务层不反向依赖终端渲染细节。非 TTY/未启用着色时只保留
终态统计，避免污染管道与 CI 日志。
"""

from __future__ import annotations

import sys
import threading
import time
from collections import deque
from typing import TextIO

from .metrics import SegmentStat, compute_rate, count_cjk, format_duration
from .ui import Palette

PANEL_WIDTH = 52


def _clip(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)] + "…"


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
            self._write(text + "\n")

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
        separator = "─" * PANEL_WIDTH
        return [
            separator,
            self._palette.ok("翻译任务已完成"),
            self._palette.meta(f"统计段数：{self._segments} 段"),
            self._palette.meta(
                f"累计汉字：{self._zh_chars} 字（不含时间戳与排版符号）"
            ),
            self._palette.meta(f"总计耗时：{format_duration(elapsed)}"),
            self._palette.meta(
                f"平均速率：{compute_rate(self._tokens, elapsed):.2f} tok/s"
            ),
            self._palette.meta(
                f"吞吐性能：{compute_rate(self._zh_chars, elapsed):.2f} 汉字/秒"
            ),
            separator,
        ]

    def _rows(self) -> list[str]:
        return [
            f"[{index:02d}] {_clip(stat.source, 24)} → "
            f"{_clip(stat.translation, 30)} | "
            f"{stat.elapsed_s:.1f}s | {stat.tps:.1f} tok/s"
            for index, stat in self._records
        ]

    def _draw_locked(self) -> None:
        if not self._enabled:
            return
        rows = self._rows()
        height = max(self._drawn, len(rows))
        if self._drawn:
            self._stream.write(f"\x1b[{self._drawn}A")
        for index in range(height):
            self._stream.write("\r\x1b[2K")
            if index < len(rows):
                self._stream.write(rows[index])
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
