"""纯文本流式渲染：增量上屏、代码块等宽缩进、围栏标记不上屏。

不使用 ANSI 光标控制：输出可被 tee / 重定向 / CI 原样抓取。
围栏行（``` 开头）必须先判定再输出，因此行首的 1–2 个反引号会短暂挂起，
其余字符一律立即写出并 flush。
"""

from __future__ import annotations

import sys
import time
from io import StringIO

FENCE = "```"
INDENT = "    "


class StreamRenderer:
    def __init__(self, stream=None) -> None:
        self._stream = stream if stream is not None else sys.stdout
        self._buffer = ""
        self._at_line_start = True
        self._in_code = False
        self._swallow = False
        self._chars = 0
        self._first_at: float | None = None
        self._started = time.perf_counter()
        self._collected: list[str] = []

    @property
    def text(self) -> str:
        return "".join(self._collected)

    def feed(self, text: str) -> None:
        for char in text:
            self._feed_char(char)

    def close(self) -> dict:
        if self._buffer:
            self._flush_line_buffer()
        if not self._at_line_start:
            self._write("\n")
        wall_ms = (time.perf_counter() - self._started) * 1000.0
        first_ms = (
            (self._first_at - self._started) * 1000.0 if self._first_at is not None else None
        )
        return {
            "chars": self._chars,
            "first_ms": round(first_ms, 1) if first_ms is not None else None,
            "wall_ms": round(wall_ms, 1),
        }

    # ------------------------------------------------------------------ #

    def _feed_char(self, char: str) -> None:
        self._chars += 1
        self._collected.append(char)
        if char == "\n":
            if self._buffer:
                self._flush_line_buffer()
            self._write("\n")
            self._buffer = ""
            self._at_line_start = True
            self._swallow = False
            return
        if self._swallow:
            return
        if not self._at_line_start:
            self._write(char)
            return

        self._buffer += char
        stripped = self._buffer.lstrip(" ")
        if stripped.startswith(FENCE):
            self._in_code = not self._in_code
            self._buffer = ""
            self._swallow = True
            self._at_line_start = False
            return
        if FENCE.startswith(stripped):
            # 可能是围栏的前缀，等下一个字符再判定
            return
        self._flush_line_buffer()

    def _flush_line_buffer(self) -> None:
        prefix = INDENT if self._in_code else ""
        self._write(prefix + self._buffer)
        self._buffer = ""
        self._at_line_start = False

    def _write(self, text: str) -> None:
        if not text:
            return
        if self._first_at is None:
            self._first_at = time.perf_counter()
        self._stream.write(text)
        self._stream.flush()


def format_text(text: str) -> str:
    """把带围栏的 Markdown 片段渲染为纯文本（批量输出/测试用）。"""
    buffer = StringIO()
    renderer = StreamRenderer(buffer)
    renderer.feed(text)
    renderer.close()
    return buffer.getvalue()
