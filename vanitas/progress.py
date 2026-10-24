"""TTY 覆盖式单行进度；非 TTY 全程静默。

只依赖 ui（Palette），不 import config / repl / launcher：
等待进度是「渲染细节」，不该反过来知道业务语义。
非 TTY 时 update/enter/exit 全是空操作，done() 仍打印最终状态——
管道与 CI 里看不到动画，但能看到结论。
"""

from __future__ import annotations

import sys
import time

from .ui import Palette

_ERASE_LINE = "\r\x1b[2K"
_TRIM_TO_EOL = "\x1b[K"


class Spinner:
    """覆盖式单行进度。TTY 默认启用；enabled=False 时行为等同非 TTY。"""

    def __init__(
        self,
        label: str,
        palette: Palette,
        stream: object = None,
        interval_ms: int = 300,
        enabled: bool | None = None,
    ) -> None:
        self._label = label
        self._palette = palette
        self._stream = sys.stderr if stream is None else stream
        self._interval = max(0.05, interval_ms / 1000.0)
        self._enabled = self._detect_tty() if enabled is None else bool(enabled)
        self._started = 0.0
        self._last_draw = 0.0
        self._active = False
        self.completed = False

    def _detect_tty(self) -> bool:
        try:
            return bool(self._stream.isatty())
        except (AttributeError, ValueError, OSError):
            return False

    def __enter__(self) -> "Spinner":
        self._started = time.monotonic()
        self._last_draw = 0.0
        if self._enabled:
            self._active = True
            self._draw(force=True)
        return self

    def __exit__(self, *exc) -> bool:
        self._erase()
        return False  # 绝不吞异常

    def update(self, label: str | None = None) -> None:
        if label is not None:
            self._label = label
        if self._enabled and self._active:
            self._draw()

    def done(self, final_text: str) -> None:
        """擦除进度行并打印最终状态；之后 __exit__ 不再重复擦除。"""
        self._erase()
        self.completed = True
        print(self._palette.ok(final_text), file=self._stream)
        try:
            self._stream.flush()
        except (AttributeError, ValueError):
            pass

    def _draw(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_draw < self._interval:
            return
        self._last_draw = now
        elapsed = now - self._started
        text = self._palette.spin(f"{self._label} {elapsed:.1f}s")
        self._stream.write("\r" + text + _TRIM_TO_EOL)
        self._stream.flush()

    def _erase(self) -> None:
        if not self._enabled or not self._active:
            return
        self._stream.write(_ERASE_LINE)
        self._stream.flush()
        self._active = False
