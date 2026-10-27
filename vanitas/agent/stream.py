"""Agent 最终答复的流式过滤：抑制 `<tool_call>` / `<function>` 块与裸 JSON。

字符级五态机；批次 6 会换成 chunk 级实现。
"""
from __future__ import annotations

import sys

from .parser import (
    FUNCTION_CLOSE,
    FUNCTION_OPEN,
    FUNCTION_RESULTS_CLOSE,
    FUNCTION_RESULTS_OPEN,
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
)


class _AgentAnswerStream:
    """把 Agent 的流式输出分成“动作块”和“最终自然语言”。

    初始阶段先看首个非空白字符：`{` / `[` 进入 JSON 抑制，`<tool_call>`
    进入标签抑制。进入自然语言后，行首的 `{` / `[` 仍按旧兼容处理；
    `<tool_call>` 可在任意位置识别，且允许标签跨流式 chunk。
    """

    def __init__(self) -> None:
        self._target = sys.stdout
        self._buffer = ""
        self._suppressed = ""
        self._suppress_close = TOOL_CALL_CLOSE
        self._restore: list[str] = []
        self._mode = "initial"  # initial | text | probe_tag | suppress_tag | suppress_json
        self._line_start = False
        self._wrote = False

    @property
    def emitted(self) -> bool:
        return self._wrote

    @property
    def suppressed(self) -> bool:
        return bool(self._restore) or self._mode in {
            "probe_tag",
            "suppress_tag",
            "suppress_json",
        }

    def write(self, text: str) -> int:
        if (
            self._mode == "text"
            and not self._line_start
            and "<" not in text
            and "\n" not in text
            and "\r" not in text
            and text
        ):
            # 快速路径：行中纯文本，无标签起点、无换行；换行会改变行首
            # 状态并可能让后续 `{`/`[` 进入抑制，必须走慢速路径。
            self._wrote = True
            self._target.write(text)
            self._target.flush()
            self._line_start = text.endswith(("\n", "\r"))
            return len(text)
        for char in text:
            self._feed(char)
        return len(text)

    def _begin_suppression(self, close_tag: str) -> None:
        self._suppress_close = close_tag
        self._suppressed = self._buffer
        self._buffer = ""
        self._line_start = False
        self._mode = "suppress_tag"

    def _try_start_suppression(self, stripped: str) -> bool:
        """识别 <tool_call>、<function …>、<function_results…> 三种开标签。

        返回 True 表示已进入抑制或仍需等待后续字符；False 表示普通文本。
        """
        lowered = stripped.lower()
        if lowered.startswith(FUNCTION_RESULTS_OPEN):
            if (
                len(lowered) == len(FUNCTION_RESULTS_OPEN)
                or lowered[len(FUNCTION_RESULTS_OPEN)] in " \t\r\n>"
            ):
                self._begin_suppression(FUNCTION_RESULTS_CLOSE)
                return True
            return False
        if FUNCTION_RESULTS_OPEN.startswith(lowered):
            return True
        if lowered.startswith(FUNCTION_OPEN):
            if (
                len(lowered) == len(FUNCTION_OPEN)
                or lowered[len(FUNCTION_OPEN)] in " \t\r\n>"
            ):
                self._begin_suppression(FUNCTION_CLOSE)
                return True
            return False
        if FUNCTION_OPEN.startswith(lowered):
            return True
        if TOOL_CALL_OPEN.startswith(lowered):
            if lowered == TOOL_CALL_OPEN:
                self._begin_suppression(TOOL_CALL_CLOSE)
            return True
        return False

    def _feed(self, char: str) -> None:
        if self._mode == "suppress_tag":
            self._suppressed += char
            if self._suppressed.lower().endswith(self._suppress_close):
                self._restore.append(self._suppressed)
                self._suppressed = ""
                self._mode = "text"
                self._line_start = False
            return
        if self._mode == "suppress_json":
            self._suppressed += char
            return
        if self._mode in {"initial", "probe_tag"}:
            self._probe(char)
            return
        if self._mode != "text":
            self._send(char)
            return
        if self._line_start:
            self._buffer += char
            if char in " \t":
                return
            stripped = self._buffer.lstrip(" \t")
            if stripped.startswith(("{", "[")):
                self._enter_json()
                return
            if stripped == "<":
                self._mode = "probe_tag"
                return
            if self._try_start_suppression(stripped):
                return
            pending = self._buffer
            self._buffer = ""
            self._line_start = pending.endswith(("\n", "\r"))
            self._send(pending)
            return

        if char == "<":
            self._mode = "probe_tag"
            self._buffer = char
            return
        if char in "\r\n":
            self._send(char)
            self._line_start = True
            return
        self._send(char)

    def _probe(self, char: str) -> None:
        self._buffer += char
        stripped = self._buffer.lstrip(" \t")
        if not stripped:
            return
        if stripped.startswith(("{", "[")):
            self._enter_json()
            return
        if self._try_start_suppression(stripped):
            return
        pending = self._buffer
        self._buffer = ""
        self._mode = "text"
        self._line_start = pending.endswith(("\n", "\r"))
        self._send(pending)

    def _enter_json(self) -> None:
        self._suppressed = self._buffer
        self._buffer = ""
        self._mode = "suppress_json"

    def flush(self) -> None:
        self._target.flush()

    def flush_suppressed(self) -> None:
        parts = list(self._restore)
        if self._mode in {"suppress_tag", "suppress_json"}:
            parts.append(self._suppressed)
        elif self._mode == "probe_tag":
            parts.append(self._buffer)
        if not parts:
            return
        pending = "".join(parts)
        self._restore = []
        self._suppressed = ""
        self._buffer = ""
        self._mode = "text"
        self._line_start = pending.endswith(("\n", "\r"))
        self._send(pending)

    def _send(self, text: str) -> None:
        if not text:
            return
        self._wrote = True
        self._target.write(text)
        self._target.flush()
