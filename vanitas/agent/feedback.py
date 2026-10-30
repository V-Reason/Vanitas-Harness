"""工具反馈的双端折叠、`<tool_result>` 封装与解析错误回灌消息。

本模块不做 I/O、不感知 UI。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..tools import ExecResult


def _clip_feedback(
    text: str, limit: int, head_lines: int = 20, tail_lines: int = 40
) -> str:
    """双端安全折叠（Head+Tail Folding）。

    当反馈文本超过 limit 时，保留前部的调用概况与尾部的关键报错堆栈，
    并在中间插入结构化折叠标记，杜绝 Context Window 击穿。
    """
    if len(text) <= limit:
        return text

    lines = text.splitlines(keepends=True)
    if len(lines) <= (head_lines + tail_lines):
        # 行数少但单行超长时按字符首尾切分
        head_len = max(10, int(limit * 0.3))
        tail_len = max(10, limit - head_len)
        omitted = max(0, len(text) - (head_len + tail_len))
        marker = f"\n[... 输出过长已截断：省略中间 {omitted} 字符 ...]\n"
        return text[:head_len] + marker + text[-tail_len:]

    # 多行输出按行切分首尾
    head = "".join(lines[:head_lines])
    tail = "".join(lines[-tail_lines:])
    if len(head) + len(tail) > limit:
        head_len = max(10, int(limit * 0.3))
        tail_len = max(10, limit - head_len)
        omitted_chars = max(0, len(text) - (head_len + tail_len))
        marker = f"\n[... 输出过长已截断：省略中间 {omitted_chars} 字符 ...]\n"
        return head[:head_len] + marker + tail[-tail_len:]

    omitted_lines = len(lines) - head_lines - tail_lines
    omitted_chars = len(text) - len(head) - len(tail)
    marker = (
        f"\n[... 输出过长已截断：省略中间 {omitted_lines} 行 "
        f"({omitted_chars} 字符) ...]\n"
    )
    return head + marker + tail


def _shell_detail(result: "ExecResult") -> str:
    pieces = [f"退出码 {result.exit_code}"]
    if result.stdout.strip():
        pieces.append(result.stdout.rstrip())
    if result.stderr.strip():
        pieces.append("stderr：" + result.stderr.rstrip())
    if len(pieces) == 1:
        pieces.append("无输出")
    return "\n".join(pieces)


def _parse_error_feedback(exc: ValueError) -> str:
    """把解析错误转成可回灌给模型的结构化纠偏 Observation。"""
    return (
        "<tool_result>\n"
        f"执行失败：动作解析错误（{exc}）。\n"
        "【纠偏要求】请严格按照协议仅输出 "
        '<tool_call>[{"action": "工具名", "parameters": {...}}]</tool_call>'
        " 格式，确保 JSON 语法合法且 Windows 路径使用正斜杠（/）"
        "或双反斜杠（\\\\）。\n"
        "</tool_result>"
    )


def wrap_tool_result(
    body: str,
    max_feedback_chars: int,
    head_lines: int = 20,
    tail_lines: int = 40,
) -> str:
    """把工具执行结果封装为 `<tool_result>...</tool_result>` 并做双端折叠。"""
    clipped = _clip_feedback(
        body,
        max_feedback_chars,
        head_lines=head_lines,
        tail_lines=tail_lines,
    )
    return f"<tool_result>\n{clipped}\n</tool_result>"


__all__ = [
    "_clip_feedback", "_parse_error_feedback", "_shell_detail", "wrap_tool_result",
]
