"""较早工具观测的双端折叠（原 Session.collapse_intermediate_observations 的算法本体）。

本模块不依赖 Session 类型，接收 messages 列表并原地修改。
Session.collapse_intermediate_observations 改为薄转发，保证行为不变。
"""
from __future__ import annotations

from typing import Callable

COLLAPSED_OBSERVATION_MARKER = "<tool_result [collapsed]>"


def collapse_intermediate_observations(
    messages: list[dict],
    budget: int,
    *,
    threshold: float = 0.65,
    keep_rounds: int = 2,
    min_chars: int = 1000,
    token_counter: Callable[[list[dict]], int],
) -> int:
    """折叠较早的超长工具 Observation，保留动作签名与最近两轮。"""
    if budget <= 0 or threshold <= 0 or min_chars <= 0:
        return 0
    keep_messages = max(0, int(keep_rounds) * 2)
    if len(messages) <= max(keep_messages, 6):
        return 0
    if token_counter(messages) / budget <= threshold:
        return 0

    cutoff = max(0, len(messages) - keep_messages)
    collapsed = 0
    for index in range(cutoff):
        message = messages[index]
        if message.get("role") != "user":
            continue
        content = str(message.get("content", ""))
        if "<tool_result" not in content:
            continue
        if COLLAPSED_OBSERVATION_MARKER in content:
            continue
        if len(content) < min_chars:
            continue
        messages[index] = {
            **message,
            "content": (
                f"{COLLAPSED_OBSERVATION_MARKER}\n"
                f"[历史工具观测已折叠：原内容 {len(content)} 字符，"
                "执行动作与成败摘要保留；如仍需细节请重新调用工具]\n"
                "</tool_result>"
            ),
        }
        collapsed += 1
    return collapsed


__all__ = ["COLLAPSED_OBSERVATION_MARKER", "collapse_intermediate_observations"]
