"""停滞检测：连续 / 震荡 / 重复 / 批次重复 / 批次震荡。

本模块只提供纯函数；deque 窗口与状态仍在 AgentLoop 上，
批次 5 会替换为策略化检测器列表。
"""
from __future__ import annotations

from collections import Counter


def evaluate_stagnation(
    recent: list[tuple[str, str]],
    rounds: list[tuple[str, ...]],
    stagnation_limit: int,
) -> str | None:
    """返回停滞原因（"连续" / "震荡" / "重复" / "批次重复" / "批次震荡"）或 None。"""
    if (
        len(recent) >= stagnation_limit
        and len(set(recent[-stagnation_limit:])) == 1
    ):
        return "连续"
    if len(recent) >= 4:
        if (
            recent[-1] == recent[-3]
            and recent[-2] == recent[-4]
            and recent[-1] != recent[-2]
        ):
            return "震荡"
    if len(recent) >= 6:
        if (
            recent[-1] == recent[-4]
            and recent[-2] == recent[-5]
            and recent[-3] == recent[-6]
        ):
            return "震荡"
    window = recent[-6:]
    if window:
        if max(Counter(window).values()) >= 3:
            return "重复"
    if len(rounds) >= 3 and len(set(rounds[-3:])) == 1:
        return "批次重复"
    if len(rounds) >= 4:
        if (
            rounds[-1] == rounds[-3]
            and rounds[-2] == rounds[-4]
            and rounds[-1] != rounds[-2]
        ):
            return "批次震荡"
    if len(rounds) >= 6:
        if (
            rounds[-1] == rounds[-4]
            and rounds[-2] == rounds[-5]
            and rounds[-3] == rounds[-6]
        ):
            return "批次震荡"
    return None


__all__ = ["evaluate_stagnation"]
