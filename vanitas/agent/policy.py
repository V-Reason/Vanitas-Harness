"""审批策略：判断一批动作是否需要人工确认。"""
from __future__ import annotations

from .actions import AgentAction


def _needs_confirmation(
    actions: list[AgentAction], confirm_actions: frozenset[str]
) -> bool:
    # 审批策略来自 settings.agent；批量里有一个确认动作就只问一次。
    return any(action.name in confirm_actions for action in actions)
