"""上下文预算的唯一出口：历史预算、折叠、裁剪、单轮输出上限。

本模块不 import Session；对 Session 的依赖通过类型字符串 + TYPE_CHECKING 保持轻量。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..config import AppConfig, ProfileSpec
from ..metrics import chat_max_tokens, context_budget

if TYPE_CHECKING:
    from .session import Session


@dataclass(frozen=True)
class ContextBudget:
    """一次请求所需的上下文预算快照。"""

    ctx: int
    max_context_tokens: int
    reserve: int
    min_output: int
    collapse_ratio: float
    collapse_keep_rounds: int
    collapse_min_chars: int

    @classmethod
    def from_config(
        cls, config: AppConfig, profile: ProfileSpec
    ) -> "ContextBudget":
        repl = config.settings.repl
        return cls(
            ctx=profile.ctx,
            max_context_tokens=repl.max_context_tokens,
            reserve=repl.context_reserve_tokens,
            min_output=repl.min_output_tokens,
            collapse_ratio=repl.context_collapse_ratio,
            collapse_keep_rounds=repl.context_collapse_keep_rounds,
            collapse_min_chars=repl.context_collapse_min_chars,
        )

    def history_tokens(self) -> int:
        """历史消息允许占用的 token 上限。"""
        return context_budget(self.ctx, self.max_context_tokens, self.reserve)

    def collapse(self, session: "Session") -> int:
        """折叠较早的超长工具观测；返回折叠条数。"""
        return session.collapse_intermediate_observations(
            self.history_tokens(),
            threshold=self.collapse_ratio,
            keep_rounds=self.collapse_keep_rounds,
            min_chars=self.collapse_min_chars,
        )

    def trim(self, session: "Session") -> int:
        """超预算时丢弃最早轮次；返回丢弃轮数。"""
        return session.trim(self.history_tokens())

    def max_output(self, session: "Session") -> int:
        """单轮生成预算。"""
        return chat_max_tokens(
            self.ctx,
            session.message_tokens(),
            self.reserve,
            self.min_output,
        )


__all__ = ["ContextBudget"]
