"""vanitas.context 公开门面：Session / 预算 / 折叠 / 度量兼容 re-export。

所有旧的 `from vanitas.session import Session` 通过 `from vanitas.context import Session` 保持可用。
"""
from __future__ import annotations

from ..metrics import MESSAGE_OVERHEAD, chat_max_tokens, context_budget, estimate_tokens
from .budget import ContextBudget
from .collapse import COLLAPSED_OBSERVATION_MARKER, collapse_intermediate_observations
from .session import Session, default_save_path

__all__ = [
    "Session",
    "ContextBudget",
    "COLLAPSED_OBSERVATION_MARKER",
    "collapse_intermediate_observations",
    "default_save_path",
    "context_budget",
    "chat_max_tokens",
    "estimate_tokens",
    "MESSAGE_OVERHEAD",
]
