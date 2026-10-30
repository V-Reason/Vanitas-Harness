"""失败动作签名的幂等集合：跨批次阻断重复提交，成功同签名即移除。"""

from __future__ import annotations

from typing import Iterable

from .actions import AgentAction, _action_signature


class FailedSignatures:
    """失败动作签名集合。

    - record_failure(signature, reason)：写入失败原因；
    - record_success(signature)：同签名成功时移除；
    - block(actions)：返回本批中已失败的动作及其历史失败原因；
    - clear()：/undo 或 run 重置时清空。
    """

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], str] = {}

    def record_failure(self, signature: tuple[str, str], reason: str) -> None:
        self._entries[signature] = reason

    def record_success(self, signature: tuple[str, str]) -> None:
        self._entries.pop(signature, None)

    def block(
        self, actions: Iterable[AgentAction]
    ) -> list[tuple[AgentAction, str]]:
        """返回本批中已存在失败记录的动作及原因列表；顺序与 actions 一致。"""
        blocked: list[tuple[AgentAction, str]] = []
        for action in actions:
            signature = _action_signature(action)
            reason = self._entries.get(signature)
            if reason is not None:
                blocked.append((action, reason))
        return blocked

    def clear(self) -> None:
        self._entries.clear()

    def __contains__(self, signature: object) -> bool:
        return signature in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def get(
        self, signature: tuple[str, str], default: str | None = None
    ) -> str | None:
        return self._entries.get(signature, default)


__all__ = ["FailedSignatures"]
