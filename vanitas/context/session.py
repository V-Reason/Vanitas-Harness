"""会话状态：历史、系统提示、任务模板、上下文预算调用、存档/恢复。

本模块只负责消息容器与存档；预算与折叠算法分别由 ..metrics 与 .collapse 提供。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .. import prompts
from ..errors import ConfigError
from ..metrics import MESSAGE_OVERHEAD, context_budget, estimate_tokens
from .collapse import (
    COLLAPSED_OBSERVATION_MARKER,
    collapse_intermediate_observations as _collapse_messages,
)

__all__ = ["Session", "context_budget", "default_save_path", "estimate_tokens"]


@dataclass
class Session:
    model_id: str = ""
    profile: str = ""
    system: str = ""
    task: str = ""
    messages: list = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )
    is_dirty: bool = False
    _cached_tokens: int | None = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------ #
    # 消息
    # ------------------------------------------------------------------ #

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})
        self.is_dirty = True
        self._cached_tokens = None

    def add_assistant(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})
        self.is_dirty = True
        self._cached_tokens = None

    def pop_last_user(self) -> None:
        if self.messages and self.messages[-1]["role"] == "user":
            self.messages.pop()
            self.is_dirty = True
            self._cached_tokens = None

    def clear_messages(self) -> None:
        self.messages.clear()
        self.is_dirty = False
        self._cached_tokens = None

    def pop_last_turn(self) -> tuple[int, int]:
        """原子回退整轮人类交互。

        持续弹出末尾消息，跳过中间 assistant 动作与 <tool_result>/[系统反馈]
        工具回显，直到弹出该轮最初的人类真实输入。返回 (条数, 释放 tokens)。
        """
        if not self.messages:
            return 0, 0
        initial_tokens = self.message_tokens()
        popped = 0
        while self.messages:
            message = self.messages.pop()
            popped += 1
            role = message.get("role")
            content = str(message.get("content", "")).strip()
            if role == "user":
                is_tool_feedback = (
                    content.startswith("<tool_result")
                    or content.startswith("[系统反馈")
                    or content.startswith("【系统反馈")
                )
                if not is_tool_feedback:
                    break
        if popped:
            self._cached_tokens = None
        released = max(0, initial_tokens - self.message_tokens())
        if popped:
            self.is_dirty = True
        return popped, released

    def api_messages(self) -> list[dict]:
        if self.system.strip():
            return [{"role": "system", "content": self.system}, *self.messages]
        return list(self.messages)

    def message_tokens(self) -> int:
        if self._cached_tokens is None:
            self._cached_tokens = sum(
                estimate_tokens(str(item.get("content", ""))) + MESSAGE_OVERHEAD
                for item in self.messages
            )
        return self._cached_tokens

    def collapse_intermediate_observations(
        self,
        budget: int,
        *,
        threshold: float = 0.65,
        keep_rounds: int = 2,
        min_chars: int = 1000,
    ) -> int:
        """折叠较早的超长工具 Observation；算法本体在 .collapse。"""
        collapsed = _collapse_messages(
            self.messages,
            budget,
            threshold=threshold,
            keep_rounds=keep_rounds,
            min_chars=min_chars,
            token_counter=lambda msgs: sum(
                estimate_tokens(str(item.get("content", ""))) + MESSAGE_OVERHEAD
                for item in msgs
            ),
        )
        if collapsed:
            self.is_dirty = True
            self._cached_tokens = None
        return collapsed

    def trim(self, budget: int) -> int:
        """超预算时丢弃最早的用户/助手对；返回丢弃的轮数。"""
        dropped = 0
        while self.messages and self.message_tokens() > budget:
            if len(self.messages) >= 2 and self.messages[0].get("role") == "user":
                del self.messages[:2]
                dropped += 1
                self._cached_tokens = None
            else:
                if self.messages:
                    self.messages.pop(0)
                    self._cached_tokens = None
                break
        if dropped:
            self.is_dirty = True
        return dropped

    # ------------------------------------------------------------------ #
    # 任务模板
    # ------------------------------------------------------------------ #

    def build_user_message(
        self,
        text: str,
        prompt_dir: Path,
        model_alias: str,
        *,
        cwd: str | None = None,
    ) -> str:
        if not self.task:
            # 无任务模板时仍注入秒级运行时信息；有模板时由模板的 {date} 负责。
            return prompts.render("当前时间：{date}\n\n{input}", {"input": text})
        # bat 会先 cd 到仓库根以保证 python -m vanitas 可导入；原 CWD 由
        # VANITAS_ORIG_CWD 透传，优先于调用方传入值和进程 CWD。
        cwd = os.environ.get("VANITAS_ORIG_CWD") or cwd or os.getcwd()
        # Agent 首轮使用完整模板；已有历史后只带 prompts/agent_continue.md
        # 的短提醒。完整契约仍在首轮消息与历史里。
        template = (
            prompts.load_template(prompt_dir, "agent_continue")
            if self.task == "agent" and self.messages
            else prompts.load_task(prompt_dir, self.task)
        )
        return prompts.render(
            template,
            {
                "input": text,
                "model": model_alias,
                "cwd": cwd,
            },
        )

    # ------------------------------------------------------------------ #
    # 存档
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "created_at": self.created_at,
            "model_id": self.model_id,
            "profile": self.profile,
            "system": self.system,
            "task": self.task,
            "messages": list(self.messages),
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path).expanduser()
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            raise ConfigError(f"会话保存失败：{target}；{exc.strerror or exc}") from exc
        self.is_dirty = False
        return target

    @classmethod
    def load(cls, path: str | Path) -> "Session":
        source = Path(path).expanduser()
        try:
            raw = source.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"会话读取失败：{source}；{exc.strerror or exc}") from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ConfigError(
                f"会话文件不是合法 JSON：{source}；第 {exc.lineno} 行：{exc.msg}"
            ) from exc
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ConfigError(f"会话文件版本不支持：{source}")
        messages = data.get("messages")
        if not isinstance(messages, list):
            raise ConfigError(f"会话文件缺少 messages 数组：{source}")
        cleaned: list[dict] = []
        for item in messages:
            if not isinstance(item, dict):
                raise ConfigError(f"会话消息必须是对象：{source}")
            role = item.get("role")
            content = item.get("content")
            if role not in {"system", "user", "assistant"} or not isinstance(content, str):
                raise ConfigError(f"会话消息缺少合法的 role/content：{source}")
            if role == "system":
                continue
            cleaned.append({"role": role, "content": content})
        return cls(
            model_id=str(data.get("model_id", "")),
            profile=str(data.get("profile", "")),
            system=str(data.get("system", "")),
            task=str(data.get("task", "")),
            messages=cleaned,
            created_at=str(data.get("created_at", datetime.now().isoformat(timespec="seconds"))),
        )


def default_save_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path.home() / ".vanitas" / "sessions" / f"{stamp}.json"
