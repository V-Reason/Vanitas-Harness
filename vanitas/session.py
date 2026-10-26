"""会话状态（范式笔记 D5）：历史、系统提示、任务模板、上下文预算、存档/恢复。

预算按「档位 ctx 与 settings.repl.max_context_tokens 取小，再留输出余量」计算，
不靠猜；超预算时从最早的一轮对话开始丢，并让调用方告知用户裁剪了多少轮。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from . import prompts
from .errors import ConfigError
from .metrics import MESSAGE_OVERHEAD, context_budget, estimate_tokens

# 预算与估算集中在 metrics；此处保留同名导入，兼容旧的
# `from vanitas.session import estimate_tokens` 用法。
__all__ = ["Session", "context_budget", "default_save_path", "estimate_tokens"]

# Agent 的首轮消息已经带入完整 task_agent 契约；后续用户输入只需短提醒，
# 否则每轮重复 1,450 tokens 的模板会在 4K 档位下很快触发上下文裁剪。
_AGENT_CONTINUATION = (
    "【Agent 继续】沿用当前工具契约；需要工具时只输出 JSON，普通文本表示完成。\n"
    "当前工作目录：{cwd}\n任务：\n{input}"
)


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

    # ------------------------------------------------------------------ #
    # 消息
    # ------------------------------------------------------------------ #

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def pop_last_user(self) -> None:
        if self.messages and self.messages[-1]["role"] == "user":
            self.messages.pop()

    def clear_messages(self) -> None:
        self.messages.clear()

    def api_messages(self) -> list[dict]:
        if self.system.strip():
            return [{"role": "system", "content": self.system}, *self.messages]
        return list(self.messages)

    def message_tokens(self) -> int:
        return sum(
            estimate_tokens(str(item.get("content", ""))) + MESSAGE_OVERHEAD
            for item in self.messages
        )

    def trim(self, budget: int) -> int:
        """超预算时丢弃最早的用户/助手对；返回丢弃的轮数。"""
        dropped = 0
        while self.messages and self.message_tokens() > budget:
            if len(self.messages) >= 2 and self.messages[0].get("role") == "user":
                del self.messages[:2]
                dropped += 1
            else:
                if self.messages:
                    self.messages.pop(0)
                break
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
        # Agent 首轮使用完整模板；已有历史后只带短提醒。完整契约仍在首轮
        # 消息与历史里，不需要每轮重复 1,450 tokens。
        template = (
            _AGENT_CONTINUATION
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
