"""单轮对话管线：预算、流式渲染、指标汇总与中断语义。

本模块不碰命令表、不打印错误、不决定模型生命周期：
- `Repl` 负责 UI/命令/模型切换；
- `ChatRunner` 只回答「这一轮发生了什么」，把 `TurnResult` 交回。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .client import LlamaClient
from .config import AppConfig, ModelSpec, ProfileSpec
from .errors import EngineError
from .metrics import chat_max_tokens, context_budget
from .progress import Spinner
from .render import StreamRenderer
from .session import Session
from .ui import Palette


@dataclass(frozen=True)
class TurnResult:
    text: str
    tokens: int
    elapsed_s: float
    tps: float
    ttft_ms: float | None
    finish_reason: str | None
    interrupted: bool = False
    reasoning_chars: int = 0
    dropped_turns: int = 0


class _DiscardStream:
    """Agent 闭环不把原始 JSON 写屏；仍需一个可写流让 StreamRenderer 正常工作。"""

    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        return None


class ChatRunner:
    """把一次用户输入变成一次流式请求 + 一条历史记录。"""

    def __init__(
        self,
        config: AppConfig,
        client: LlamaClient | None,
        session: Session,
        model: ModelSpec,
        profile: ProfileSpec,
        palette: Palette,
        *,
        verbose: bool = False,
    ) -> None:
        self.config = config
        self.client = client
        self.session = session
        self.model = model
        self.profile = profile
        self.palette = palette
        self.verbose = verbose

    def run(
        self,
        text: str,
        *,
        on_dropped: Callable[[int], None] | None = None,
        display: bool = True,
    ) -> TurnResult:
        if self.client is None:
            raise EngineError("服务端没有运行：先运行 vanitas serve <id>")
        budget = context_budget(
            self.profile.ctx,
            self.config.settings.repl.max_context_tokens,
            self.config.settings.repl.context_reserve_tokens,
        )
        user_message = self.session.build_user_message(
            text,
            self.prompt_dir,
            self.model.alias,
            cwd=os.getcwd(),
        )
        self.session.add_user(user_message)
        return self._run_turn(
            budget,
            on_dropped=on_dropped,
            display=display,
        )

    def run_followup(
        self,
        feedback: str,
        *,
        on_dropped: Callable[[int], None] | None = None,
        display: bool = True,
    ) -> TurnResult:
        """把系统反馈直接作为 user 消息追加，不重复套任务模板。"""
        if self.client is None:
            raise EngineError("服务端没有运行：先运行 vanitas serve <id>")
        budget = context_budget(
            self.profile.ctx,
            self.config.settings.repl.max_context_tokens,
            self.config.settings.repl.context_reserve_tokens,
        )
        self.session.add_user(feedback)
        return self._run_turn(
            budget,
            on_dropped=on_dropped,
            display=display,
        )

    def _run_turn(
        self,
        budget: int,
        *,
        on_dropped: Callable[[int], None] | None,
        display: bool,
    ) -> TurnResult:
        if self.client is None:
            raise EngineError("服务端没有运行：先运行 vanitas serve <id>")
        dropped = self.session.trim(budget)
        if dropped and on_dropped is not None:
            on_dropped(dropped)

        max_tokens = chat_max_tokens(
            self.profile.ctx,
            self.session.message_tokens(),
            self.config.settings.repl.context_reserve_tokens,
            self.config.settings.repl.min_output_tokens,
        )
        renderer = StreamRenderer(stream=None if display else _DiscardStream())
        usage: dict | None = None
        done: dict | None = None
        reasoning_chars = 0
        try:
            for event in self._stream_events(
                self.session.api_messages(), max_tokens
            ):
                if event["type"] == "delta":
                    renderer.feed(event["content"])
                elif event["type"] == "reasoning":
                    reasoning_chars += len(event["content"])
                elif event["type"] == "usage":
                    raw = event.get("usage")
                    if isinstance(raw, dict):
                        usage = raw
                elif event["type"] == "done":
                    done = event
                    raw = event.get("usage")
                    if isinstance(raw, dict) and raw:
                        usage = raw
        except EngineError:
            self.session.pop_last_user()
            raise
        except KeyboardInterrupt:
            renderer.close()
            self.session.pop_last_user()
            return TurnResult(
                text="",
                tokens=0,
                elapsed_s=0.0,
                tps=0.0,
                ttft_ms=None,
                finish_reason=None,
                interrupted=True,
                reasoning_chars=reasoning_chars,
            )

        meta = renderer.close()
        answer = renderer.text
        if answer.strip():
            self.session.add_assistant(answer)

        payload = done or {}
        tokens = int(payload.get("tokens") or 0)
        if tokens <= 0:
            tokens = int(len(answer) / 3)
        elapsed_s = float(meta["wall_ms"]) / 1000.0
        tps = float(payload.get("tps") or 0.0)
        if tps <= 0 and elapsed_s > 0:
            tps = tokens / elapsed_s
        ttft_ms = payload.get("ttft_ms")
        if ttft_ms is None:
            ttft_ms = meta["first_ms"]
        finish_reason = payload.get("finish_reason")
        return TurnResult(
            text=answer,
            tokens=tokens,
            elapsed_s=elapsed_s,
            tps=tps,
            ttft_ms=float(ttft_ms) if ttft_ms is not None else None,
            finish_reason=str(finish_reason) if finish_reason else None,
            interrupted=False,
            reasoning_chars=reasoning_chars,
            dropped_turns=dropped,
        )

    @property
    def prompt_dir(self) -> Path:
        return self.config.config_dir / "prompts"

    def _stream_events(self, messages: list[dict], max_tokens: int):
        """首字前显示覆盖式进度，首字到达立刻擦除，避免等待行与 stdout 混屏。"""
        spin = Spinner("正在生成", self.palette)
        spin.__enter__()
        first = True
        generator = None
        try:
            generator = self.client.stream_chat(
                messages, max_tokens=max_tokens, temperature=0.2
            )
            for event in generator:
                if first and event["type"] == "delta":
                    first = False
                    spin.__exit__(None, None, None)
                yield event
        finally:
            # 生成器被提前关闭（Ctrl+C / 上层异常）时，显式释放底层响应。
            if generator is not None:
                close = getattr(generator, "close", None)
                if close is not None:
                    close()
            spin.__exit__(None, None, None)
