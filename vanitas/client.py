"""高层客户端门面：暴露 `stream_chat` / `stream_completion` 业务 API。

`engine.py` 仍是传输层（HTTP、SSE、重试、超时、响应关闭）；
本模块只做命名与消息构造，不复制传输逻辑，也不偷偷改变重试语义。
"""

from __future__ import annotations

from typing import Callable, Iterator

from .engine import EngineClient


class LlamaClient(EngineClient):
    """业务层统一入口；`for_model()` 继承后返回本类实例。"""

    def stream_chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float = 0.2,
        timeout: float | None = None,
        on_progress: Callable[[dict], None] | None = None,
    ) -> Iterator[dict]:
        return self.chat_stream(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            on_progress=on_progress,
        )

    def stream_completion(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int,
        temperature: float = 0.2,
        timeout: float | None = None,
        on_progress: Callable[[dict], None] | None = None,
    ) -> Iterator[dict]:
        messages: list[dict] = []
        if system and system.strip():
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.stream_chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            on_progress=on_progress,
        )


Client = LlamaClient
__all__ = ["Client", "LlamaClient"]
