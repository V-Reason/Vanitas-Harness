"""OpenAI 兼容客户端：连接/读超时、指数退避 + 抖动、SSE 解析。

只用 /v1/chat/completions、/v1/models、/health 三个端点。
必须绕开系统代理：HTTP_PROXY 会把 127.0.0.1 也劫持走代理，表现为「服务未启动」。
"""

from __future__ import annotations

import http.client
import json
import random
import time
import urllib.error
import urllib.request
from typing import Iterator

from .config import AppConfig, ModelSpec, RequestSettings
from .errors import EngineError

DONE_TYPES = frozenset({"done"})


def _build_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _error_body(exc: urllib.error.HTTPError) -> str:
    try:
        raw = exc.read(400)
    except OSError:
        return ""
    finally:
        try:
            exc.close()
        except OSError:
            pass
    return raw.decode("utf-8", "replace").strip()


class EngineClient:
    def __init__(
        self,
        settings: RequestSettings,
        host: str,
        port: int,
        template_kwargs: dict | None = None,
    ) -> None:
        self._settings = settings
        self._host = host
        self._port = port
        self._template_kwargs = dict(template_kwargs or {})
        self._opener = _build_opener()

    @classmethod
    def for_model(
        cls, config: AppConfig, model: ModelSpec, port: int | None = None
    ) -> "EngineClient":
        return cls(
            settings=config.settings.request,
            host=config.settings.host,
            port=port if port is not None else config.effective_port(model),
            template_kwargs=model.template_kwargs,
        )

    @property
    def endpoint(self) -> str:
        return f"http://{self._host}:{self._port}"

    # ------------------------------------------------------------------ #
    # 传输层
    # ------------------------------------------------------------------ #

    def _url(self, path: str) -> str:
        return f"{self.endpoint}{path}"

    def _sleep_backoff(self, attempt: int) -> None:
        base = self._settings.backoff_base_ms / 1000.0
        jitter = self._settings.backoff_jitter_ms / 1000.0
        delay = base * (2**attempt) + random.uniform(0.0, jitter)
        if delay > 0:
            time.sleep(delay)

    def _open(
        self,
        request: urllib.request.Request,
        timeout: float,
        retries: int | None = None,
    ) -> object:
        """打开请求；仅对连接错误、超时与 5xx 重试，4xx 立即失败。"""
        attempts = (self._settings.max_retries if retries is None else retries) + 1
        attempts = max(1, attempts)
        last_failure: EngineError | None = None
        for attempt in range(attempts):
            try:
                return self._opener.open(request, timeout=timeout)
            except urllib.error.HTTPError as exc:
                detail = _error_body(exc)
                if exc.code < 500:
                    raise EngineError(
                        f"{request.full_url} 返回 HTTP {exc.code}"
                        + (f"：{detail}" if detail else ""),
                        retryable=False,
                    ) from exc
                last_failure = EngineError(
                    f"{request.full_url} 返回 HTTP {exc.code}"
                    + (f"：{detail}" if detail else "")
                )
            except urllib.error.URLError as exc:
                last_failure = EngineError(
                    f"无法连接 {request.full_url}：{exc.reason}（服务端未启动？）"
                )
            except (TimeoutError, OSError) as exc:
                last_failure = EngineError(
                    f"请求 {request.full_url} 超时或中断（{timeout}s）：{exc}"
                )
            if attempt + 1 >= attempts:
                break
            self._sleep_backoff(attempt)
        assert last_failure is not None
        raise last_failure

    def _get_json(self, path: str, timeout: float, retries: int | None = None) -> dict:
        request = urllib.request.Request(self._url(path), method="GET")
        response = self._open(request, timeout, retries=retries)
        try:
            raw = response.read()  # type: ignore[attr-defined]
        except (OSError, http.client.HTTPException) as exc:
            raise EngineError(f"读取 {path} 响应失败：{exc}") from exc
        finally:
            try:
                response.close()  # type: ignore[attr-defined]
            except OSError:
                pass
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EngineError(
                f"{path} 返回非 JSON（前 200 字节：{raw[:200]!r}）：{exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise EngineError(f"{path} 返回的顶层不是 JSON 对象")
        return payload

    # ------------------------------------------------------------------ #
    # 端点
    # ------------------------------------------------------------------ #

    def health(self, timeout: float | None = None) -> bool:
        """加载期 503 属于正常中间态，这里只回答「现在能不能用」。"""
        request = urllib.request.Request(self._url("/health"), method="GET")
        try:
            response = self._open(request, timeout or 3.0, retries=0)
        except EngineError:
            return False
        try:
            raw = response.read()  # type: ignore[attr-defined]
        except (OSError, http.client.HTTPException):
            return False
        finally:
            try:
                response.close()  # type: ignore[attr-defined]
            except OSError:
                pass
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return False
        return isinstance(payload, dict) and payload.get("status") == "ok"

    def model_ids(self) -> list[str]:
        payload = self._get_json("/v1/models", self._settings.timeout_sec)
        items = payload.get("data") or []
        return [str(item.get("id", "")) for item in items if isinstance(item, dict)]

    def chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float = 0.2,
        response_format: dict | None = None,
        timeout: float | None = None,
    ) -> dict:
        payload = self._build_payload(messages, max_tokens, temperature, response_format)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self._url("/v1/chat/completions"),
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        response = self._open(request, timeout or self._settings.timeout_sec)
        try:
            raw = response.read()  # type: ignore[attr-defined]
        except (OSError, http.client.HTTPException) as exc:
            raise EngineError(f"读取对话响应失败：{exc}") from exc
        finally:
            try:
                response.close()  # type: ignore[attr-defined]
            except OSError:
                pass
        try:
            result = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EngineError(
                f"/v1/chat/completions 返回非 JSON（前 200 字节：{raw[:200]!r}）：{exc}"
            ) from exc
        if not isinstance(result, dict) or not result.get("choices"):
            raise EngineError("/v1/chat/completions 返回体缺少 choices")
        return result

    def chat_stream(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float = 0.2,
        timeout: float | None = None,
    ) -> Iterator[dict]:
        """产出 delta/reasoning/usage/done 事件；首字前失败可重试，首字后失败直接报错。"""
        payload = self._build_payload(messages, max_tokens, temperature, None)
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self._url("/v1/chat/completions"),
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        read_timeout = timeout or self._settings.timeout_sec
        attempts = max(1, self._settings.max_retries + 1)
        last_failure: EngineError | None = None
        for attempt in range(attempts):
            emitted = False
            try:
                response = self._open(request, read_timeout, retries=0)
                try:
                    for event in self._iter_sse(response, read_timeout):
                        if event["type"] == "delta":
                            emitted = True
                        yield event
                finally:
                    try:
                        response.close()  # type: ignore[attr-defined]
                    except OSError:
                        pass
                return
            except EngineError as exc:
                last_failure = exc
                if emitted or not exc.retryable:
                    raise
                if attempt + 1 >= attempts:
                    break
                self._sleep_backoff(attempt)
        assert last_failure is not None
        raise last_failure

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #

    def _build_payload(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        response_format: dict | None,
    ) -> dict:
        payload: dict = {
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
        if response_format is not None:
            payload["response_format"] = response_format
        if self._template_kwargs:
            payload["chat_template_kwargs"] = self._template_kwargs
        return payload

    def _iter_sse(self, response: object, read_timeout: float) -> Iterator[dict]:
        usage: dict | None = None
        try:
            for raw_line in response:  # type: ignore[union-attr]
                line = raw_line.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if not isinstance(chunk, dict):
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content") or ""
                reasoning = delta.get("reasoning_content") or ""
                if content:
                    yield {"type": "delta", "content": content}
                elif reasoning:
                    yield {"type": "reasoning", "content": reasoning}
        except (OSError, http.client.HTTPException) as exc:
            raise EngineError(
                f"流式读取中断（超时 {read_timeout}s）：{exc}"
            ) from exc
        yield {"type": "done", "usage": usage}
