"""服务端物理截断：编号推断、请求体 stop 与鸭子客户端兼容。

全部离线，不启动 llama-server。
"""

from __future__ import annotations

import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vanitas.config import RequestSettings  # noqa: E402
from vanitas.engine import EngineClient  # noqa: E402
from vanitas.translate import (  # noqa: E402
    StreamingAbortDetector,
    TranslationEngine,
    TranslationOptions,
    next_window_id,
    window_stop_tokens,
)

PROMPT_DIR = ROOT / "config" / "prompts"
SETTINGS = RequestSettings(
    timeout_sec=10,
    max_retries=0,
    backoff_base_ms=0,
    backoff_jitter_ms=0,
)
OPTIONS = TranslationOptions(
    n_ctx=8192,
    context_window=5,
    context_summary_chars=200,
    min_output_tokens=64,
    output_ratio=2.5,
    prompt_safety_tokens=64,
    temperature=0.2,
    max_split_depth=3,
)

WINDOW = "\n".join(f"[#{index}] 台詞{index}" for index in range(7, 13))


class FakeResponse:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines
        self.closed = False

    def __iter__(self):
        return iter(self._lines)

    def close(self) -> None:
        self.closed = True


SSE = [
    'data: {"choices":[{"delta":{"content":"你好"},"finish_reason":null}]}\n'.encode(
        "utf-8"
    ),
    b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n',
    b"data: [DONE]\n",
]


class IdInferenceTests(unittest.TestCase):
    def test_window_uses_max_id_plus_one(self) -> None:
        self.assertEqual(next_window_id(WINDOW), 13)
        self.assertEqual(
            next_window_id("[#1] あ\n[#2] い\n[#3] う"), 4
        )

    def test_single_cue_and_sparse_ids(self) -> None:
        self.assertEqual(next_window_id("[#7] あ"), 8)
        self.assertEqual(next_window_id("[#3] あ\n[#9] い"), 10)

    def test_text_without_ids_has_none(self) -> None:
        self.assertIsNone(next_window_id("こんにちは"))
        self.assertIsNone(next_window_id(""))

    def test_stop_tokens_cover_protocol_variants(self) -> None:
        self.assertEqual(
            window_stop_tokens(WINDOW),
            ["[#13]", "#13]", "【#13】", "\n\n\n\n"],
        )
        self.assertEqual(window_stop_tokens("こんにちは"), ["\n\n\n\n"])


class PayloadTests(unittest.TestCase):
    def test_payload_includes_stop(self) -> None:
        client = EngineClient(SETTINGS, "127.0.0.1", 4060)
        payload = client._build_payload([], 16, 0.2, None, stop=["[#13]"])
        self.assertEqual(payload["stop"], ["[#13]"])

    def test_payload_omits_empty_stop(self) -> None:
        client = EngineClient(SETTINGS, "127.0.0.1", 4060)
        self.assertNotIn("stop", client._build_payload([], 16, 0.2, None))
        self.assertNotIn(
            "stop", client._build_payload([], 16, 0.2, None, stop=[])
        )

    def test_stream_request_body_contains_stop(self) -> None:
        response = FakeResponse(SSE)
        client = EngineClient(SETTINGS, "127.0.0.1", 4060)
        with mock.patch.object(client, "_open", return_value=response) as opened:
            list(
                client.chat_stream(
                    [{"role": "user", "content": "hi"}],
                    max_tokens=8,
                    temperature=0.2,
                    stop=["[#13]", "\n\n\n\n"],
                )
            )
        request = opened.call_args.args[0]
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["stop"], ["[#13]", "\n\n\n\n"])
        self.assertTrue(response.closed)


class _StreamClient:
    def __init__(self) -> None:
        self.kwargs: list[dict] = []

    def stream_chat(self, messages, *, max_tokens, temperature=0.2, **kwargs):
        self.kwargs.append(dict(kwargs))
        yield {"type": "delta", "content": "译七\n译八"}
        yield {"type": "done", "finish_reason": "stop", "tokens": 2}


class _ChatWithStopClient:
    def __init__(self) -> None:
        self.stops: list[object] = []

    def chat(self, messages, *, max_tokens, temperature=0.2, stop=None):
        self.stops.append(stop)
        return {
            "choices": [{"message": {"content": "译文"}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 1},
        }


class _ChatWithoutStopClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, *, max_tokens, temperature=0.2):
        self.calls += 1
        return {
            "choices": [{"message": {"content": "译文"}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 1},
        }


class EngineStopTests(unittest.TestCase):
    def test_streaming_client_receives_dynamic_stop(self) -> None:
        client = _StreamClient()
        engine = TranslationEngine(client, PROMPT_DIR, "", options=OPTIONS)
        self.assertEqual(engine(WINDOW), "译七\n译八")
        self.assertEqual(
            client.kwargs[0]["stop"], ["[#13]", "#13]", "【#13】", "\n\n\n\n"]
        )

    def test_blocking_client_receives_stop_when_supported(self) -> None:
        client = _ChatWithStopClient()
        engine = TranslationEngine(client, PROMPT_DIR, "", options=OPTIONS)
        self.assertEqual(engine("[#7] あ"), "译文")
        self.assertEqual(
            client.stops[0], ["[#8]", "#8]", "【#8】", "\n\n\n\n"]
        )

    def test_client_without_stop_kwarg_is_untouched(self) -> None:
        client = _ChatWithoutStopClient()
        engine = TranslationEngine(client, PROMPT_DIR, "", options=OPTIONS)
        self.assertEqual(engine("[#7] あ"), "译文")
        self.assertEqual(client.calls, 1)

    def test_config_switch_disables_stop(self) -> None:
        client = _StreamClient()
        engine = TranslationEngine(
            client,
            PROMPT_DIR,
            "",
            options=replace(OPTIONS, enable_server_stop_tokens=False),
        )
        engine(WINDOW)
        self.assertNotIn("stop", client.kwargs[0])

    def test_detector_second_window_is_not_aborted(self) -> None:
        detector = StreamingAbortDetector(next_window_id(WINDOW))
        self.assertTrue(detector.feed("[#7] 译七\n"))
        self.assertFalse(detector.aborted)
        self.assertFalse(detector.soft_stop)


if __name__ == "__main__":
    unittest.main()
