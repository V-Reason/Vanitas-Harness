"""engine：SSE 指标、finish_reason、progress 回调、响应释放与重复惩罚。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vanitas import engine  # noqa: E402
from vanitas.config import RequestSettings  # noqa: E402
from vanitas.engine import EngineClient  # noqa: E402

SETTINGS = RequestSettings(
    timeout_sec=10,
    max_retries=0,
    backoff_base_ms=0,
    backoff_jitter_ms=0,
)
SSE = [
    'data: {"choices":[{"delta":{"content":"你"},"finish_reason":null}]}\n'.encode(
        "utf-8"
    ),
    (
        'data: {"choices":[{"delta":{"content":"好"},"finish_reason":null}],'
        '"usage":{"completion_tokens":2},'
        '"timings":{"predicted_per_second":40.0,"predicted_n":2}}\n'
    ).encode("utf-8"),
    b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n',
    b"data: [DONE]\n",
]


class FakeResponse:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines
        self.closed = False

    def __iter__(self):
        return iter(self._lines)

    def close(self) -> None:
        self.closed = True


class EngineTests(unittest.TestCase):
    def test_stream_metrics_and_finish_reason(self) -> None:
        response = FakeResponse(SSE)
        client = EngineClient(SETTINGS, "127.0.0.1", 4060)
        with mock.patch.object(client, "_open", return_value=response):
            events = list(
                client.chat_stream([{"role": "user", "content": "hi"}], max_tokens=8)
            )
        types = [event["type"] for event in events]
        self.assertIn("usage", types)
        self.assertIn("delta", types)
        done = events[-1]
        self.assertEqual(done["type"], "done")
        self.assertEqual(done["tokens"], 2)
        self.assertEqual(done["tps"], 40.0)
        self.assertEqual(done["tps_source"], "server.timings")
        self.assertEqual(done["finish_reason"], "stop")
        self.assertIsNotNone(done["ttft_ms"])
        self.assertGreaterEqual(done["elapsed_ms"], 0)
        self.assertTrue(response.closed)

    def test_progress_callback_is_invoked(self) -> None:
        response = FakeResponse(SSE)
        client = EngineClient(SETTINGS, "127.0.0.1", 4060)
        received: list[dict] = []
        with (
            mock.patch.object(client, "_open", return_value=response),
            mock.patch.object(engine, "PROGRESS_INTERVAL_SEC", 0.0),
        ):
            list(
                client.chat_stream(
                    [{"role": "user", "content": "hi"}],
                    max_tokens=8,
                    on_progress=received.append,
                )
            )
        self.assertTrue(received)
        self.assertIn("tokens_est", received[0])
        self.assertIn("tps_est", received[0])

    def test_generator_close_closes_response(self) -> None:
        response = FakeResponse(SSE)
        client = EngineClient(SETTINGS, "127.0.0.1", 4060)
        with mock.patch.object(client, "_open", return_value=response):
            stream = client.chat_stream(
                [{"role": "user", "content": "hi"}], max_tokens=8
            )
            next(stream)
            stream.close()
        self.assertTrue(response.closed)

    def test_repeat_penalty_only_above_one(self) -> None:
        client = EngineClient(SETTINGS, "127.0.0.1", 4062, repeat_penalty=1.1)
        payload = client._build_payload([], 16, 0.2, None)
        self.assertEqual(payload["repeat_penalty"], 1.1)

        plain = EngineClient(SETTINGS, "127.0.0.1", 4062)
        self.assertNotIn("repeat_penalty", plain._build_payload([], 16, 0.2, None))


if __name__ == "__main__":
    unittest.main()
