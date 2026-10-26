"""client：高层门面的 stream_chat / stream_completion 委托。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vanitas.client import Client, LlamaClient  # noqa: E402
from vanitas.config import RequestSettings, load_config  # noqa: E402

SETTINGS = RequestSettings(
    timeout_sec=10,
    max_retries=0,
    backoff_base_ms=0,
    backoff_jitter_ms=0,
)


class ClientFacadeTests(unittest.TestCase):
    def test_client_alias(self) -> None:
        self.assertIs(Client, LlamaClient)

    def test_stream_chat_delegates(self) -> None:
        client = LlamaClient(SETTINGS, "127.0.0.1", 4060)
        events = [{"type": "delta", "content": "x"}]
        with mock.patch.object(client, "chat_stream", return_value=iter(events)) as prop:
            result = list(
                client.stream_chat(
                    [{"role": "user", "content": "hi"}],
                    max_tokens=8,
                    temperature=0.2,
                )
            )
        self.assertEqual(result, events)
        props = prop.call_args.kwargs
        self.assertEqual(props["max_tokens"], 8)
        self.assertIsNone(props["on_progress"])

    def test_stream_completion_builds_messages(self) -> None:
        client = LlamaClient(SETTINGS, "127.0.0.1", 4060)
        with mock.patch.object(client, "chat_stream", return_value=iter([])) as prop:
            list(
                client.stream_completion(
                    "hello",
                    system="be nice",
                    max_tokens=16,
                    temperature=0.2,
                )
            )
        messages = prop.call_args.args[0]
        self.assertEqual(messages[0], {"role": "system", "content": "be nice"})
        self.assertEqual(messages[1], {"role": "user", "content": "hello"})

    def test_for_model_returns_facade(self) -> None:
        config = load_config()
        client = LlamaClient.for_model(config, config.model("qwen7b"))
        self.assertIsInstance(client, LlamaClient)


if __name__ == "__main__":
    unittest.main()
