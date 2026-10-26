"""chat.ChatRunner：预算、usage 回退、中断与 finish_reason。"""

from __future__ import annotations

import contextlib
import io
import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vanitas.chat import ChatRunner  # noqa: E402
from vanitas.config import load_config  # noqa: E402
from vanitas.errors import EngineError  # noqa: E402
from vanitas.metrics import MESSAGE_OVERHEAD, chat_max_tokens, estimate_tokens  # noqa: E402
from vanitas.session import Session  # noqa: E402
from vanitas.ui import Palette  # noqa: E402


class FakeClient:
    def __init__(self, events: list[object]) -> None:
        self.events = events
        self.calls: list[dict] = []

    def stream_chat(self, messages, *, max_tokens, temperature=0.2, on_progress=None):
        self.calls.append(
            {
                "messages": list(messages),
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        for event in self.events:
            if isinstance(event, BaseException):
                raise event
            yield event


class ChatRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config()
        self.model = self.config.model("qwen7b")
        self.profile = self.config.resolve_profile(self.model)
        self.session = Session(
            model_id=self.model.id,
            profile=self.profile.key,
            system="sys",
        )

    def make_runner(self, client: FakeClient) -> ChatRunner:
        return ChatRunner(
            self.config,
            client,
            self.session,
            self.model,
            self.profile,
            Palette({}, enabled=False),
        )

    def run_turn(self, runner: ChatRunner):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            result = runner.run("hi")
        return result, out.getvalue(), err.getvalue()

    def test_temperature_comes_from_settings(self) -> None:
        self.config = replace(
            self.config,
            settings=replace(
                self.config.settings,
                chat=replace(self.config.settings.chat, temperature=0.9),
            ),
        )
        client = FakeClient(
            [
                {"type": "delta", "content": "ok"},
                {"type": "done", "tokens": 1, "tps": 1.0, "finish_reason": "stop"},
            ]
        )
        runner = self.make_runner(client)
        self.run_turn(runner)
        self.assertEqual(client.calls[0]["temperature"], 0.9)

    def test_usage_metrics_and_history(self) -> None:
        client = FakeClient(
            [
                {"type": "delta", "content": "你好"},
                {
                    "type": "done",
                    "tokens": 128,
                    "tps": 36.5,
                    "ttft_ms": 12.0,
                    "finish_reason": "stop",
                    "usage": {"completion_tokens": 128},
                },
            ]
        )
        runner = self.make_runner(client)
        result, out, _err = self.run_turn(runner)
        self.assertEqual(result.tokens, 128)
        self.assertEqual(result.tps, 36.5)
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.reasoning_chars, 0)
        self.assertIn("你好", out)
        self.assertEqual(len(self.session.messages), 2)
        self.assertEqual(self.session.messages[1]["content"], "你好")

        call = client.calls[0]
        history_tokens = sum(
            estimate_tokens(str(item["content"])) + MESSAGE_OVERHEAD
            for item in call["messages"]
            if item["role"] != "system"
        )
        self.assertEqual(
            call["max_tokens"],
            chat_max_tokens(
                self.profile.ctx,
                history_tokens,
                self.config.settings.repl.context_reserve_tokens,
                self.config.settings.repl.min_output_tokens,
            ),
        )

    def test_character_fallback_tokens(self) -> None:
        client = FakeClient(
            [
                {"type": "delta", "content": "abcdef"},
                {"type": "done", "tokens": 0, "tps": 0.0, "finish_reason": "stop"},
            ]
        )
        runner = self.make_runner(client)
        result, _out, _err = self.run_turn(runner)
        self.assertEqual(result.tokens, 2)

    def test_keyboard_interrupt_drops_partial_history(self) -> None:
        client = FakeClient(
            [
                {"type": "delta", "content": "半句"},
                KeyboardInterrupt(),
            ]
        )
        runner = self.make_runner(client)
        result, out, _err = self.run_turn(runner)
        self.assertTrue(result.interrupted)
        self.assertEqual(self.session.messages, [])
        self.assertIn("半句", out)

    def test_engine_error_pops_user(self) -> None:
        client = FakeClient([EngineError("模拟失败")])
        runner = self.make_runner(client)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(EngineError):
                runner.run("hi")
        self.assertEqual(self.session.messages, [])

    def test_finish_reason_length_is_exposed(self) -> None:
        client = FakeClient(
            [
                {"type": "delta", "content": "被截断"},
                {
                    "type": "done",
                    "tokens": 4,
                    "tps": 1.0,
                    "finish_reason": "length",
                },
            ]
        )
        runner = self.make_runner(client)
        result, _out, _err = self.run_turn(runner)
        self.assertEqual(result.finish_reason, "length")

    def test_display_false_does_not_write_model_text(self) -> None:
        client = FakeClient(
            [
                {"type": "delta", "content": '{"action":"shell"}'},
                {"type": "done", "tokens": 3, "tps": 1.0, "finish_reason": "stop"},
            ]
        )
        runner = self.make_runner(client)
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            result = runner.run("hi", display=False)
        self.assertEqual(result.text, '{"action":"shell"}')
        self.assertNotIn('{"action"', out.getvalue())
        self.assertEqual(self.session.messages[-1]["content"], '{"action":"shell"}')

    def test_display_false_can_stream_to_explicit_sink(self) -> None:
        client = FakeClient(
            [
                {"type": "delta", "content": "流式"},
                {"type": "done", "tokens": 2, "tps": 1.0, "finish_reason": "stop"},
            ]
        )
        runner = self.make_runner(client)
        sink = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()):
            result = runner.run("hi", display=False, stream=sink)
        self.assertEqual(result.text, "流式")
        self.assertIn("流式", sink.getvalue())

    def test_followup_appends_raw_system_feedback(self) -> None:
        client = FakeClient(
            [
                {"type": "delta", "content": "完成"},
                {"type": "done", "tokens": 2, "tps": 1.0, "finish_reason": "stop"},
            ]
        )
        runner = self.make_runner(client)
        feedback = "<tool_result>\n执行成功：已写入 a.txt\n</tool_result>"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            runner.run_followup(feedback, display=False)
        self.assertEqual(self.session.messages[0]["role"], "user")
        self.assertEqual(self.session.messages[0]["content"], feedback)
        self.assertEqual(self.session.messages[1]["content"], "完成")
        self.assertEqual(client.calls[0]["messages"][-1]["content"], feedback)

    def test_no_client_raises_engine_error(self) -> None:
        runner = ChatRunner(
            self.config,
            None,
            self.session,
            self.model,
            self.profile,
            Palette({}, enabled=False),
        )
        with self.assertRaises(EngineError):
            runner.run("hi")


if __name__ == "__main__":
    unittest.main()
