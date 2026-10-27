"""恢复能力：目录删除、震荡熔断、失败动作阻断、/undo 与解析错误。"""

from __future__ import annotations

import io
import contextlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vanitas.agent import (  # noqa: E402
    AgentAction,
    AgentLoop,
    _parse_actions,
)
from vanitas.chat import TurnResult  # noqa: E402
from vanitas.config import load_config  # noqa: E402
from vanitas.repl import Repl  # noqa: E402
from vanitas.tools import AgentTools  # noqa: E402
from vanitas.ui import Palette  # noqa: E402


def quiet_run(fn):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        value = fn()
    return value, out.getvalue(), err.getvalue()


def action(name: str, **params) -> AgentAction:
    return AgentAction(name=name, parameters=params)


def turn(text: str) -> TurnResult:
    return TurnResult(
        text=text,
        tokens=1,
        elapsed_s=0.0,
        tps=0.0,
        ttft_ms=None,
        finish_reason="stop",
    )


class StagnationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.config = load_config(
            preferences_path=self.root / "preferences.json"
        )
        self.loop = AgentLoop(
            self.config,
            AgentTools(self.config, self.root),
            palette=Palette({}, enabled=False),
        )

    def _feed(self, sequence: list[AgentAction]) -> tuple[bool, str]:
        result = False
        for item in sequence:
            result = self.loop._stagnates([item])
        return result, self.loop._stagnation_reason

    def test_classic_repeat(self) -> None:
        a = action("list_dir", path=".")
        result, reason = self._feed([a, a, a])
        self.assertTrue(result)
        self.assertEqual(reason, "连续")

    def test_ping_pong(self) -> None:
        a = action("list_dir", path=".")
        b = action("read_file", path="a.txt")
        result, reason = self._feed([a, b, a, b])
        self.assertTrue(result)
        self.assertEqual(reason, "震荡")

    def test_three_cycle(self) -> None:
        a = action("list_dir", path="a")
        b = action("list_dir", path="b")
        c = action("list_dir", path="c")
        result, reason = self._feed([a, b, c, a, b, c])
        self.assertTrue(result)
        self.assertEqual(reason, "震荡")

    def test_frequency_and_unique_sequence(self) -> None:
        a = action("list_dir", path="a")
        b = action("list_dir", path="b")
        c = action("list_dir", path="c")
        result, reason = self._feed([a, b, c, a, b, a])
        self.assertTrue(result)
        self.assertEqual(reason, "重复")

        other = AgentLoop(
            self.config,
            AgentTools(self.config, self.root),
            palette=Palette({}, enabled=False),
        )
        result = False
        for name in ("a", "b", "c", "d", "e", "f"):
            result = other._stagnates([action("list_dir", path=name)])
        self.assertFalse(result)

    def test_failed_action_is_blocked_without_second_confirmation(self) -> None:
        bad = (
            '<tool_call>[{"action":"patch_file","parameters":'
            '{"path":"missing.py","search":"a","replace":"b"}}]</tool_call>'
        )
        answers = iter([turn(bad), turn(bad), turn("done")])
        confirm = mock.Mock(return_value=True)

        def converse(_text, _feedback, _stream):
            return next(answers)

        def finalize(_text):
            return None

        _value, out, err = quiet_run(
            lambda: self.loop.run(
                "test",
                converse=converse,
                confirm=confirm,
                announce=lambda _actions: None,
                finalize=finalize,
            )
        )
        self.assertEqual(confirm.call_count, 1)
        self.assertIn("不再二次确认", out + err)
        self.assertIn("已失败", out + err)
        self.assertIsNotNone(self.loop._last_failed_signatures)


class DeletePathTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.config = load_config(
            preferences_path=self.root / "preferences.json"
        )
        self.toolbox = AgentTools(self.config, self.root)

    def test_symlink_deletes_link_not_target(self) -> None:
        target = self.root / "target.txt"
        target.write_text("keep", encoding="utf-8")
        link = self.root / "link.txt"
        try:
            os.symlink(target, link)
        except OSError as exc:
            self.skipTest(f"symlink unsupported: {exc}")
        self.assertIn("删除链接", self.toolbox.delete_path("link.txt"))
        self.assertFalse(link.exists())
        self.assertTrue(target.exists())


class UndoTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.config = load_config(
            preferences_path=Path(self._tmp.name) / "preferences.json"
        )
        self.repl = Repl(self.config, palette=Palette({}, enabled=False))

    def test_pop_last_turn_only(self) -> None:
        self.repl.session.messages = [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "new"},
            {"role": "assistant", "content": "new answer"},
        ]
        removed, released = self.repl.session.pop_last_turn()
        self.assertEqual(removed, 2)
        self.assertGreater(released, 0)
        self.assertEqual(
            [item["content"] for item in self.repl.session.messages],
            ["old", "old answer"],
        )
        self.assertTrue(self.repl.session.is_dirty)

    def test_undo_command_and_aliases(self) -> None:
        self.repl.session.messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        _value, out, _err = quiet_run(lambda: self.repl._cmd_undo(""))
        self.assertIn("已回溯上一轮完整对话", out)
        self.assertEqual(out.count("✓"), 1)
        self.assertNotIn("✓ ✓", out)
        self.assertEqual(self.repl.session.messages, [])
        self.assertIn("undo", __import__("vanitas.repl", fromlist=["COMMAND_MAP"]).COMMAND_MAP)
        self.assertIn("pop", __import__("vanitas.repl", fromlist=["COMMAND_MAP"]).COMMAND_MAP)
        self.assertIn("rollback", __import__("vanitas.repl", fromlist=["COMMAND_MAP"]).COMMAND_MAP)

    def test_undo_without_history_warns(self) -> None:
        _value, out, _err = quiet_run(lambda: self.repl._cmd_undo(""))
        self.assertIn("可供回溯", out)

    def test_undo_rolls_back_full_agent_turn_in_one_shot(self) -> None:
        self.repl.session.messages = [
            {"role": "user", "content": "human request"},
            {
                "role": "assistant",
                "content": '<tool_call>[{"action":"list_dir"}]</tool_call>',
            },
            {"role": "user", "content": "<tool_result>ok</tool_result>"},
            {"role": "assistant", "content": "final answer"},
        ]
        _value, out, _err = quiet_run(lambda: self.repl._cmd_undo(""))
        self.assertEqual(self.repl.session.messages, [])
        self.assertIn("移除 4 条记录", out)
        self.assertEqual(out.count("✓"), 1)
        self.assertNotIn("✓ ✓", out)


class ActionParserTests(unittest.TestCase):
    def test_new_action_does_not_fall_back_to_shell(self) -> None:
        actions = _parse_actions(
            '<tool_call>[{"action":"patch_file","parameters":'
            '{"path":"a.py","search":"a","replace":"b"}}]</tool_call>',
            max_batch_actions=10,
            action_aliases={},
        )
        self.assertIsNotNone(actions)
        self.assertEqual(actions[0].name, "patch_file")
        self.assertEqual(actions[0].parameters["search"], "a")

    def test_empty_parameters_names_action_and_field(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _parse_actions(
                '<tool_call>[{"action":"write_file","parameters":{}}]</tool_call>',
                max_batch_actions=10,
                action_aliases={},
            )
        message = str(ctx.exception)
        self.assertIn("write_file", message)
        self.assertIn("parameters.path", message)
        self.assertIn("收到参数键", message)

    def test_delete_path_action(self) -> None:
        actions = _parse_actions(
            '<tool_call>[{"action":"delete_path","parameters":'
            '{"path":"sub"}}]</tool_call>',
            max_batch_actions=10,
            action_aliases={},
        )
        self.assertEqual(actions[0].name, "delete_path")
