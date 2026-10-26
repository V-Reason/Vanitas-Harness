"""Agent 闭环：动作解析、确认、执行、系统反馈与步数上限。"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vanitas import agent, tools  # noqa: E402
from vanitas.chat import TurnResult  # noqa: E402
from vanitas.config import load_config  # noqa: E402
from vanitas.ui import Palette  # noqa: E402


class ScriptedRunner:
    """替代 ChatRunner：按脚本返回模型输出，并模拟历史写入。"""

    def __init__(self, session, turns: list[str]) -> None:
        self.session = session
        self.turns = list(turns)
        self.calls: list[tuple[str, str, bool]] = []

    def _next(self, kind: str, payload: str, display: bool) -> TurnResult:
        self.calls.append((kind, payload, display))
        if not self.turns:
            raise AssertionError("脚本轮次已耗尽")
        text = self.turns.pop(0)
        self.session.add_user(payload)
        self.session.add_assistant(text)
        return TurnResult(
            text=text,
            tokens=1,
            elapsed_s=0.1,
            tps=10.0,
            ttft_ms=None,
            finish_reason="stop",
        )

    def run(self, text: str, *, display: bool = True, on_dropped=None) -> TurnResult:
        return self._next("run", text, display)

    def run_followup(
        self, feedback: str, *, display: bool = True, on_dropped=None
    ) -> TurnResult:
        return self._next("followup", feedback, display)


def quiet_run(fn):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        value = fn()
    return value, out.getvalue(), err.getvalue()


class AgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config()
        self.model = self.config.model("qwen7b")
        self.profile = self.config.resolve_profile(self.model)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def make_repl(
        self,
        turns: list[str],
        *,
        max_steps: int = 5,
        confirm=None,
        toolbox=None,
    ):
        repl = agent.AgentRepl(
            self.config,
            self.model,
            self.profile.key,
            palette=Palette({}, enabled=False),
            tools=toolbox or tools.AgentTools(self.config, self.root),
            max_steps=max_steps,
            confirm=confirm if confirm is not None else (lambda _plan: True),
        )
        repl.client = mock.Mock()
        runner = ScriptedRunner(repl.session, turns)
        repl.chat_runner = runner
        return repl, runner

    def test_run_agent_uses_agent_repl_and_task(self) -> None:
        fake_repl = mock.Mock()
        fake_repl.run.return_value = 0
        out = io.StringIO()
        with (
            mock.patch.object(agent, "AgentRepl", return_value=fake_repl) as repl_cls,
            contextlib.redirect_stdout(out),
        ):
            code = agent.run_agent(
                self.config,
                self.model,
                self.profile.key,
                palette=Palette({}, enabled=False),
            )
        self.assertEqual(code, 0)
        repl_cls.assert_called_once()
        fake_repl.configure.assert_called_once_with(system=None, task="agent")
        fake_repl.run.assert_called_once()
        self.assertIn("Agent 已就绪", out.getvalue())
        self.assertNotIn("不自动执行", out.getvalue())

    def test_plain_text_finishes_without_confirmation(self) -> None:
        plans: list[str] = []
        repl, runner = self.make_repl(
            ["我在 T:\\demo"], confirm=lambda plan: plans.append(plan) or True
        )
        _value, out, _err = quiet_run(lambda: repl._chat("我在哪个目录？"))
        self.assertIn("我在 T:\\demo", out)
        self.assertEqual(plans, [])
        self.assertEqual(len(runner.calls), 1)
        self.assertEqual(runner.calls[0][0], "run")
        self.assertFalse(runner.calls[0][2])

    def test_write_file_executes_and_feeds_result_back(self) -> None:
        action = json.dumps(
            {
                "action": "write_file",
                "parameters": {"path": "a.txt", "content": "你好"},
            }
        )
        plans: list[str] = []
        repl, runner = self.make_repl(
            [action, "已经写好了"],
            confirm=lambda plan: plans.append(plan) or True,
        )
        _value, out, _err = quiet_run(lambda: repl._chat("写文件"))
        self.assertEqual((self.root / "a.txt").read_text(encoding="utf-8"), "你好")
        self.assertTrue(any("将写入文件 a.txt" in plan for plan in plans))
        self.assertIn("已经写好了", out)
        self.assertNotIn('{"action"', out)
        self.assertEqual(runner.calls[0][0], "run")
        self.assertEqual(runner.calls[1][0], "followup")
        self.assertEqual(runner.calls[1][2], False)
        feedbacks = [
            item["content"]
            for item in repl.session.messages
            if item["role"] == "user" and item["content"].startswith("[系统反馈]")
        ]
        self.assertTrue(feedbacks)
        self.assertTrue(feedbacks[0].startswith("[系统反馈] 执行成功："))
        self.assertIn("a.txt", feedbacks[0])

    def test_user_refusal_cancels_without_execution(self) -> None:
        action = json.dumps(
            {
                "action": "write_file",
                "parameters": {"path": "a.txt", "content": "x"},
            }
        )
        repl, runner = self.make_repl([action], confirm=lambda _plan: False)
        _value, out, _err = quiet_run(lambda: repl._chat("写文件"))
        self.assertFalse((self.root / "a.txt").exists())
        self.assertIn("已取消", out)
        self.assertEqual(len(runner.calls), 1)

    def test_outside_write_is_blocked_after_confirmation(self) -> None:
        action = json.dumps(
            {
                "action": "write_file",
                "parameters": {"path": "../escape.txt", "content": "x"},
            }
        )
        repl, runner = self.make_repl([action, "不会继续"])
        outside = self.root.parent / "escape.txt"
        if outside.exists():
            outside.unlink()
        _value, out, err = quiet_run(lambda: repl._chat("越界写"))
        self.assertIn("越界操作", err)
        self.assertNotIn("不会继续", out)
        self.assertEqual(len(runner.calls), 1)
        self.assertFalse(outside.exists())

    def test_shell_success_and_failure_are_fed_back(self) -> None:
        action = json.dumps(
            {
                "action": "shell",
                "parameters": {"command": "ren a.txt b.txt"},
            }
        )
        success = mock.Mock(
            return_value=mock.Mock(returncode=0, stdout=b"", stderr=b"")
        )
        repl, runner = self.make_repl([action, "改名完成"])
        with mock.patch("vanitas.tools.subprocess.run", success):
            _value, out, _err = quiet_run(lambda: repl._chat("改名"))
        self.assertIn("改名完成", out)
        self.assertEqual(success.call_count, 1)
        self.assertTrue(
            any(
                item["content"].startswith("[系统反馈] 执行成功：")
                for item in repl.session.messages
                if item["role"] == "user"
            )
        )

        failure = mock.Mock(
            return_value=mock.Mock(returncode=1, stdout=b"", stderr=b"boom")
        )
        repl2, _runner2 = self.make_repl([action, "失败后收尾"])
        with mock.patch("vanitas.tools.subprocess.run", failure):
            _value2, out2, _err2 = quiet_run(lambda: repl2._chat("改名"))
        self.assertIn("失败后收尾", out2)
        self.assertTrue(
            any(
                item["content"].startswith("[系统反馈] 执行失败：")
                for item in repl2.session.messages
                if item["role"] == "user"
            )
        )

    def test_max_steps_stops_before_sixth_action(self) -> None:
        first = json.dumps(
            {
                "action": "write_file",
                "parameters": {"path": "one.txt", "content": "1"},
            }
        )
        second = json.dumps(
            {
                "action": "write_file",
                "parameters": {"path": "two.txt", "content": "2"},
            }
        )
        repl, runner = self.make_repl([first, second, "结束"], max_steps=1)
        _value, out, _err = quiet_run(lambda: repl._chat("连续写"))
        self.assertTrue((self.root / "one.txt").exists())
        self.assertFalse((self.root / "two.txt").exists())
        self.assertIn("已达最大步数 1", out)
        self.assertEqual(len(runner.calls), 2)

    def test_bad_json_and_unknown_action_do_not_execute(self) -> None:
        for text in ('{"action":', '{"action":"nope","parameters":{}}'):
            with self.subTest(text=text):
                repl, runner = self.make_repl([text])
                _value, _out, err = quiet_run(lambda: repl._chat("坏动作"))
                self.assertIn("动作", err + _out)
                self.assertEqual(len(runner.calls), 1)

    def test_json_without_action_is_final_text(self) -> None:
        repl, _runner = self.make_repl(['{"result":"ok"}'])
        _value, out, _err = quiet_run(lambda: repl._chat("输出"))
        self.assertIn('{"result":"ok"}', out)


if __name__ == "__main__":
    unittest.main()
