"""Agent 闭环：分级审批、批量动作、执行反馈与步数上限。"""

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


def action_payload(name: str, **parameters) -> str:
    return json.dumps({"action": name, "parameters": parameters}, ensure_ascii=False)


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

    def test_read_only_action_runs_without_confirmation(self) -> None:
        (self.root / "a.txt").write_text("alpha", encoding="utf-8")
        plans: list[str] = []
        repl, runner = self.make_repl(
            [action_payload("read_file", path="a.txt"), "读完了"],
            confirm=lambda plan: plans.append(plan) or True,
        )
        _value, out, _err = quiet_run(lambda: repl._chat("读文件"))
        self.assertIn("读完了", out)
        self.assertEqual(plans, [])
        self.assertEqual(len(runner.calls), 2)
        self.assertEqual(runner.calls[1][0], "followup")
        feedback = runner.calls[1][1]
        self.assertTrue(feedback.startswith("[系统反馈] 执行成功："))
        self.assertIn("alpha", feedback)

    def test_write_file_executes_and_feeds_result_back(self) -> None:
        action = action_payload("write_file", path="a.txt", content="你好")
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
        action = action_payload("write_file", path="a.txt", content="x")
        repl, runner = self.make_repl([action], confirm=lambda _plan: False)
        _value, out, _err = quiet_run(lambda: repl._chat("写文件"))
        self.assertFalse((self.root / "a.txt").exists())
        self.assertIn("已取消", out)
        self.assertEqual(len(runner.calls), 1)

    def test_outside_read_is_denied_and_fed_back(self) -> None:
        action = action_payload("read_file", path="../escape.txt")
        repl, runner = self.make_repl([action, "改用工作区文件"])
        _value, out, err = quiet_run(lambda: repl._chat("越界读"))
        self.assertIn("越界", err)
        self.assertIn("改用工作区文件", out)
        self.assertEqual(len(runner.calls), 2)
        feedback = runner.calls[1][1]
        self.assertTrue(feedback.startswith("[系统反馈] 执行失败："))
        self.assertIn("越界", feedback)

    def test_outside_write_is_blocked_after_confirmation(self) -> None:
        action = action_payload("write_file", path="../escape.txt", content="x")
        repl, runner = self.make_repl([action, "改用工作区"])
        outside = self.root.parent / "escape.txt"
        if outside.exists():
            outside.unlink()
        _value, out, err = quiet_run(lambda: repl._chat("越界写"))
        self.assertIn("越界", err)
        self.assertIn("改用工作区", out)
        self.assertEqual(len(runner.calls), 2)
        self.assertFalse(outside.exists())
        self.assertIn("执行失败", runner.calls[1][1])

    def test_batch_move_uses_one_summary_and_one_confirmation(self) -> None:
        (self.root / "a.md").write_text("a", encoding="utf-8")
        (self.root / "b.md").write_text("b", encoding="utf-8")
        batch = json.dumps(
            [
                {"action": "move_file", "parameters": {"src": "a.md", "dst": "a.srt"}},
                {"action": "move_file", "parameters": {"src": "b.md", "dst": "b.srt"}},
            ]
        )
        plans: list[str] = []
        repl, runner = self.make_repl(
            [batch, "批量完成"],
            confirm=lambda plan: plans.append(plan) or True,
        )
        _value, out, _err = quiet_run(lambda: repl._chat("把 md 改成 srt"))
        self.assertEqual(len(plans), 1)
        self.assertIn("批量操作（2 项）", plans[0])
        self.assertIn("1. 重命名 a.md -> a.srt", plans[0])
        self.assertIn("2. 重命名 b.md -> b.srt", plans[0])
        self.assertTrue((self.root / "a.srt").exists())
        self.assertTrue((self.root / "b.srt").exists())
        self.assertFalse((self.root / "a.md").exists())
        self.assertIn("批量完成", out)
        self.assertEqual(len(runner.calls), 2)
        self.assertIn("批量执行完成：2 项，成功 2，失败 0", runner.calls[1][1])

    def test_batch_read_only_actions_use_one_confirmation(self) -> None:
        (self.root / "a.txt").write_text("a", encoding="utf-8")
        (self.root / "b.txt").write_text("b", encoding="utf-8")
        batch = json.dumps(
            [
                {"action": "read_file", "parameters": {"path": "a.txt"}},
                {"action": "read_file", "parameters": {"path": "b.txt"}},
            ]
        )
        plans: list[str] = []
        repl, runner = self.make_repl(
            [batch, "读完了"],
            confirm=lambda plan: plans.append(plan) or True,
        )
        _value, out, _err = quiet_run(lambda: repl._chat("读两个文件"))
        self.assertEqual(len(plans), 1)
        self.assertIn("批量操作（2 项）", plans[0])
        self.assertIn("读完了", out)
        self.assertEqual(len(runner.calls), 2)
        feedback = runner.calls[1][1]
        self.assertIn("批量执行完成：2 项，成功 2，失败 0", feedback)

    def test_batch_partial_failure_continues_and_reports(self) -> None:
        (self.root / "a.md").write_text("a", encoding="utf-8")
        batch = json.dumps(
            [
                {"action": "move_file", "parameters": {"src": "a.md", "dst": "a.srt"}},
                {"action": "move_file", "parameters": {"src": "nope.md", "dst": "x.srt"}},
            ]
        )
        plans: list[str] = []
        repl, runner = self.make_repl(
            [batch, "收尾"],
            confirm=lambda plan: plans.append(plan) or True,
        )
        _value, out, err = quiet_run(lambda: repl._chat("批量改名"))
        self.assertEqual(len(plans), 1)
        self.assertIn("收尾", out)
        self.assertIn("工具拒绝", err)
        self.assertTrue((self.root / "a.srt").exists())
        feedback = runner.calls[1][1]
        self.assertIn("批量执行完成：2 项，成功 1，失败 1", feedback)
        self.assertIn("2. 失败", feedback)

    def test_shell_modification_is_denied_and_fed_back(self) -> None:
        action = action_payload("run_shell_ro", command="ren a.txt b.txt")
        repl, runner = self.make_repl([action, "改用 move_file"])
        with mock.patch("vanitas.tools.subprocess.run") as runner_proc:
            _value, out, err = quiet_run(lambda: repl._chat("用 shell 改名"))
        runner_proc.assert_not_called()
        self.assertIn("只允许只读命令", err)
        self.assertIn("改用 move_file", out)
        self.assertEqual(len(runner.calls), 2)
        self.assertIn("执行失败", runner.calls[1][1])

    def test_shell_read_only_echo_runs_without_confirmation(self) -> None:
        action = action_payload("run_shell_ro", command="echo hi")
        plans: list[str] = []
        repl, runner = self.make_repl(
            [action, "执行完成"],
            confirm=lambda plan: plans.append(plan) or True,
        )
        _value, out, _err = quiet_run(lambda: repl._chat("echo"))
        self.assertEqual(plans, [])
        self.assertIn("执行完成", out)
        self.assertIn("hi", runner.calls[1][1])

    def test_max_steps_stops_before_second_action(self) -> None:
        first = action_payload("write_file", path="one.txt", content="1")
        second = action_payload("write_file", path="two.txt", content="2")
        repl, runner = self.make_repl([first, second, "结束"], max_steps=1)
        _value, out, _err = quiet_run(lambda: repl._chat("连续写"))
        self.assertTrue((self.root / "one.txt").exists())
        self.assertFalse((self.root / "two.txt").exists())
        self.assertIn("已达最大步数 1", out)
        self.assertEqual(len(runner.calls), 2)

    def test_bad_json_and_unknown_action_do_not_execute(self) -> None:
        for text in (
            '{"action":',
            '{"action":"nope","parameters":{}}',
            '[{"action":"nope","parameters":{}}]',
        ):
            with self.subTest(text=text):
                repl, runner = self.make_repl([text])
                _value, out, err = quiet_run(lambda: repl._chat("坏动作"))
                self.assertIn("动作", err + out)
                self.assertEqual(len(runner.calls), 1)

    def test_json_without_action_is_final_text(self) -> None:
        repl, _runner = self.make_repl(['{"result":"ok"}'])
        _value, out, _err = quiet_run(lambda: repl._chat("输出"))
        self.assertIn('{"result":"ok"}', out)


class ActionParserTests(unittest.TestCase):
    def test_repairs_untranslated_windows_backslashes(self) -> None:
        raw = r'{"action":"list_dir","parameters":{"path":"<WORKSPACE>"}}'
        actions = agent._parse_actions(raw)
        self.assertIsNotNone(actions)
        self.assertEqual(actions[0].parameters["path"], r"<WORKSPACE>")

    def test_repairs_backslashes_in_batch_paths(self) -> None:
        raw = (
            r'[{"action":"move_file","parameters":'
            r'{"src":"<WORKSPACE>\a.md","dst":"<WORKSPACE>\a.srt"}}]'
        )
        actions = agent._parse_actions(raw)
        self.assertIsNotNone(actions)
        self.assertEqual(actions[0].parameters["src"], r"<WORKSPACE>\a.md")
        self.assertEqual(actions[0].parameters["dst"], r"<WORKSPACE>\a.srt")

    def test_valid_double_backslashes_remain_single_literal(self) -> None:
        raw = '{"action":"list_dir","parameters":{"path":"T:\\\\TMP\\\\test"}}'
        actions = agent._parse_actions(raw)
        self.assertIsNotNone(actions)
        self.assertEqual(actions[0].parameters["path"], r"<WORKSPACE>")

    def test_parses_single_alias_and_batch(self) -> None:
        single = agent._parse_actions(action_payload("shell", command="echo hi"))
        self.assertEqual(single[0].name, "run_shell_ro")
        batch = agent._parse_actions(
            '[{"action":"list_dir","parameters":{}},'
            '{"action":"delete_file","parameters":{"path":"a"}}]'
        )
        self.assertEqual([item.name for item in batch], ["list_dir", "delete_file"])

    def test_move_accepts_source_destination_aliases(self) -> None:
        actions = agent._parse_actions(
            '{"action":"move_file","parameters":'
            '{"source":"a.md","destination":"a.srt"}}'
        )
        self.assertEqual(actions[0].parameters, {"src": "a.md", "dst": "a.srt"})

    def test_any_array_requires_confirmation(self) -> None:
        actions = agent._parse_actions(
            '[{"action":"read_file","parameters":{"path":"a.txt"}}]'
        )
        self.assertTrue(agent._needs_confirmation(actions, is_batch=True))
        self.assertFalse(agent._needs_confirmation(actions))

    def test_rejects_empty_or_oversized_batch(self) -> None:
        with self.assertRaises(ValueError):
            agent._parse_actions("[]")
        oversized = "[" + ",".join(
            '{"action":"list_dir","parameters":{}}'
            for _ in range(agent.MAX_BATCH_ACTIONS + 1)
        ) + "]"
        with self.assertRaises(ValueError):
            agent._parse_actions(oversized)


if __name__ == "__main__":
    unittest.main()
