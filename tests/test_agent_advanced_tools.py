"""高阶工具、动态提示符、观测折叠与状态看板。"""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vanitas import guard, prompts, tools  # noqa: E402
from vanitas.agent import AgentAction, _run_action  # noqa: E402
from vanitas.chat import TurnResult  # noqa: E402
from vanitas.config import DEFAULT_THEME, load_config  # noqa: E402
from vanitas.errors import ToolDenied  # noqa: E402
from vanitas.launcher import RunRecord  # noqa: E402
from vanitas.repl import Repl  # noqa: E402
from vanitas.context import COLLAPSED_OBSERVATION_MARKER, Session  # noqa: E402
from vanitas.ui import Palette  # noqa: E402


def quiet_run(fn):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        value = fn()
    return value, out.getvalue(), err.getvalue()


class AdvancedToolsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.config = load_config(
            preferences_path=self.root / "preferences.json"
        )
        self.toolbox = tools.AgentTools(self.config, self.root)

    def test_patch_file_single_match_and_bom_preserved(self) -> None:
        target = self.root / "a.py"
        target.write_bytes(b"\xef\xbb\xbfvalue = 1\n")
        message = self.toolbox.patch_file("a.py", "value = 1", "value = 2")
        self.assertIn("补丁应用成功", message)
        raw = target.read_bytes()
        self.assertTrue(raw.startswith(b"\xef\xbb\xbf"))
        self.assertIn(b"value = 2", raw)

    def test_patch_file_no_match_rejected(self) -> None:
        (self.root / "a.py").write_text("value = 1\n", encoding="utf-8")
        with self.assertRaises(ToolDenied) as ctx:
            self.toolbox.patch_file("a.py", "missing", "x")
        self.assertIn("未在文件", str(ctx.exception))

    def test_patch_file_multiple_matches_rejected(self) -> None:
        (self.root / "a.py").write_text("x\nx\n", encoding="utf-8")
        with self.assertRaises(ToolDenied) as ctx:
            self.toolbox.patch_file("a.py", "x", "y")
        self.assertIn("定位不唯一", str(ctx.exception))

    def test_patch_file_outside_root_rejected(self) -> None:
        with self.assertRaises(ToolDenied):
            self.toolbox.patch_file("../outside.py", "a", "b")

    def test_run_action_formats_new_tools(self) -> None:
        (self.root / "a.py").write_text("value = 1\n", encoding="utf-8")
        ok, detail = _run_action(
            self.toolbox,
            AgentAction(
                name="patch_file",
                parameters={
                    "path": "a.py",
                    "search": "value = 1",
                    "replace": "value = 2",
                },
            ),
        )
        self.assertTrue(ok)
        self.assertIn("补丁应用成功", detail)
        ok, detail = _run_action(
            self.toolbox,
            AgentAction(
                name="grep_workspace",
                parameters={"pattern": "value"},
            ),
        )
        self.assertTrue(ok)
        self.assertIn("a.py:1", detail)

    def test_read_file_range_normal_and_clamped_end(self) -> None:
        (self.root / "a.txt").write_text(
            "one\ntwo\nthree\n", encoding="utf-8"
        )
        text = self.toolbox.read_file_range("a.txt", 2, 99)
        self.assertIn("第 2–3 行", text)
        self.assertIn("   2 | two", text)
        self.assertIn("   3 | three", text)

    def test_read_file_range_invalid_bounds(self) -> None:
        (self.root / "a.txt").write_text("one\ntwo\n", encoding="utf-8")
        for start, end in ((0, 1), (2, 1), (3, 3)):
            with self.subTest(start=start, end=end):
                with self.assertRaises(ToolDenied):
                    self.toolbox.read_file_range("a.txt", start, end)

    def test_read_file_range_max_span(self) -> None:
        (self.root / "a.txt").write_text(
            "\n".join(f"line{i}" for i in range(20)), encoding="utf-8"
        )
        with self.assertRaises(ToolDenied) as ctx:
            tools.read_file_range(
                "a.txt", 1, 20, root=self.root, max_lines=5
            )
        self.assertIn("超过上限", str(ctx.exception))

    def test_grep_workspace_regex_and_ignore_dirs(self) -> None:
        (self.root / "src").mkdir()
        (self.root / "src" / "a.py").write_text(
            "Needle = 1\nother\n", encoding="utf-8"
        )
        (self.root / "__pycache__").mkdir()
        (self.root / "__pycache__" / "b.py").write_text(
            "needle\n", encoding="utf-8"
        )
        matches = self.toolbox.grep_workspace("needle")
        self.assertEqual(len(matches), 1)
        self.assertIn("src/a.py:1", matches[0])

    def test_grep_workspace_invalid_regex_and_outside_path(self) -> None:
        with self.assertRaises(ToolDenied):
            self.toolbox.grep_workspace("[")
        with self.assertRaises(ToolDenied):
            tools.grep_workspace(".", str(self.root.parent), root=self.root)

    def test_grep_workspace_max_results(self) -> None:
        (self.root / "a.txt").write_text(
            "hit\nhit\nhit\n", encoding="utf-8"
        )
        matches = tools.grep_workspace(
            "hit", root=self.root, max_results=2
        )
        self.assertEqual(matches[-1], "…（结果超出上限，已截断）")

    def test_agent_action_names_cover_new_tools(self) -> None:
        from vanitas.config import AGENT_ACTION_NAMES

        for name in ("patch_file", "read_file_range", "grep_workspace"):
            self.assertIn(name, AGENT_ACTION_NAMES)


class AdvancedPromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.config = load_config(
            preferences_path=Path(self._tmp.name) / "preferences.json"
        )
        self.repl = Repl(
            self.config,
            palette=Palette(DEFAULT_THEME, enabled=True),
        )

    def test_context_tag_and_three_color_bands(self) -> None:
        cases = ((0, "\x1b[90m"), (6000, "\x1b[33m"), (7000, "\x1b[31m"))
        for used, color in cases:
            with self.subTest(used=used):
                with mock.patch.object(
                    self.repl.session, "message_tokens", return_value=used
                ):
                    text = self.repl._prompt_text()
                self.assertIn("[ctx:", text)
                self.assertIn(color, text)
                self.assertIn("/8.2k]", text)

    def test_context_tag_percentage(self) -> None:
        with mock.patch.object(
            self.repl.session, "message_tokens", return_value=1966
        ):
            text = self.repl._prompt_text()
        self.assertIn("ctx: 23%", text)
        self.assertIn("2.0k/8.2k", text)

    def test_sleeping_tag_visible(self) -> None:
        self.repl._sleeping = True
        with mock.patch.object(
            self.repl.session, "message_tokens", return_value=0
        ):
            text = self.repl._prompt_text()
        self.assertIn("[休眠]", text)
        self.assertIn("ctx: 0%", text)

    def test_static_system_prefix_has_no_date(self) -> None:
        prompt_dir = Path(self.config.config_dir) / "prompts"
        system = prompts.system_prompt(prompt_dir)
        self.assertNotIn("{date}", system)
        agent_task = prompts.load_task(prompt_dir, "agent")
        self.assertIn("{date}", agent_task)


class ObservationCollapseTests(unittest.TestCase):
    @staticmethod
    def _session(large: int = 2000) -> Session:
        big = "<tool_result>\n" + ("x" * large) + "\n</tool_result>"
        return Session(
            messages=[
                {"role": "user", "content": "task"},
                {
                    "role": "assistant",
                    "content": '<tool_call>[{"action":"read_file"}]</tool_call>',
                },
                {"role": "user", "content": big},
                {
                    "role": "assistant",
                    "content": '<tool_call>[{"action":"list_dir"}]</tool_call>',
                },
                {"role": "user", "content": big},
                {"role": "assistant", "content": "done"},
                {"role": "user", "content": big},
                {"role": "assistant", "content": "done"},
            ]
        )

    def test_collapses_only_old_observations(self) -> None:
        session = self._session()
        collapsed = session.collapse_intermediate_observations(100)
        self.assertEqual(collapsed, 1)
        self.assertIn(
            COLLAPSED_OBSERVATION_MARKER,
            session.messages[2]["content"],
        )
        self.assertNotIn("read_file", session.messages[2]["content"])
        self.assertIn("<tool_call>", session.messages[1]["content"])
        self.assertNotIn(
            COLLAPSED_OBSERVATION_MARKER, session.messages[6]["content"]
        )
        self.assertTrue(session.is_dirty)

    def test_idempotent_and_threshold(self) -> None:
        session = self._session()
        self.assertEqual(
            session.collapse_intermediate_observations(100), 1
        )
        self.assertEqual(
            session.collapse_intermediate_observations(100), 0
        )
        other = self._session()
        self.assertEqual(
            other.collapse_intermediate_observations(1_000_000), 0
        )


class StatusDashboardTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.config = load_config(
            preferences_path=Path(self._tmp.name) / "preferences.json"
        )
        self.repl = Repl(self.config, palette=Palette({}, enabled=False))
        self.repl.client = mock.Mock()
        self.repl.client.health.return_value = True
        self.repl.last_turn = TurnResult(
            text="ok",
            tokens=8,
            elapsed_s=0.5,
            tps=36.4,
            ttft_ms=68.0,
            finish_reason="stop",
        )
        self.repl.session.messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]

    def test_status_reports_window_gpu_process_and_rates(self) -> None:
        record = RunRecord(
            model_id="qwen_coder",
            pid=12480,
            port=4060,
            profile="1",
            alias="qwen2.5-coder-7b-instruct-q6_k",
            started_at="2026-10-27T00:00:00",
            log_path="x.log",
        )
        with (
            mock.patch.object(guard, "gpu_memory", return_value=(7000, 8188)),
            mock.patch.object(guard, "process_memory_kb", return_value=114688),
            mock.patch.object(
                sys.modules["vanitas.repl"].launcher,
                "read_records",
                return_value=[record],
            ),
            mock.patch.object(
                sys.modules["vanitas.repl"].launcher,
                "pid_running",
                return_value=True,
            ),
        ):
            _value, out, _err = quiet_run(lambda: self.repl._cmd_status(""))
        self.assertIn("视窗监控", out)
        self.assertIn("已用 7,000 MiB / 总量 8,188 MiB", out)
        self.assertIn("PID 12480", out)
        self.assertIn("RSS 112 MiB", out)
        self.assertIn("TTFT 68 ms", out)
        self.assertIn("36.4 t/s", out)
        self.assertIn("Agent 步数配额 -/10", out)
