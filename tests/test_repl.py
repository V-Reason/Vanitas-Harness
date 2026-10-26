"""repl：Tab 补全、模型切换、档位待生效、meta 与中断提示。"""

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

from vanitas import repl as repl_mod  # noqa: E402
from vanitas.chat import TurnResult  # noqa: E402
from vanitas.config import DEFAULT_THEME, load_config  # noqa: E402
from vanitas.errors import ToolDenied  # noqa: E402
from vanitas.launcher import RunRecord  # noqa: E402
from vanitas.repl import Repl  # noqa: E402
from vanitas.ui import Palette  # noqa: E402


def quiet_run(fn):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        value = fn()
    return value, out.getvalue(), err.getvalue()


class ReplCompletionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config()
        self.repl = Repl(self.config, palette=Palette({}, enabled=False))

    def test_commands_aliases_and_prefix(self) -> None:
        candidates = self.repl._completion_candidates("/")
        texts = [item[0] for item in candidates]
        self.assertIn("/help", texts)
        self.assertIn("/model", texts)
        self.assertIn("/exit", texts)

        partial = self.repl._completion_candidates("/mod")
        self.assertEqual([item[0] for item in partial], ["/model"])
        self.assertEqual(partial[0][1], -4)

    def test_model_and_profile_arguments(self) -> None:
        ids = [item[0] for item in self.repl._completion_candidates("/model ")]
        self.assertEqual(ids, ["qwen7b", "minicpm", "hymt", "qwen3b"])

        prefix = self.repl._completion_candidates("/model qw")
        self.assertEqual([item[0] for item in prefix], ["qwen7b", "qwen3b"])
        self.assertEqual(prefix[0][1], -2)

        profiles = [
            item[0] for item in self.repl._completion_candidates("/profile ")
        ]
        self.assertEqual(profiles, ["1", "2", "3"])

    def test_non_commands_and_second_argument(self) -> None:
        self.assertEqual(self.repl._completion_candidates("hello"), [])
        self.assertEqual(self.repl._completion_candidates("/model q w"), [])

    def test_prompt_session_receives_completer(self) -> None:
        with (
            mock.patch("prompt_toolkit.PromptSession") as session_cls,
            mock.patch("prompt_toolkit.history.FileHistory"),
        ):
            self.repl._get_prompt_session()
        completer = session_cls.call_args.kwargs["completer"]
        self.assertIsNotNone(completer)

        class Document:
            text_before_cursor = "/model "

        completions = list(completer.get_completions(Document(), None))
        self.assertEqual(
            [item.text for item in completions],
            ["qwen7b", "minicpm", "hymt", "qwen3b"],
        )


class ReplPromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config()
        self.repl = Repl(self.config, palette=Palette({}, enabled=False))

    def test_prompt_text_is_magenta_when_palette_enabled(self) -> None:
        from prompt_toolkit.formatted_text import ANSI

        repl = Repl(self.config, palette=Palette(DEFAULT_THEME, enabled=True))
        prompt = repl._prompt_text()
        self.assertIn("\x1b[35m", prompt)
        fragments = ANSI(prompt).__pt_formatted_text__()
        self.assertTrue(any(style == "ansimagenta" for style, _text in fragments))

    def test_prompt_session_receives_ansi_object(self) -> None:
        from prompt_toolkit.formatted_text import ANSI

        class FakeSession:
            def __init__(self) -> None:
                self.value = None

            def prompt(self, value):
                self.value = value
                return "hello"

        fake = FakeSession()
        self.repl._prompt_ready = True
        self.repl._prompt_session = fake
        self.repl._ansi_formatter = ANSI
        with mock.patch("sys.stdin.isatty", return_value=True):
            line = self.repl._read_line("\x1b[35mqwen7b@1 › \x1b[0m")
        self.assertEqual(line, "hello")
        self.assertIsInstance(fake.value, ANSI)
        fragments = fake.value.__pt_formatted_text__()
        self.assertTrue(any(style == "ansimagenta" for style, _text in fragments))

    def test_fallback_input_has_no_raw_ansi(self) -> None:
        self.repl._prompt_ready = True
        self.repl._prompt_session = None
        err = io.StringIO()
        with (
            mock.patch("sys.stdin.isatty", return_value=True),
            mock.patch("builtins.input", return_value="hello") as reader,
            contextlib.redirect_stderr(err),
        ):
            self.assertEqual(
                self.repl._read_line("\x1b[35mqwen7b@1 › \x1b[0m"), "hello"
            )
        reader.assert_called_once_with()
        self.assertNotIn("\x1b[", err.getvalue())
        self.assertIn("qwen7b@1", err.getvalue())

    def test_non_tty_prompt_is_plain_stderr(self) -> None:
        self.repl._prompt_ready = True
        self.repl._prompt_session = None
        err = io.StringIO()
        with (
            mock.patch("sys.stdin.isatty", return_value=False),
            mock.patch("builtins.input", return_value="hello"),
            contextlib.redirect_stderr(err),
        ):
            self.assertEqual(
                self.repl._read_line("\x1b[35mqwen7b@1 › \x1b[0m"), "hello"
            )
        self.assertNotIn("\x1b[", err.getvalue())


class ReplSwitchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config()
        self.repl = Repl(self.config, palette=Palette({}, enabled=False))

    def test_switch_not_ready_stops_and_starts(self) -> None:
        record = RunRecord(
            model_id="qwen3b",
            pid=123,
            port=4063,
            profile="1",
            alias="qwen2.5-3b-instruct-q4_k_m",
            started_at="2026-10-26T00:00:00",
            log_path="x.log",
        )
        with (
            mock.patch.object(Repl, "_probe", return_value=None),
            mock.patch.object(repl_mod.launcher, "stop_all", return_value=[1]) as stop,
            mock.patch.object(repl_mod.launcher, "start", return_value=record) as start,
            mock.patch.object(repl_mod.launcher, "wait_ready", return_value=0.0),
        ):
            _value, out, _err = quiet_run(lambda: self.repl._cmd_model("qwen3b"))
        stop.assert_called_once()
        start.assert_called_once()
        self.assertEqual(self.repl.model.id, "qwen3b")
        self.assertIsNotNone(self.repl.client)
        self.assertIsNotNone(self.repl.chat_runner)
        self.assertIn("已切换", out)

    def test_switch_ready_reuses_connection(self) -> None:
        fake = mock.Mock()
        with (
            mock.patch.object(Repl, "_probe", return_value=(fake, 4061)),
            mock.patch.object(repl_mod.launcher, "read_records", return_value=[]),
            mock.patch.object(repl_mod.launcher, "stop_all") as stop,
        ):
            _value, out, _err = quiet_run(lambda: self.repl._cmd_model("minicpm"))
        stop.assert_not_called()
        self.assertEqual(self.repl.model.id, "minicpm")
        self.assertIs(self.repl.client, fake)
        self.assertIn("已切换到", out)

    def test_same_model_is_noop(self) -> None:
        with (
            mock.patch.object(Repl, "_probe") as probe,
            mock.patch.object(repl_mod.launcher, "stop_all") as stop,
        ):
            _value, out, _err = quiet_run(lambda: self.repl._cmd_model("qwen7b"))
        probe.assert_not_called()
        stop.assert_not_called()
        self.assertIn("已经是当前模型", out)

    def test_profile_is_pending_not_active(self) -> None:
        _value, out, _err = quiet_run(lambda: self.repl._cmd_profile("2"))
        self.assertEqual(self.repl.profile.key, "1")
        self.assertEqual(self.repl.pending_profile.key, "2")
        self.assertIn("已暂存，重启生效", out)

    def test_configure_system_and_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sys.md"
            path.write_text("自定义系统提示", encoding="utf-8")
            self.repl.configure(system=str(path), task="chat")
        self.assertEqual(self.repl.session.system, "自定义系统提示")
        self.assertEqual(self.repl.session.task, "chat")


class ReplMetaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config()
        self.repl = Repl(self.config, palette=Palette({}, enabled=False))

    def install_result(self, **overrides):
        fields = {
            "text": "answer",
            "tokens": 128,
            "elapsed_s": 3.5,
            "tps": 36.5,
            "ttft_ms": 75.0,
            "finish_reason": "stop",
        }
        fields.update(overrides)
        fake = mock.Mock()
        fake.run.return_value = TurnResult(**fields)
        self.repl.client = mock.Mock()
        self.repl.chat_runner = fake

    def test_meta_format(self) -> None:
        self.install_result()
        _value, out, _err = quiet_run(lambda: self.repl._chat("hi"))
        self.assertIn("· 128 tokens · 3.5s · 36.5 t/s", out)
        self.assertNotIn("首字", out)

    def test_verbose_meta_includes_ttft(self) -> None:
        self.repl.verbose = True
        self.install_result()
        _value, out, _err = quiet_run(lambda: self.repl._chat("hi"))
        self.assertIn("首字 75 ms", out)

    def test_interrupt_warning(self) -> None:
        self.install_result(interrupted=True)
        _value, out, _err = quiet_run(lambda: self.repl._chat("hi"))
        self.assertIn("已中止生成", out)

    def test_length_warning(self) -> None:
        self.install_result(finish_reason="length")
        _value, out, _err = quiet_run(lambda: self.repl._chat("hi"))
        self.assertIn("达到长度上限，输出被截断", out)

    def test_model_table_colors_state_and_port(self) -> None:
        palette = Palette(DEFAULT_THEME, enabled=True)
        repl = Repl(self.config, palette=palette)
        with mock.patch.object(repl_mod.launcher, "read_records", return_value=[]):
            _value, out, _err = quiet_run(lambda: repl._cmd_model(""))
        self.assertIn("\x1b[31m✗ 停止\x1b[0m", out)
        self.assertIn("\x1b[1m\x1b[35m", out)
        self.assertIn("\x1b[36mport", out)
        self.assertNotIn("✓ 未运行", out)

    def test_status_uses_semantic_symbols(self) -> None:
        palette = Palette(DEFAULT_THEME, enabled=True)
        repl = Repl(self.config, palette=palette)
        with (
            mock.patch.object(repl_mod.launcher, "read_records", return_value=[]),
            mock.patch.object(repl_mod.guard, "gpu_status", return_value=None),
        ):
            _value, out, _err = quiet_run(lambda: repl._cmd_status(""))
        self.assertIn("\x1b[31m✗ 停止\x1b[0m", out)
        self.assertNotIn("✓ 就绪", out)

    def test_exec_denial_is_printed_without_exit_code(self) -> None:
        with mock.patch.object(
            repl_mod.tools,
            "run_exec",
            side_effect=ToolDenied("安全限制：禁止修改或删除工作目录之外的文件/目录"),
        ):
            _value, out, err = quiet_run(lambda: self.repl._cmd_exec("del C:\\x"))
        self.assertIn("安全限制", err)
        self.assertNotIn("退出码", out)
        self.assertNotIn("退出码", err)

    def test_exec_cancel_is_meta_not_error(self) -> None:
        with mock.patch.object(
            repl_mod.tools,
            "run_exec",
            side_effect=ToolDenied(repl_mod.tools.CANCEL_MESSAGE),
        ):
            _value, out, err = quiet_run(lambda: self.repl._cmd_exec("echo hi"))
        self.assertIn("· 已取消", out)
        self.assertEqual(err, "")


if __name__ == "__main__":
    unittest.main()
