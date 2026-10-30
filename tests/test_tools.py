"""人类 `/exec`：原生 shell、无白名单/路径限制、确认与输出保护。"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vanitas import tools  # noqa: E402
from vanitas.config import load_config  # noqa: E402
from vanitas.errors import ToolDenied  # noqa: E402

DESTRUCTIVE = load_config().settings.tools.destructive_commands


def make_config(enabled=True, confirm=True):
    base = load_config()
    tool_settings = replace(
        base.settings.tools,
        exec_enabled=enabled,
        exec_require_confirm=confirm,
    )
    return replace(base, settings=replace(base.settings, tools=tool_settings))


def completed(stdout=b"ok\n", stderr=b"", code=0):
    return subprocess.CompletedProcess([], code, stdout=stdout, stderr=stderr)


class ToolsTests(unittest.TestCase):
    def test_disabled_rejects(self) -> None:
        config = make_config(enabled=False)
        with self.assertRaises(ToolDenied) as ctx:
            tools.run_exec(config, "echo hi", confirm=lambda _: "y")
        self.assertIn("exec_enabled", str(ctx.exception))

    def test_empty_command_rejected(self) -> None:
        config = make_config()
        with self.assertRaises(ToolDenied):
            tools.run_exec(config, "   ", confirm=lambda _: "y")

    def test_confirm_no_does_not_execute(self) -> None:
        config = make_config(confirm=True)
        with mock.patch("vanitas.tools.subprocess.run") as runner:
            with self.assertRaises(ToolDenied) as ctx:
                tools.run_exec(config, "echo hi", confirm=lambda _: "n")
        self.assertIn("取消", str(ctx.exception))
        runner.assert_not_called()

    def test_native_shell_keeps_pipe_string_and_shell_true(self) -> None:
        config = make_config(confirm=True)
        fake = mock.Mock(return_value=completed(b"hi\n"))
        with mock.patch("vanitas.tools.subprocess.run", fake):
            result = tools.run_exec(
                config, "echo hi | findstr hi", confirm=lambda _: "y"
            )
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(fake.call_args.args[0], "echo hi | findstr hi")
        self.assertTrue(fake.call_args.kwargs["shell"])

    def test_native_shell_executes_pipeline(self) -> None:
        config = make_config(confirm=True)
        result = tools.run_exec(
            config, "echo hello | findstr hello", confirm=lambda _: "y"
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("hello", result.stdout)

    def test_destructive_command_outside_workspace_is_allowed(self) -> None:
        config = make_config(confirm=False)
        fake = mock.Mock(return_value=completed())
        with mock.patch("vanitas.tools.subprocess.run", fake):
            result = tools.run_exec(config, r"del C:\Windows\System32\xxx")
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(
            fake.call_args.args[0], r"del C:\Windows\System32\xxx"
        )
        self.assertTrue(fake.call_args.kwargs["shell"])

    def test_cwd_uses_original_user_cwd(self) -> None:
        config = make_config(confirm=False)
        with tempfile.TemporaryDirectory() as tmp:
            fake = mock.Mock(return_value=completed())
            with (
                mock.patch.dict(os.environ, {"VANITAS_ORIG_CWD": tmp}),
                mock.patch("vanitas.tools.subprocess.run", fake),
            ):
                tools.run_exec(config, "echo hi")
        self.assertEqual(fake.call_args.kwargs["cwd"], str(Path(tmp)))

    def test_missing_shell_reports_denial(self) -> None:
        config = make_config(confirm=False)
        with mock.patch(
            "vanitas.tools.subprocess.run", side_effect=FileNotFoundError
        ):
            with self.assertRaises(ToolDenied) as ctx:
                tools.run_exec(config, "echo hi")
        self.assertIn("启动失败", str(ctx.exception))

    def test_timeout_reports_denial(self) -> None:
        config = make_config(confirm=False)
        with mock.patch(
            "vanitas.tools.subprocess.run",
            side_effect=subprocess.TimeoutExpired("echo hi", 1),
        ):
            with self.assertRaises(ToolDenied) as ctx:
                tools.run_exec(config, "echo hi")
        self.assertIn("未结束", str(ctx.exception))

    def test_large_output_is_truncated(self) -> None:
        config = make_config(confirm=False)
        huge = b"A" * (tools.MAX_CAPTURE_BYTES + 1024)
        fake = mock.Mock(return_value=completed(huge))
        with mock.patch("vanitas.tools.subprocess.run", fake):
            result = tools.run_exec(config, "echo big")
        self.assertIn("底层标准输出超出", result.stdout)
        self.assertIn("标准输出", result.stdout)

    def test_capture_limit_is_configurable(self) -> None:
        base = make_config(confirm=False)
        tools_settings = replace(
            base.settings.tools,
            max_capture_bytes=1024,
        )
        config = replace(
            base,
            settings=replace(base.settings, tools=tools_settings),
        )
        fake = mock.Mock(return_value=completed(b"A" * 4096))
        with mock.patch("vanitas.tools.subprocess.run", fake):
            result = tools.run_exec(config, "echo big")
        self.assertIn("底层标准输出超出 1 KiB", result.stdout)

    def test_split_command_keeps_quoted_argument(self) -> None:
        self.assertEqual(tools.split_command('type "a b"'), ["type", "a b"])

    def test_destructive_helper_still_recognizes_cmd_wrapper(self) -> None:
        argv = tools.split_command(r'cmd /c "del C:\Windows\System32\xxx"')
        self.assertEqual(
            tools._is_destructive_command(argv, DESTRUCTIVE),
            [r"C:\Windows\System32\xxx"],
        )


if __name__ == "__main__":
    unittest.main()
