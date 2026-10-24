"""工具白名单与确认流程（不执行真实命令的用例全部走 mock）。"""

from __future__ import annotations

import subprocess
import sys
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


def make_config(enabled=True, whitelist=("echo",), confirm=True):
    base = load_config()
    tool_settings = replace(
        base.settings.tools,
        exec_enabled=enabled,
        exec_whitelist=tuple(whitelist),
        exec_require_confirm=confirm,
    )
    return replace(base, settings=replace(base.settings, tools=tool_settings))


class ToolsTests(unittest.TestCase):
    def test_disabled_rejects(self) -> None:
        config = make_config(enabled=False)
        with self.assertRaises(ToolDenied) as ctx:
            tools.run_exec(config, "echo hi", confirm=lambda _: "y")
        self.assertIn("exec_enabled", str(ctx.exception))

    def test_not_whitelisted_rejects(self) -> None:
        config = make_config(whitelist=("echo",))
        with self.assertRaises(ToolDenied) as ctx:
            tools.run_exec(config, "dir C:\\", confirm=lambda _: "y")
        self.assertIn("白名单", str(ctx.exception))

    def test_confirm_no_does_not_execute(self) -> None:
        config = make_config(whitelist=("echo",), confirm=True)
        with mock.patch("vanitas.tools.subprocess.run") as runner:
            with self.assertRaises(ToolDenied) as ctx:
                tools.run_exec(config, "echo hi", confirm=lambda _: "n")
        self.assertIn("取消", str(ctx.exception))
        runner.assert_not_called()

    def test_metacharacters_are_literal_arguments(self) -> None:
        config = make_config(whitelist=("echo",))
        fake = mock.Mock(
            return_value=subprocess.CompletedProcess([], 0, stdout=b"ok\n", stderr=b"")
        )
        with mock.patch("vanitas.tools.subprocess.run", fake):
            result = tools.run_exec(config, "echo a & dir", confirm=lambda _: "y")
        argv = fake.call_args.args[0]
        self.assertEqual(argv, ["echo", "a", "&", "dir"])
        kwargs = fake.call_args.kwargs
        self.assertFalse(kwargs["shell"])
        self.assertEqual(result.exit_code, 0)

    def test_quoted_argument_kept_whole(self) -> None:
        self.assertEqual(tools.split_command('type "a b"'), ["type", "a b"])

    def test_empty_command_rejected(self) -> None:
        config = make_config()
        with self.assertRaises(ToolDenied):
            tools.run_exec(config, "   ", confirm=lambda _: "y")

    def test_real_execution_of_whitelisted_command(self) -> None:
        config = make_config(whitelist=("python --version",), confirm=True)
        result = tools.run_exec(config, "python --version", confirm=lambda _: "y")
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Python", result.stdout)

    def test_missing_executable_reports_denial(self) -> None:
        config = make_config(whitelist=("definitely-not-here",))
        with self.assertRaises(ToolDenied) as ctx:
            tools.run_exec(config, "definitely-not-here", confirm=lambda _: "y")
        self.assertIn("不存在", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
