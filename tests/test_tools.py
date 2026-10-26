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

    def test_metacharacters_are_literal_arguments_for_executables(self) -> None:
        config = make_config(whitelist=("python",))
        fake = mock.Mock(
            return_value=subprocess.CompletedProcess([], 0, stdout=b"ok\n", stderr=b"")
        )
        with mock.patch("vanitas.tools.subprocess.run", fake):
            result = tools.run_exec(config, "python a & dir", confirm=lambda _: "y")
        argv = fake.call_args.args[0]
        self.assertEqual(argv, ["python", "a", "&", "dir"])
        kwargs = fake.call_args.kwargs
        self.assertFalse(kwargs["shell"])
        self.assertEqual(result.exit_code, 0)

    def test_builtin_command_goes_through_cmd(self) -> None:
        config = make_config(whitelist=("echo",), confirm=True)
        result = tools.run_exec(config, "echo hi", confirm=lambda _: "y")
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.stdout.strip(), "hi")

    def test_builtin_rejects_chaining(self) -> None:
        config = make_config(whitelist=("echo",), confirm=True)
        with self.assertRaises(ToolDenied) as ctx:
            tools.run_exec(config, "echo a & dir", confirm=lambda _: "y")
        self.assertIn("内置命令", str(ctx.exception))

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

    def test_destructive_command_extracts_paths(self) -> None:
        paths = tools._is_destructive_command(
            ["del", "/f", "/s", r"C:\Windows\System32\xxx"]
        )
        self.assertEqual(paths, [r"C:\Windows\System32\xxx"])

    def test_destructive_command_skips_switches_and_keeps_relative_path(self) -> None:
        self.assertEqual(
            tools._is_destructive_command(["del", "/q", r".\test.txt"]),
            [r".\test.txt"],
        )

    def test_non_destructive_command_has_no_targets(self) -> None:
        self.assertEqual(tools._is_destructive_command(["echo", "del", "x"]), [])

    def test_cmd_wrapper_detects_quoted_del(self) -> None:
        argv = tools.split_command(r'cmd /c "del C:\Windows\System32\xxx"')
        self.assertEqual(
            tools._is_destructive_command(argv),
            [r"C:\Windows\System32\xxx"],
        )

    def test_cmd_wrapper_detects_raw_quoted_argv(self) -> None:
        argv = ["cmd", "/c", r'"del C:\Windows\System32\xxx"']
        self.assertEqual(
            tools._is_destructive_command(argv),
            [r"C:\Windows\System32\xxx"],
        )

    def test_within_cwd_accepts_relative_and_rejects_traversal(self) -> None:
        base = Path(load_config().base_dir)
        self.assertTrue(tools._is_within_cwd(r".\test.txt", base))
        self.assertTrue(tools._is_within_cwd(str(base / "nested" / "a.txt"), base))
        self.assertFalse(tools._is_within_cwd(r"..\outside.txt", base))
        self.assertFalse(tools._is_within_cwd(r"C:\Windows\System32\xxx", base))

    def test_outside_del_rejected_before_subprocess(self) -> None:
        config = make_config(whitelist=("del",), confirm=True)
        with mock.patch("vanitas.tools.subprocess.run") as runner:
            with self.assertRaises(ToolDenied) as ctx:
                tools.run_exec(
                    config,
                    r"del C:\Windows\System32\xxx",
                    confirm=lambda _: "y",
                )
        self.assertIn(
            "安全限制：禁止修改或删除工作目录之外的文件/目录",
            str(ctx.exception),
        )
        runner.assert_not_called()

    def test_outside_del_is_rejected_even_when_exec_is_disabled(self) -> None:
        config = make_config(enabled=False, whitelist=("del",), confirm=False)
        with mock.patch("vanitas.tools.subprocess.run") as runner:
            with self.assertRaises(ToolDenied) as ctx:
                tools.run_exec(config, r"del C:\Windows\System32\xxx")
        self.assertIn("安全限制", str(ctx.exception))
        runner.assert_not_called()

    def test_inside_del_reaches_subprocess(self) -> None:
        config = make_config(whitelist=("del",), confirm=False)
        fake = mock.Mock(
            return_value=subprocess.CompletedProcess([], 0, stdout=b"", stderr=b"")
        )
        with mock.patch("vanitas.tools.subprocess.run", fake):
            result = tools.run_exec(config, r"del .\test.txt")
        self.assertEqual(result.exit_code, 0)
        argv = fake.call_args.args[0]
        self.assertEqual(argv, ["cmd", "/d", "/s", "/c", r"del .\test.txt"])
        self.assertEqual(fake.call_args.kwargs["cwd"], str(config.base_dir))

    def test_destructive_command_without_target_rejected(self) -> None:
        config = make_config(whitelist=("del",), confirm=False)
        with mock.patch("vanitas.tools.subprocess.run") as runner:
            with self.assertRaises(ToolDenied) as ctx:
                tools.run_exec(config, "del /q")
        self.assertIn("必须显式指定目标路径", str(ctx.exception))
        runner.assert_not_called()

    def test_copy_outside_target_rejected(self) -> None:
        config = make_config(whitelist=("copy",), confirm=False)
        with mock.patch("vanitas.tools.subprocess.run") as runner:
            with self.assertRaises(ToolDenied) as ctx:
                tools.run_exec(config, r"copy .\a.txt C:\Windows\b.txt")
        self.assertIn("安全限制", str(ctx.exception))
        runner.assert_not_called()

    def test_cmd_wrapper_outside_del_rejected(self) -> None:
        config = make_config(whitelist=("cmd",), confirm=False)
        with mock.patch("vanitas.tools.subprocess.run") as runner:
            with self.assertRaises(ToolDenied) as ctx:
                tools.run_exec(config, r'cmd /c "del C:\Windows\System32\xxx"')
        self.assertIn("安全限制", str(ctx.exception))
        runner.assert_not_called()

    def test_destructive_help_switch_is_not_treated_as_target(self) -> None:
        config = make_config(whitelist=("del",), confirm=False)
        fake = mock.Mock(
            return_value=subprocess.CompletedProcess([], 0, stdout=b"help", stderr=b"")
        )
        with mock.patch("vanitas.tools.subprocess.run", fake):
            result = tools.run_exec(config, "del /?")
        self.assertEqual(result.exit_code, 0)
        fake.assert_called_once()


if __name__ == "__main__":
    unittest.main()
