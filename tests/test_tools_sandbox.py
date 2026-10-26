"""Agent 工具沙盒：路径边界、强类型文件工具与只读 Shell。"""

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


def make_config(enabled: bool = True):
    base = load_config()
    tool_settings = replace(base.settings.tools, exec_enabled=enabled)
    return replace(base, settings=replace(base.settings, tools=tool_settings))


class WorkspaceBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        (self.root / "sub").mkdir()
        (self.root / "sub" / "a.txt").write_text("x", encoding="utf-8")

    def test_accepts_relative_absolute_and_case_variant(self) -> None:
        self.assertTrue(tools.is_inside_workspace("a.txt", self.root))
        self.assertTrue(tools.is_inside_workspace(".", self.root))
        self.assertTrue(
            tools.is_inside_workspace(str(self.root / "sub" / "a.txt"), self.root)
        )
        self.assertTrue(
            tools.is_inside_workspace(str(self.root).upper(), self.root)
        )

    def test_rejects_traversal_absolute_and_empty(self) -> None:
        for path in ("../outside.txt", str(self.root.parent / "outside.txt"), ""):
            with self.subTest(path=path):
                with self.assertRaises(ToolDenied) as ctx:
                    tools.is_inside_workspace(path, self.root)
                self.assertIn("越界", str(ctx.exception))
                self.assertFalse(
                    tools.is_inside_workspace(path, self.root, raise_on_error=False)
                )

    def test_default_function_uses_process_cwd_not_env(self) -> None:
        with (
            mock.patch.object(Path, "cwd", return_value=self.root),
            mock.patch.dict(
                os.environ, {"VANITAS_ORIG_CWD": str(self.root.parent)}
            ),
        ):
            self.assertTrue(tools.is_inside_workspace("a.txt"))
            with self.assertRaises(ToolDenied):
                tools.is_inside_workspace("../outside.txt")

    def test_rejects_windows_system_path_and_null_byte(self) -> None:
        with self.assertRaises(ToolDenied):
            tools.is_inside_workspace(
                r"C:\Windows\System32\drivers\etc\hosts", self.root
            )
        with self.assertRaises(ToolDenied):
            tools.is_inside_workspace("bad\x00name", self.root)


class FileToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.config = make_config()
        self.toolbox = tools.AgentTools(self.config, self.root)

    def test_read_write_list_move_delete(self) -> None:
        self.assertEqual(self.toolbox.write_file("sub/a.md", "你好"), 6)
        self.assertEqual(self.toolbox.read_file("sub/a.md"), "你好")
        self.assertEqual(self.toolbox.list_dir("."), ["sub/"])
        self.assertEqual(self.toolbox.list_dir("sub"), ["a.md"])
        self.assertIn(
            "sub/a.md -> sub/a.srt",
            self.toolbox.move_file("sub/a.md", "sub/a.srt"),
        )
        self.assertFalse((self.root / "sub" / "a.md").exists())
        self.assertEqual(
            (self.root / "sub" / "a.srt").read_text(encoding="utf-8"), "你好"
        )
        self.assertIn("a.srt", self.toolbox.delete_file("sub/a.srt"))
        self.assertFalse((self.root / "sub" / "a.srt").exists())

    def test_write_rejects_non_string_content(self) -> None:
        with self.assertRaises(ToolDenied):
            self.toolbox.write_file("a.txt", 123)  # type: ignore[arg-type]

    def test_move_rejects_existing_destination(self) -> None:
        self.toolbox.write_file("a.txt", "a")
        self.toolbox.write_file("b.txt", "b")
        with self.assertRaises(ToolDenied) as ctx:
            self.toolbox.move_file("a.txt", "b.txt")
        self.assertIn("目标已存在", str(ctx.exception))

    def test_move_rejects_missing_source_with_clear_message(self) -> None:
        with self.assertRaises(ToolDenied) as ctx:
            self.toolbox.move_file("nope.md", "x.srt")
        self.assertEqual(str(ctx.exception), "源文件不存在或路径非法")

    def test_move_rejects_illegal_path_characters(self) -> None:
        self.toolbox.write_file("a.md", "a")
        for src, dst in (
            ("a#.md", "a.srt"),
            ("a.md", "a#.srt"),
            ("a?.md", "a.srt"),
            ("a.md", "a*.srt"),
        ):
            with self.subTest(src=src, dst=dst):
                with self.assertRaises(ToolDenied) as ctx:
                    self.toolbox.move_file(src, dst)
                self.assertEqual(str(ctx.exception), "源文件不存在或路径非法")
        self.assertTrue((self.root / "a.md").exists())

    def test_delete_rejects_directory(self) -> None:
        (self.root / "sub").mkdir()
        with self.assertRaises(ToolDenied):
            self.toolbox.delete_file("sub")

    def test_all_file_tools_reject_outside_paths(self) -> None:
        operations = (
            lambda: self.toolbox.read_file("../outside.txt"),
            lambda: self.toolbox.write_file("../outside.txt", "x"),
            lambda: self.toolbox.move_file("a.txt", "../outside.txt"),
            lambda: self.toolbox.delete_file("../outside.txt"),
            lambda: self.toolbox.list_dir(".."),
        )
        for operation in operations:
            with self.subTest(operation=operation):
                with self.assertRaises(ToolDenied) as ctx:
                    operation()
                self.assertIn("越界", str(ctx.exception))
        self.assertFalse((self.root.parent / "outside.txt").exists())

    def test_default_root_prefers_original_cwd(self) -> None:
        with mock.patch.dict(os.environ, {"VANITAS_ORIG_CWD": str(self.root)}):
            toolbox = tools.AgentTools(self.config)
        self.assertEqual(toolbox.root, self.root.resolve())


class ReadonlyShellTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        (self.root / "a.txt").write_text("alpha", encoding="utf-8")
        (self.root / "b.txt").write_text("beta", encoding="utf-8")
        self.config = make_config()
        self.toolbox = tools.AgentTools(self.config, self.root)

    def test_builtins_are_readonly_and_metachars_are_literal(self) -> None:
        result = self.toolbox.run_shell_ro("echo hi & del a.txt > out.txt")
        self.assertEqual(result.exit_code, 0)
        self.assertIn("hi & del a.txt > out.txt", result.stdout)
        self.assertTrue((self.root / "a.txt").exists())
        self.assertFalse((self.root / "out.txt").exists())

    def test_type_and_dir_are_served_without_shell(self) -> None:
        typed = self.toolbox.run_shell_ro("type a.txt")
        self.assertEqual((typed.exit_code, typed.stdout), (0, "alpha"))
        catted = self.toolbox.run_shell_ro("cat b.txt")
        self.assertEqual((catted.exit_code, catted.stdout), (0, "beta"))
        listed = self.toolbox.run_shell_ro("dir")
        self.assertEqual(listed.exit_code, 0)
        self.assertIn("a.txt", listed.stdout)
        lsed = self.toolbox.run_shell_ro("ls")
        self.assertEqual(lsed.exit_code, 0)
        self.assertIn("b.txt", lsed.stdout)
        pwd = self.toolbox.run_shell_ro("pwd")
        self.assertEqual(pwd.stdout.strip(), str(self.root.resolve()))

    def test_modification_commands_are_rejected(self) -> None:
        for command in (
            "ren a.txt c.txt",
            "rename a.txt c.txt",
            "move a.txt c.txt",
            "del a.txt",
            "erase a.txt",
            "copy a.txt c.txt",
            "mkdir sub",
        ):
            with self.subTest(command=command):
                with self.assertRaises(ToolDenied) as ctx:
                    self.toolbox.run_shell_ro(command)
                self.assertIn("只允许只读命令", str(ctx.exception))
        self.assertTrue((self.root / "a.txt").exists())
        self.assertFalse((self.root / "c.txt").exists())

    def test_write_flags_are_rejected(self) -> None:
        for command in (
            "find . -exec del {} ;",
            "find . -delete",
            "find . -o -name x",
            "git diff --output=x.txt",
            "git log -o",
        ):
            with self.subTest(command=command):
                with self.assertRaises(ToolDenied) as ctx:
                    self.toolbox.run_shell_ro(command)
                self.assertIn("写操作参数", str(ctx.exception))

    def test_shell_paths_are_workspace_bound(self) -> None:
        with self.assertRaises(ToolDenied) as ctx:
            self.toolbox.run_shell_ro(
                r"type C:\Windows\System32\drivers\etc\hosts"
            )
        self.assertIn("越界", str(ctx.exception))
        with self.assertRaises(ToolDenied):
            self.toolbox.run_shell_ro(r"dir C:\Windows")

    def test_git_readonly_uses_shell_false_and_blocks_path_options(self) -> None:
        fake = mock.Mock(
            return_value=subprocess.CompletedProcess([], 0, stdout=b"ok", stderr=b"")
        )
        with mock.patch("vanitas.tools.subprocess.run", fake):
            result = self.toolbox.run_shell_ro("git status --short")
        self.assertEqual(result.exit_code, 0)
        kwargs = fake.call_args.kwargs
        self.assertFalse(kwargs["shell"])
        self.assertEqual(kwargs["cwd"], str(self.root.resolve()))
        self.assertEqual(kwargs["env"]["GIT_PAGER"], "cat")
        with self.assertRaises(ToolDenied):
            self.toolbox.run_shell_ro("git -C .. status")
        with self.assertRaises(ToolDenied):
            self.toolbox.run_shell_ro("git diff --no-index a.txt C:\\Windows\\x")
        with self.assertRaises(ToolDenied):
            self.toolbox.run_shell_ro("git branch -D main")

    def test_readonly_shell_does_not_depend_on_manual_exec_toggle(self) -> None:
        disabled = make_config(enabled=False)
        toolbox = tools.AgentTools(disabled, self.root)
        result = toolbox.run_shell_ro("echo hi")
        self.assertEqual((result.exit_code, result.stdout), (0, "hi\n"))


if __name__ == "__main__":
    unittest.main()
