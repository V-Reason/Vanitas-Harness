"""注释门自身的测试：每条规则都要“能报警”，且仓库当前必须是绿的。

为什么值得单独测：门最容易的坏法不是误报，而是**悄悄失效**——正则写错、
锚点文件改名、钉子值读不到，结果永远 0 违规，看起来“很干净”。
所以每条规则都给一正一反两个样本，再加一条“真实仓库 = 0 硬违规”。
"""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import check_comments  # noqa: E402


def run(root: Path, *extra: str) -> tuple[int, str]:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = check_comments.main(["--root", str(root), *extra])
    return code, buffer.getvalue()


class CommentGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        (self.root / "vanitas").mkdir()
        (self.root / "docs").mkdir()
        (self.root / "tests").mkdir()
        (self.root / "docs" / "变更历史.md").write_text(
            "## 52. 旧章\n\n## 54. V5\n", encoding="utf-8"
        )
        # 钉子值来源：注释门会在运行时从这里读出“已钉死的字节/指纹”。
        (self.root / "tests" / "test_prompts.py").write_text(
            "self.assertEqual(len(production), 13108)\n"
            'self.assertEqual(hashlib.sha256(production).hexdigest().upper(),\n'
            '    "74324996DA7786F6A2BC4E1FDA6C5972E0E3AAC2B2BDB0C1A7887F5AFB268F0A")\n',
            encoding="utf-8",
        )

    def write(self, name: str, body: str) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
        return path

    def test_selftest_passes(self) -> None:
        code, out = run(ROOT, "--selftest")
        self.assertEqual(code, 0, out)
        self.assertIn("失败 0", out)

    def test_forensics_leak_is_hard_violation(self) -> None:
        self.write("vanitas/a.py", "# 实测 6 条重复，见 run 20261031-005132\nX = 1\n")
        code, out = run(self.root)
        self.assertEqual(code, 1, out)
        self.assertIn("[G1]", out)

    def test_forensics_with_history_pointer_is_clean(self) -> None:
        self.write("vanitas/a.py", "# 为什么提到 11：见 §52\nX = 1\n")
        code, out = run(self.root)
        self.assertEqual(code, 0, out)

    def test_ghost_symbol_is_hard_violation(self) -> None:
        self.write("vanitas/a.py", "# 由 `removed_helper_xyz` 统一控制\nX = 1\n")
        code, out = run(self.root)
        self.assertEqual(code, 1, out)
        self.assertIn("[G2]", out)

    def test_existing_symbol_is_clean(self) -> None:
        self.write("vanitas/a.py", "# 由 `helper_here` 统一控制\nhelper_here = 1\n")
        code, out = run(self.root)
        self.assertEqual(code, 0, out)

    def test_dangling_anchor_is_hard_violation(self) -> None:
        self.write("vanitas/a.py", "# 细节见 §99\nX = 1\n")
        code, out = run(self.root)
        self.assertEqual(code, 1, out)
        self.assertIn("[G3]", out)

    def test_resolvable_anchor_is_clean(self) -> None:
        self.write("vanitas/a.py", "# 细节见 §54\nX = 1\n")
        code, out = run(self.root)
        self.assertEqual(code, 0, out)

    def test_pinned_literal_is_hard_violation(self) -> None:
        self.write("vanitas/a.py", "# 提示词 13108 B，指纹 74324996\nX = 1\n")
        code, out = run(self.root)
        self.assertEqual(code, 1, out)
        self.assertIn("[G4]", out)

    def test_inline_exemption_is_counted_and_clean(self) -> None:
        self.write(
            "vanitas/a.py",
            "# gate-allow: 这一段要引用历史批次号\n# 实测 6 条\nX = 1\n",
        )
        code, out = run(self.root)
        self.assertEqual(code, 1, out)  # 第二行仍然违规
        self.assertIn("豁免 1", out)

    def test_tests_root_ignores_cue_numbers_but_not_run_stamps(self) -> None:
        self.write("tests/test_x.py", "# file1: cue2/3 各重译一次\nX = 1\n")
        self.assertEqual(run(self.root)[0], 0)
        self.write("tests/test_y.py", "# run 20261031-005132\nY = 1\n")
        self.assertEqual(run(self.root)[0], 1)

    def test_syntax_error_fails_closed(self) -> None:
        self.write("vanitas/broken.py", "def x(:\n")
        code, out = run(self.root)
        self.assertEqual(code, 2, out)
        self.assertIn("PARSE ERROR", out)

    def test_empty_todo_is_warning_not_blocking(self) -> None:
        self.write("vanitas/a.py", "# TODO 还没做\nX = 1\n")
        code, out = run(self.root)
        self.assertEqual(code, 0, out)
        self.assertIn("警告 1", out)

    def test_repository_itself_is_clean(self) -> None:
        """最重要的一条：门必须和仓库焊死，仓库现在就得是绿的。"""
        code, out = run(ROOT)
        self.assertEqual(code, 0, out)
        self.assertIn("硬违规 0", out)


if __name__ == "__main__":
    unittest.main()
