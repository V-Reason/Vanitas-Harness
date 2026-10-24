"""progress.Spinner：TTY 覆盖行、非 TTY 静默、done 只擦一次、异常不吞。"""

from __future__ import annotations

import io
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vanitas.config import DEFAULT_THEME  # noqa: E402
from vanitas.progress import Spinner  # noqa: E402
from vanitas.ui import Palette  # noqa: E402


class TtyStream(io.StringIO):
    """伪终端：只有 isatty 返回 True，其余与 StringIO 一致。"""

    def isatty(self) -> bool:
        return True


class SpinnerTests(unittest.TestCase):
    @staticmethod
    def palette(enabled: bool = False) -> Palette:
        return Palette(DEFAULT_THEME, enabled)

    def test_non_tty_is_silent_until_done(self) -> None:
        stream = io.StringIO()
        spin = Spinner("正在加载模型", self.palette(), stream=stream)
        spin.__enter__()
        spin.update()
        spin.update("正在翻译 a.srt（1/3）")
        self.assertEqual(stream.getvalue(), "")
        spin.done("已就绪 · 0.1s")
        spin.__exit__(None, None, None)
        # 非 TTY 只保留最终状态，且 done 之后不再追加
        self.assertEqual(stream.getvalue(), "✓ 已就绪 · 0.1s\n")

    def test_tty_draws_overwrite_line(self) -> None:
        stream = TtyStream()
        with Spinner("正在加载模型", self.palette(), stream=stream, interval_ms=0) as spin:
            spin.update()
        raw = stream.getvalue()
        self.assertTrue(raw.startswith("\r"), repr(raw))
        self.assertIn("⋯ 正在加载模型", raw)
        self.assertIn("\x1b[2K", raw)  # __exit__ 擦除整行

    def test_update_redraws_with_new_label(self) -> None:
        stream = TtyStream()
        with Spinner(
            "正在翻译 a.srt", self.palette(), stream=stream, interval_ms=0
        ) as spin:
            time.sleep(0.06)  # 越过最小重绘间隔
            spin.update("正在翻译 a.srt（2/3）")
        self.assertIn("（2/3）", stream.getvalue())

    def test_exception_path_erases_and_propagates(self) -> None:
        stream = TtyStream()
        with self.assertRaises(RuntimeError):
            with Spinner("x", self.palette(), stream=stream, interval_ms=0) as spin:
                spin.update()
                raise RuntimeError("boom")
        self.assertIn("\x1b[2K", stream.getvalue())

    def test_done_erases_exactly_once(self) -> None:
        stream = TtyStream()
        spin = Spinner("x", self.palette(), stream=stream, interval_ms=0)
        with spin:
            spin.update()
            spin.done("完成")
        self.assertEqual(stream.getvalue().count("\x1b[2K"), 1)
        self.assertTrue(spin.completed)

    def test_done_is_plain_when_palette_disabled(self) -> None:
        stream = TtyStream()
        spin = Spinner("x", self.palette(), stream=stream, enabled=False)
        spin.done("已就绪 · 1.0s")
        self.assertEqual(stream.getvalue(), "✓ 已就绪 · 1.0s\n")
        self.assertNotIn("\x1b[", stream.getvalue())

    def test_disabled_never_animates(self) -> None:
        stream = TtyStream()  # 是 TTY，但显式禁用
        with Spinner("x", self.palette(), stream=stream, interval_ms=0, enabled=False) as spin:
            spin.update("y")
        self.assertEqual(stream.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
