"""monitor：紧凑摘要、多文件隔离、TTY 看板边界与非 TTY 降级。"""

from __future__ import annotations

import io
import os
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vanitas.metrics import SegmentStat  # noqa: E402
from vanitas.monitor import TranslationMonitor, _display_width  # noqa: E402
from vanitas.config import DEFAULT_THEME  # noqa: E402
from vanitas.ui import Palette  # noqa: E402

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
VERBOSE_PHRASES = (
    "翻译任务已完成",
    "统计段数",
    "累计汉字",
    "总计耗时",
    "平均速率",
    "吞吐性能",
    "不含时间戳",
    "排版符号",
)


def stat(source: str, translation: str, tokens: int = 10) -> SegmentStat:
    return SegmentStat(
        source=source,
        translation=translation,
        elapsed_s=1.0,
        tokens=tokens,
        tps=10.0,
        tps_source="server.timings",
    )


def make_monitor(stream: io.StringIO, enabled: bool = False, capacity: int = 5):
    return TranslationMonitor(
        stream=stream,
        enabled=enabled,
        capacity=capacity,
        palette=Palette({}, enabled=False),
    )


class MonitorTests(unittest.TestCase):
    def test_non_tty_only_prints_compact_summary(self) -> None:
        stream = io.StringIO()
        monitor = make_monitor(stream)
        monitor.record(stat("one", "一", 10))
        monitor.record(stat("two", "二", 20))
        summary = monitor.finish()
        text = stream.getvalue()
        self.assertNotIn("\x1b[", text)
        self.assertEqual(summary, text.rstrip("\n"))
        self.assertRegex(
            summary,
            r"^2 段 · 2 字 · \d{2}分\d{2}秒 · "
            r"\d+\.\d{2} tok/s · \d+\.\d{2} 字/秒$",
        )
        for phrase in VERBOSE_PHRASES:
            self.assertNotIn(phrase, text)

    def test_finish_summary_is_single_line(self) -> None:
        stream = io.StringIO()
        monitor = make_monitor(stream)
        monitor.record(stat("one", "一"))
        summary = monitor.finish()
        self.assertNotIn("\n", summary)
        self.assertEqual(len(summary.splitlines()), 1)

    def test_multiple_monitors_are_independent(self) -> None:
        first_stream = io.StringIO()
        second_stream = io.StringIO()
        first = make_monitor(first_stream)
        second = make_monitor(second_stream)
        first.record(stat("a", "一", 7))
        second.record(stat("b", "二", 3))
        second.record(stat("c", "三", 4))
        first_summary = first.finish()
        second_summary = second.finish()
        self.assertTrue(first_summary.startswith("1 段 · 1 字"))
        self.assertTrue(second_summary.startswith("2 段 · 2 字"))
        self.assertNotIn("2 段", first_stream.getvalue())

    def test_reset_starts_new_independent_batch(self) -> None:
        stream = io.StringIO()
        monitor = make_monitor(stream)
        monitor.record(stat("one", "一", 10))
        first = monitor.finish()
        monitor.reset()
        monitor.record(stat("two", "二三", 20))
        second = monitor.finish()
        self.assertTrue(first.startswith("1 段 · 1 字"))
        self.assertTrue(second.startswith("1 段 · 2 字"))
        self.assertEqual(second.count("段"), 1)

    def test_ring_buffer_keeps_latest_n(self) -> None:
        stream = io.StringIO()
        monitor = TranslationMonitor(
            stream=stream,
            enabled=True,
            capacity=2,
            palette=Palette({}, enabled=False),
        )
        with mock.patch(
            "vanitas.monitor.shutil.get_terminal_size",
            return_value=os.terminal_size((80, 24)),
        ):
            monitor.record(stat("one", "一"))
            monitor.record(stat("two", "二"))
            monitor.record(stat("three", "三"))
        text = stream.getvalue()
        self.assertIn("\x1b[", text)
        self.assertIn("[02]", text)
        self.assertIn("[03]", text)
        self.assertEqual(
            [row.split("]")[0] + "]" for row in monitor._rows()],
            ["[02]", "[03]"],
        )
        monitor.finish()

    def test_tty_panel_clips_to_width_and_height(self) -> None:
        stream = io.StringIO()
        monitor = make_monitor(stream, enabled=True, capacity=10)
        long_source = "原" * 80
        long_translation = "译" * 80
        with mock.patch(
            "vanitas.monitor.shutil.get_terminal_size",
            return_value=os.terminal_size((40, 5)),
        ):
            for index in range(8):
                monitor.record(stat(f"{index}{long_source}", long_translation))
        self.assertLessEqual(monitor._drawn, 3)
        visible = ANSI_RE.sub("", stream.getvalue()).replace("\r", "")
        self.assertNotIn("\n\n", visible)
        for line in visible.splitlines():
            if line:
                self.assertLessEqual(_display_width(line), 39)

    def test_print_line_clears_panel_without_blank_lines(self) -> None:
        stream = io.StringIO()
        monitor = make_monitor(stream, enabled=True, capacity=2)
        with mock.patch(
            "vanitas.monitor.shutil.get_terminal_size",
            return_value=os.terminal_size((80, 24)),
        ):
            monitor.record(stat("one", "一"))
            monitor.print_line("✓ file.srt → file.zh.srt · 1 段")
            monitor.finish()
        text = stream.getvalue()
        self.assertIn("file.zh.srt", text)
        visible = ANSI_RE.sub("", text)
        self.assertNotIn("\n\n", visible)
        self.assertEqual(monitor._drawn, 0)

    def test_clear_is_idempotent(self) -> None:
        stream = io.StringIO()
        monitor = make_monitor(stream, enabled=True)
        with mock.patch(
            "vanitas.monitor.shutil.get_terminal_size",
            return_value=os.terminal_size((80, 24)),
        ):
            monitor.record(stat("one", "一"))
        monitor.clear()
        monitor.clear()
        self.assertEqual(monitor._drawn, 0)
        self.assertEqual(monitor.segments, 1)

    def test_non_tty_never_emits_draw_or_ansi(self) -> None:
        stream = io.StringIO()
        monitor = make_monitor(stream, enabled=False)
        monitor.record(stat("one", "一"))
        monitor.print_line("line")
        monitor.finish()
        text = stream.getvalue()
        self.assertIn("line", text)
        self.assertNotIn("\x1b[", text)

    def test_non_tty_print_line_strips_caller_ansi(self) -> None:
        stream = io.StringIO()
        monitor = TranslationMonitor(
            stream=stream,
            enabled=False,
            palette=Palette(DEFAULT_THEME, enabled=True),
        )
        monitor.print_line("\x1b[32mok\x1b[0m")
        self.assertEqual(stream.getvalue(), "ok\n")


if __name__ == "__main__":
    unittest.main()
