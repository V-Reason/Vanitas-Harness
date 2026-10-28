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
from vanitas.monitor import (  # noqa: E402
    FileReport,
    TranslationMonitor,
    TranslationReport,
    _display_width,
)
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

    def test_begin_file_draws_header_with_index_and_progress(self) -> None:
        stream = io.StringIO()
        monitor = make_monitor(stream, enabled=True)
        with mock.patch(
            "vanitas.monitor.shutil.get_terminal_size",
            return_value=os.terminal_size((80, 24)),
        ):
            monitor.begin_file(2, 6, "a.srt")
        text = ANSI_RE.sub("", stream.getvalue()).replace("\r", "")
        self.assertIn("[2/6] a.srt", text)
        self.assertIn("准备中", text)
        self.assertIn("00分00秒", text)

    def test_segment_progress_updates_header(self) -> None:
        stream = io.StringIO()
        monitor = make_monitor(stream, enabled=True)
        with mock.patch(
            "vanitas.monitor.shutil.get_terminal_size",
            return_value=os.terminal_size((80, 24)),
        ):
            monitor.begin_file(1, 3, "a.srt")
            monitor.set_segment_progress(12, 56)
        text = ANSI_RE.sub("", stream.getvalue()).replace("\r", "")
        self.assertIn("[1/3] a.srt", text)
        self.assertIn("12/56 段 · 21%", text)

    def test_begin_file_header_respects_terminal_width(self) -> None:
        stream = io.StringIO()
        monitor = make_monitor(stream, enabled=True)
        with mock.patch(
            "vanitas.monitor.shutil.get_terminal_size",
            return_value=os.terminal_size((40, 8)),
        ):
            monitor.begin_file(1, 2, "很长的文件名" * 10)
        visible = ANSI_RE.sub("", stream.getvalue()).replace("\r", "")
        for line in visible.splitlines():
            if line:
                self.assertLessEqual(_display_width(line), 39)

    def test_clear_collapses_panel_without_blank_gap(self) -> None:
        stream = io.StringIO()
        monitor = make_monitor(stream, enabled=True, capacity=5)
        with mock.patch(
            "vanitas.monitor.shutil.get_terminal_size",
            return_value=os.terminal_size((80, 24)),
        ):
            monitor.begin_file(1, 2, "a.srt")
            monitor.record(stat("one", "一"))
            monitor.print_line("✓ done")
        visible = ANSI_RE.sub("", stream.getvalue()).replace("\r", "")
        self.assertIn("✓ done", visible)
        # 擦除后必须回到面板顶部并清到屏幕末尾；否则完成行前会留下 N 行空白。
        self.assertIn("\x1b[J", stream.getvalue())
        self.assertEqual(monitor._drawn, 0)

    def test_non_tty_begin_file_prints_one_plain_line(self) -> None:
        stream = io.StringIO()
        monitor = make_monitor(stream, enabled=False)
        monitor.begin_file(1, 3, "a.srt")
        self.assertEqual(
            stream.getvalue(), "· [1/3] 正在翻译 a.srt\n"
        )
        self.assertNotIn("\x1b[", stream.getvalue())

    def test_report_renders_success_partial_and_failure(self) -> None:
        report = TranslationReport()
        report.add(
            FileReport(
                index=1,
                total=3,
                label="a.srt",
                output="a.zh.srt",
                segments=10,
                elapsed_s=2.0,
                tokens=20,
                zh_chars=30,
            )
        )
        report.add(
            FileReport(
                index=2,
                total=3,
                label="b.srt",
                output="—",
                segments=0,
                elapsed_s=1.0,
                error="写入失败",
            )
        )
        report.add(
            FileReport(
                index=3,
                total=3,
                label="c.srt",
                output="c.zh.srt",
                segments=5,
                elapsed_s=1.0,
                failures=("line 2：空内容",),
                overwrote=True,
            )
        )
        text = "\n".join(
            report.render(
                Palette({}, enabled=False),
                width=100,
                elapsed_s=4.0,
                out_dir=Path("out"),
            )
        )
        self.assertIn("翻译完成 · 3 个文件 · 15 段 · 30 字", text)
        self.assertIn("成功 1 · 部分失败 1 · 失败 1 · 中断 0 · 覆盖 1", text)
        self.assertIn("1 段失败", text)
        self.assertIn("失败明细", text)
        self.assertIn("line 2：空内容", text)
        self.assertIn("写入失败", text)
        self.assertIn("输出：out", text)

    def test_report_compact_mode_hides_success_rows_over_limit(self) -> None:
        report = TranslationReport()
        for index in range(1, 13):
            report.add(
                FileReport(
                    index=index,
                    total=13,
                    label=f"{index}.srt",
                    output=f"{index}.zh.srt",
                    segments=1,
                    elapsed_s=0.1,
                )
            )
        report.add(
            FileReport(
                index=13,
                total=13,
                label="13.srt",
                output="13.zh.srt",
                segments=1,
                elapsed_s=0.1,
                error="boom",
            )
        )
        text = "\n".join(
            report.render(Palette({}, enabled=False), width=100)
        )
        self.assertNotIn("[1/13]", text)
        self.assertIn("[13/13]", text)
        self.assertIn("-v 查看全部 13", text)
        verbose = "\n".join(
            report.render(
                Palette({}, enabled=False), width=100, verbose=True
            )
        )
        self.assertIn("[1/13]", verbose)


if __name__ == "__main__":
    unittest.main()
