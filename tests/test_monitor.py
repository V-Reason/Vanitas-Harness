"""monitor：5 段环形看板、终态统计与非 TTY 降级。"""

from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vanitas.metrics import SegmentStat  # noqa: E402
from vanitas.monitor import TranslationMonitor  # noqa: E402
from vanitas.ui import Palette  # noqa: E402


def stat(source: str, translation: str, tokens: int = 10) -> SegmentStat:
    return SegmentStat(
        source=source,
        translation=translation,
        elapsed_s=1.0,
        tokens=tokens,
        tps=10.0,
        tps_source="server.timings",
    )


class MonitorTests(unittest.TestCase):
    def test_non_tty_only_prints_summary(self) -> None:
        stream = io.StringIO()
        monitor = TranslationMonitor(
            stream=stream,
            enabled=False,
            capacity=5,
            palette=Palette({}, enabled=False),
        )
        monitor.record(stat("one", "一", 10))
        monitor.record(stat("two", "二", 20))
        summary = monitor.finish()
        text = stream.getvalue()
        self.assertNotIn("\x1b[", text)
        self.assertIn("翻译任务已完成", text)
        self.assertIn("统计段数：2 段", text)
        self.assertIn("累计汉字：2 字", text)
        self.assertIn("平均速率", text)
        self.assertIn("吞吐性能", text)
        self.assertEqual(summary, text.rstrip("\n"))

    def test_ring_buffer_keeps_latest_n(self) -> None:
        stream = io.StringIO()
        monitor = TranslationMonitor(
            stream=stream,
            enabled=True,
            capacity=2,
            palette=Palette({}, enabled=False),
        )
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

    def test_print_line_clears_panel(self) -> None:
        stream = io.StringIO()
        monitor = TranslationMonitor(
            stream=stream,
            enabled=True,
            capacity=2,
            palette=Palette({}, enabled=False),
        )
        monitor.record(stat("one", "一"))
        monitor.print_line("✓ file.srt → file.zh.srt · 1 段")
        text = stream.getvalue()
        self.assertIn("file.zh.srt", text)
        monitor.finish()


if __name__ == "__main__":
    unittest.main()
