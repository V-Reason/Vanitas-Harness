"""QC 脚本：结构 error 与语言 warn 的只读检查。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import qc_srt  # noqa: E402


def srt(*cues: str) -> str:
    blocks = []
    for index, text in enumerate(cues, start=1):
        start = f"00:00:{index * 2 - 1:02d},000"
        end = f"00:00:{index * 2:02d},000"
        blocks.append(f"{index}\n{start} --> {end}\n{text}")
    return "\n\n".join(blocks) + "\n"


class QcSrtTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def pair(self, source: str, translated: str) -> qc_srt.FileQC:
        src = self.dir / "a.srt"
        zh = self.dir / "a.zh.srt"
        src.write_text(source, encoding="utf-8")
        zh.write_text(translated, encoding="utf-8")
        return qc_srt.check_pair(src, zh)

    def test_clean_pair_has_no_errors(self) -> None:
        source = srt("こんにちは", "さようなら")
        result = self.pair(source, source.replace("こんにちは", "你好").replace("さようなら", "再见"))
        self.assertEqual(result.errors, [])
        self.assertEqual(result.cues, 2)

    def test_cue_count_mismatch_is_error(self) -> None:
        result = self.pair(srt("一", "二"), srt("一"))
        self.assertTrue(any(item.code == "cue_count" for item in result.errors))

    def test_window_id_is_error(self) -> None:
        result = self.pair(srt("一"), srt("译文 [#12]"))
        self.assertTrue(any(item.code == "window_id" for item in result.errors))

    def test_duplicate_translation_for_distinct_sources_is_error(self) -> None:
        result = self.pair(
            srt("今日はとても楽しかったです", "明日の予定を確認します"),
            srt("这是完全相同的一条长译文", "这是完全相同的一条长译文"),
        )
        self.assertTrue(
            any(item.code == "duplicate_translation" for item in result.errors)
        )

    def test_language_and_format_warnings(self) -> None:
        result = self.pair(
            srt("これは最初の長い台詞です", "これは二番目の長い台詞です"),
            srt("這是繁體", "，开头有标点"),
        )
        codes = {item.code for item in result.warnings}
        self.assertIn("traditional", codes)
        self.assertIn("leading_punct", codes)

    def test_blank_and_kana_warnings(self) -> None:
        result = self.pair(srt("これは長い台詞です"), srt("まだ仮名"))
        codes = {item.code for item in result.warnings}
        self.assertIn("kana", codes)
        result = self.pair(srt("これは長い台詞です"), srt(""))
        self.assertIn("blank", {item.code for item in result.warnings})


if __name__ == "__main__":
    unittest.main()
