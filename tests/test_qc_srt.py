"""QC 脚本：结构 error、语义 actionable 与静默 auto 的只读检查。"""

from __future__ import annotations

import contextlib
import io
import json
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

    def codes(self, result: qc_srt.FileQC) -> set[str]:
        return {item.code for item in result.issues}

    def test_clean_pair_has_no_errors(self) -> None:
        source = srt("こんにちは", "さようなら")
        result = self.pair(
            source,
            source.replace("こんにちは", "你好").replace("さようなら", "再见"),
        )
        self.assertEqual(result.errors, [])
        self.assertEqual(result.warnings, [])
        self.assertEqual(result.cues, 2)
        self.assertEqual(result.silent["traditional"], 0)
        self.assertEqual(result.silent["leading_punct"], 0)

    def test_cue_count_mismatch_is_error(self) -> None:
        result = self.pair(srt("一", "二"), srt("一"))
        self.assertTrue(any(item.code == "cue_count" for item in result.errors))

    def test_index_and_timing_are_errors(self) -> None:
        translated = (
            "1\n00:00:01,000 --> 00:00:02,999\n译一\n\n"
            "3\n00:00:03,000 --> 00:00:04,000\n译二\n"
        )
        result = self.pair(srt("一", "二"), translated)
        codes = self.codes(result)
        self.assertIn("index", codes)
        self.assertIn("timing", codes)

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

    def test_duplicate_onomatopoeia_is_not_error(self) -> None:
        result = self.pair(
            srt("カリッ カリッ カリッ", "カリ カリ カリ"),
            srt("卡利……卡利……卡利……", "卡利……卡利……卡利……"),
        )
        self.assertFalse(any(item.code == "duplicate_translation" for item in result.errors))

    def test_duplicate_source_substring_is_not_error(self) -> None:
        result = self.pair(
            srt("好き好きとは言ってないけど", "好き好き"),
            srt("虽然没说喜欢喜欢啦。", "虽然没说喜欢喜欢啦。"),
        )
        self.assertFalse(any(item.code == "duplicate_translation" for item in result.errors))

    def test_duplicate_near_synonym_sources_are_not_error(self) -> None:
        result = self.pair(
            srt("挿入のイメージ", "お挿入のイメージなんだけど"),
            srt("就是插入的印象啦。", "是关于插入的印象啦。"),
        )
        self.assertFalse(any(item.code == "duplicate_translation" for item in result.errors))

        result = self.pair(
            srt("えーとか言って", "えぇーって"),
            srt("说着“呃——”之类的。", "说着“诶——”之类的。"),
        )
        self.assertFalse(any(item.code == "duplicate_translation" for item in result.errors))

    def test_latin_artifact_is_actionable(self) -> None:
        result = self.pair(srt("セックスしたい"), srt("做SEXPAR啦"))
        self.assertIn("latin", self.codes(result))
        result = self.pair(srt("トラック"), srt("Track 语"))
        self.assertIn("latin", self.codes(result))
        result = self.pair(srt("ASMR"), srt("ASMR 和 NG"))
        self.assertNotIn("latin", self.codes(result))

    def test_slash_alternative_is_actionable(self) -> None:
        result = self.pair(srt("吉田さん"), srt("吉田先生/女士"))
        self.assertIn("slash", self.codes(result))
        result = self.pair(srt("半分流"), srt("一半/二分之一"))
        self.assertIn("slash", self.codes(result))
        result = self.pair(srt("分数"), srt("1/2"))
        self.assertNotIn("slash", self.codes(result))

    def test_auto_normalizable_issues_are_silent(self) -> None:
        result = self.pair(
            srt("これは最初の長い台詞です", "これは二番目の長い台詞です"),
            srt("這是繁體", "，开头有标点"),
        )
        self.assertEqual(result.errors, [])
        self.assertEqual(result.warnings, [])
        self.assertNotIn("traditional", self.codes(result))
        self.assertNotIn("leading_punct", self.codes(result))
        self.assertEqual(result.silent["traditional"], 1)
        self.assertEqual(result.silent["leading_punct"], 1)

    def test_kana_and_blank_are_actionable(self) -> None:
        result = self.pair(srt("これは長い台詞です"), srt("まだ仮名"))
        self.assertIn("kana", self.codes(result))
        result = self.pair(srt("これは長い台詞です"), srt(""))
        self.assertIn("blank", self.codes(result))

    def test_asr_suspect_term_is_actionable(self) -> None:
        result = self.pair(srt("弁当向かってる"), srt("便当正在过去"))
        self.assertIn("asr_term", self.codes(result))

    def test_honorific_actionable_and_polite_exempt(self) -> None:
        result = self.pair(srt("ほら、見て…"), srt("您看吧……"))
        self.assertIn("honorific", self.codes(result))

        result = self.pair(srt("お騒がせしました"), srt("给您添麻烦了"))
        self.assertNotIn("honorific", self.codes(result))

        result = self.pair(srt("お客様、こちらです"), srt("您这边请"))
        self.assertNotIn("honorific", self.codes(result))

    def test_unauthorized_staging_is_actionable(self) -> None:
        result = self.pair(srt("好きだよ"), srt("（在耳后）喜欢你哦"))
        self.assertIn("staging", self.codes(result))

        result = self.pair(
            srt("(両耳) 好きだよ"), srt("（在双耳边）喜欢你哦")
        )
        self.assertNotIn("staging", self.codes(result))

    def test_length_anomaly_is_actionable(self) -> None:
        translated = "这是一个" + "非常" * 20 + "长的译文"
        result = self.pair(srt("これはとても短い台詞です"), srt(translated))
        self.assertIn("length", self.codes(result))

    def test_missing_pair_is_error(self) -> None:
        (self.dir / "a.zh.srt").write_text(srt("你好"), encoding="utf-8")
        pairs, problems = qc_srt.collect_pairs(self.dir, self.dir)
        self.assertEqual(pairs, [])
        self.assertTrue(
            any(
                item.code == "missing_source"
                for problem in problems
                for item in problem.errors
            )
        )

    def test_json_summary_and_flagged_cues(self) -> None:
        (self.dir / "a.srt").write_text(srt("ほら、見て…"), encoding="utf-8")
        (self.dir / "a.zh.srt").write_text(srt("您看吧……"), encoding="utf-8")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = qc_srt.main(
                [
                    "--source-dir",
                    str(self.dir),
                    "--zh-dir",
                    str(self.dir),
                    "--json",
                ]
            )
        payload = json.loads(out.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(payload["summary"]["files"], 1)
        self.assertEqual(payload["summary"]["errors"], 0)
        self.assertGreaterEqual(payload["summary"]["actionable"], 1)
        self.assertTrue(payload["summary"]["passed"])
        self.assertIn("files", payload)
        self.assertTrue(
            any(
                item["code"] == "honorific"
                for item in payload["flagged_cues"]
            )
        )

    def test_strict_exit_code(self) -> None:
        (self.dir / "a.srt").write_text(srt("ほら、見て…"), encoding="utf-8")
        (self.dir / "a.zh.srt").write_text(srt("您看吧……"), encoding="utf-8")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            default_code = qc_srt.main(
                ["--source-dir", str(self.dir), "--zh-dir", str(self.dir)]
            )
        with contextlib.redirect_stdout(io.StringIO()):
            strict_code = qc_srt.main(
                [
                    "--source-dir",
                    str(self.dir),
                    "--zh-dir",
                    str(self.dir),
                    "--strict",
                ]
            )
        self.assertEqual(default_code, 0)
        self.assertEqual(strict_code, 1)

    def test_structural_error_exit_one(self) -> None:
        (self.dir / "a.srt").write_text(srt("一", "二"), encoding="utf-8")
        (self.dir / "a.zh.srt").write_text(srt("一"), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            code = qc_srt.main(
                ["--source-dir", str(self.dir), "--zh-dir", str(self.dir)]
            )
        self.assertEqual(code, 1)

    def test_report_is_compact_diff(self) -> None:
        result = self.pair(srt("ほら、見て…"), srt("您看吧……"))
        report = qc_srt.render_report([result])
        self.assertIn("⚠ 语义疑似点", report)
        self.assertIn("- 原文", report)
        self.assertIn("+ 译文", report)
        self.assertIn("? ", report)
        self.assertNotIn("|---", report)


if __name__ == "__main__":
    unittest.main()
