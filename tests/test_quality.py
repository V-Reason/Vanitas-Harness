"""确定性质量守卫：术语表、语域/人称、文档级重复与审计。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vanitas.errors import TranslateError  # noqa: E402
from vanitas.quality import (  # noqa: E402
    DuplicateIndex,
    Glossary,
    TranslationProfile,
    apply_pronoun_policy,
    audit_duplicate,
    audit_honorific,
    detect_register,
    normalize_compare,
)

LONG_A = "这是完全相同的一条长译文"
LONG_B = "这是完全相同的一条长译文啊"
CROSS_A = "今天真的非常开心，发生了很多事。"
CROSS_B = "今天真的非常开心呢，发生了很多事。"


class GlossaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def write(self, text: str) -> Path:
        path = self.dir / "asmr.tsv"
        path.write_bytes(text.encode("utf-8"))
        return path

    def test_missing_and_empty_glossary_are_noops(self) -> None:
        missing = Glossary.load(self.dir / "nope.tsv")
        self.assertEqual(missing.entries, ())
        self.assertEqual(missing.apply("アマアマ", "业余"), "业余")

        empty = Glossary.load(self.write("\n# 只有注释\n\n"))
        self.assertEqual(empty.entries, ())
        self.assertEqual(empty.apply("アマアマ", "业余"), "业余")

    def test_bad_line_reports_file_and_line(self) -> None:
        path = self.write("# 注释\n只有一列\n")
        with self.assertRaises(TranslateError) as ctx:
            Glossary.load(path)
        self.assertIn(f"{path}:2:", str(ctx.exception))

        path = self.write("アマアマ\t\t业余\n")
        with self.assertRaises(TranslateError) as ctx:
            Glossary.load(path)
        self.assertIn(f"{path}:1:", str(ctx.exception))
        self.assertIn("preferred", str(ctx.exception))

    def test_load_parses_forbidden_pipe_list(self) -> None:
        glossary = Glossary.load(
            self.write("アマアマ\t甜腻\t业余水平|业余\n")
        )
        self.assertEqual(len(glossary.entries), 1)
        entry = glossary.entries[0]
        self.assertEqual(entry.source, "アマアマ")
        self.assertEqual(entry.preferred, "甜腻")
        self.assertEqual(entry.forbidden, ("业余水平", "业余"))

    def test_longest_forbidden_wins(self) -> None:
        glossary = Glossary.load(
            self.write("アマアマ\t甜腻\t业余水平|业余\n")
        )
        self.assertEqual(
            glossary.apply("アマアマな声", "业余水平的演出"), "甜腻的演出"
        )
        self.assertEqual(glossary.apply("アマアマな声", "有点业余"), "有点甜腻")

    def test_preferred_not_present_is_not_an_error(self) -> None:
        glossary = Glossary.load(
            self.write("アマアマ\t甜腻\t业余水平|业余\n")
        )
        self.assertEqual(
            glossary.apply("アマアマな声", "甜美的声音"), "甜美的声音"
        )
        # 源文未命中术语时，即使译文含 forbidden 也不动：避免误伤普通词。
        self.assertEqual(glossary.apply("普通の声", "有点业余"), "有点业余")

    def test_context_lists_matching_entries_only(self) -> None:
        glossary = Glossary.load(
            self.write("アマアマ\t甜腻\t业余水平|业余\n")
        )
        self.assertIn("アマアマ→甜腻", glossary.context("アマアマな声"))
        self.assertEqual(glossary.context("普通の声"), "")
        self.assertEqual(glossary.context(""), "")


class RegisterTests(unittest.TestCase):
    def test_service_register(self) -> None:
        self.assertEqual(
            detect_register(["いらっしゃいませ、お客様"]), "service"
        )
        self.assertEqual(
            detect_register(["ご予約ありがとうございます", "本日の施術です"]),
            "service",
        )

    def test_intimate_register(self) -> None:
        self.assertEqual(detect_register(["先輩、好きです"]), "intimate")
        self.assertEqual(
            detect_register(["膝枕してあげるね", "添い寝しようか"]),
            "intimate",
        )

    def test_neutral_when_no_evidence(self) -> None:
        self.assertEqual(detect_register(["こんにちは、いい天気ですね"]), "neutral")
        self.assertEqual(detect_register([]), "neutral")

    def test_neutral_when_evidence_is_insufficient(self) -> None:
        self.assertEqual(detect_register(["お客様"]), "neutral")
        self.assertEqual(detect_register(["好き"]), "neutral")

    def test_neutral_when_scores_tie(self) -> None:
        self.assertEqual(detect_register(["先輩、お客様"]), "neutral")

    def test_high_side_wins_on_clear_margin(self) -> None:
        self.assertEqual(
            detect_register(["いらっしゃいませ", "お客様", "ご予約", "好き"]),
            "service",
        )
        # 同一词重复出现不累积，只统计命中的不同词。
        self.assertEqual(
            detect_register(["好き好き好き", "お客様"]), "neutral"
        )


class PronounPolicyTests(unittest.TestCase):
    def test_intimate_replaces_nin(self) -> None:
        profile = TranslationProfile(register="intimate", pronoun="你")
        self.assertEqual(apply_pronoun_policy("您请坐", profile), "你请坐")

    def test_service_and_neutral_keep_nin(self) -> None:
        for register in ("service", "neutral"):
            with self.subTest(register=register):
                profile = TranslationProfile(register=register, pronoun="auto")
                self.assertEqual(apply_pronoun_policy("您请坐", profile), "您请坐")

    def test_intimate_auto_does_not_touch_text(self) -> None:
        profile = TranslationProfile(register="intimate", pronoun="auto")
        self.assertEqual(apply_pronoun_policy("您请坐", profile), "您请坐")


class DuplicateIndexTests(unittest.TestCase):
    def test_exact_duplicate_is_detected(self) -> None:
        index = DuplicateIndex()
        index.add(1, "今日は楽しかったです", LONG_A)
        self.assertEqual(index.find("明日の予定を確認します", LONG_A), 1)

    def test_near_duplicate_is_detected(self) -> None:
        index = DuplicateIndex()
        index.add(1, "今日は楽しかったです", CROSS_A)
        self.assertEqual(index.find("まったく別の長い台詞です", CROSS_B), 1)

    def test_similar_source_is_legitimate_repeat(self) -> None:
        index = DuplicateIndex()
        index.add(1, "今日は楽しかったです", LONG_A)
        self.assertIsNone(index.find("今日は楽しかったです", LONG_A))
        self.assertIsNone(
            index.find("今日は楽しかったですね", LONG_A)
        )

    def test_short_translation_is_skipped(self) -> None:
        index = DuplicateIndex()
        index.add(1, "好き", "喜欢你")
        self.assertIsNone(index.find("大好き", "喜欢你"))
        self.assertIsNone(index.find("別の原文", "短い"))

    def test_hits_item_beyond_eighty_cues(self) -> None:
        index = DuplicateIndex()
        for cue in range(1, 80):
            index.add(cue, f"source {cue}", f"第{cue}条独立的测试译文内容")
        index.add(80, "source 80", LONG_A)
        self.assertEqual(index.find("different source", LONG_A), 80)
        self.assertEqual(len(index), 80)

    def test_candidates_ranked_by_shared_trigrams(self) -> None:
        query = "甲乙丙丁戊己庚辛壬癸子寅"
        # cue 1 共享 9 个 trigram，相似度 0.9；cue 2 相似度更高但只共享 8 个。
        index = DuplicateIndex()
        index.add(1, "source one", "甲乙丙丁戊己庚辛壬癸子丑")
        index.add(2, "source two", "甲乙丙丁戊己庚辛壬癸卯寅")
        self.assertEqual(index.find("query source", query), 1)

    def test_equal_scores_tie_break_on_smaller_cue(self) -> None:
        translation = "甲乙丙丁戊己庚辛壬癸子丑"
        index = DuplicateIndex()
        index.add(9, "source nine", translation)
        index.add(4, "source four", translation)
        self.assertEqual(
            index.find("query source", "甲乙丙丁戊己庚辛壬癸子寅"), 4
        )

    def test_string_cue_round_trips_and_sorts_numerically(self) -> None:
        translation = "甲乙丙丁戊己庚辛壬癸子丑"
        index = DuplicateIndex()
        index.add("100", "source hundred", translation)
        index.add("9", "source nine", translation)
        # 同分按 cue 数值排序（9 < 100），返回值保留调用方的字符串类型。
        self.assertEqual(
            index.find("query source", "甲乙丙丁戊己庚辛壬癸子寅"), "9"
        )

    def test_reset_clears_index(self) -> None:
        index = DuplicateIndex()
        index.add(1, "今日は楽しかったです", LONG_A)
        index.reset()
        self.assertEqual(len(index), 0)
        self.assertIsNone(index.find("別の原文", LONG_A))

    def test_normalize_compare_keeps_kana_marks(self) -> None:
        self.assertEqual(normalize_compare("カリッ、 カー。"), "カリッカー")


class AuditTests(unittest.TestCase):
    def test_duplicate_audit_hint_points_to_previous_cue(self) -> None:
        index = DuplicateIndex()
        index.add(7, "今日は楽しかったです", LONG_A)
        issues = audit_duplicate("別の原文です", LONG_A, 12, index)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "duplicate_translation")
        self.assertEqual(issues[0].cue, 12)
        self.assertIn("第 7 条", issues[0].hint)
        self.assertIn("禁止复用", issues[0].hint)

    def test_duplicate_audit_is_empty_for_clean_cue(self) -> None:
        index = DuplicateIndex()
        index.add(1, "今日は楽しかったです", LONG_A)
        self.assertEqual(
            audit_duplicate("別の原文です", "完全不同的另一段译文内容", 2, index),
            [],
        )

    def test_honorific_audit_only_for_intimate(self) -> None:
        intimate = TranslationProfile(register="intimate", pronoun="你")
        issues = audit_honorific("您辛苦了", intimate, 3)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "honorific")
        self.assertEqual(issues[0].hint, "本作亲密关系，统一用你")

        for register in ("service", "neutral"):
            with self.subTest(register=register):
                profile = TranslationProfile(register=register, pronoun="auto")
                self.assertEqual(audit_honorific("您辛苦了", profile, 3), [])

        self.assertEqual(audit_honorific("你辛苦了", intimate, 3), [])


if __name__ == "__main__":
    unittest.main()
