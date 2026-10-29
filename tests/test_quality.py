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
    AuditIssue,
    DuplicateIndex,
    DuplicateMatch,
    Glossary,
    TranslationProfile,
    apply_pronoun_policy,
    audit_customer_direction,
    audit_duplicate,
    audit_first_person,
    audit_honorific,
    audit_negative_question,
    audit_quote_private,
    audit_tag_question,
    audit_translation,
    combine_hints,
    detect_register,
    normalize_compare,
    repair_first_person_name,
    repair_quote_private,
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


class DuplicateMatchTests(unittest.TestCase):
    def test_find_match_returns_previous_item_and_find_delegates(self) -> None:
        index = DuplicateIndex()
        index.add(7, "今日は楽しかったです", LONG_A)
        match = index.find_match("別の原文です", LONG_A)
        self.assertIsInstance(match, DuplicateMatch)
        self.assertEqual(match.cue, 7)
        self.assertEqual(match.source, "今日は楽しかったです")
        self.assertEqual(match.translation, LONG_A)
        self.assertEqual(index.find("別の原文です", LONG_A), 7)
        self.assertIsNone(index.find_match("別の原文です", LONG_A, cue=7))

    def test_duplicate_hint_contains_both_source_snippets(self) -> None:
        previous_source = "あ" * 30
        current_source = "い" * 30
        index = DuplicateIndex()
        index.add(7, previous_source, LONG_A)
        issues = audit_duplicate(current_source, LONG_A, 12, index)
        self.assertEqual(len(issues), 1)
        hint = issues[0].hint
        self.assertIn(f"该条原文「{'あ' * 24}…」", hint)
        self.assertIn(f"本条原文「{'い' * 24}…」", hint)
        self.assertIn("禁止复用，只按本条原文重译", hint)

    def test_shared_long_prefix_with_dissimilar_source_hits(self) -> None:
        source_two = "伊賀崎彩香の庭主催しております"
        translation_two = "这是由伊贺崎彩香主办的庭主活动。"
        source_eighty = "伊賀崎綾香の庭はね基本的に"
        index = DuplicateIndex()
        index.add(2, source_two, translation_two)

        # 实际 #80 译文（规范化后包含 #2 译文）必须命中。
        actual = "这是由伊贺崎彩香主办的庭主活动，基本上啊。"
        self.assertEqual(index.find_match(source_eighty, actual, 80).cue, 2)

        # 相似度掉到 0.80 以下、也不构成包含时，仍由 ≥6 字共同前缀兜底。
        variant = "这是由伊贺崎彩香主办的庭院聚会，基本上啊。"
        self.assertEqual(index.find_match(source_eighty, variant, 80).cue, 2)

    def test_shared_prefix_does_not_fire_for_similar_source(self) -> None:
        index = DuplicateIndex()
        index.add(
            1,
            "伊賀崎彩香の庭主催しております",
            "这是由伊贺崎彩香主办的庭主活动。",
        )
        # 源文相似度略高于 0.5，属于“同一句的近似写法”，不得因共同开头误报。
        self.assertIsNone(
            index.find_match(
                "伊賀崎彩香の庭は基本的に",
                "这是由伊贺崎彩香主办的庭院聚会，基本上啊。",
                80,
            )
        )

    def test_shared_prefix_shorter_than_six_does_not_fire(self) -> None:
        index = DuplicateIndex()
        index.add(
            1,
            "伊賀崎彩香の庭主催しております",
            "这是由伊贺崎彩香主办的庭主活动。",
        )
        self.assertIsNone(
            index.find_match(
                "伊賀崎綾香の庭はね基本的に",
                "这是伊贺崎彩香主持的庭院聚会，基本上啊。",
                80,
            )
        )


class NarrowAuditTests(unittest.TestCase):
    def test_customer_direction_reports_reversed_voice(self) -> None:
        for translation in (
            "我们不可以触碰客人。",
            "禁止触碰客人。",
            "请不要接触客人。",
        ):
            with self.subTest(translation=translation):
                issues = audit_customer_direction(
                    "お客様から私たち触れることは禁止となっております。",
                    translation,
                    16,
                )
                self.assertEqual(len(issues), 1)
                self.assertEqual(issues[0].code, "customer_direction")
                self.assertIn("客人→我们", issues[0].hint)

    def test_customer_direction_accepts_correct_voice(self) -> None:
        for translation in (
            "禁止客人触碰我们。",
            "客人不能触碰我们。",
            "客人不可以碰我们。",
        ):
            with self.subTest(translation=translation):
                self.assertEqual(
                    audit_customer_direction(
                        "お客様から私たち触れることは禁止となっております。",
                        translation,
                        16,
                    ),
                    [],
                )

    def test_customer_direction_requires_all_source_conditions(self) -> None:
        source = "お客様から私たち触れることは禁止となっております。"
        self.assertEqual(
            audit_customer_direction(source, "", 1), []
        )
        self.assertEqual(
            audit_customer_direction("", "我们触碰客人。", 1), []
        )
        self.assertEqual(
            audit_customer_direction(
                "お客様から私たち触れることは許されております。",
                "我们触碰客人。",
                1,
            ),
            [],
        )

    def test_quote_private_only_narrow_first_person_modes(self) -> None:
        for translation in (
            "让我叫你雫",
            "我的名字是雫",
            "让你叫我雫",
            # 叠词口误“叫我叫”是把引用句写成了第一人称，必须报。
            "说叫我叫雫就好",
        ):
            with self.subTest(translation=translation):
                issues = audit_quote_private(
                    "私のことは雫でいいって", translation, 7
                )
                self.assertEqual(len(issues), 1)
                self.assertEqual(issues[0].code, "quote_private")
                self.assertIn("引用他人", issues[0].hint)

        for translation in (
            "他说，叫我雫就好",
            "会长说叫她雫就好",
            # 单独“叫我”在语境中可能正确，保留为人工/QC residual，
            # 不触发自动重译。
        ):
            with self.subTest(translation=translation):
                self.assertEqual(
                    audit_quote_private(
                        "私のことは雫でいいって", translation, 7
                    ),
                    [],
                )
        # 没有“って”引用标记时不报，避免把普通自我介绍当引用。
        self.assertEqual(
            audit_quote_private("私のことは雫でいい", "我的名字是雫", 7),
            [],
        )
        self.assertEqual(audit_quote_private("", "", 7), [])

    def test_repair_quote_private_only_for_narrow_quote_source(self) -> None:
        source = "私のことは雫でいいって"
        self.assertEqual(
            repair_quote_private(source, "说叫我叫雫就好"),
            "说叫我雫就好",
        )
        # 普通“叫我”在语境中可能正确，不得顺手改写。
        self.assertEqual(
            repair_quote_private(source, "他说，叫我雫就好"),
            "他说，叫我雫就好",
        )
        # 源文没有引用标记“って”时原样返回，避免改普通自我介绍。
        self.assertEqual(
            repair_quote_private("私のことは雫でいい", "说叫我叫雫就好"),
            "说叫我叫雫就好",
        )
        # 空输入原样返回。
        self.assertEqual(repair_quote_private("", ""), "")
        self.assertEqual(repair_quote_private(source, ""), "")
        self.assertEqual(
            repair_quote_private("", "说叫我叫雫就好"), "说叫我叫雫就好"
        )

    def test_negative_question_keeps_polarity(self) -> None:
        source = "私のこと、覚えていませんか?"
        issues = audit_negative_question(source, "你还记得我吗？", 12)
        self.assertEqual([issue.code for issue in issues], ["negative_question"])
        self.assertIn("否定", issues[0].hint)
        self.assertEqual(
            audit_negative_question(source, "你不记得我了吗？", 12), []
        )
        self.assertEqual(
            audit_negative_question(source, "难道你忘了我吗？", 12), []
        )
        # 「〜ませんか」是邀请，不是持续体否定疑问，不报。
        self.assertEqual(
            audit_negative_question("一緒に行きませんか", "要不要一起去？", 1),
            [],
        )
        self.assertEqual(audit_negative_question("", "", 1), [])

    def test_first_person_requires_absent_source_pronoun_and_cue_word(self) -> None:
        issues = audit_first_person(
            "名前を呼んでください", "请叫我的名字", 5
        )
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "first_person")
        self.assertIn("不是第一人称", issues[0].hint)

        self.assertEqual(
            audit_first_person("私の名前を呼んで", "请叫我的名字", 5), []
        )
        self.assertEqual(
            audit_first_person("名前を呼んで", "请叫他的名字", 5), []
        )
        self.assertEqual(
            audit_first_person("名前を呼んで", "请让我来", 5)[0].code,
            "first_person",
        )
        self.assertEqual(audit_first_person("", "", 5), [])

    def test_repair_first_person_name_only_for_narrow_source(self) -> None:
        source = "ちゃんと下の名前で呼んであげてよ"
        self.assertEqual(
            repair_first_person_name(
                source, "请用我的名字好好叫我哦"
            ),
            "请用她的名字好好叫她哦",
        )
        self.assertEqual(
            repair_first_person_name(
                source, "请用她的名字好好叫她哦"
            ),
            "请用她的名字好好叫她哦",
        )
        # 不含固定源短语或源文已有第一人称时不得改写。
        self.assertEqual(
            repair_first_person_name(
                "名前を呼んでください", "请用我的名字好好叫我哦"
            ),
            "请用我的名字好好叫我哦",
        )
        self.assertEqual(
            repair_first_person_name(
                "私の下の名前で呼んであげてよ", "请用我的名字好好叫我哦"
            ),
            "请用我的名字好好叫我哦",
        )

    def test_tag_question_only_when_source_has_no_tag(self) -> None:
        issues = audit_tag_question(
            "予算くださいってこと", "对吧？", 17
        )
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "tag_question")
        self.assertIn("标签问", issues[0].hint)

        self.assertEqual(
            audit_tag_question("予算くださいってこと", "也就是说要预算", 17),
            [],
        )
        self.assertEqual(
            audit_tag_question("そうですね", "对吧？", 17), []
        )
        self.assertEqual(
            audit_tag_question("予算くださいってこと", "……对吧？", 17)[0].code,
            "tag_question",
        )
        self.assertEqual(
            audit_tag_question("予算くださいってこと", "对不对", 17)[0].code,
            "tag_question",
        )
        self.assertEqual(audit_tag_question("", "", 17), [])


class AuditAggregatorTests(unittest.TestCase):
    def test_combine_hints_deduplicates_and_limits(self) -> None:
        issues = [
            AuditIssue("a", 1, "m", "提示一"),
            AuditIssue("b", 1, "m", "提示一"),
            AuditIssue("c", 1, "m", "提示二"),
            AuditIssue("d", 1, "m", "提示三"),
        ]
        self.assertEqual(combine_hints(issues), "提示一；提示二")
        self.assertEqual(combine_hints(issues, limit=0), "")
        self.assertEqual(combine_hints([]), "")

    def test_audit_translation_order_duplicate_then_semantic(self) -> None:
        translation = "我们不可以触碰客人，这不行的"
        index = DuplicateIndex()
        index.add(1, "今日は楽しかったです", translation)
        issues = audit_translation(
            "お客様から私たち触れることは禁止となっております。",
            translation,
            16,
            index=index,
            profile=TranslationProfile(register="service", pronoun="auto"),
        )
        self.assertEqual(
            [issue.code for issue in issues],
            ["duplicate_translation", "customer_direction"],
        )

    def test_audit_translation_is_empty_for_empty_inputs(self) -> None:
        self.assertEqual(
            audit_translation(
                "",
                "",
                1,
                index=DuplicateIndex(),
                profile=TranslationProfile(register="intimate", pronoun="你"),
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
