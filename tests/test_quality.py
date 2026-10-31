"""确定性质量守卫：术语表、语域/人称、文档级重复与审计。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vanitas.errors import TranslateError  # noqa: E402
from vanitas.quality import (  # noqa: E402
    ASMR_T1_CODES,
    PROMPT_EXAMPLE_NAMES,
    strip_example_names,
    AuditIssue,
    DuplicateIndex,
    DuplicateMatch,
    Glossary,
    TranslationProfile,
    apply_pronoun_policy,
    audit_address_term,
    audit_customer_direction,
    audit_duplicate,
    audit_first_person,
    audit_honorific,
    audit_invented_name,
    audit_invented_tool,
    audit_negative_question,
    audit_quote_private,
    audit_short_duplicate,
    audit_tag_question,
    audit_translation,
    audit_under_translation,
    combine_hints,
    detect_register,
    is_onomatopoeia_or_reduplication,
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


class T1ShortDuplicateTests(unittest.TestCase):
    TRANSLATION = "今天真的非常开心呀"

    def index(self) -> DuplicateIndex:
        # 与 translate.py 的第二个索引参数一致：min_chars=8 / 0.85。
        return DuplicateIndex(min_chars=8, translation_ratio=0.85)

    def test_four_cue_group_hits_three_of_three(self) -> None:
        index = self.index()
        index.add(1, "はい", self.TRANSLATION)
        hits: list[tuple[int, str]] = []
        for cue, source in ((2, "うん"), (3, "そう"), (4, "だよね")):
            issues = audit_short_duplicate(
                source, self.TRANSLATION, cue, index
            )
            self.assertEqual([issue.code for issue in issues], ["short_duplicate"])
            self.assertIn("第 1 条", issues[0].hint)
            hits.append((cue, source))
            index.add(cue, source, self.TRANSLATION)
        # 4 句同译组覆盖 4/4，后续 3 条各命中 1 次（3/3）。
        self.assertEqual([cue for cue, _ in hits], [2, 3, 4])
        self.assertEqual(len(index), 4)

    def test_same_cue_is_never_matched_against_itself(self) -> None:
        index = self.index()
        index.add(5, "はい", self.TRANSLATION)
        self.assertEqual(
            audit_short_duplicate("うん", self.TRANSLATION, 5, index), []
        )

    def test_near_synonym_pair_outside_ceiling_is_clean(self) -> None:
        """天花板之外（源文比上限更长）的近义对不得被判 short_duplicate。

        上限本身随语料演进（8→11），所以这里引用常量而不是写死数字——
        否则每次调阈值都要改测试，测试就从契约退化成绊脚石。
        """
        from vanitas.quality import SHORT_DUPLICATE_MAX_SOURCE_CHARS  # noqa: PLC0415

        long_source = "これはとても長い原文のテストですね"
        short_source = "まったく別の話題を話していますよ"
        self.assertGreater(len(short_source), SHORT_DUPLICATE_MAX_SOURCE_CHARS)
        self.assertLess(
            SequenceMatcher(None, long_source, short_source).ratio(), 0.75
        )
        index = self.index()
        index.add(1, long_source, self.TRANSLATION)
        self.assertEqual(
            audit_short_duplicate(short_source, self.TRANSLATION, 2, index), []
        )

    def test_nine_char_source_is_inside_the_ceiling_and_fires(self) -> None:
        """上限提到 11 的直接收益：9 字短句复用同一译文必须被抓到。

        真实用例：`ご無理はなさらずに`(9 字) 在 V1~V3 都跨 cue 复用了同一译文。
        """
        from vanitas.quality import SHORT_DUPLICATE_MAX_SOURCE_CHARS  # noqa: PLC0415

        short_source = "ご無理はなさらずに"
        self.assertEqual(len(short_source), 9)
        self.assertLessEqual(len(short_source), SHORT_DUPLICATE_MAX_SOURCE_CHARS)
        index = self.index()
        index.add(1, "これはとても長い原文のテストですね", self.TRANSLATION)
        issues = audit_short_duplicate(short_source, self.TRANSLATION, 2, index)
        self.assertEqual([issue.code for issue in issues], ["short_duplicate"])

    def test_shorter_source_side_at_eight_char_boundary_fires(self) -> None:
        # 契约写死“较短源文 ≤8 字”：8/9 的异质短句同译仍应命中。
        index = self.index()
        index.add(1, "甲乙丙丁戊己庚辛", self.TRANSLATION)
        issues = audit_short_duplicate(
            "子丑寅卯辰巳午未申", self.TRANSLATION, 2, index
        )
        self.assertEqual([issue.code for issue in issues], ["short_duplicate"])

    def test_translation_similarity_threshold_is_point_eight_five(self) -> None:
        index = self.index()
        index.add(1, "はい", self.TRANSLATION)
        near = "今天真的非常开心啊"
        self.assertGreaterEqual(
            SequenceMatcher(None, self.TRANSLATION, near).ratio(), 0.85
        )
        self.assertEqual(
            [
                issue.code
                for issue in audit_short_duplicate(
                    "子丑寅卯辰巳午未申", near, 2, index
                )
            ],
            ["short_duplicate"],
        )
        far = "今天真的特别高兴呀"
        self.assertLess(
            SequenceMatcher(None, self.TRANSLATION, far).ratio(), 0.85
        )
        self.assertEqual(
            audit_short_duplicate(
                "子丑寅卯辰巳午未申", far, 2, index
            ),
            [],
        )

    def test_strict_mode_ignores_containment_and_long_prefix(self) -> None:
        index = self.index()
        index.add(1, "はい", self.TRANSLATION)
        longer = self.TRANSLATION + "，不过现在有点累了"
        # 非 strict 的“包含/长共同前缀”窄规则本会命中；short_duplicate
        # 只认相同/≥0.85，必须放行。
        self.assertIsNotNone(index.find("まったく別の長い原文です", longer))
        self.assertEqual(
            audit_short_duplicate(
                "まったく別の長い原文です", longer, 2, index
            ),
            [],
        )

    def test_long_sources_or_short_translation_do_not_fire(self) -> None:
        index = self.index()
        index.add(1, "これはとても長い原文のテストですね", self.TRANSLATION)
        self.assertEqual(
            audit_short_duplicate(
                "まったく別の長い原文です", self.TRANSLATION, 2, index
            ),
            [],
        )
        short_index = self.index()
        short_index.add(1, "はい", "开心呀")
        self.assertEqual(
            audit_short_duplicate("うん", "开心呀", 2, short_index), []
        )

    def test_onomatopoeia_and_reduplication_are_exempt(self) -> None:
        for text in (
            "沙沙",
            "软绵绵",
            "コショコショ",
            "はむはむ",
            "カリッ",
            "沙沙沙沙沙沙沙沙",
        ):
            with self.subTest(text=text):
                self.assertTrue(is_onomatopoeia_or_reduplication(text))
        for text in ("ミルク", "好き", "だめ", "おかえり"):
            with self.subTest(text=text):
                self.assertFalse(is_onomatopoeia_or_reduplication(text))

        index = self.index()
        index.add(1, "コショコショ", "沙沙沙沙沙沙沙沙")
        self.assertEqual(
            audit_short_duplicate(
                "はむはむ", "沙沙沙沙沙沙沙沙", 2, index
            ),
            [],
        )

    def test_normal_katakana_long_vowel_is_not_onomatopoeia(self) -> None:
        for text in ("コーヒー", "ケーキ", "メイド", "らーめん"):
            with self.subTest(text=text):
                self.assertFalse(is_onomatopoeia_or_reduplication(text))

        # 正常片假名长音词不因“全假名+ー”被豁免：short_duplicate 仍能报。
        index = self.index()
        index.add(1, "コーヒー", self.TRANSLATION)
        issues = audit_short_duplicate(
            "ケーキ", self.TRANSLATION, 2, index
        )
        self.assertEqual([issue.code for issue in issues], ["short_duplicate"])


class T1AddressTermTests(unittest.TestCase):
    def test_closed_set_with_teacher_translation_fires(self) -> None:
        for source in ("旦那様、おかえりなさい", "ご主人様、お疲れさまです"):
            for bad in ("先生", "老师"):
                with self.subTest(source=source, bad=bad):
                    issues = audit_address_term(source, f"{bad}，您回来了", 3)
                    self.assertEqual(len(issues), 1)
                    self.assertEqual(issues[0].code, "address_term")
                    self.assertEqual(
                        issues[0].hint,
                        "称谓按人物关系译，不译“先生/老师”，不写死单一译名",
                    )
                    # hint 不得回退成封闭译名列表。
                    for banned in ("老公", "主人", "老爷"):
                        self.assertNotIn(banned, issues[0].hint)

    def test_surname_sama_customer_and_correct_choice_are_clean(self) -> None:
        for source, translation in (
            ("南様、こんにちは", "南先生，你好"),
            ("梅地様、お願いします", "梅地先生，拜托了"),
            ("お客様、こちらへどうぞ", "客人，这边请"),
            ("旦那様、おかえりなさい", "主人，你回来了"),
            ("旦那様、おかえりなさい", "当家的，你回来了"),
        ):
            with self.subTest(source=source, translation=translation):
                self.assertEqual(
                    audit_address_term(source, translation, 4), []
                )
        self.assertEqual(audit_address_term("", "", 4), [])


class T1InventedToolTests(unittest.TestCase):
    SOURCE = "ここで静かにするのを"
    TRANSLATION = "先用掏耳勺轻轻掏一下"

    def test_source_without_tool_but_translation_with_tool_fires(self) -> None:
        issues = audit_invented_tool(
            self.SOURCE, self.TRANSLATION, 42
        )
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "invented_tool")
        self.assertIn("±3", issues[0].hint)
        self.assertIn("道具", issues[0].hint)

    def test_cross_cue_correct_candidate_is_not_triggered(self) -> None:
        # 前后 ±3 内有「綿棒で」时，本 cue 的“棉签”有跨 cue 依据，不得误报。
        self.assertEqual(
            audit_invented_tool(
                self.SOURCE,
                "先用棉签轻轻转一下",
                42,
                nearby_sources=("綿棒でいきますね",),
            ),
            [],
        )

    def test_source_with_tool_or_non_tool_latin_is_clean(self) -> None:
        for source, translation in (
            ("耳かきでゆっくり", "用掏耳勺慢慢来"),
            ("綿棒でいきますね", "用棉签来哦"),
            ("梵天で耳を撫でる", "用梵天轻抚耳边"),
            ("ぼんてんを使います", "用梵天"),
            # G0 dev 实测别名：源文有真实采耳刷时不得误报“棕刷”是臆造。
            ("ホルホールで擦って", "用棕刷"),
            # 具体耳刷复合词可放行；通用“ブラシ”不作为别名（见下一测试）。
            ("耳ブラシで優しく", "用采耳刷"),
            ("KU100 ASMR NG BGM", "KU100 ASMR NG BGM"),
        ):
            with self.subTest(source=source, translation=translation):
                self.assertEqual(
                    audit_invented_tool(source, translation, 5), []
                )
        self.assertEqual(audit_invented_tool("", "", 5), [])

    def test_generic_brush_is_not_a_tool_alias(self) -> None:
        # 歯ブラシ/ヘアブラシ 是通用刷子，不是采耳道具；译文出现采耳道具
        # 时必须报 invented_tool，不能因“ブラシ”子串被放行。
        for source in ("歯ブラシで磨く", "ヘアブラシでとかす"):
            with self.subTest(source=source):
                issues = audit_invented_tool(source, "用掏耳勺", 5)
                self.assertEqual(
                    [issue.code for issue in issues], ["invented_tool"]
                )


class T1InventedNameTests(unittest.TestCase):
    DOCUMENT = "お耳の\nコリコリ固いところを\nほぐしていきます\nコーリ\nいかがですか"

    def test_example_name_without_any_source_name_fires(self) -> None:
        # 真实取证：曲目5 #39 源文「コーリ」，译文「茜音……」，全篇没有あかね。
        issues = audit_invented_name("コーリ", "茜音……", 39, self.DOCUMENT)
        self.assertEqual([issue.code for issue in issues], ["invented_name"])
        self.assertIn("茜音", issues[0].message)
        self.assertIn("人名", issues[0].hint)

    def test_document_containing_kana_name_is_clean(self) -> None:
        # 同一篇别处真的出现过 あかね 时，本条补名不算渗漏（宁漏不错）。
        self.assertEqual(
            audit_invented_name(
                "コーリ", "茜音……", 39, "あかねちゃん、おはよう"
            ),
            [],
        )
        # 片假名写法同样算"出现过"：靶场 CAT_C_003「ミサキヒヨリ」不能误报。
        self.assertEqual(
            audit_invented_name(
                "コーリ", "日和……", 39, "ミサキヒヨリという名のサクラです"
            ),
            [],
        )

    def test_source_with_han_name_is_clean(self) -> None:
        self.assertEqual(audit_invented_name("茜音ちゃん", "茜音……", 3), [])

    def test_other_example_name_and_empty_inputs_are_clean(self) -> None:
        self.assertEqual(audit_invented_name("こんにちは", "你好", 3), [])
        self.assertEqual(audit_invented_name("", "茜音……", 3), [])
        # 日和/ひより 同样是提示词示例，同样只在整篇没有时才算渗漏。
        self.assertEqual(
            [issue.code for issue in audit_invented_name("コーリ", "日和……", 3)],
            ["invented_name"],
        )
        self.assertEqual(
            audit_invented_name("コーリ", "日和……", 3, "日和って呼んで"), []
        )

    def test_forbidden_spelling_of_example_name_also_fires(self) -> None:
        # 复跑取证：同一份语料里 コーリ 全部落成“赤音……”，而“赤音”正是提示词
        # 明令禁止的写法——补的是哪个写法不影响判定（取证细节见 变更历史 §53）。
        self.assertEqual(
            [
                issue.code
                for issue in audit_invented_name(
                    "コーリ", "赤音……", 39, self.DOCUMENT
                )
            ],
            ["invented_name"],
        )
        # 源文真的写成“赤音”（ASR 汉字写法）时不算渗漏。
        self.assertEqual(
            audit_invented_name("赤音、おはよう", "赤音……", 3), []
        )

    def test_strip_example_names_only_when_whole_cue_is_the_name(self) -> None:
        # 拟声位被填人名：删完只剩标点 → 确定性降级成省略号。
        self.assertEqual(
            strip_example_names("茜音……", "コーリ", self.DOCUMENT), "……"
        )
        self.assertEqual(strip_example_names("茜音", "コーリ", self.DOCUMENT), "……")
        # 同一条里反复填空（“赤音……赤音……赤音……”）同样收敛成一个省略号。
        self.assertEqual(
            strip_example_names("赤音……赤音……赤音……", "コーリ", self.DOCUMENT),
            "……",
        )
        # 名字嵌在正常句子里：原样返回，交给 audit_invented_name 重译，不删残。
        self.assertEqual(
            strip_example_names(
                "说不出来呢，日和的两次连续……", "言えないヒヨノア2連続"
            ),
            "说不出来呢，日和的两次连续……",
        )
        # 整篇真出现过该名字时不动。
        self.assertEqual(
            strip_example_names("茜音……", "コーリ", "あかねちゃん、おはよう"),
            "茜音……",
        )
        self.assertEqual(strip_example_names("", "コーリ"), "")

    def test_prompt_example_names_stay_in_sync_with_template(self) -> None:
        # 提示词换示例专名时，PROMPT_EXAMPLE_NAMES 必须同步，否则审计形同虚设。
        template = (
            ROOT / "config" / "prompts" / "task_translate_asmr.md"
        ).read_text(encoding="utf-8")
        for zh_name, kana in PROMPT_EXAMPLE_NAMES:
            with self.subTest(name=zh_name):
                self.assertIn(zh_name, template)
                self.assertIn(kana, template)
                self.assertIn(f"「{zh_name}（{kana}）」", template)


class T1UnderTranslationTests(unittest.TestCase):
    def test_single_word_for_long_source_fires(self) -> None:
        # 真实取证：⑤#23「たのも知ってましたけど。」→"哦"（前半句被并进上一条）。
        issues = audit_under_translation(
            "たのも知ってましたけど。", "哦", 23
        )
        self.assertEqual([issue.code for issue in issues], ["under_translation"])
        self.assertIn("漏译", issues[0].message)

    def test_legit_compression_is_clean(self) -> None:
        # 金标 CAT_A_001：25 字 → 9 字（比例 0.36）是合法压缩，不是漏译。
        # 这条否掉了"按比例判漏译"的设计：真漏译 0.35 与合法压缩 0.36 不可分。
        self.assertEqual(
            audit_under_translation(
                "お客様から私たちに触れることは禁止となっております",
                "禁止客人触碰我们哦",
                1,
            ),
            [],
        )
        self.assertEqual(
            audit_under_translation(
                "お客様、ゆっくりと仰向けになっていただけますでしょうか。",
                "客人，可以请您慢慢躺平吗？",
                16,
            ),
            [],
        )

    def test_onomatopoeia_and_short_sources_are_clean(self) -> None:
        for source, translation in (
            ("さらさら、さらさら", "滑溜溜的，好滑溜溜的~"),  # 拟声按设计短译
            ("ふわふわ", "柔软柔软的~"),
            ("はい", "嗯……"),  # 源文不足 10 字
            ("もう先生", "已经硬得这么厉害了……"),
        ):
            with self.subTest(source=source):
                self.assertEqual(
                    audit_under_translation(source, translation, 7), []
                )

    def test_empty_inputs_are_clean(self) -> None:
        self.assertEqual(audit_under_translation("", "", 1), [])
        self.assertEqual(audit_under_translation("あ" * 20, "", 1), [])


class T1CodeSetTests(unittest.TestCase):
    def test_t1_closed_set_has_five_codes(self) -> None:
        self.assertEqual(
            ASMR_T1_CODES,
            frozenset(
                {
                    "short_duplicate",
                    "address_term",
                    "invented_tool",
                    "invented_name",
                    "under_translation",
                }
            ),
        )

    def test_new_codes_are_gated_by_asmr_guard(self) -> None:
        index = DuplicateIndex()
        # 不传 asmr_guard 时，两个新 code 都不参与汇总。
        issues = audit_translation(
            "コーリ",
            "茜音……",
            1,
            index=index,
            profile=TranslationProfile(register="intimate", pronoun="auto"),
        )
        self.assertEqual(
            [issue.code for issue in issues if issue.code in ASMR_T1_CODES], []
        )


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

    def test_audit_translation_t1_fixed_order_and_asmr_gate(self) -> None:
        source = "旦那様、お客様から私たち触れることは禁止となっていませんか"
        translation = "对吧，我们触碰客人，先生，您用掏耳勺吧"
        profile = TranslationProfile(register="intimate", pronoun="你")
        index = DuplicateIndex()
        index.add(1, "今日はとても楽しかったです、本当にいい一日でした", translation)
        short_index = DuplicateIndex(min_chars=8, translation_ratio=0.85)
        short_index.add(1, "はい", translation)

        issues = audit_translation(
            source,
            translation,
            5,
            index=index,
            profile=profile,
            short_index=short_index,
            asmr_guard=True,
        )
        self.assertEqual(
            [issue.code for issue in issues],
            [
                "duplicate_translation",
                "short_duplicate",
                "customer_direction",
                "negative_question",
                "tag_question",
                "address_term",
                "invented_tool",
                "honorific",
            ],
        )

        # 非 ASMR 门控时 T1 三个 code 全部不出现；既有语义审计不变。
        gated_off = audit_translation(
            source,
            translation,
            5,
            index=DuplicateIndex(),
            profile=profile,
            short_index=short_index,
            asmr_guard=False,
        )
        codes = [issue.code for issue in gated_off]
        for t1_code in ("short_duplicate", "address_term", "invented_tool"):
            self.assertNotIn(t1_code, codes)
        self.assertIn("customer_direction", codes)

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


