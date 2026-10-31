"""``vanitas.qc`` 纯函数内核：code 目录、不变量、定义类与统计分组。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vanitas import qc  # noqa: E402
from vanitas.quality import Glossary  # noqa: E402

GLOSSARY_PATH = ROOT / "config" / "glossary" / "asmr.tsv"
ASR_PATH = ROOT / "config" / "qc" / "asr_suspect.tsv"


def srt(*texts: str) -> str:
    blocks = []
    for index, text in enumerate(texts, start=1):
        start = f"00:00:{index * 2 - 1:02d},000"
        end = f"00:00:{index * 2:02d},000"
        blocks.append(f"{index}\n{start} --> {end}\n{text}")
    return "\n\n".join(blocks) + "\n"


class CodeCatalogTests(unittest.TestCase):
    def test_catalog_has_expected_size_and_severities(self) -> None:
        self.assertEqual(len(qc.CODES), 26)
        self.assertEqual(
            {spec.severity for spec in qc.CODES.values()},
            {qc.SEVERITY_BLOCK, qc.SEVERITY_REVIEW, qc.SEVERITY_INFO},
        )
        for code, spec in qc.CODES.items():
            self.assertEqual(code, spec.code)
            self.assertTrue(spec.rule.startswith("qc."))
            self.assertTrue(spec.definition.strip())

    def test_unknown_code_defaults_to_info(self) -> None:
        self.assertEqual(qc.severity_of("not_a_code"), qc.SEVERITY_INFO)
        self.assertEqual(qc.rule_of("not_a_code"), "qc.unknown.not_a_code")

    def test_level_keeps_legacy_mapping(self) -> None:
        issue = qc.Issue(
            severity=qc.SEVERITY_REVIEW,
            code="kana",
            rule="qc.legacy.kana",
            message="x",
        )
        self.assertEqual(issue.level, "actionable")
        self.assertEqual(issue.as_dict()["severity"], "review")


class InvariantTests(unittest.TestCase):
    def test_garbled_is_block(self) -> None:
        result = qc.check_pair(srt("コーリ"), srt("\ufffd茜音……"))
        self.assertTrue(any(i.code == "garbled" and i.severity == "block" for i in result.issues))

    def test_pipeline_drift_is_info(self) -> None:
        result = qc.check_pair(srt("これは長い台詞です"), srt("这是一段很长的台词。"))
        self.assertEqual(result.errors, [])
        self.assertEqual(result.warnings, [])
        drift = [i for i in result.issues if i.code == "pipeline_drift"]
        self.assertEqual(len(drift), 1)
        self.assertEqual(drift[0].severity, "info")
        self.assertEqual(result.fixpoint, {"stable": 0, "total": 1, "drift": 1})

    def test_clean_postprocess_is_stable(self) -> None:
        result = qc.check_pair(srt("こんにちは"), srt("你好"))
        self.assertEqual(result.fixpoint, {"stable": 1, "total": 1, "drift": 0})
        self.assertFalse(any(i.code == "pipeline_drift" for i in result.issues))

    def test_glossary_forbidden_and_converged(self) -> None:
        glossary = Glossary.load(GLOSSARY_PATH)
        result = qc.check_pair(
            srt("コショコショ…"), srt("咕噜咕噜……"), glossary=glossary
        )
        self.assertTrue(
            any(i.code == "glossary_forbidden" for i in result.issues)
        )
        result = qc.check_pair(
            srt("コショコショ…"), srt("沙沙……"), glossary=glossary
        )
        self.assertFalse(
            any(i.code == "glossary_forbidden" for i in result.issues)
        )


class DefinitionTests(unittest.TestCase):
    def test_invented_name_uses_document_text(self) -> None:
        result = qc.check_pair(
            srt("コーリ", "あかねちゃん、おはよう"),
            srt("茜音……", "茜音，早上好"),
        )
        self.assertFalse(any(i.code == "invented_name" for i in result.issues))
        result = qc.check_pair(
            srt("コーリ", "こんにちは"),
            srt("茜音……", "你好"),
        )
        self.assertTrue(any(i.code == "invented_name" for i in result.issues))

    def test_under_translation_skips_asr_hallucination(self) -> None:
        result = qc.check_pair(
            srt("たのも知ってましたけど。"), srt("哦")
        )
        self.assertTrue(any(i.code == "under_translation" for i in result.issues))
        result = qc.check_pair(
            srt("ご視聴ありがとうございました"), srt("")
        )
        self.assertFalse(any(i.code == "under_translation" for i in result.issues))
        self.assertFalse(any(i.code == "blank" for i in result.issues))

    def test_invented_tool_needs_nearby_check(self) -> None:
        result = qc.check_pair(
            srt("こんにちは", "くすぐったいですか?"),
            srt("梵天老师，你好", "会痒吗？"),
        )
        self.assertTrue(any(i.code == "invented_tool" for i in result.issues))
        result = qc.check_pair(
            srt("綿棒で掃除します", "くすぐったいですか?"),
            srt("用棉签清洁", "会痒吗？"),
        )
        self.assertFalse(any(i.code == "invented_tool" for i in result.issues))

    def test_address_term(self) -> None:
        result = qc.check_pair(
            srt("ご主人様、おかえりなさい"), srt("先生，欢迎回来")
        )
        self.assertTrue(any(i.code == "address_term" for i in result.issues))

    def test_numeral_truncation(self) -> None:
        result = qc.check_pair(
            srt("十一人で来ました"), srt("一个人来的")
        )
        self.assertTrue(
            any(i.code == "numeral_truncation" for i in result.issues)
        )
        for translation in ("11个人来的", "十一个人来的"):
            result = qc.check_pair(
                srt("十一人で来ました"), srt(translation)
            )
            self.assertFalse(
                any(i.code == "numeral_truncation" for i in result.issues)
            )

    def test_organ_swap_ear_to_vagina(self) -> None:
        result = qc.check_pair(
            srt("お耳の中も失礼して"),
            srt("（轻轻探入）你的小穴里面也要好好检查哦"),
        )
        issues = [i for i in result.issues if i.code == "organ_mismatch"]
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, "review")
        self.assertIn("小穴", issues[0].evidence)

        for translation in ("你的耳朵里面也要好好检查哦",):
            clean = qc.check_pair(srt("お耳の中も失礼して"), srt(translation))
            self.assertFalse(
                any(i.code == "organ_mismatch" for i in clean.issues)
            )

    def test_organ_swap_needs_narrow_guard(self) -> None:
        # 源文同时提到两处器官：不改写、不报。
        both = qc.check_pair(
            srt("お耳もおまんこも両方"), srt("耳朵和小穴都要")
        )
        self.assertFalse(any(i.code == "organ_mismatch" for i in both.issues))
        # 源文没有耳词：不属于本 code 的判定面（改用 review/人工）。
        no_ear = qc.check_pair(
            srt("っぱいどろどろのつゆもついちゃってます"), srt("小穴里也沾满了汁水呢")
        )
        self.assertFalse(any(i.code == "organ_mismatch" for i in no_ear.issues))

    def test_staging_any_unsourced_bracket(self) -> None:
        result = qc.check_pair(srt("好きだよ"), srt("（轻轻探入）喜欢你哦"))
        staging = [i for i in result.issues if i.code == "staging"]
        self.assertEqual(len(staging), 1)
        self.assertIn("轻轻探入", staging[0].evidence)
        # 源文自带括号属于有源标注，不报。
        clean = qc.check_pair(srt("(両耳) 好きだよ"), srt("（在双耳边）喜欢你哦"))
        self.assertFalse(any(i.code == "staging" for i in clean.issues))


class DuplicateTests(unittest.TestCase):
    def test_duplicate_long(self) -> None:
        result = qc.check_pair(
            srt("今日はとても楽しかったです", "明日の予定を確認します"),
            srt("这是完全相同的一条长译文", "这是完全相同的一条长译文"),
        )
        self.assertTrue(
            any(i.code == "duplicate_translation" for i in result.issues)
        )

    def test_short_duplicate_exact_and_sentence_override(self) -> None:
        result = qc.check_pair(
            srt("みっともないですわ", "ストップですわ"),
            srt("真是太没面子了", "真是太没面子了"),
        )
        self.assertTrue(any(i.code == "short_duplicate" for i in result.issues))

    def test_onomatopoeia_and_near_synonym_are_exempt(self) -> None:
        result = qc.check_pair(
            srt("しこしこ", "つりつり"),
            srt("沙沙地响个不停", "沙沙地响个不停"),
        )
        self.assertFalse(
            any(i.code in ("duplicate_translation", "short_duplicate") for i in result.issues)
        )
        result = qc.check_pair(
            srt("気に入って", "お気に召して"),
            srt("您喜欢就好", "您喜欢就好"),
        )
        self.assertFalse(
            any(i.code in ("duplicate_translation", "short_duplicate") for i in result.issues)
        )

    def test_name_variant_is_exempt(self) -> None:
        result = qc.check_pair(
            srt(
                "伊賀崎綾香の庭はね基本的に",
                "伊賀崎彩香の庭主催しております",
            ),
            srt("这是由伊贺崎彩香主办的庭院哦", "这是由伊贺崎彩香主办的庭院哦"),
        )
        self.assertFalse(
            any(i.code in ("duplicate_translation", "short_duplicate") for i in result.issues)
        )


class StatsAndRegisterTests(unittest.TestCase):
    def test_onomatopoeia_collapse_groups_three(self) -> None:
        result = qc.check_pair(
            srt("シーコ", "シーコ", "シーコ"),
            srt("嗖……", "嗖……", "嗖……"),
        )
        collapses = [i for i in result.issues if i.code == "onomatopoeia_collapse"]
        self.assertEqual(len(collapses), 3)
        self.assertTrue(all(i.severity == "info" for i in collapses))

    def test_term_drift_is_per_cue_info(self) -> None:
        glossary = Glossary.load(GLOSSARY_PATH)
        result = qc.check_pair(
            srt("耳かきをしますね", "耳かきをしますね"),
            srt("采耳哦", "掏耳朵哦"),
            glossary=glossary,
        )
        drifts = [i for i in result.issues if i.code == "term_drift"]
        self.assertEqual(len(drifts), 2)
        self.assertTrue(all(i.severity == "info" for i in drifts))

    def test_register_summary_and_honorific_rules(self) -> None:
        result = qc.check_pair(
            srt("先輩、好きです", "膝枕してあげるね"),
            srt("您真好呢", "给您膝枕哦"),
        )
        self.assertEqual(result.register, "intimate")
        self.assertEqual(result.pronoun["您"], 2)
        self.assertEqual(len(result.reviews), 2)
        self.assertTrue(all(i.code == "honorific" for i in result.reviews))

        result = qc.check_pair(
            srt("いらっしゃいませ、お客様", "ご予約ありがとうございます"),
            srt("您这边请", "您的预约已确认"),
        )
        self.assertEqual(result.register, "service")
        self.assertFalse(any(i.code == "honorific" for i in result.issues))

        result = qc.check_pair(
            srt("先輩、好きです"), srt("他说“您真好呢”")
        )
        self.assertFalse(any(i.code == "honorific" for i in result.issues))

    def test_pronoun_mixed_anchors_on_first_switch(self) -> None:
        result = qc.check_pair(
            srt("こんにちは", "はい", "いいえ", "そうですね"),
            srt("您看吧", "你看吧", "您听着", "你试试"),
        )
        mixed = [i for i in result.issues if i.code == "pronoun_mixed"]
        self.assertEqual(len(mixed), 1)
        self.assertEqual(mixed[0].severity, "review")
        self.assertEqual(mixed[0].cue, "2")
        self.assertIn("切换 3 次", mixed[0].evidence)

    def test_pronoun_mixed_needs_both_and_switches(self) -> None:
        # 只有「您」不报；只有一次切换（服务→亲密）也不报。
        only_nin = qc.check_pair(
            srt("こんにちは", "はい"), srt("您看吧", "您听着")
        )
        self.assertFalse(
            any(i.code == "pronoun_mixed" for i in only_nin.issues)
        )
        one_switch = qc.check_pair(
            srt("こんにちは", "はい", "いいえ"),
            srt("您看吧", "你看吧", "你听着"),
        )
        self.assertFalse(
            any(i.code == "pronoun_mixed" for i in one_switch.issues)
        )


class OptionsAndCorpusTests(unittest.TestCase):
    def test_only_filters_issues(self) -> None:
        result = qc.check_pair(
            srt("先輩、好きです"),
            srt("您まだ"),
            options=qc.QCOptions(only=frozenset({"kana"})),
        )
        self.assertTrue(result.issues)
        self.assertEqual({i.code for i in result.issues}, {"kana"})

    def test_check_corpus_names_and_counts(self) -> None:
        results = qc.check_corpus(
            [("a.zh.srt", srt("こんにちは"), srt("你好"))]
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].source, "a.srt")
        self.assertEqual(results[0].translated, "a.zh.srt")
        self.assertEqual(results[0].cues, 1)

    def test_parse_asr_terms(self) -> None:
        terms = qc.parse_asr_terms("南様\tlooks wrong\tverify\n# comment\n")
        self.assertEqual(len(terms), 1)
        self.assertEqual(terms[0].term, "南様")
        with self.assertRaises(ValueError):
            qc.parse_asr_terms("only-one-column\n")
        self.assertTrue(ASR_PATH.is_file())


class RenderTests(unittest.TestCase):
    def test_render_sorts_by_file_cue_severity(self) -> None:
        first = qc.check_pair(
            srt("こんにちは", "先輩、好きです"),
            srt("你好", "您真好呢"),
        )
        first.translated = "b.zh.srt"
        second = qc.check_pair(srt("こんにちは"), srt("你好"))
        second.translated = "a.zh.srt"
        report = qc.render_report([first, second])
        self.assertIn("⚠ 语义疑似点", report)
        self.assertIn("- 原文", report)
        self.assertIn("+ 译文", report)
        self.assertIn("? ", report)


if __name__ == "__main__":
    unittest.main()
