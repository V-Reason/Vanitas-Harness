"""数据驱动规则：rules.tsv 解析/执行与实体停用词外置。

这些用例只读仓库自带的数据文件与临时文件，不调用模型、不联网。
"""

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
    DEFAULT_RULES_PATH,
    RULE_TYPES,
    apply_context_rules,
    default_context_rules,
    detect_organ_swap,
    load_context_rules,
    repair_first_person_name,
    repair_organ_swap,
    strip_example_names,
)
from vanitas.translate.entity import (  # noqa: E402
    CORE_STOPWORDS,
    ENTITY_STOPWORDS,
    STOPWORDS_PATH,
    load_entity_stopwords,
)

RULES = ROOT / "config" / "glossary" / "rules.tsv"
STOPWORDS = ROOT / "config" / "filters" / "entity_stopwords.txt"


def read_bytes(path: Path) -> bytes:
    return path.read_bytes()


class RulesFileTests(unittest.TestCase):
    def test_rules_file_is_utf8_lf_without_bom(self) -> None:
        raw = read_bytes(RULES)
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        self.assertNotIn(b"\r", raw)
        raw.decode("utf-8")

    def test_stopwords_file_is_utf8_lf_without_bom(self) -> None:
        raw = read_bytes(STOPWORDS)
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        self.assertNotIn(b"\r", raw)
        raw.decode("utf-8")

    def test_missing_file_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope.tsv"
            self.assertEqual(load_context_rules(missing), ())

    def test_wrong_column_count_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.tsv"
            path.write_text("context_sub\tonly-three\tcolumns\n", encoding="utf-8")
            with self.assertRaises(TranslateError) as caught:
                load_context_rules(path)
            self.assertIn("4 列", str(caught.exception))

    def test_unknown_rule_type_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.tsv"
            path.write_text(
                "\t".join(["unknown", "a", "b", "c"]) + "\n", encoding="utf-8"
            )
            with self.assertRaises(TranslateError) as caught:
                load_context_rules(path)
            self.assertIn("未知规则类型", str(caught.exception))

    def test_bad_regex_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.tsv"
            path.write_text(
                "\t".join(["context_sub", "(", "b", "c"]) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(TranslateError) as caught:
                load_context_rules(path)
            self.assertIn("正则非法", str(caught.exception))

    def test_repo_rules_cover_all_types(self) -> None:
        rules = load_context_rules(RULES)
        types = [rule.rule_type for rule in rules]
        self.assertEqual(set(types), RULE_TYPES)
        for rule_type in RULE_TYPES:
            self.assertIn(rule_type, types)
        self.assertEqual(default_context_rules(), rules)
        self.assertEqual(DEFAULT_RULES_PATH, RULES)


class ApplyRulesTests(unittest.TestCase):
    def test_organ_guard_repairs_only_without_vagina_in_source(self) -> None:
        source = "お耳の中も失礼して"
        self.assertEqual(
            repair_organ_swap(source, "你的小穴里面也要好好检查"),
            "你的耳朵里面也要好好检查",
        )
        self.assertEqual(
            detect_organ_swap(source, "你的小穴里面也要好好检查"),
            "源=耳 译=小穴",
        )
        # 源文同时提到两处器官：不动，避免改写合法译法。
        both = "お耳もおまんこも両方"
        self.assertIsNone(detect_organ_swap(both, "耳朵和小穴都要"))
        self.assertEqual(repair_organ_swap(both, "耳朵和小穴都要"), "耳朵和小穴都要")
        # 源文没有耳部词：不属于本规则。
        self.assertIsNone(detect_organ_swap("っぱいどろどろ", "小穴里沾满了汁水"))

    def test_context_sub_requires_no_first_person_in_source(self) -> None:
        source = "ちゃんと下の名前で呼んであげてよ"
        self.assertEqual(
            repair_first_person_name(source, "请用我的名字好好叫我哦"),
            "请用她的名字好好叫她哦",
        )
        self.assertEqual(
            repair_first_person_name(
                "私の下の名前で呼んであげてよ", "请用我的名字好好叫我哦"
            ),
            "请用我的名字好好叫我哦",
        )
        self.assertEqual(
            repair_first_person_name("名前を呼んでください", "请用我的名字好好叫我哦"),
            "请用我的名字好好叫我哦",
        )

    def test_leak_guard_degrades_whole_cue_only(self) -> None:
        self.assertEqual(strip_example_names("茜音……", "コーリ"), "……")
        self.assertEqual(
            strip_example_names("赤音……赤音……赤音……", "コーリ"), "……"
        )
        # 名字嵌在正常句子里：原样返回，交给审计重译，不删残。
        self.assertEqual(
            strip_example_names("说不出来呢，日和的两次连续……", "言えない"),
            "说不出来呢，日和的两次连续……",
        )

    def test_leak_guard_folds_katakana_and_checks_document(self) -> None:
        self.assertEqual(
            strip_example_names("茜音……", "コーリ", "あかねちゃん、おはよう"),
            "茜音……",
        )
        self.assertEqual(
            strip_example_names("日和……", "コーリ", "ミサキヒヨリという名"),
            "日和……",
        )

    def test_rule_types_filter_skips_other_families(self) -> None:
        rules = default_context_rules()
        text = "你的小穴里面也要好好检查"
        self.assertEqual(
            apply_context_rules(
                "お耳の中も失礼して", text, rules, rule_types=("context_sub",)
            ),
            text,
        )
        self.assertEqual(
            apply_context_rules(
                "お耳の中も失礼して", text, rules, rule_types=("organ_guard",)
            ),
            "你的耳朵里面也要好好检查",
        )

    def test_empty_inputs_are_noops(self) -> None:
        self.assertEqual(apply_context_rules("", ""), "")
        self.assertEqual(repair_organ_swap("", ""), "")
        self.assertEqual(strip_example_names("", "コーリ"), "")

    def test_quality_module_has_no_migrated_literals(self) -> None:
        # 把"DoD grep"变成回归护栏：样本级字面不得回流到 Python 源码。
        text = (ROOT / "vanitas" / "quality.py").read_text(encoding="utf-8")
        for literal in (
            "下の名前で呼んであげて",
            "お耳",
            "茜音",
            "あかね",
            "日和",
            "ひより",
            "PROMPT_EXAMPLE_NAMES",
            "ORGAN_EAR_SOURCE_TERMS",
        ):
            with self.subTest(literal=literal):
                self.assertNotIn(literal, text)


class StopwordsTests(unittest.TestCase):
    def test_repo_stopwords_are_loaded(self) -> None:
        self.assertTrue(STOPWORDS.is_file())
        self.assertEqual(ENTITY_STOPWORDS, load_entity_stopwords())
        self.assertGreaterEqual(len(ENTITY_STOPWORDS), 99)
        self.assertIn("トラック", ENTITY_STOPWORDS)
        self.assertIn("美容師", ENTITY_STOPWORDS)
        self.assertTrue(CORE_STOPWORDS <= ENTITY_STOPWORDS)

    def test_custom_file_merges_with_core(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "words.txt"
            path.write_text(
                "# 注释\n\n自定义噪声词\n另一個词\n", encoding="utf-8"
            )
            loaded = load_entity_stopwords(path)
            self.assertIn("自定义噪声词", loaded)
            self.assertIn("另一個词", loaded)
            self.assertTrue(CORE_STOPWORDS <= loaded)

    def test_missing_file_falls_back_to_core(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loaded = load_entity_stopwords(Path(tmp) / "missing.txt")
            self.assertEqual(loaded, CORE_STOPWORDS)


if __name__ == "__main__":
    unittest.main()
