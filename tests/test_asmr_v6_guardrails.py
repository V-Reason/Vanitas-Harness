"""V6 语义守卫：器官互斥、T1 确定性兜底、和制词/拟坏拟声与 top_p。

全部离线；提示词字节锁在 ``tests/test_prompts.py`` 里单独验证。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vanitas.config import RequestSettings, TranslateSettings  # noqa: E402
from vanitas.engine import EngineClient  # noqa: E402
from vanitas.quality import (  # noqa: E402
    apply_context_rules,
    default_context_rules,
    enforce_gender_anatomy_invariant,
    salvage_t1_rejection,
)
from vanitas.translate import TranslationEngine, TranslationOptions  # noqa: E402

PROMPT_DIR = ROOT / "config" / "prompts"
REQUEST = RequestSettings(
    timeout_sec=10,
    max_retries=0,
    backoff_base_ms=0,
    backoff_jitter_ms=0,
)
OPTIONS = TranslationOptions(
    n_ctx=8192,
    context_window=5,
    context_summary_chars=200,
    min_output_tokens=64,
    output_ratio=2.5,
    prompt_safety_tokens=64,
    temperature=0.2,
    max_split_depth=3,
)


class AnatomyInvariantTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rules = default_context_rules()

    def test_penis_context_converges_to_penis(self) -> None:
        self.assertEqual(
            enforce_gender_anatomy_invariant(
                "落ちんぱに", "小穴上", self.rules
            ),
            "肉棒上",
        )

    def test_fluid_context_converges_to_front(self) -> None:
        self.assertEqual(
            enforce_gender_anatomy_invariant(
                "っぱいどろどろのつゆ", "小穴里", self.rules
            ),
            "前端里",
        )

    def test_no_context_converges_to_neutral(self) -> None:
        self.assertEqual(
            enforce_gender_anatomy_invariant(
                "優気に向かって", "你的小穴", self.rules
            ),
            "你的这里",
        )

    def test_source_with_female_organ_is_untouched(self) -> None:
        self.assertEqual(
            enforce_gender_anatomy_invariant("おまんこ", "小穴", self.rules),
            "小穴",
        )


class SalvageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rules = default_context_rules()

    def test_embedded_example_name_is_removed(self) -> None:
        result = salvage_t1_rejection(
            "说不出口的日和二次体验连发呢~",
            "言えないヒヨノア2連続",
            {"invented_name"},
            "",
            self.rules,
        )
        self.assertNotIn("日和", result)
        self.assertTrue(result)

    def test_leading_name_is_removed_with_punctuation_cleanup(self) -> None:
        self.assertEqual(
            salvage_t1_rejection(
                "茜音，我们先开始吧", "コーリ", {"invented_name"}, "", self.rules
            ),
            "我们先开始吧",
        )

    def test_whole_name_becomes_ellipsis(self) -> None:
        self.assertEqual(
            salvage_t1_rejection(
                "茜音……", "コーリ", {"invented_name"}, "", self.rules
            ),
            "……",
        )

    def test_tool_before_honorific_keeps_honorific(self) -> None:
        self.assertEqual(
            salvage_t1_rejection(
                "梵天老师", "オーラ先生", {"invented_tool"}, "", self.rules
            ),
            "老师",
        )

    def test_embedded_tool_becomes_pronoun(self) -> None:
        self.assertEqual(
            salvage_t1_rejection(
                "用棉签来哦", "ここ", {"invented_tool"}, "", self.rules
            ),
            "用它来哦",
        )

    def test_long_translation_for_short_source_becomes_ellipsis(self) -> None:
        self.assertEqual(
            salvage_t1_rejection(
                "居然已经硬得这么厉害了~",
                "もう先生",
                {"short_duplicate"},
                "",
                self.rules,
            ),
            "……",
        )

    def test_short_translation_is_not_truncated(self) -> None:
        self.assertEqual(
            salvage_t1_rejection(
                "不行不行", "うん", {"short_duplicate"}, "", self.rules
            ),
            "不行不行",
        )

    def test_empty_input_is_ellipsis(self) -> None:
        self.assertEqual(salvage_t1_rejection("", "x", set()), "……")


class WaseiAndSfxTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rules = default_context_rules()

    def test_wasei_words_are_localized(self) -> None:
        cases = {
            "部活": "社团活动",
            "生徒会": "学生会",
            "一応": "姑且",
            "一人二役": "一人分饰两角",
            "乳首": "乳头",
            "耳垢": "耳屎",
            "隣": "隔壁",
            "性液": "爱液",
            "恋人妄想": "恋人幻想",
            "先辈": "前辈",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(
                    apply_context_rules(
                        "テスト",
                        source,
                        self.rules,
                        rule_types=("wasei_clean",),
                    ),
                    expected,
                )

    def test_sfx_repair_requires_source_trigger(self) -> None:
        self.assertEqual(
            apply_context_rules(
                "すりすり",
                "蠢蹭……",
                self.rules,
                rule_types=("sfx_repair",),
            ),
            "蹭蹭……",
        )
        self.assertEqual(
            apply_context_rules(
                "別の台詞",
                "蠢蹭……",
                self.rules,
                rule_types=("sfx_repair",),
            ),
            "蠢蹭……",
        )

    def test_kosho_variants_are_repaired(self) -> None:
        for source in ("こしょこしょ", "こしゃこしゃ", "こっしょ"):
            with self.subTest(source=source):
                result = apply_context_rules(
                    source, "刮刮……", self.rules, rule_types=("sfx_repair",)
                )
                self.assertNotIn("刮刮", result)


class TopPSettingsTests(unittest.TestCase):
    def test_payload_includes_top_p_only_below_one(self) -> None:
        client = EngineClient(REQUEST, "127.0.0.1", 4060)
        payload = client._build_payload([], 16, 0.2, None, top_p=0.85)
        self.assertEqual(payload["top_p"], 0.85)
        self.assertNotIn(
            "top_p", client._build_payload([], 16, 0.2, None, top_p=1.0)
        )

    def test_translate_settings_defaults(self) -> None:
        settings = TranslateSettings(
            source_suffixes=(".srt",), chunk_chars=800, output_suffix=".zh"
        )
        self.assertEqual(settings.temperature, 0.15)
        self.assertEqual(settings.top_p, 0.85)

    def test_engine_passes_top_p_to_supporting_client(self) -> None:
        class Streaming:
            def __init__(self) -> None:
                self.kwargs: list[dict] = []

            def stream_chat(
                self, messages, *, max_tokens, temperature=0.2, **kwargs
            ):
                self.kwargs.append(dict(kwargs))
                yield {"type": "delta", "content": "译"}
                yield {"type": "done", "finish_reason": "stop", "tokens": 1}

        client = Streaming()
        engine = TranslationEngine(
            client, PROMPT_DIR, "", options=OPTIONS
        )
        self.assertEqual(engine("[#1] あ"), "译")
        self.assertEqual(client.kwargs[0].get("top_p"), 0.85)

    def test_engine_skips_top_p_for_old_client(self) -> None:
        class Blocking:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages, *, max_tokens, temperature=0.2):
                self.calls += 1
                return {
                    "choices": [
                        {"message": {"content": "译"}, "finish_reason": "stop"}
                    ],
                    "usage": {"completion_tokens": 1},
                }

        client = Blocking()
        engine = TranslationEngine(
            client, PROMPT_DIR, "", options=OPTIONS
        )
        self.assertEqual(engine("[#1] あ"), "译")
        self.assertEqual(client.calls, 1)


if __name__ == "__main__":
    unittest.main()
