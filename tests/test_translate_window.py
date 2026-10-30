"""P2 局部回退调度：只补损坏 cue，灾难性损坏才整窗熔断。

覆盖三件事：
1. ``align_window_response`` 的三态判定（完全对齐 / 缺项部分可用 / 灾难）；
2. ``identify_suspicious_cues`` 能定位到具体 cue，且同窗复制成对标记；
3. ``translate_srt`` 的请求数契约：W=6 窗内只坏 1 条时单句请求恰好 1 次，
   而灾难性输出仍退化为整窗 6 次（与 P1 行为一致）。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vanitas.translate import (  # noqa: E402
    Subtitle,
    _window_translation_is_suspicious,
    align_window_response,
    identify_suspicious_cues,
    parse_srt,
    translate_srt,
)

SOURCES = [
    "今日はいい天気ですね",
    "明日の予定を確認します",
    "窓を閉めてください",
    "コーヒーを入れます",
    "駅まで歩きます",
    "本を読みます",
]
TRANSLATIONS = [
    "今天天气真好",
    "确认明天的安排",
    "请把窗户关上",
    "泡一杯咖啡",
    "走到车站去",
    "看看书",
]
# 同窗复制用例专用：两行不同原文都拿到同一句够长的译文。
COPIED = "这是一句被复制过来的长译文"


def srt_text(texts: list[str]) -> str:
    blocks = []
    for index, text in enumerate(texts, start=1):
        start = f"00:00:{index * 2 - 1:02d},000"
        end = f"00:00:{index * 2:02d},000"
        blocks.append(f"{index}\n{start} --> {end}\n{text}")
    return "\n\n".join(blocks) + "\n"


def batch_of(texts: dict[str, str]) -> str:
    return "\n".join(f"[#{cue}] {text}" for cue, text in texts.items())


def single_cue_of(prompt: str) -> str:
    """从单句提示 ``[#N] 原文`` 里取出编号。"""
    return prompt.split(" ", 1)[0][2:-1]


def window_of(texts: list[str]) -> list[tuple[int, Subtitle]]:
    return [
        (position, Subtitle(index=str(position + 1), timing="", text=text))
        for position, text in enumerate(texts)
    ]


class PartialWindowRecoveryTests(unittest.TestCase):
    """局部回退的请求数与结果正确性。"""

    def run_window(
        self,
        batch_output: str,
        texts: list[str],
        *,
        single=None,
        window_size: int = 6,
    ) -> tuple[list[str], str, list]:
        calls: list[str] = []

        def chat(prompt: str) -> str:
            calls.append(prompt)
            if "\n" in prompt:
                return batch_output
            cue = single_cue_of(prompt)
            if single is not None:
                return single(cue)
            return f"补译{cue}"

        failures: list = []
        output = translate_srt(
            chat,
            srt_text(texts),
            chunk_chars=800,
            source="partial.srt",
            eol="\n",
            failures=failures,
            window_size=window_size,
            max_window_chars=2000,
        )
        return calls, output, failures

    @staticmethod
    def batch_calls(calls: list[str]) -> list[str]:
        return [call for call in calls if "\n" in call]

    @staticmethod
    def single_calls(calls: list[str]) -> list[str]:
        return [call for call in calls if "\n" not in call]

    def assert_structure(self, output: str) -> None:
        blocks = parse_srt(output, "\n")
        self.assertEqual(
            [block.index for block, _ in blocks],
            [str(index) for index in range(1, len(SOURCES) + 1)],
        )
        source = parse_srt(srt_text(SOURCES), "\n")
        self.assertEqual(
            [block.timing for block, _ in blocks],
            [block.timing for block, _ in source],
        )

    # ------------------------------------------------------------------ #
    # 1) 部分编号缺失：只补缺失那条
    # ------------------------------------------------------------------ #
    def test_missing_window_id_retranslates_only_that_cue(self) -> None:
        batch = batch_of(
            {
                "1": TRANSLATIONS[0],
                "2": TRANSLATIONS[1],
                "3": TRANSLATIONS[2],
                "5": TRANSLATIONS[4],
                "6": TRANSLATIONS[5],
            }
        )
        calls, output, failures = self.run_window(batch, SOURCES)

        self.assertEqual(failures, [])
        self.assertEqual(len(self.batch_calls(calls)), 1)
        self.assertEqual(len(self.single_calls(calls)), 1, calls)
        self.assertEqual(single_cue_of(self.single_calls(calls)[0]), "4")
        self.assertEqual(
            [block.text for block, _ in parse_srt(output, "\n")],
            [
                TRANSLATIONS[0],
                TRANSLATIONS[1],
                TRANSLATIONS[2],
                "补译4",
                TRANSLATIONS[4],
                TRANSLATIONS[5],
            ],
        )
        self.assert_structure(output)

    # ------------------------------------------------------------------ #
    # 2) 单条嫌疑（正文泄漏编号）：只补那一条
    # ------------------------------------------------------------------ #
    def test_inline_window_id_marks_only_that_cue(self) -> None:
        batch = batch_of(
            {
                str(index + 1): text for index, text in enumerate(TRANSLATIONS)
            }
        ).replace(
            TRANSLATIONS[2], f"{TRANSLATIONS[2]} [#99]"
        )
        calls, output, failures = self.run_window(batch, SOURCES)

        self.assertEqual(failures, [])
        singles = self.single_calls(calls)
        self.assertEqual([single_cue_of(prompt) for prompt in singles], ["3"])
        self.assertEqual(
            [block.text for block, _ in parse_srt(output, "\n")][2], "补译3"
        )
        self.assertNotIn("[#99]", output)

    # ------------------------------------------------------------------ #
    # 3) 单条硬校验失败（残留假名）：只补那一条
    # ------------------------------------------------------------------ #
    def test_residual_kana_marks_only_that_cue(self) -> None:
        texts = list(TRANSLATIONS)
        texts[1] = "こんにちは"
        batch = batch_of(
            {str(index + 1): text for index, text in enumerate(texts)}
        )
        calls, output, failures = self.run_window(batch, SOURCES)

        self.assertEqual(failures, [])
        self.assertEqual(
            [single_cue_of(prompt) for prompt in self.single_calls(calls)], ["2"]
        )
        self.assertEqual(
            [block.text for block, _ in parse_srt(output, "\n")][1], "补译2"
        )

    # ------------------------------------------------------------------ #
    # 4) 同窗复制：两条同时判坏、各自重译
    # ------------------------------------------------------------------ #
    def test_pairwise_duplicate_marks_both_cues(self) -> None:
        texts = list(TRANSLATIONS)
        texts[1] = COPIED
        texts[4] = COPIED
        batch = batch_of(
            {str(index + 1): text for index, text in enumerate(texts)}
        )
        calls, output, failures = self.run_window(batch, SOURCES)

        self.assertEqual(failures, [])
        self.assertEqual(
            sorted(single_cue_of(prompt) for prompt in self.single_calls(calls)),
            ["2", "5"],
        )
        blocks = [block.text for block, _ in parse_srt(output, "\n")]
        self.assertEqual(blocks[1], "补译2")
        self.assertEqual(blocks[4], "补译5")
        self.assertEqual(output.count(COPIED), 0)

        suspicious = identify_suspicious_cues(
            window_of(SOURCES),
            {str(index + 1): text for index, text in enumerate(texts)},
        )
        self.assertEqual(suspicious, {"2", "5"})

    # ------------------------------------------------------------------ #
    # 5) 灾难性破坏：整窗熔断，退化为 6 次单句（P1 行为）
    # ------------------------------------------------------------------ #
    def test_catastrophic_order_falls_back_to_whole_window(self) -> None:
        batch = "\n".join(
            [
                f"[#3] {TRANSLATIONS[2]}",
                f"[#1] {TRANSLATIONS[0]}",
                f"[#2] {TRANSLATIONS[1]}",
                f"[#4] {TRANSLATIONS[3]}",
                f"[#5] {TRANSLATIONS[4]}",
                f"[#6] {TRANSLATIONS[5]}",
            ]
        )
        calls, output, failures = self.run_window(batch, SOURCES)

        self.assertEqual(failures, [])
        self.assertEqual(len(self.batch_calls(calls)), 1)
        self.assertEqual(len(self.single_calls(calls)), 6, calls)
        self.assertEqual(
            [block.text for block, _ in parse_srt(output, "\n")],
            [f"补译{index}" for index in range(1, 7)],
        )
        self.assert_structure(output)

    def test_unparsable_batch_falls_back_to_whole_window(self) -> None:
        calls, output, failures = self.run_window(
            "抱歉，我无法完成这个请求。", SOURCES
        )

        self.assertEqual(failures, [])
        self.assertEqual(len(self.single_calls(calls)), 6, calls)
        self.assert_structure(output)

    def test_majority_damage_falls_back_to_whole_window(self) -> None:
        # 6 条里 3 条被压成极短译文：半数损坏 => 整窗不可信，6 次单句。
        batch = "\n".join(
            [
                "[#1] 好",
                "[#2] 嗯",
                "[#3] 是",
                f"[#4] {TRANSLATIONS[3]}",
                f"[#5] {TRANSLATIONS[4]}",
                f"[#6] {TRANSLATIONS[5]}",
            ]
        )
        long_sources = [
            "今日はとても楽しかったです、本当にいい一日でした、また明日も来たいですね",
            "明日の予定をちゃんと確認しておいてくださいね、忘れないでください",
            "窓を閉めてください、風が強くなってきましたから、風邪をひきますよ",
            SOURCES[3],
            SOURCES[4],
            SOURCES[5],
        ]
        calls, output, failures = self.run_window(
            batch, long_sources, window_size=6
        )

        self.assertEqual(failures, [])
        self.assertEqual(len(self.single_calls(calls)), 6, calls)

    # ------------------------------------------------------------------ #
    # 6) 兼容性：旧布尔接口与新的定位接口同规则
    # ------------------------------------------------------------------ #
    def test_suspicion_bool_stays_compatible_with_new_locator(self) -> None:
        window = window_of(SOURCES)
        clean = {str(index + 1): text for index, text in enumerate(TRANSLATIONS)}
        self.assertFalse(_window_translation_is_suspicious(window, clean))
        self.assertEqual(identify_suspicious_cues(window, clean), set())

        leaked = dict(clean)
        leaked["3"] = f"{TRANSLATIONS[2]} [#7]"
        self.assertTrue(_window_translation_is_suspicious(window, leaked))
        self.assertEqual(identify_suspicious_cues(window, leaked), {"3"})

        empty = dict(clean)
        empty["2"] = ""
        self.assertTrue(_window_translation_is_suspicious(window, empty))
        self.assertEqual(identify_suspicious_cues(window, empty), {"2"})

        history = ((SOURCES[0], COPIED),)
        copied = dict(clean)
        copied["6"] = COPIED
        self.assertTrue(
            _window_translation_is_suspicious(window, copied, history)
        )
        self.assertEqual(
            identify_suspicious_cues(window, copied, history), {"6"}
        )

    # ------------------------------------------------------------------ #
    # 7) 对齐三态：完整 / 重编号 / 缺项 / 灾难
    # ------------------------------------------------------------------ #
    def test_align_window_response_three_states(self) -> None:
        ids = ["1", "2", "3"]

        full = align_window_response("[#1] a\n[#2] b\n[#3] c", ids)
        self.assertFalse(full.is_catastrophic)
        self.assertEqual(full.aligned, {"1": "a", "2": "b", "3": "c"})
        self.assertEqual(full.missing_ids, [])
        self.assertFalse(full.reordered)

        reordered = align_window_response("[#1] a\n[#2] b\n[#3] c", ["7", "8", "9"])
        self.assertFalse(reordered.is_catastrophic)
        self.assertTrue(reordered.reordered)
        self.assertEqual(reordered.aligned, {"7": "a", "8": "b", "9": "c"})

        partial = align_window_response("[#1] a\n[#3] c", ids)
        self.assertFalse(partial.is_catastrophic)
        self.assertEqual(partial.aligned, {"1": "a", "3": "c"})
        self.assertEqual(partial.missing_ids, ["2"])

        duplicated = align_window_response("[#1] a\n[#1] b\n[#2] c", ids)
        self.assertTrue(duplicated.is_catastrophic)
        self.assertEqual(duplicated.aligned, {})

        unknown = align_window_response("[#1] a\n[#4] d", ids)
        self.assertTrue(unknown.is_catastrophic)

        prefix = align_window_response("[#1] a\n[#2] b", ids)
        # 1..M 残缺前缀与整窗重编号不可区分，必须熔断而不是位置猜测。
        self.assertTrue(prefix.is_catastrophic)


if __name__ == "__main__":
    unittest.main()
