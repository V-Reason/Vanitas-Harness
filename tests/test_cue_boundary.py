"""跨 cue 断词右吸附：真实治疗对、守卫、幂等与坏表。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vanitas.asr_repair import load_joins as compat_load_joins  # noqa: E402
from vanitas.asr_repair import parse_cues  # noqa: E402
from vanitas.cue_boundary import (  # noqa: E402
    JoinRule,
    load_joins,
    parse_timing,
    transfer_texts,
)
from vanitas.errors import TranslateError  # noqa: E402

JOINS_PATH = ROOT / "config" / "asr_joins.tsv"
JOINS = load_joins(JOINS_PATH)

TIMING_0_1 = "00:00:00,000 --> 00:00:01,000"
TIMING_1_2 = "00:00:01,000 --> 00:00:02,000"
TIMING_2_3 = "00:00:02,000 --> 00:00:03,000"

THERAPY_CUES = [
    (
        "00:00:15,400",
        "00:00:22,070",
        "お履き物はこちらに履き替えていただき、\n"
        "履き替えていただきましたらカ",
    ),
    ("00:00:22,071", "00:00:23,390", "ウンターへお越しください。"),
]


def make_srt(cues, *, eol: str = "\n", bom: bool = False) -> str:
    blocks = []
    for index, (start, end, text) in enumerate(cues, start=1):
        body = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", eol)
        blocks.append(f"{index}{eol}{start} --> {end}{eol}{body}")
    prefix = "\ufeff" if bom else ""
    return prefix + (eol * 2).join(blocks) + eol


def texts_and_timings(cues) -> tuple[list[str], list[str]]:
    parsed = parse_cues(make_srt(cues))
    return [cue.text for cue in parsed], [cue.timing for cue in parsed]


class RuleTableTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def write(self, text: str) -> Path:
        path = self.dir / "joins.tsv"
        path.write_text(text, encoding="utf-8")
        return path

    def test_asr_repair_reexports_the_same_loader(self) -> None:
        self.assertIs(compat_load_joins, load_joins)
        self.assertEqual(compat_load_joins(JOINS_PATH), JOINS)

    def test_bad_line_reports_path_and_line(self) -> None:
        path = self.write("只有一列\n")
        with self.assertRaises(TranslateError) as ctx:
            load_joins(path)
        self.assertIn(f"{path}:1:", str(ctx.exception))
        self.assertIn("3 列", str(ctx.exception))

    def test_duplicate_key_reports_path_and_line(self) -> None:
        path = self.write(
            "カ\tウンター\tカウンター\nカ\tウンター\t別の語\n"
        )
        with self.assertRaises(TranslateError) as ctx:
            load_joins(path)
        self.assertIn(f"{path}:2:", str(ctx.exception))
        self.assertIn("重复键", str(ctx.exception))

    def test_parse_timing_accepts_comma_and_dot(self) -> None:
        parsed = parse_timing("00:00:01,5 --> 00:00:02.250")
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed[:2], (1500, 2250))
        self.assertEqual(
            parsed[2:], ("00:00:01,500", "00:00:02,250")
        )
        self.assertIsNone(parse_timing("不是时间轴"))


class TransferTests(unittest.TestCase):
    def test_real_therapy_cues_2_3_move_counter_tail(self) -> None:
        texts, timings = texts_and_timings(THERAPY_CUES)
        moved, changed = transfer_texts(texts, timings, JOINS)
        self.assertEqual(changed, 1)
        self.assertEqual(len(moved), 2)
        self.assertEqual(
            moved[0],
            "お履き物はこちらに履き替えていただき、"
            "履き替えていただきましたら",
        )
        self.assertEqual(moved[1], "カウンターへお越しください。")
        # 入参不被就地修改。
        self.assertTrue(texts[0].endswith("カ"))
        self.assertTrue(timings[0].startswith("00:00:15,400"))

    def test_gap_boundary_500_passes_and_501_skips(self) -> None:
        texts = ["ましたらカ", "ウンターへ"]
        passing = [
            "00:00:00,000 --> 00:00:01,000",
            "00:00:01,500 --> 00:00:02,000",
        ]
        moved, changed = transfer_texts(texts, passing, JOINS)
        self.assertEqual(changed, 1)
        self.assertEqual(moved, ["ましたら", "カウンターへ"])

        skipped = [
            "00:00:00,000 --> 00:00:01,000",
            "00:00:01,501 --> 00:00:02,000",
        ]
        moved, changed = transfer_texts(texts, skipped, JOINS)
        self.assertEqual(changed, 0)
        self.assertEqual(moved, texts)

    def test_empty_prefix_is_skipped(self) -> None:
        texts = ["カ", "ウンターへ"]
        moved, changed = transfer_texts(
            texts, [TIMING_0_1, TIMING_1_2], JOINS
        )
        self.assertEqual(changed, 0)
        self.assertEqual(moved, texts)

    def test_punctuation_only_prefix_is_skipped(self) -> None:
        texts = ["、カ", "ウンターへ"]
        moved, changed = transfer_texts(
            texts, [TIMING_0_1, TIMING_1_2], JOINS
        )
        self.assertEqual(changed, 0)
        self.assertEqual(moved, texts)

    def test_prefix_ending_with_sentence_particle_is_allowed(self) -> None:
        texts = ["そうですねカ", "ウンターへ"]
        moved, changed = transfer_texts(
            texts, [TIMING_0_1, TIMING_1_2], JOINS
        )
        self.assertEqual(changed, 1)
        self.assertEqual(moved, ["そうですね", "カウンターへ"])

    def test_prefix_ending_with_punctuation_is_allowed(self) -> None:
        texts = ["ましたら、カ", "ウンターへ"]
        moved, changed = transfer_texts(
            texts, [TIMING_0_1, TIMING_1_2], JOINS
        )
        self.assertEqual(changed, 1)
        self.assertEqual(moved, ["ましたら、", "カウンターへ"])

    def test_empty_side_is_skipped(self) -> None:
        for texts in (["ましたらカ", ""], ["", "ウンターへ"]):
            with self.subTest(texts=texts):
                moved, changed = transfer_texts(
                    texts, [TIMING_0_1, TIMING_1_2], JOINS
                )
                self.assertEqual(changed, 0)
                self.assertEqual(moved, texts)

    def test_multiline_source_keeps_internal_newlines(self) -> None:
        # 真实治疗源 #2 是多行；tail 在最后一行末尾，仍应吸附。
        texts = ["お履き物はこちらに、\nましたらカ", "ウンターへ"]
        moved, changed = transfer_texts(
            texts, [TIMING_0_1, TIMING_1_2], JOINS
        )
        self.assertEqual(changed, 1)
        self.assertEqual(moved, ["お履き物はこちらに、\nましたら", "カウンターへ"])

    def test_multiline_right_side_is_preserved(self) -> None:
        texts = ["ましたらカ", "ウンター\nへお越しください。"]
        moved, changed = transfer_texts(
            texts, [TIMING_0_1, TIMING_1_2], JOINS
        )
        self.assertEqual(changed, 1)
        self.assertEqual(moved, ["ましたら", "カウンター\nへお越しください。"])

    def test_second_run_is_idempotent(self) -> None:
        texts, timings = texts_and_timings(THERAPY_CUES)
        first, changed = transfer_texts(texts, timings, JOINS)
        second, again = transfer_texts(first, timings, JOINS)
        self.assertEqual(changed, 1)
        self.assertEqual(again, 0)
        self.assertEqual(second, first)

    def test_chained_boundary_is_not_moved_twice(self) -> None:
        texts = ["ましたらカ", "ウンター、手", "を繋ぐ"]
        moved, changed = transfer_texts(
            texts, [TIMING_0_1, TIMING_1_2, TIMING_2_3], JOINS
        )
        self.assertEqual(changed, 1)
        self.assertEqual(moved, ["ましたらカ", "ウンター、", "手を繋ぐ"])

    def test_overlapping_timeline_is_skipped(self) -> None:
        texts = ["ましたらカ", "ウンターへ"]
        timings = [
            "00:00:00,000 --> 00:00:03,000",
            "00:00:02,000 --> 00:00:04,000",
        ]
        moved, changed = transfer_texts(texts, timings, JOINS)
        self.assertEqual(changed, 0)
        self.assertEqual(moved, texts)

    def test_unparseable_timing_is_skipped(self) -> None:
        texts = ["ましたらカ", "ウンターへ"]
        moved, changed = transfer_texts(
            texts, ["不是时间轴", TIMING_1_2], JOINS
        )
        self.assertEqual(changed, 0)
        self.assertEqual(moved, texts)

    def test_rule_replacement_must_equal_tail_plus_head(self) -> None:
        texts = ["ましたらカ", "ウンターへ"]
        rules = (JoinRule("カ", "ウンター", "カウンタ"),)
        moved, changed = transfer_texts(
            texts, [TIMING_0_1, TIMING_1_2], rules
        )
        self.assertEqual(changed, 0)
        self.assertEqual(moved, texts)

    def test_rule_with_newline_or_particle_tail_is_skipped(self) -> None:
        texts = ["ましたらカ", "ウンターへ"]
        rules = (JoinRule("カ", "ウンター", "カウン\nター"),)
        moved, changed = transfer_texts(
            texts, [TIMING_0_1, TIMING_1_2], rules
        )
        self.assertEqual(changed, 0)
        self.assertEqual(moved, texts)

        particle_rules = (JoinRule("を", "持", "を持"),)
        particle_texts = ["ましたらを", "持ち帰る"]
        moved, changed = transfer_texts(
            particle_texts, [TIMING_0_1, TIMING_1_2], particle_rules
        )
        self.assertEqual(changed, 0)
        self.assertEqual(moved, particle_texts)

    def test_fewer_than_two_cues_returns_unchanged(self) -> None:
        self.assertEqual(transfer_texts([], [], JOINS), ([], 0))
        self.assertEqual(
            transfer_texts(["ましたらカ"], [TIMING_0_1], JOINS),
            (["ましたらカ"], 0),
        )


if __name__ == "__main__":
    unittest.main()
