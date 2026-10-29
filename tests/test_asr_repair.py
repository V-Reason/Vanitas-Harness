"""ASR 跨 cue 断词修复：真实断词、守卫、幂等与 CLI 退出码。"""

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

import repair_asr_srt  # noqa: E402
from vanitas.asr_repair import (  # noqa: E402
    JoinRule,
    load_joins,
    parse_cues,
    render_srt,
    repair_cues,
)
from vanitas.errors import TranslateError  # noqa: E402

JOINS_PATH = ROOT / "config" / "asr_joins.tsv"
JOINS = load_joins(JOINS_PATH)


def make_srt(cues, *, eol: str = "\n", bom: bool = False) -> str:
    blocks = []
    for index, (start, end, text) in enumerate(cues, start=1):
        body = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", eol)
        blocks.append(f"{index}{eol}{start} --> {end}{eol}{body}")
    prefix = "\ufeff" if bom else ""
    return prefix + (eol * 2).join(blocks) + eol


THERAPY_CUES = [
    (
        "00:00:15,400",
        "00:00:22,070",
        "お履き物はこちらに履き替えていただき、\n"
        "履き替えていただきましたらカ",
    ),
    ("00:00:22,071", "00:00:23,390", "ウンターへお越しください。"),
]


class JoinTableTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def write(self, text: str) -> Path:
        path = self.dir / "joins.tsv"
        path.write_text(text, encoding="utf-8")
        return path

    def test_default_table_has_seven_ordered_rows(self) -> None:
        expected = [
            ("カ", "ウンター", "カウンター"),
            ("手", "を", "手を"),
            ("寝", "込", "寝込"),
            ("綿", "棒", "綿棒"),
            ("心", "を", "心を"),
            ("気", "持ち", "気持ち"),
            ("ホテル", "じゃ", "ホテルじゃ"),
        ]
        self.assertEqual(
            [
                (rule.left_tail, rule.right_head, rule.replacement)
                for rule in JOINS
            ],
            expected,
        )

    def test_bad_line_reports_path_and_line(self) -> None:
        path = self.write("カ\tウンター\tカウンター\n只有一列\n")
        with self.assertRaises(TranslateError) as ctx:
            load_joins(path)
        self.assertIn(f"{path}:2:", str(ctx.exception))

    def test_duplicate_key_reports_path_and_line(self) -> None:
        path = self.write(
            "カ\tウンター\tカウンター\nカ\tウンター\t別の語\n"
        )
        with self.assertRaises(TranslateError) as ctx:
            load_joins(path)
        self.assertIn(f"{path}:2:", str(ctx.exception))
        self.assertIn("重复键", str(ctx.exception))


class RepairAlgorithmTests(unittest.TestCase):
    def test_real_therapy_cues_merge_to_counter(self) -> None:
        result = repair_cues(parse_cues(make_srt(THERAPY_CUES)), JOINS)
        self.assertEqual(result.merged, 1)
        self.assertEqual(len(result.cues), 1)
        self.assertEqual(result.cues[0].index, "1")
        self.assertEqual(
            result.cues[0].text,
            "お履き物はこちらに履き替えていただき、"
            "履き替えていただきましたらカウンターへお越しください。",
        )
        self.assertEqual(
            result.cues[0].timing,
            "00:00:15,400 --> 00:00:23,390",
        )

    def test_timeline_uses_left_start_right_end_and_renumbers(self) -> None:
        text = make_srt(
            [
                ("00:00:00,000", "00:00:01,000", "aaカ"),
                ("00:00:01,000", "00:00:02,000", "ウンターbb"),
                ("00:00:03,000", "00:00:04,000", "cc"),
            ]
        )
        result = repair_cues(parse_cues(text), JOINS)
        self.assertEqual(result.merged, 1)
        self.assertEqual(
            [(cue.index, cue.timing, cue.text) for cue in result.cues],
            [
                ("1", "00:00:00,000 --> 00:00:02,000", "aaカウンターbb"),
                ("2", "00:00:03,000 --> 00:00:04,000", "cc"),
            ],
        )
        rendered = render_srt(result.cues)
        self.assertTrue(rendered.startswith("1\n"))
        self.assertIn("\n\n2\n", rendered)

    def test_merged_duration_over_eight_seconds_is_skipped(self) -> None:
        text = make_srt(
            [
                ("00:00:00,000", "00:00:08,000", "aaカ"),
                ("00:00:08,000", "00:00:09,000", "ウンター"),
            ]
        )
        result = repair_cues(parse_cues(text), JOINS)
        self.assertEqual(result.merged, 0)
        self.assertEqual(len(result.cues), 2)
        self.assertIn("超过 8000ms", result.skipped[0].reason)

    def test_gap_boundary_is_inclusive(self) -> None:
        passing = make_srt(
            [
                ("00:00:00,000", "00:00:01,000", "aaカ"),
                ("00:00:02,000", "00:00:03,000", "ウンター"),
            ]
        )
        self.assertEqual(
            repair_cues(parse_cues(passing), JOINS).merged, 1
        )

        skipped = make_srt(
            [
                ("00:00:00,000", "00:00:01,000", "aaカ"),
                ("00:00:02,001", "00:00:03,000", "ウンター"),
            ]
        )
        result = repair_cues(parse_cues(skipped), JOINS)
        self.assertEqual(result.merged, 0)
        self.assertIn("1001ms", result.skipped[0].reason)
        self.assertIn("1000ms", result.skipped[0].reason)

    def test_overlap_is_skipped(self) -> None:
        text = make_srt(
            [
                ("00:00:00,000", "00:00:03,000", "aaカ"),
                ("00:00:02,000", "00:00:04,000", "ウンター"),
            ]
        )
        result = repair_cues(parse_cues(text), JOINS)
        self.assertEqual(result.merged, 0)
        self.assertEqual(result.skipped[0].reason, "right.start < left.end")

    def test_left_prefix_empty_is_skipped(self) -> None:
        text = make_srt(
            [
                ("00:00:00,000", "00:00:01,000", "カ"),
                ("00:00:01,000", "00:00:02,000", "ウンターへ"),
            ]
        )
        result = repair_cues(parse_cues(text), JOINS)
        self.assertEqual(result.merged, 0)
        self.assertEqual(result.skipped[0].reason, "left 前置为空")

    def test_bom_crlf_and_multiline_body_are_normalized(self) -> None:
        text = make_srt(
            [
                ("00:00:00,000", "00:00:01,000", "前半\nカ"),
                ("00:00:01,000", "00:00:02,000", "ウンター後半"),
            ],
            eol="\r\n",
            bom=True,
        )
        result = repair_cues(parse_cues(text), JOINS)
        self.assertEqual(result.merged, 1)
        self.assertEqual(result.cues[0].text, "前半カウンター後半")
        self.assertNotIn("\n", result.cues[0].text)
        rendered = render_srt(result.cues)
        self.assertNotIn("\r", rendered)
        self.assertFalse(rendered.startswith("\ufeff"))

    def test_second_run_is_idempotent(self) -> None:
        first = repair_cues(parse_cues(make_srt(THERAPY_CUES)), JOINS)
        rendered = render_srt(first.cues)
        second = repair_cues(parse_cues(rendered), JOINS)
        self.assertEqual(second.merged, 0)
        self.assertEqual(render_srt(second.cues), rendered)

    def test_chained_merge_runs_to_fixed_point(self) -> None:
        rules = (
            JoinRule("カ", "ウンター", "カウンター"),
            JoinRule("ー", "へ", "ーへ"),
        )
        text = make_srt(
            [
                ("00:00:00,000", "00:00:01,000", "aaカ"),
                ("00:00:01,000", "00:00:02,000", "ウンター"),
                ("00:00:02,000", "00:00:03,000", "へcc"),
            ]
        )
        result = repair_cues(parse_cues(text), rules)
        self.assertEqual(result.merged, 2)
        self.assertEqual(len(result.cues), 1)
        self.assertEqual(result.cues[0].text, "aaカウンターへcc")

    def test_non_monotonic_result_is_rejected(self) -> None:
        text = make_srt(
            [
                ("00:00:05,000", "00:00:06,000", "aa"),
                ("00:00:01,000", "00:00:02,000", "bb"),
            ]
        )
        with self.assertRaises(TranslateError):
            repair_cues(parse_cues(text), JOINS)


class RepairCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.src_dir = self.dir / "src"
        self.src_dir.mkdir()
        self.out_dir = self.dir / "out"

    def write_source(self, name: str = "a.srt", text: str | None = None) -> Path:
        path = self.src_dir / name
        path.write_text(
            text if text is not None else make_srt(THERAPY_CUES),
            encoding="utf-8",
        )
        return path

    def run_cli(self, args: list[str]) -> tuple[int, str, str]:
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
            stderr
        ):
            code = repair_asr_srt.main(args)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_dry_run_does_not_write_and_write_outputs_stem(self) -> None:
        source = self.write_source()
        original = source.read_text(encoding="utf-8")
        code, stdout, _stderr = self.run_cli(
            ["--source-dir", str(self.src_dir), "--out-dir", str(self.out_dir)]
        )
        self.assertEqual(code, 0)
        self.assertIn("dry-run", stdout)
        target = self.out_dir / "a.srt"
        self.assertFalse(target.exists())

        code, stdout, _stderr = self.run_cli(
            [
                "--source-dir",
                str(self.src_dir),
                "--out-dir",
                str(self.out_dir),
                "--write",
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn("已写", stdout)
        self.assertTrue(target.exists())
        self.assertIn("カウンター", target.read_text(encoding="utf-8"))
        self.assertFalse((self.out_dir / "a.join.srt").exists())
        self.assertEqual(source.read_text(encoding="utf-8"), original)

    def test_check_returns_one_when_pending(self) -> None:
        source = self.write_source()
        code, stdout, _stderr = self.run_cli(
            ["--check", "--out-dir", str(self.out_dir), str(source)]
        )
        self.assertEqual(code, 1)
        self.assertIn("待合并", stdout)
        self.assertFalse((self.out_dir / "a.srt").exists())

    def test_check_returns_zero_when_clean(self) -> None:
        source = self.write_source(
            text=make_srt(
                [("00:00:00,000", "00:00:01,000", "手を繋ぐ")]
            )
        )
        code, stdout, _stderr = self.run_cli(
            ["--check", "--out-dir", str(self.out_dir), str(source)]
        )
        self.assertEqual(code, 0)
        self.assertIn("没有待合并", stdout)

    def test_same_directory_is_rejected(self) -> None:
        source = self.write_source()
        code, _stdout, stderr = self.run_cli(
            ["--out-dir", str(source.parent), str(source)]
        )
        self.assertEqual(code, 2)
        self.assertIn("不能等于源目录", stderr)

    def test_zh_srt_is_rejected(self) -> None:
        source = self.write_source("a.zh.srt")
        code, _stdout, stderr = self.run_cli(
            ["--out-dir", str(self.out_dir), str(source)]
        )
        self.assertEqual(code, 2)
        self.assertIn(".zh.srt", stderr)

    def test_bad_join_line_returns_two(self) -> None:
        source = self.write_source()
        joins = self.dir / "bad.tsv"
        joins.write_text("只有一列\n", encoding="utf-8")
        code, _stdout, stderr = self.run_cli(
            [
                "--joins",
                str(joins),
                "--out-dir",
                str(self.out_dir),
                str(source),
            ]
        )
        self.assertEqual(code, 2)
        self.assertIn(f"{joins}:1:", stderr)

    def test_existing_target_is_not_overwritten(self) -> None:
        source = self.write_source()
        self.out_dir.mkdir()
        target = self.out_dir / "a.srt"
        target.write_text("保留", encoding="utf-8")
        code, _stdout, stderr = self.run_cli(
            [
                "--write",
                "--out-dir",
                str(self.out_dir),
                str(source),
            ]
        )
        self.assertEqual(code, 2)
        self.assertIn("拒绝覆盖", stderr)
        self.assertEqual(target.read_text(encoding="utf-8"), "保留")

    def test_json_reports_merged_and_skipped(self) -> None:
        source = self.write_source()
        code, stdout, _stderr = self.run_cli(
            [
                "--check",
                "--json",
                "--out-dir",
                str(self.out_dir),
                str(source),
            ]
        )
        self.assertEqual(code, 1)
        payload = json.loads(stdout)
        self.assertEqual(payload["summary"]["files"], 1)
        self.assertEqual(payload["summary"]["merged"], 1)
        self.assertTrue(payload["summary"]["pending"])
        self.assertEqual(payload["files"][0]["merged"], 1)


if __name__ == "__main__":
    unittest.main()
