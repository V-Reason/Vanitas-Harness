"""语义黄金靶场基建的离线单测：数据契约、Oracle、聚合数学、Mock 全链路。

全部用例不触网：真机入口 ``build_chat_factory`` 不在测试里调用，
``run_benchmark`` 通过 ``chat_factory`` 注入 FakeEngine。
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.benchmarks import bench_semantic_quality as bench  # noqa: E402

TSV = ROOT / "tests" / "data" / "golden_asmr_benchmark.tsv"
QUOTA = {
    "direction": 20,
    "negation": 16,
    "pronoun": 16,
    "conjunction": 16,
    "lexicon": 12,
}


class FakeEngine:
    """按源文回放预设译文的假引擎；实现 P3 复审通道与历史接口。"""

    task_template = "translate_asmr"
    glossary = None
    join_rules = ()

    def __init__(self, script: dict[str, str], critique_map: dict[str, str] | None = None):
        self.script = dict(script)
        self.critique_map = dict(critique_map or {})
        self.prompts: list[str] = []
        self.critique_prompts: list[str] = []
        self._history: list[tuple[str, str]] = []

    # -- 翻译通道 ----------------------------------------------------------
    def _lookup(self, text: str) -> str:
        for source, translation in self.script.items():
            if source in text:
                return translation
        return "（未提供译文）"

    @staticmethod
    def _parse_batch(prompt: str) -> list[tuple[str, str]]:
        items: list[tuple[str, str]] = []
        current_id: str | None = None
        buffer: list[str] = []
        for line in prompt.splitlines():
            match = re.match(r"^\[#([^\]]+)\]\s*(.*)$", line)
            if match:
                if current_id is not None:
                    items.append((current_id, "\n".join(buffer).strip()))
                current_id = match.group(1)
                buffer = [match.group(2)]
            elif current_id is not None:
                buffer.append(line)
        if current_id is not None:
            items.append((current_id, "\n".join(buffer).strip()))
        return items

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        rendered = []
        for cue_id, text in self._parse_batch(prompt):
            translation = self._lookup(text)
            self._history.append((text, translation))
            rendered.append(f"[#{cue_id}] {translation}")
        return "\n".join(rendered)

    # -- 复审通道 ----------------------------------------------------------
    def critique(self, text: str, *, max_tokens: int = 96) -> str:
        self.critique_prompts.append(text)
        for source, corrected in self.critique_map.items():
            if source in text:
                return corrected
        return "【保持】"

    # -- 引擎历史接口 ------------------------------------------------------
    def history_snapshot(self):
        return tuple(self._history)

    def reset_history(self) -> None:
        self._history.clear()

    def restore_history(self, snapshot) -> None:
        self._history = list(snapshot)

    def set_entity_engine(self, engine) -> None:
        return None


def _write_tsv(directory: Path, rows: list[list[str]]) -> Path:
    path = directory / "bench.tsv"
    header = "#id\tcategory\tsource\tcontext_cues\texpected_pattern\tforbidden_pattern\treference_zh\n"
    body = "\n".join("\t".join(row) for row in rows) + "\n"
    path.write_text(header + body, encoding="utf-8", newline="\n")
    return path


class BenchmarkDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cues = bench.load_golden_benchmark(TSV)

    def test_row_count_and_quota(self) -> None:
        self.assertEqual(len(self.cues), 80)
        counted: dict[str, int] = {}
        for cue in self.cues:
            counted[cue.category] = counted.get(cue.category, 0) + 1
        self.assertEqual(counted, QUOTA)

    def test_ids_unique_and_prefix_matches_category(self) -> None:
        seen = set()
        for cue in self.cues:
            self.assertNotIn(cue.id, seen)
            seen.add(cue.id)
            self.assertTrue(cue.id.startswith(bench.ID_PREFIX[cue.category]))

    def test_every_data_line_has_seven_columns(self) -> None:
        raw = TSV.read_text(encoding="utf-8-sig")
        index = 0
        for line in raw.splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            index += 1
            self.assertEqual(len(line.split("\t")), bench.COLUMNS, line[:40])
        self.assertEqual(index, 80)

    def test_all_patterns_compile(self) -> None:
        for cue in self.cues:
            re.compile(cue.expected_pattern)
            re.compile(cue.forbidden_pattern)

    def test_reference_satisfies_its_own_oracle(self) -> None:
        for cue in self.cues:
            passed, reason = bench.evaluate_cue(cue.reference_zh, cue)
            self.assertTrue(passed, f"{cue.id}: {reason}")

    def test_source_is_japanese_and_context_optional(self) -> None:
        for cue in self.cues:
            self.assertRegex(cue.source, r"[\u3040-\u30ff]")
            self.assertEqual(hasattr(cue.context_cues, "__len__"), True)
            for line in cue.context_cues:
                self.assertTrue(line.strip())

    def test_rejects_wrong_column_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_tsv(
                Path(tmp),
                [["CAT_A_001", "direction", "あ", "NONE", "あ", "い"]],
            )
            with self.assertRaises(bench.BenchDataError):
                bench.load_golden_benchmark(path)

    def test_rejects_bad_regex(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_tsv(
                Path(tmp),
                [["CAT_A_001", "direction", "あ", "NONE", "(", "い", "あ"]],
            )
            with self.assertRaises(bench.BenchDataError):
                bench.load_golden_benchmark(path)

    def test_rejects_id_category_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_tsv(
                Path(tmp),
                [["CAT_B_001", "direction", "あ", "NONE", "あ", "い", "あ"]],
            )
            with self.assertRaises(bench.BenchDataError):
                bench.load_golden_benchmark(path)

    def test_rejects_reference_that_fails_its_own_pattern(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_tsv(
                Path(tmp),
                [["CAT_A_001", "direction", "あ", "NONE", "触碰", "い", "你好"]],
            )
            with self.assertRaises(bench.BenchDataError):
                bench.load_golden_benchmark(path)


class OracleTests(unittest.TestCase):
    def _cue(self) -> bench.GoldenCue:
        return bench.GoldenCue(
            id="CAT_A_001",
            category="direction",
            source="お客様から私たちに触れることは禁止となっております",
            context_cues=(),
            expected_pattern=r"(?:禁止|不允许).{0,8}(?:客人|您).{0,6}(?:触碰|碰|摸)",
            forbidden_pattern=r"(?:我们|我).{0,8}(?:触碰|碰|摸).{0,8}(?:客人|您)",
            reference_zh="禁止客人触碰我们哦",
        )

    def test_pass(self) -> None:
        passed, reason = bench.evaluate_cue("禁止客人触碰我们哦", self._cue())
        self.assertTrue(passed)
        self.assertIsNone(reason)

    def test_empty_translation(self) -> None:
        passed, reason = bench.evaluate_cue("   ", self._cue())
        self.assertFalse(passed)
        self.assertIn("为空", reason or "")

    def test_hard_validation_blocks_kana(self) -> None:
        passed, reason = bench.evaluate_cue("禁止客人触碰我们哦だよ", self._cue())
        self.assertFalse(passed)
        self.assertIn("硬校验失败", reason or "")

    def test_missing_expected(self) -> None:
        passed, reason = bench.evaluate_cue("客人请不要触碰哦", self._cue())
        self.assertFalse(passed)
        self.assertIn("未匹配必要特征", reason or "")

    def test_forbidden_hit(self) -> None:
        cue = bench.GoldenCue(
            id="CAT_E_001",
            category="lexicon",
            source="梵天で耳の奥を撫でてあげるね",
            context_cues=(),
            expected_pattern=r"梵天",
            forbidden_pattern=r"(?:羽毛|绒球|焚天)",
            reference_zh="用梵天轻抚耳道深处哦",
        )
        passed, reason = bench.evaluate_cue("把梵天当成羽毛绒球来用", cue)
        self.assertFalse(passed)
        self.assertIn("触发禁忌特征", reason or "")


class TransitionAndAggregateTests(unittest.TestCase):
    def test_classify_transition_matrix(self) -> None:
        cases = {
            (False, True, True): "repaired",
            (True, False, True): "degraded",
            (True, False, False): "kept",
            (False, False, False): "rejected",
            (False, False, True): "rejected",
        }
        for args, expected in cases.items():
            self.assertEqual(bench.classify_transition(*args), expected, args)

    def _report(self) -> bench.BenchmarkReport:
        def cue(cid: str, category: str) -> bench.GoldenCue:
            return bench.GoldenCue(
                id=cid,
                category=category,
                source="あ",
                context_cues=(),
                expected_pattern="x",
                forbidden_pattern="y",
                reference_zh="x",
            )

        results = [
            bench.CueEvalResult(cue("CAT_A_001", "direction"), "", "", "repaired", False, True, "r", None),
            bench.CueEvalResult(cue("CAT_A_002", "direction"), "", "", "kept", True, True, None, None),
            bench.CueEvalResult(cue("CAT_A_003", "direction"), "", "", "degraded", True, False, None, "r"),
            bench.CueEvalResult(cue("CAT_B_001", "negation"), "", "", "rejected", False, False, "r", "r"),
        ]
        return bench.BenchmarkReport(
            mode="critique",
            window=5,
            model="hymt",
            profile="2",
            port=4062,
            elapsed_s=1.0,
            results=results,
        )

    def test_rates_and_totals(self) -> None:
        report = self._report()
        self.assertEqual(report.total_cues, 4)
        self.assertEqual(report.passed_draft, 2)
        self.assertEqual(report.passed_final, 2)
        self.assertEqual(report.repaired, 1)
        self.assertEqual(report.degraded, 1)
        self.assertAlmostEqual(report.repair_rate, 0.5)  # 1 / (4-2)
        self.assertAlmostEqual(report.degrade_rate, 0.5)  # 1 / 2

    def test_category_rows(self) -> None:
        rows = {row["category"]: row for row in self._report().rows()}
        self.assertEqual(rows["direction"]["cues"], 3)
        self.assertEqual(rows["direction"]["passed_draft"], 2)
        self.assertEqual(rows["direction"]["repaired"], 1)
        self.assertEqual(rows["negation"]["final_rate"], 0.0)

    def test_zero_denominator_is_none(self) -> None:
        report = bench.BenchmarkReport(
            mode="baseline",
            window=5,
            model="hymt",
            profile="2",
            port=4062,
            elapsed_s=0.1,
            results=[],
        )
        self.assertIsNone(report.repair_rate)
        self.assertIsNone(report.degrade_rate)
        self.assertEqual(report.rows(), [])

    def test_dashboard_and_payload(self) -> None:
        report = self._report()
        text = bench.render_dashboard(report)
        self.assertIn("ASMR", text)
        self.assertIn("CAT-A", text)
        payload = bench.report_payload(report)
        json.dumps(payload, ensure_ascii=False)
        self.assertEqual(payload["totals"]["cues"], 4)
        self.assertEqual(payload["mode"], "critique")


class MockPipelineTests(unittest.TestCase):
    """三条样本跑通 run_benchmark：基线 1/3 通过，复审修复 1 条。"""

    A_SRC = "お客様から私たちに触れることは禁止となっております"
    B_SRC = "私のこと、覚えていませんか?"
    E_SRC = "梵天で耳の奥を撫でてあげるね"

    def _tsv(self, directory: Path) -> Path:
        rows = [
            [
                "CAT_A_001",
                "direction",
                self.A_SRC,
                "NONE",
                r"(?:禁止|不允许).{0,8}(?:客人|您).{0,6}(?:触碰|碰|摸)",
                r"(?:我们|我).{0,8}(?:触碰|碰|摸).{0,8}(?:客人|您)",
                "禁止客人触碰我们哦",
            ],
            [
                "CAT_B_001",
                "negation",
                self.B_SRC,
                "久しぶりだね。",
                r"(?:不|没|难道不).{0,3}(?:记得|认出)",
                r"(?:你还记得|还记得我吗)",
                "你不记得我了吗？",
            ],
            [
                "CAT_E_001",
                "lexicon",
                self.E_SRC,
                "NONE",
                r"梵天",
                r"(?:羽毛|绒球|焚天)",
                "用梵天轻抚耳道深处哦",
            ],
        ]
        return _write_tsv(directory, rows)

    def test_baseline_and_critique(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cues = bench.load_golden_benchmark(self._tsv(Path(tmp)))
        script = {
            self.A_SRC: "禁止客人触碰我们哦",
            self.B_SRC: "我完全记得你哦",
            self.E_SRC: "用羽毛轻抚耳道深处哦",
        }
        engine = FakeEngine(
            script,
            critique_map={self.B_SRC: "你不记得我了吗？"},
        )
        baseline = bench.run_benchmark(
            cues, lambda on_stat=None: engine, mode="baseline", window=3
        )
        self.assertEqual(baseline.total_cues, 3)
        self.assertEqual(baseline.passed_draft, 1)
        self.assertEqual(baseline.passed_final, 1)
        self.assertTrue(all(item.transition == "unscored" for item in baseline.results))
        self.assertGreater(len(engine.prompts), 0)

        engine2 = FakeEngine(script, critique_map={self.B_SRC: "你不记得我了吗？"})
        critique = bench.run_benchmark(
            cues, lambda on_stat=None: engine2, mode="critique", window=3
        )
        self.assertEqual(critique.passed_final, 2)
        self.assertEqual(critique.repaired, 1)
        self.assertEqual(critique.degraded, 0)
        self.assertEqual(critique.critique_used, 1)
        transitions = {item.cue.id: item.transition for item in critique.results}
        self.assertEqual(transitions["CAT_B_001"], "repaired")
        self.assertEqual(transitions["CAT_A_001"], "kept")
        self.assertEqual(transitions["CAT_E_001"], "rejected")

    def test_force_critique_bypasses_triage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cues = bench.load_golden_benchmark(self._tsv(Path(tmp)))
        script = {
            self.A_SRC: "禁止客人触碰我们哦",
            self.B_SRC: "你不记得我了吗？",
            self.E_SRC: "用羽毛轻抚耳道深处哦",
        }
        engine = FakeEngine(script)
        report = bench.run_benchmark(
            cues,
            lambda on_stat=None: engine,
            mode="critique",
            window=3,
            force_critique=True,
        )
        self.assertEqual(report.critique_requests, 1)
        self.assertEqual(len(engine.critique_prompts), 1)
        self.assertIn(self.E_SRC, engine.critique_prompts[0])

    def test_report_payload_is_json_serializable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cues = bench.load_golden_benchmark(self._tsv(Path(tmp)))
        engine = FakeEngine(
            {
                self.A_SRC: "禁止客人触碰我们哦",
                self.B_SRC: "你不记得我了吗？",
                self.E_SRC: "用梵天轻抚耳道深处哦",
            }
        )
        report = bench.run_benchmark(
            cues, lambda on_stat=None: engine, mode="baseline", window=3
        )
        payload = bench.report_payload(report)
        text = json.dumps(payload, ensure_ascii=False)
        self.assertIn("CAT_A_001", text)
        self.assertEqual(payload["totals"]["passed_final"], 3)

    def test_srt_timeline_is_monotonic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cues = bench.load_golden_benchmark(self._tsv(Path(tmp)))
        text = bench.build_benchmark_srt(cues)
        blocks = list(bench.parse_srt(text, "\n"))
        self.assertEqual(len(blocks), 3)
        previous_end = -1
        for subtitle, _separator in blocks:
            start_text, end_text = subtitle.timing.split("-->")
            start = bench._timestamp  # 仅为覆盖命名空间，实际解析见下
            start_ms = self._to_ms(start_text.strip())
            end_ms = self._to_ms(end_text.strip())
            self.assertGreater(end_ms, start_ms)
            self.assertGreaterEqual(start_ms, previous_end)
            previous_end = end_ms

    @staticmethod
    def _to_ms(stamp: str) -> int:
        clock, milli = stamp.split(",")
        hour, minute, second = (int(part) for part in clock.split(":"))
        return ((hour * 60 + minute) * 60 + second) * 1000 + int(milli)


if __name__ == "__main__":
    unittest.main()
