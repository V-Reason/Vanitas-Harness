"""QC 标定台的数据集门槛与 gate 行为。"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "tests" / "benchmarks"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import bench_qc  # noqa: E402


class BenchQcTests(unittest.TestCase):
    def test_dataset_size_and_negative_floor(self) -> None:
        rows = bench_qc.load_rows(bench_qc.DEFAULT_GOLDEN)
        negatives = sum(1 for row in rows if not row["expected"])
        self.assertGreaterEqual(len(rows), bench_qc.MIN_ROWS)
        self.assertGreaterEqual(negatives, bench_qc.MIN_NEGATIVES)

    def test_bench_passes(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = bench_qc.main(["--json"])
        payload = json.loads(out.getvalue())
        self.assertEqual(code, 0)
        self.assertTrue(payload["passed"])
        self.assertEqual(payload["failures"], [])
        self.assertGreaterEqual(payload["dataset"]["rows"], 120)

    def test_shrunk_dataset_fails(self) -> None:
        tmp = ROOT / "tests" / "reports" / "_qc_golden_shrunk.tsv"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        try:
            tmp.write_text(
                "case\t1\tこんにちは\t你好\t\tone\n", encoding="utf-8"
            )
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = bench_qc.main(["--golden", str(tmp), "--json"])
            payload = json.loads(out.getvalue())
            self.assertEqual(code, 1)
            self.assertFalse(payload["passed"])
            self.assertTrue(
                any("行数" in item or "负例" in item for item in payload["failures"])
            )
        finally:
            tmp.unlink(missing_ok=True)

    def test_review_gate_rejects_low_precision(self) -> None:
        ok, verdict = bench_qc.code_verdict(
            "kana", {"tp": 1, "fp": 1, "fn": 0}
        )
        self.assertFalse(ok)
        self.assertIn("P", verdict)
        ok, _ = bench_qc.code_verdict("kana", {"tp": 2, "fp": 0, "fn": 0})
        self.assertTrue(ok)

    def test_info_code_is_not_gated(self) -> None:
        ok, verdict = bench_qc.code_verdict(
            "term_drift", {"tp": 1, "fp": 9, "fn": 0}
        )
        self.assertTrue(ok)
        self.assertEqual(verdict, "info")


if __name__ == "__main__":
    unittest.main()
