"""QC 标定台：在 ``tests/data/qc_golden.tsv`` 上算 per-code P/R/F1。

口径
----
- 每个 TSV 行是一个“cue 级标注点”；同一 ``case_id`` 拼成一个 SRT 对。
- expected_code 为空表示该 cue 必须干净（不得出现任何 block/review）；
  多个 code 用 ``|`` 并列。- ``block`` 类只看精确率（必须 1.00）；``review`` 类 P≥0.85 且 R≥0.60；
  ``info`` 类只展示，不进 gate。文件级 ``pipeline_drift`` 不参与 cue 级 P/R。
- 数据集缩水也算失败：总行数 <120 或负例 <25 直接红。

用法：
    python tests/benchmarks/bench_qc.py [--json] [--golden path]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vanitas import qc  # noqa: E402
from vanitas.quality import Glossary  # noqa: E402

DEFAULT_GOLDEN = ROOT / "tests" / "data" / "qc_golden.tsv"
DEFAULT_GLOSSARY = ROOT / "config" / "glossary" / "asmr.tsv"
DEFAULT_ASR_TERMS = ROOT / "config" / "qc" / "asr_suspect.tsv"

MIN_ROWS = 120
MIN_NEGATIVES = 25
BLOCK_MIN_PRECISION = 1.00
REVIEW_MIN_PRECISION = 0.85
REVIEW_MIN_RECALL = 0.60

GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"


class BenchDataError(ValueError):
    """标定集格式错误；加载即失败，不静默跳过。"""


def load_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise BenchDataError(f"找不到标定集：{path}")
    rows: list[dict] = []
    for line_no, raw in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        parts = raw.split("\t")
        if len(parts) != 6:
            raise BenchDataError(
                f"{path}:{line_no}: 必须是 6 列 case_id/cue/source/translation/expected/note"
            )
        case_id, cue_text, source, translation, expected_text, note = parts
        try:
            cue = int(cue_text)
        except ValueError as exc:
            raise BenchDataError(f"{path}:{line_no}: cue 必须是整数") from exc
        expected = tuple(
            item.strip() for item in expected_text.split("|") if item.strip()
        )
        unknown = [code for code in expected if code not in qc.CODES]
        if unknown:
            raise BenchDataError(
                f"{path}:{line_no}: 未知 expected_code {unknown}"
            )
        rows.append(
            {
                "case_id": case_id,
                "cue": cue,
                "source": source,
                "translation": translation,
                "expected": expected,
                "note": note,
            }
        )
    if not rows:
        raise BenchDataError(f"{path}: 标定集为空")
    return rows


def _timestamp(seconds: int) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},000"


def build_srt(texts: list[str]) -> str:
    blocks = []
    for index, text in enumerate(texts, start=1):
        start = _timestamp(index * 2 - 1)
        end = _timestamp(index * 2)
        blocks.append(f"{index}\n{start} --> {end}\n{text}")
    return "\n\n".join(blocks) + "\n"


def group_cases(rows: list[dict]) -> dict[str, list[dict]]:
    cases: dict[str, list[dict]] = {}
    for row in rows:
        cases.setdefault(row["case_id"], []).append(row)
    for case_id, members in cases.items():
        members.sort(key=lambda item: item["cue"])
        cues = [item["cue"] for item in members]
        if cues != list(range(1, len(cues) + 1)):
            raise BenchDataError(
                f"{case_id}: cue 必须从 1 连续递增，实际为 {cues}"
            )
    return cases


def evaluate(
    rows: list[dict],
    cases: dict[str, list[dict]],
    glossary: Glossary | None,
    asr_terms,
) -> tuple[dict[str, dict[str, int]], list[dict], list[str]]:
    predictions: dict[tuple[str, int], set[str]] = {}
    file_level: list[dict] = []
    for case_id, members in cases.items():
        source_srt = build_srt([item["source"] for item in members])
        zh_srt = build_srt([item["translation"] for item in members])
        result = qc.check_pair(
            source_srt,
            zh_srt,
            glossary=glossary,
            asr_terms=asr_terms,
        )
        for issue in result.issues:
            if issue.cue is None:
                file_level.append(
                    {
                        "case_id": case_id,
                        "code": issue.code,
                        "severity": issue.severity,
                    }
                )
                continue
            try:
                cue = int(issue.cue)
            except ValueError:
                continue
            predictions.setdefault((case_id, cue), set()).add(issue.code)

    stats: dict[str, dict[str, int]] = {}
    for row in rows:
        key = (row["case_id"], row["cue"])
        expected = set(row["expected"])
        predicted = predictions.get(key, set())
        for code in expected | predicted:
            entry = stats.setdefault(code, {"tp": 0, "fp": 0, "fn": 0})
            if code in expected and code in predicted:
                entry["tp"] += 1
            elif code in predicted:
                entry["fp"] += 1
            elif code in expected:
                entry["fn"] += 1

    # 失败关闭：expected 里要求的 code 若一条都没被预测，FN 仍然计入；
    # 未出现在 expected/predicted 的 code 保持 0 样本，表格里显示 n/a。
    failures: list[str] = []
    negatives = sum(1 for row in rows if not row["expected"])
    if len(rows) < MIN_ROWS:
        failures.append(f"标定集行数 {len(rows)} < {MIN_ROWS}")
    if negatives < MIN_NEGATIVES:
        failures.append(f"负例行数 {negatives} < {MIN_NEGATIVES}")
    return stats, file_level, failures


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def code_verdict(code: str, entry: dict[str, int]) -> tuple[bool, str]:
    spec = qc.CODES.get(code)
    if spec is None:
        return True, "n/a"
    tp, fp, fn = entry["tp"], entry["fp"], entry["fn"]
    if spec.severity == qc.SEVERITY_INFO:
        return True, "info"
    if spec.severity == qc.SEVERITY_BLOCK:
        if tp + fp == 0:
            return True, "n/a"
        precision = tp / (tp + fp)
        if precision < BLOCK_MIN_PRECISION:
            return False, f"P {precision:.2f} < {BLOCK_MIN_PRECISION:.2f}"
        return True, "pass"
    if tp + fn == 0:
        return True, "n/a"
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)
    reasons = []
    if precision < REVIEW_MIN_PRECISION:
        reasons.append(f"P {precision:.2f} < {REVIEW_MIN_PRECISION:.2f}")
    if recall < REVIEW_MIN_RECALL:
        reasons.append(f"R {recall:.2f} < {REVIEW_MIN_RECALL:.2f}")
    if reasons:
        return False, "；".join(reasons)
    return True, "pass"


def _format_metric(value: float | None) -> str:
    return "  — " if value is None else f"{value:.2f}"


def render_table(stats: dict[str, dict[str, int]], *, color: bool) -> tuple[list[str], list[str]]:
    lines = [
        "code                     sev     TP   FP   FN     P     R    F1  gate",
        "---------------------------------------------------------------------",
    ]
    failures: list[str] = []
    for code in qc.CODES:
        entry = stats.get(code, {"tp": 0, "fp": 0, "fn": 0})
        tp, fp, fn = entry["tp"], entry["fp"], entry["fn"]
        precision = _ratio(tp, tp + fp)
        recall = _ratio(tp, tp + fn)
        f1 = None
        if precision is not None and recall is not None and precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        ok, verdict = code_verdict(code, entry)
        if not ok:
            failures.append(f"{code}: {verdict}")
        spec = qc.CODES[code]
        row = (
            f"{code:<24} {spec.severity:<6} {tp:>4} {fp:>4} {fn:>4} "
            f"{_format_metric(precision):>5} {_format_metric(recall):>5} "
            f"{_format_metric(f1):>5}  {verdict}"
        )
        if color and spec.severity != qc.SEVERITY_INFO:
            row = f"{GREEN if ok else RED}{row}{RESET}"
        lines.append(row)
    return lines, failures


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    parser = argparse.ArgumentParser(description="QC 标定台：per-code P/R/F1")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--glossary", type=Path, default=DEFAULT_GLOSSARY)
    parser.add_argument("--asr-terms", type=Path, default=DEFAULT_ASR_TERMS)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args(argv)

    try:
        rows = load_rows(args.golden)
        cases = group_cases(rows)
        glossary = (
            Glossary.load(args.glossary) if args.glossary.is_file() else None
        )
        asr_terms = (
            qc.parse_asr_terms(
                args.asr_terms.read_text(encoding="utf-8-sig")
            )
            if args.asr_terms.is_file()
            else ()
        )
    except (BenchDataError, OSError, UnicodeDecodeError, ValueError) as exc:
        print(f"标定集加载失败：{exc}", file=sys.stderr)
        return 2

    stats, file_level, dataset_failures = evaluate(
        rows, cases, glossary, asr_terms
    )
    use_color = not args.json and not args.no_color
    table, gate_failures = render_table(stats, color=use_color)
    failures = dataset_failures + gate_failures
    negatives = sum(1 for row in rows if not row["expected"])
    fail_prefix = f"{RED}[FAIL]{RESET}" if use_color else "[FAIL]"
    fail_color = RED if use_color else ""
    reset_color = RESET if use_color else ""
    ok_color = GREEN if use_color else ""

    if args.json:
        payload = {
            "dataset": {
                "rows": len(rows),
                "cases": len(cases),
                "negatives": negatives,
                "min_rows": MIN_ROWS,
                "min_negatives": MIN_NEGATIVES,
            },
            "stats": stats,
            "file_level": file_level,
            "failures": failures,
            "passed": not failures,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(
            f"标定集：{len(rows)} 行（负例 {negatives}）/ "
            f"{len(cases)} 个 case"
        )
        print("\n".join(table))
        print("")
        if file_level:
            print(f"文件级 info（不进 cue 级 P/R）：{len(file_level)} 条")
        if dataset_failures:
            for item in dataset_failures:
                print(f"{fail_prefix} {item}")
        if gate_failures:
            for item in gate_failures:
                print(f"{fail_prefix} {item}")
        if failures:
            print(f"{fail_color}结论：未达标（{len(failures)} 项）{reset_color}")
        else:
            print(f"{ok_color}结论：全部达标{reset_color}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
