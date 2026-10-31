"""只读 QC CLI：参数解析、目录遍历、读文件、渲染与退出码。

判定全部在 ``vanitas.qc``（纯函数、无文件 IO）。本文件只做三件事：
把磁盘上的 SRT 读成文本、把默认术语表/ASR 数据交给内核、把 ``Issue``
渲染成文本或 JSON。``check_pair`` / ``render_report`` 保留为薄转发，
既有测试仍直接调用它们。

用法：
    python scripts/qc_srt.py --source-dir <dir> [--zh-dir <dir>] [--json]
        [--strict] [--only code1,code2] [--diff old.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vanitas import qc  # noqa: E402
from vanitas.errors import TranslateError  # noqa: E402
from vanitas.quality import Glossary  # noqa: E402
from vanitas.translate.format import parse_srt as _parse_srt  # noqa: E402

DEFAULT_GLOSSARY = ROOT / "config" / "glossary" / "asmr.tsv"
DEFAULT_ASR_TERMS = ROOT / "config" / "qc" / "asr_suspect.tsv"

Issue = qc.Issue
FileQC = qc.FileQC

_UNSET = object()
_GLOSSARY_CACHE: dict[str, Glossary | None] = {}
_ASR_CACHE: dict[str, tuple[qc.ASRTerm, ...]] = {}


@dataclass(frozen=True)
class Cue:
    index: str
    timing: str
    text: str


def parse_srt(text: str) -> list[Cue]:
    """薄包装：与旧脚本同名的解析入口，返回简单 ``Cue``。"""
    return [
        Cue(index=sub.index, timing=sub.timing, text=sub.text)
        for sub, _separator in _parse_srt(text)
        if sub.index and sub.timing
    ]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _default_glossary() -> Glossary | None:
    key = str(DEFAULT_GLOSSARY)
    if key not in _GLOSSARY_CACHE:
        _GLOSSARY_CACHE[key] = (
            Glossary.load(DEFAULT_GLOSSARY) if DEFAULT_GLOSSARY.is_file() else None
        )
    return _GLOSSARY_CACHE[key]


def _default_asr_terms() -> tuple[qc.ASRTerm, ...]:
    key = str(DEFAULT_ASR_TERMS)
    if key not in _ASR_CACHE:
        if DEFAULT_ASR_TERMS.is_file():
            _ASR_CACHE[key] = qc.parse_asr_terms(
                DEFAULT_ASR_TERMS.read_text(encoding="utf-8-sig")
            )
        else:
            _ASR_CACHE[key] = ()
    return _ASR_CACHE[key]


def check_pair(
    source: Path,
    translated: Path,
    *,
    glossary=_UNSET,
    asr_terms=_UNSET,
    options: qc.QCOptions | None = None,
) -> FileQC:
    """读两个文件并转发给 ``qc.check_pair``；保留旧签名给既有测试。"""
    if glossary is _UNSET:
        glossary = _default_glossary()
    if asr_terms is _UNSET:
        asr_terms = _default_asr_terms()
    try:
        source_text = _read(source)
        zh_text = _read(translated)
    except (OSError, UnicodeDecodeError) as exc:
        result = FileQC(source=source.name, translated=translated.name)
        result.issues.append(
            qc.Issue(
                severity=qc.SEVERITY_BLOCK,
                code="read",
                rule="qc.io.read",
                message=f"读取失败：{exc}",
                suggestion="检查文件存在性与 UTF-8 编码",
            )
        )
        return result
    result = qc.check_pair(
        source_text,
        zh_text,
        glossary=glossary,
        asr_terms=asr_terms,
        options=options,
    )
    result.source = source.name
    result.translated = translated.name
    return result


def collect_pairs(
    source_dir: Path, zh_dir: Path
) -> tuple[list[tuple[Path, Path]], list[FileQC]]:
    pairs: list[tuple[Path, Path]] = []
    problems: list[FileQC] = []
    for translated in sorted(zh_dir.glob("*.zh.srt")):
        source = source_dir / (translated.name[:-7] + ".srt")
        if not source.is_file():
            result = FileQC(source=source.name, translated=translated.name)
            result.issues.append(
                qc.Issue(
                    severity=qc.SEVERITY_BLOCK,
                    code="missing_source",
                    rule="qc.io.missing_source",
                    message=f"找不到源 SRT：{source}",
                    suggestion="把源文件放到同一目录或指定 --source-dir",
                )
            )
            problems.append(result)
            continue
        pairs.append((source, translated))
    if not pairs and not problems:
        result = FileQC(source=str(source_dir), translated=str(zh_dir))
        result.issues.append(
            qc.Issue(
                severity=qc.SEVERITY_BLOCK,
                code="no_pairs",
                rule="qc.io.no_pairs",
                message="没有找到 *.zh.srt 配对",
                suggestion="确认 --zh-dir/--source-dir 与输出后缀",
            )
        )
        problems.append(result)
    return pairs, problems


def render_report(results: list[FileQC]) -> str:
    """薄转发：报告排版在内核里，脚本不复制模板。"""
    return qc.render_report(results)


def _parse_only(text: str | None) -> frozenset[str] | None:
    if not text:
        return None
    codes = frozenset(item.strip() for item in text.split(",") if item.strip())
    unknown = sorted(code for code in codes if code not in qc.CODES and code != "read")
    if unknown:
        raise ValueError(f"未知 code：{', '.join(unknown)}")
    return codes


def _load_previous(path: Path) -> dict[str, int]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise ValueError(f"找不到 diff 文件：{path}（{exc.strerror or exc}）") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"diff 文件不是 UTF-8：{exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"diff 文件不是合法 JSON：{exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("diff 文件顶层必须是 JSON 对象")
    summary = payload.get("summary")
    if isinstance(summary, dict) and isinstance(summary.get("by_code"), dict):
        return {
            str(code): int(count)
            for code, count in summary["by_code"].items()
            if isinstance(count, (int, float))
        }
    flagged = payload.get("flagged_cues")
    if isinstance(flagged, list):
        counts: dict[str, int] = {}
        for item in flagged:
            if isinstance(item, dict) and item.get("code"):
                code = str(item["code"])
                counts[code] = counts.get(code, 0) + 1
        return counts
    files = payload.get("files")
    if isinstance(files, list):
        counts = {}
        for item in files:
            if not isinstance(item, dict):
                continue
            for issue in item.get("issues", ()):
                if isinstance(issue, dict) and issue.get("code"):
                    code = str(issue["code"])
                    counts[code] = counts.get(code, 0) + 1
        return counts
    raise ValueError("diff 文件缺少 summary.by_code / flagged_cues / files")


def _render_diff(previous: dict[str, int], current: dict[str, int]) -> list[str]:
    lines = ["· 与上一版 JSON 的 per-code 计数："]
    for code in sorted(set(previous) | set(current)):
        old = previous.get(code, 0)
        new = current.get(code, 0)
        delta = new - old
        if delta > 0:
            change = f"+{delta}"
        elif delta < 0:
            change = f"{delta}"
        else:
            change = "不变"
        lines.append(f"  {code}: {old} → {new}  {change}")
    return lines


def _build_payload(
    results: list[FileQC],
    *,
    strict: bool,
    only: frozenset[str] | None,
    passed: bool,
) -> dict:
    rows = qc.sorted_issues(results)
    flagged = [
        {
            "file": result.translated,
            "cue": issue.cue,
            "level": issue.level,
            "severity": issue.severity,
            "code": issue.code,
            "rule": issue.rule,
            "message": issue.message,
            "evidence": issue.evidence,
            "source_text": issue.source_text,
            "translation_text": issue.translation_text,
            "suggestion": issue.suggestion,
        }
        for result, issue in rows
        if issue.severity in (qc.SEVERITY_BLOCK, qc.SEVERITY_REVIEW)
    ]
    by_code: dict[str, int] = {}
    severity_counts = {severity: 0 for severity in qc.SEVERITIES}
    for _result, issue in rows:
        by_code[issue.code] = by_code.get(issue.code, 0) + 1
        severity_counts[issue.severity] += 1
    return {
        "summary": {
            "files": len(results),
            "cues": sum(result.cues for result in results),
            "errors": severity_counts[qc.SEVERITY_BLOCK],
            "warnings": severity_counts[qc.SEVERITY_REVIEW],
            "actionable": severity_counts[qc.SEVERITY_REVIEW],
            "block": severity_counts[qc.SEVERITY_BLOCK],
            "review": severity_counts[qc.SEVERITY_REVIEW],
            "info": severity_counts[qc.SEVERITY_INFO],
            "by_code": dict(sorted(by_code.items())),
            "silent": {
                "traditional": sum(
                    result.silent.get("traditional", 0) for result in results
                ),
                "leading_punct": sum(
                    result.silent.get("leading_punct", 0) for result in results
                ),
            },
            "passed": passed,
            "strict": bool(strict),
            "only": sorted(only) if only else [],
        },
        "flagged_cues": flagged,
        "files": [result.as_dict() for result in results],
        "errors": severity_counts[qc.SEVERITY_BLOCK],
        "warnings": severity_counts[qc.SEVERITY_REVIEW],
    }


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    parser = argparse.ArgumentParser(description="只读检查源/中文 SRT 配对")
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--zh-dir", type=Path)
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    parser.add_argument("--strict", action="store_true", help="review 也返回 1")
    parser.add_argument(
        "--only", help="只跑指定 code，逗号分隔（如 duplicate_translation,kana）"
    )
    parser.add_argument("--diff", type=Path, help="与上一版 JSON 比 per-code 计数")
    parser.add_argument(
        "--glossary", type=Path, default=DEFAULT_GLOSSARY, help="术语表路径"
    )
    parser.add_argument(
        "--asr-terms",
        type=Path,
        default=DEFAULT_ASR_TERMS,
        help="ASR 疑似错词表路径；缺文件=不启用",
    )
    args = parser.parse_args(argv)

    try:
        only = _parse_only(args.only)
    except ValueError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    previous: dict[str, int] | None = None
    if args.diff is not None:
        try:
            previous = _load_previous(args.diff)
        except ValueError as exc:
            print(f"错误：{exc}", file=sys.stderr)
            return 2

    try:
        glossary = (
            Glossary.load(args.glossary) if args.glossary.is_file() else None
        )
        asr_terms = (
            qc.parse_asr_terms(args.asr_terms.read_text(encoding="utf-8-sig"))
            if args.asr_terms.is_file()
            else ()
        )
    except (TranslateError, OSError, UnicodeDecodeError, ValueError) as exc:
        print(f"错误：QC 数据表读取失败：{exc}", file=sys.stderr)
        return 2

    options = qc.QCOptions(only=only)
    zh_dir = args.zh_dir or args.source_dir
    pairs, problems = collect_pairs(args.source_dir, zh_dir)
    results: list[FileQC] = list(problems)
    for source, translated in pairs:
        results.append(
            check_pair(
                source,
                translated,
                glossary=glossary,
                asr_terms=asr_terms,
                options=options,
            )
        )
    if only is not None:
        keep_always = {"read", "missing_source", "no_pairs"}
        for result in results:
            result.issues = [
                issue
                for issue in result.issues
                if issue.code in only or issue.code in keep_always
            ]
        for result in results:
            result.issues = qc._sort_issues(result.issues)

    blocks = sum(len(result.blocks) for result in results)
    reviews = sum(len(result.reviews) for result in results)
    io_error = any(
        issue.code == "read" for result in results for issue in result.issues
    )
    passed = blocks == 0 and (not args.strict or reviews == 0) and not io_error

    if args.json:
        payload = _build_payload(
            results, strict=args.strict, only=only, passed=passed
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(render_report(results))

    if previous is not None:
        current = {
            code: sum(
                1
                for result in results
                for issue in result.issues
                if issue.code == code
            )
            for code in {issue.code for result in results for issue in result.issues}
        }
        lines = _render_diff(previous, current)
        target = sys.stderr if args.json else sys.stdout
        if args.json:
            print("\n".join(lines), file=target)
        else:
            print("")
            print("\n".join(lines))

    if io_error:
        return 2
    if blocks:
        return 1
    if args.strict and reviews:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
