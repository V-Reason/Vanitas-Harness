"""只读 QC：检查源 SRT 与中文 SRT 的结构/语言风险。

用法：
    python scripts/qc_srt.py --source-dir <dir> [--zh-dir <dir>] [--json] [--strict]

error 表示必须修（cue 数/时间轴/编号泄漏/错位重复）；warn 表示需要人看
（空 cue、开头标点、残留假名、繁体、敬称计数）。脚本不写任何文件。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

BLOCK_SEP = re.compile(r"(\r?\n[ \t]*\r?\n)")
WINDOW_ID = re.compile(r"\[#\d+\]")
TRADITIONAL = set(
    "個這個們來時說會發現實體為與從對開關頭間問題點還麼樣過讓給帶嗎"
    "讀寫聽覺愛戀臉頰臺灣專輯關於潔話謝請誰後裡裏妳"
)
KANA = re.compile(r"[\u3040-\u30ff]")
LEADING_PUNCT = re.compile(r"^[，、；：]")


@dataclass
class Issue:
    level: str
    code: str
    message: str
    cue: str | None = None

    def as_dict(self) -> dict:
        data = {"level": self.level, "code": self.code, "message": self.message}
        if self.cue is not None:
            data["cue"] = self.cue
        return data


@dataclass
class FileQC:
    source: str
    translated: str
    cues: int = 0
    honorific: int = 0
    blank: int = 0
    kana: int = 0
    traditional: int = 0
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [item for item in self.issues if item.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [item for item in self.issues if item.level == "warn"]

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "translated": self.translated,
            "cues": self.cues,
            "honorific": self.honorific,
            "blank": self.blank,
            "kana": self.kana,
            "traditional": self.traditional,
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "issues": [item.as_dict() for item in self.issues],
        }


@dataclass(frozen=True)
class Cue:
    index: str
    timing: str
    text: str


def parse_srt(text: str) -> list[Cue]:
    cues: list[Cue] = []
    for block in BLOCK_SEP.split(text):
        lines = block.splitlines()
        if len(lines) < 2:
            continue
        index = lines[0].strip()
        timing = lines[1].strip()
        if not index.isdigit() or "-->" not in timing:
            continue
        cues.append(Cue(index=index, timing=timing, text="\n".join(lines[2:]).strip()))
    return cues


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _normalize(text: str) -> str:
    return re.sub(r"[\s。！？，、…—]+", "", text)


def _source_texts_are_similar(left: str, right: str) -> bool:
    if not left or not right:
        return False
    return SequenceMatcher(None, left, right).ratio() >= 0.75


def check_pair(source: Path, translated: Path) -> FileQC:
    result = FileQC(source=source.name, translated=translated.name)
    try:
        src_cues = parse_srt(_read(source))
        zh_cues = parse_srt(_read(translated))
    except OSError as exc:
        result.issues.append(Issue("error", "read", f"读取失败：{exc}"))
        return result
    result.cues = len(zh_cues)
    result.honorific = sum(cue.text.count("您") for cue in zh_cues)

    if len(src_cues) != len(zh_cues):
        result.issues.append(
            Issue(
                "error",
                "cue_count",
                f"cue 数不一致：源 {len(src_cues)}，译 {len(zh_cues)}",
            )
        )

    for src, zh in zip(src_cues, zh_cues):
        if src.index != zh.index:
            result.issues.append(
                Issue("error", "index", f"序号不一致：{src.index} → {zh.index}", zh.index)
            )
        if src.timing != zh.timing:
            result.issues.append(
                Issue("error", "timing", f"时间轴不一致：{src.timing} → {zh.timing}", zh.index)
            )
        if WINDOW_ID.search(zh.text):
            result.issues.append(
                Issue("error", "window_id", "译文正文泄漏 [#N]", zh.index)
            )
        if not zh.text.strip():
            result.blank += 1
            result.issues.append(Issue("warn", "blank", "译文为空", zh.index))
        if LEADING_PUNCT.search(zh.text):
            result.issues.append(
                Issue("warn", "leading_punct", "译文以标点开头", zh.index)
            )
        if KANA.search(zh.text):
            result.kana += 1
            result.issues.append(Issue("warn", "kana", "译文残留日文假名", zh.index))
        chars = sorted(set(zh.text) & TRADITIONAL)
        if chars:
            result.traditional += 1
            result.issues.append(
                Issue(
                    "warn",
                    "traditional",
                    "疑似繁体字：" + "、".join(chars),
                    zh.index,
                )
            )

    if len(src_cues) == len(zh_cues):
        for left in range(len(zh_cues)):
            source_left = _normalize(src_cues[left].text)
            translation_left = _normalize(zh_cues[left].text)
            if len(translation_left) < 8 or len(source_left) < 4:
                continue
            for right in range(left + 1, len(zh_cues)):
                source_right = _normalize(src_cues[right].text)
                translation_right = _normalize(zh_cues[right].text)
                if (
                    source_left != source_right
                    and len(source_right) >= 4
                    and not _source_texts_are_similar(source_left, source_right)
                    and (
                        translation_left == translation_right
                        or (
                            len(translation_left) >= 8
                            and len(translation_right) >= 8
                            and (
                                translation_left in translation_right
                                or translation_right in translation_left
                                or SequenceMatcher(
                                    None, translation_left, translation_right
                                ).ratio()
                                >= 0.8
                            )
                        )
                    )
                ):
                    result.issues.append(
                        Issue(
                            "error",
                            "duplicate_translation",
                            f"不同原文得到相同译文：{zh_cues[left].index} / {zh_cues[right].index}",
                            zh_cues[right].index,
                        )
                    )
    return result


def collect_pairs(source_dir: Path, zh_dir: Path) -> tuple[list[tuple[Path, Path]], list[FileQC]]:
    pairs: list[tuple[Path, Path]] = []
    problems: list[FileQC] = []
    for translated in sorted(zh_dir.glob("*.zh.srt")):
        source = source_dir / (translated.name[:-7] + ".srt")
        if not source.is_file():
            qc = FileQC(source=source.name, translated=translated.name)
            qc.issues.append(Issue("error", "missing_source", f"找不到源 SRT：{source}"))
            problems.append(qc)
            continue
        pairs.append((source, translated))
    if not pairs and not problems:
        qc = FileQC(source=str(source_dir), translated=str(zh_dir))
        qc.issues.append(Issue("error", "no_pairs", "没有找到 *.zh.srt 配对"))
        problems.append(qc)
    return pairs, problems


def render_markdown(results: list[FileQC]) -> str:
    lines = [
        "| 文件 | cue | error | warn | 您 | 空 | 假名 | 繁体 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        lines.append(
            f"| {item.translated} | {item.cues} | {len(item.errors)} | "
            f"{len(item.warnings)} | {item.honorific} | {item.blank} | "
            f"{item.kana} | {item.traditional} |"
        )
    issues = [
        (item, issue)
        for item in results
        for issue in item.issues
        if issue.level == "error"
    ]
    if issues:
        lines.extend(["", "### Errors", ""])
        for item, issue in issues:
            cue = f" cue {issue.cue}" if issue.cue else ""
            lines.append(f"- {item.translated}{cue}：{issue.message}")
    warnings = [
        (item, issue)
        for item in results
        for issue in item.issues
        if issue.level == "warn"
    ]
    if warnings:
        lines.extend(["", "### Warnings", ""])
        for item, issue in warnings[:80]:
            cue = f" cue {issue.cue}" if issue.cue else ""
            lines.append(f"- {item.translated}{cue}：{issue.message}")
        if len(warnings) > 80:
            lines.append(f"- … 另有 {len(warnings) - 80} 条")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    parser = argparse.ArgumentParser(description="只读检查源/中文 SRT 配对")
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--zh-dir", type=Path)
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    parser.add_argument("--strict", action="store_true", help="warn 也返回 1")
    args = parser.parse_args(argv)

    zh_dir = args.zh_dir or args.source_dir
    pairs, problems = collect_pairs(args.source_dir, zh_dir)
    results = list(problems)
    for source, translated in pairs:
        results.append(check_pair(source, translated))

    if args.json:
        payload = {
            "files": [item.as_dict() for item in results],
            "errors": sum(len(item.errors) for item in results),
            "warnings": sum(len(item.warnings) for item in results),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(render_markdown(results))

    errors = sum(len(item.errors) for item in results)
    warnings = sum(len(item.warnings) for item in results)
    if errors:
        return 1
    if args.strict and warnings:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
