"""只读 QC：源 SRT 与中文 SRT 的三层检查。

用法：
    python scripts/qc_srt.py --source-dir <dir> [--zh-dir <dir>] [--json] [--strict]

三层分工：
1. 静默 auto：繁体、行首标点等由翻译后处理覆盖，只计数不刷 warning；
2. 结构 error：cue 数/序号/时间轴、[#N] 泄漏、不同原文同长译文，阻断；
3. 语义 actionable：ASR 疑似错词、亲昵语境“您”、未授权动作括号、
   译文暴增、空译、假名残留、可疑拉丁词与斜杠并列，默认只提示，
   --strict 才阻断。

脚本只读，不写任何文件。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vanitas.translate import (  # noqa: E402
    ASR_HALLUCINATION_PATTERNS,
    LATIN_ALLOW,
    LATIN_TOKEN_RE,
    SLASH_ALTERNATIVE_RE,
    SOURCE_HAS_BRACKETS_RE,
    STAGING_BRACKETS_RE,
    TRADITIONAL_CHARS,
    TRADITIONAL_TO_SIMPLIFIED_MAP,
)

BLOCK_SEP = re.compile(r"(\r?\n[ \t]*\r?\n)")
WINDOW_ID = re.compile(r"\[#\d+\]")
KANA = re.compile(r"[\u3040-\u30ff]")
LEADING_PUNCT = re.compile(r"^\s*[，、；：。！？!?]")
TRADITIONAL = TRADITIONAL_CHARS
ASR_SUSPECT_TERMS = ("南様", "弁当", "無聴", "テコッキー")
POLITE_SOURCE_RE = re.compile(
    r"(?:ます|です|ござ|ください|お願い|様|さん|先生|先輩|いただ|申し|失礼|お騒がせ)"
)
INTIMATE_SOURCE_RE = re.compile(
    r"(?:好き|大好き|抱|キス|ちゅ|耳|舐|噛|添い寝|膝枕|ねぇ|ねえ|あなた|君|お前|気持ち)"
)


@dataclass
class Issue:
    level: str
    code: str
    message: str
    cue: str | None = None
    source_text: str = ""
    translation_text: str = ""
    suggestion: str = ""

    def as_dict(self) -> dict:
        data = {
            "level": self.level,
            "code": self.code,
            "message": self.message,
            "source_text": self.source_text,
            "translation_text": self.translation_text,
            "suggestion": self.suggestion,
        }
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
    silent: dict[str, int] = field(
        default_factory=lambda: {"traditional": 0, "leading_punct": 0}
    )
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [item for item in self.issues if item.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [item for item in self.issues if item.level == "actionable"]

    @property
    def actionable(self) -> list[Issue]:
        return self.warnings

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "translated": self.translated,
            "cues": self.cues,
            "honorific": self.honorific,
            "blank": self.blank,
            "kana": self.kana,
            "traditional": self.traditional,
            "silent": dict(self.silent),
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "actionable": len(self.warnings),
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
        cues.append(
            Cue(index=index, timing=timing, text="\n".join(lines[2:]).strip())
        )
    return cues


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _normalize(text: str, *, translate: bool = False) -> str:
    if translate:
        text = text.translate(TRADITIONAL_TO_SIMPLIFIED_MAP)
    return re.sub(r"[\s。！？，、…—]+", "", text)


def _clip(text: str, limit: int = 80) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)] + "…"


def _source_texts_are_similar(left: str, right: str) -> bool:
    if not left or not right:
        return False
    # 促音/长音在拟声词里没有语义差异；先抹平再比较，避免“カリッ/カリ”误报。
    left = left.replace("ッ", "").replace("ー", "")
    right = right.replace("ッ", "").replace("ー", "")
    if left in right or right in left:
        return True
    return SequenceMatcher(None, left, right).ratio() >= 0.60


def _issue(
    level: str,
    code: str,
    message: str,
    cue: str | None,
    source_text: str,
    translation_text: str,
    suggestion: str,
) -> Issue:
    return Issue(
        level=level,
        code=code,
        message=message,
        cue=cue,
        source_text=source_text,
        translation_text=translation_text,
        suggestion=suggestion,
    )


def check_pair(source: Path, translated: Path) -> FileQC:
    result = FileQC(source=source.name, translated=translated.name)
    try:
        src_cues = parse_srt(_read(source))
        zh_cues = parse_srt(_read(translated))
    except OSError as exc:
        result.issues.append(
            _issue("error", "read", f"读取失败：{exc}", None, "", "", "检查文件存在性与 UTF-8 编码")
        )
        return result

    result.cues = len(zh_cues)
    result.honorific = sum(cue.text.count("您") for cue in zh_cues)

    if len(src_cues) != len(zh_cues):
        result.issues.append(
            _issue(
                "error",
                "cue_count",
                f"cue 数不一致：源 {len(src_cues)}，译 {len(zh_cues)}",
                None,
                "",
                "",
                "物理时间轴不得重切；请重跑或回退该文件",
            )
        )

    for src, zh in zip(src_cues, zh_cues):
        source_text = src.text
        translation_text = zh.text
        if src.index != zh.index:
            result.issues.append(
                _issue(
                    "error",
                    "index",
                    f"序号不一致：{src.index} → {zh.index}",
                    zh.index,
                    source_text,
                    translation_text,
                    "序号必须与源文件逐条一致",
                )
            )
        if src.timing != zh.timing:
            result.issues.append(
                _issue(
                    "error",
                    "timing",
                    f"时间轴不一致：{src.timing} → {zh.timing}",
                    zh.index,
                    source_text,
                    translation_text,
                    "物理时间轴不得漂移；请重跑或回退该文件",
                )
            )
        if WINDOW_ID.search(zh.text):
            result.issues.append(
                _issue(
                    "error",
                    "window_id",
                    "译文正文泄漏 [#N]",
                    zh.index,
                    source_text,
                    translation_text,
                    "删除编号泄漏；重跑后处理或该窗口",
                )
            )

        if LEADING_PUNCT.search(zh.text):
            result.silent["leading_punct"] += 1
        chars = sorted(set(zh.text) & TRADITIONAL)
        if chars:
            result.traditional += 1
            result.silent["traditional"] += 1

        if not zh.text.strip():
            if source_text.strip() and not ASR_HALLUCINATION_PATTERNS.search(source_text):
                result.blank += 1
                result.issues.append(
                    _issue(
                        "actionable",
                        "blank",
                        "非空原文得到空译文",
                        zh.index,
                        source_text,
                        translation_text,
                        "确认是否漏译；ASR 片尾幻觉之外不应为空",
                    )
                )
        if KANA.search(zh.text):
            result.kana += 1
            result.issues.append(
                _issue(
                    "actionable",
                    "kana",
                    "译文残留日文假名",
                    zh.index,
                    source_text,
                    translation_text,
                    "确认是否漏译或专名未本地化",
                )
            )
        latin_tokens = [
            token
            for token in LATIN_TOKEN_RE.findall(zh.text)
            if token.upper() not in LATIN_ALLOW
        ]
        if latin_tokens:
            result.issues.append(
                _issue(
                    "actionable",
                    "latin",
                    f"译文含可疑拉丁词：{'、'.join(latin_tokens[:3])}",
                    zh.index,
                    source_text,
                    translation_text,
                    "确认是否模型臆造词或未本地化的 Track/SEXPAR",
                )
            )
        if SLASH_ALTERNATIVE_RE.search(zh.text):
            result.issues.append(
                _issue(
                    "actionable",
                    "slash",
                    "译文含中日文斜杠并列",
                    zh.index,
                    source_text,
                    translation_text,
                    "改成一个确定译名；不要用 A/B 或 先生/女士 并列",
                )
            )

        for term in ASR_SUSPECT_TERMS:
            if term in source_text:
                result.issues.append(
                    _issue(
                        "actionable",
                        "asr_term",
                        f"源文含疑似 ASR 错词：{term}",
                        zh.index,
                        source_text,
                        translation_text,
                        "听音确认原文；不要直接凭猜测改源",
                    )
                )

        if "您" in zh.text and (
            INTIMATE_SOURCE_RE.search(source_text)
            or not POLITE_SOURCE_RE.search(source_text)
        ):
            result.issues.append(
                _issue(
                    "actionable",
                    "honorific",
                    "亲昵/非正式语境出现敬称“您”",
                    zh.index,
                    source_text,
                    translation_text,
                    "确认是否为正式/服务语境；亲昵耳语建议改为“你”",
                )
            )

        if (
            not SOURCE_HAS_BRACKETS_RE.search(source_text)
            and STAGING_BRACKETS_RE.search(zh.text)
        ):
            result.issues.append(
                _issue(
                    "actionable",
                    "staging",
                    "原文无括号，译文新增动作舞台指示",
                    zh.index,
                    source_text,
                    translation_text,
                    "删除未授权动作括号；不要凭剧情脑补",
                )
            )

        source_norm = _normalize(source_text)
        translation_norm = _normalize(translation_text, translate=True)
        if len(source_norm) >= 6 and len(translation_norm) >= max(
            20, len(source_norm) * 4
        ):
            result.issues.append(
                _issue(
                    "actionable",
                    "length",
                    f"译文长度异常：源 {len(source_norm)} 字，译 {len(translation_norm)} 字",
                    zh.index,
                    source_text,
                    translation_text,
                    "检查是否合并相邻行、解释性扩写或加戏",
                )
            )

    if len(src_cues) == len(zh_cues):
        for left in range(len(zh_cues)):
            source_left = _normalize(src_cues[left].text)
            translation_left = _normalize(zh_cues[left].text, translate=True)
            if len(translation_left) < 8 or len(source_left) < 4:
                continue
            for right in range(left + 1, len(zh_cues)):
                source_right = _normalize(src_cues[right].text)
                translation_right = _normalize(zh_cues[right].text, translate=True)
                if source_left == source_right or len(source_right) < 4:
                    continue
                if _source_texts_are_similar(source_left, source_right):
                    continue
                duplicated = translation_left == translation_right
                overlapped = (
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
                if not duplicated and not overlapped:
                    continue
                result.issues.append(
                    _issue(
                        "error",
                        "duplicate_translation",
                        f"不同原文得到相同/覆盖译文："
                        f"{zh_cues[left].index} / {zh_cues[right].index}",
                        zh_cues[right].index,
                        src_cues[right].text,
                        zh_cues[right].text,
                        "检查窗口错位/复制相邻行；重跑该窗口",
                    )
                )
    return result


def collect_pairs(
    source_dir: Path, zh_dir: Path
) -> tuple[list[tuple[Path, Path]], list[FileQC]]:
    pairs: list[tuple[Path, Path]] = []
    problems: list[FileQC] = []
    for translated in sorted(zh_dir.glob("*.zh.srt")):
        source = source_dir / (translated.name[:-7] + ".srt")
        if not source.is_file():
            qc = FileQC(source=source.name, translated=translated.name)
            qc.issues.append(
                _issue(
                    "error",
                    "missing_source",
                    f"找不到源 SRT：{source}",
                    None,
                    "",
                    "",
                    "把源文件放到同一目录或指定 --source-dir",
                )
            )
            problems.append(qc)
            continue
        pairs.append((source, translated))
    if not pairs and not problems:
        qc = FileQC(source=str(source_dir), translated=str(zh_dir))
        qc.issues.append(
            _issue(
                "error",
                "no_pairs",
                "没有找到 *.zh.srt 配对",
                None,
                "",
                "",
                "确认 --zh-dir/--source-dir 与输出后缀",
            )
        )
        problems.append(qc)
    return pairs, problems


def _render_issue(file_name: str, issue: Issue) -> list[str]:
    cue = f":{issue.cue}" if issue.cue else ""
    lines = [f"  [{file_name}{cue}] {issue.code}：{issue.message}"]
    if issue.source_text:
        lines.append(f"  - 原文：{_clip(issue.source_text)}")
    if issue.translation_text:
        lines.append(f"  + 译文：{_clip(issue.translation_text)}")
    if issue.suggestion:
        lines.append(f"  ? {issue.suggestion}")
    return lines


def render_report(results: list[FileQC]) -> str:
    errors = [
        (item.translated, issue)
        for item in results
        for issue in item.issues
        if issue.level == "error"
    ]
    actionable = [
        (item.translated, issue)
        for item in results
        for issue in item.issues
        if issue.level == "actionable"
    ]
    cues = sum(item.cues for item in results)
    silent_traditional = sum(
        item.silent.get("traditional", 0) for item in results
    )
    silent_leading = sum(
        item.silent.get("leading_punct", 0) for item in results
    )

    marker = "✗" if errors else "✓"
    lines = [
        f"{marker} 结构核验：{len(results)} 个文件 / {cues} cues，"
        f"error {len(errors)}"
    ]
    silent_parts = []
    if silent_traditional:
        silent_parts.append(f"繁体 {silent_traditional} 处")
    if silent_leading:
        silent_parts.append(f"行首标点 {silent_leading} 处")
    if silent_parts:
        lines.append(
            "· 自动核验：" + "，".join(silent_parts) + "（后处理已覆盖）"
        )

    if errors:
        lines.append("")
        lines.append(f"✗ 结构错误 {len(errors)} 处：")
        for file_name, issue in errors:
            lines.extend(_render_issue(file_name, issue))
    if actionable:
        lines.append("")
        lines.append(f"⚠ 语义疑似点 {len(actionable)} 处：")
        for file_name, issue in actionable:
            lines.extend(_render_issue(file_name, issue))
    if not errors and not actionable:
        lines.append("✓ 未发现结构错误或语义疑似点")
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
    parser.add_argument("--strict", action="store_true", help="actionable 也返回 1")
    args = parser.parse_args(argv)

    zh_dir = args.zh_dir or args.source_dir
    pairs, problems = collect_pairs(args.source_dir, zh_dir)
    results = list(problems)
    for source, translated in pairs:
        results.append(check_pair(source, translated))

    errors = sum(len(item.errors) for item in results)
    actionable = sum(len(item.warnings) for item in results)
    passed = errors == 0 and (not args.strict or actionable == 0)

    if args.json:
        flagged = [
            {
                "file": item.translated,
                "cue": issue.cue,
                "level": issue.level,
                "code": issue.code,
                "message": issue.message,
                "source_text": issue.source_text,
                "translation_text": issue.translation_text,
                "suggestion": issue.suggestion,
            }
            for item in results
            for issue in item.issues
            if issue.level in ("error", "actionable")
        ]
        payload = {
            "summary": {
                "files": len(results),
                "cues": sum(item.cues for item in results),
                "errors": errors,
                "actionable": actionable,
                "silent": {
                    "traditional": sum(
                        item.silent.get("traditional", 0) for item in results
                    ),
                    "leading_punct": sum(
                        item.silent.get("leading_punct", 0) for item in results
                    ),
                },
                "passed": passed,
                "strict": bool(args.strict),
            },
            "flagged_cues": flagged,
            "files": [item.as_dict() for item in results],
            "errors": errors,
            "warnings": actionable,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(render_report(results))

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
