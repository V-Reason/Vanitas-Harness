"""只读 QC 的纯函数内核：给定源/译文文本，产出可复核的 ``Issue``。

设计边界
--------
- **不做文件 IO**：读写、目录遍历、参数解析与退出码都在 ``scripts/qc_srt.py``。
- **不调用模型**：所有判定都是确定性字符串规则或复用生产审计函数。
- **不改翻译逻辑**：``vanitas.translate`` 的 cleaner/format 与
  ``vanitas.quality`` 只被读取调用，本模块不写回、不替换、不重译。

三条口径
--------
1. ``CODES`` 是唯一 code 目录：每个 code 的 severity、rule id 与定义都写在
   这里，bench 直接读它，避免“代码一套、标定一套”。
2. severity 三档：``block`` 必须重跑；``review`` 人工复核；``info`` 只统计，
   永不进退出码。``level`` 保留旧值（error/actionable/info）只做兼容。
3. 不变量优先：``garbled`` 是确定性坏字符；``pipeline_drift`` 与
   ``glossary_forbidden`` 直接调用生产后处理/术语表，不重新发明规则。

未知即不报：任何“比例阈值”都只用于 ``review``，没有一条进入 ``block``。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Iterable, Sequence

from .quality import (
    REGISTER_INTIMATE,
    DuplicateIndex,
    Glossary,
    audit_address_term,
    audit_duplicate,
    audit_invented_name,
    audit_invented_tool,
    audit_short_duplicate,
    audit_under_translation,
    detect_organ_swap,
    detect_register,
    is_onomatopoeia_or_reduplication,
    normalize_compare,
)
from .translate.cleaner import (
    LATIN_ALLOW,
    LATIN_TOKEN_RE,
    LEADING_INVALID_PUNCT_RE,
    QUOTE_CHARS,
    REPLACEMENT_CHARS_RE,
    SLASH_ALTERNATIVE_RE,
    SOURCE_HAS_BRACKETS_RE,
    TONE_ANNOTATION_BRACKETS_RE,
    TRADITIONAL_CHARS,
    TRADITIONAL_TO_SIMPLIFIED_MAP,
    UNTRANSLATED_KANA_RE,
    WINDOW_ID_ANY_RE,
    postprocess_cue_text,
)
from .translate.format import parse_srt
from .translate.types import ASR_HALLUCINATION_PATTERNS

SEVERITY_BLOCK = "block"
SEVERITY_REVIEW = "review"
SEVERITY_INFO = "info"
SEVERITY_ORDER = {
    SEVERITY_BLOCK: 0,
    SEVERITY_REVIEW: 1,
    SEVERITY_INFO: 2,
}
LEVEL_BY_SEVERITY = {
    SEVERITY_BLOCK: "error",
    SEVERITY_REVIEW: "actionable",
    SEVERITY_INFO: "info",
}
SEVERITIES = (SEVERITY_BLOCK, SEVERITY_REVIEW, SEVERITY_INFO)


@dataclass(frozen=True)
class CodeSpec:
    """一个 QC code 的完整口径；``definition`` 与 bench/文档同源。"""

    code: str
    severity: str
    rule: str
    definition: str


# 唯一 code 目录。``register_summary`` 是 FileQC 的文件级统计字段，不产生
# issue code；其 review 升级落到 ``honorific``。
CODE_SPECS: tuple[CodeSpec, ...] = (
    CodeSpec(
        "cue_count",
        SEVERITY_BLOCK,
        "qc.structure.cue_count",
        "源/译 cue 数不一致；物理时间轴不得重切。",
    ),
    CodeSpec(
        "index",
        SEVERITY_BLOCK,
        "qc.structure.index",
        "逐条序号与源文件不一致。",
    ),
    CodeSpec(
        "timing",
        SEVERITY_BLOCK,
        "qc.structure.timing",
        "逐条时间轴与源文件不一致。",
    ),
    CodeSpec(
        "protocol_residue",
        SEVERITY_BLOCK,
        "qc.structure.protocol_residue",
        "译文正文残留 [#N] 协议编号（含单边括号、全角、Markdown 变体）。",
    ),
    CodeSpec(
        "garbled",
        SEVERITY_BLOCK,
        "qc.invariant.garbled",
        "译文含 U+FFFD/U+FFFE/U+FFFF 替换符或非字符。",
    ),
    CodeSpec(
        "pipeline_drift",
        SEVERITY_INFO,
        "qc.invariant.postprocess_fixpoint",
        "文件内存在不是当前 postprocess_cue_text 不动点的 cue；文件级 info。",
    ),
    CodeSpec(
        "glossary_forbidden",
        SEVERITY_REVIEW,
        "qc.invariant.glossary_forbidden",
        "源文命中术语表条目，译文仍含该条 forbidden 形态。",
    ),
    CodeSpec(
        "invented_name",
        SEVERITY_REVIEW,
        "qc.definition.invented_name",
        "译文出现整篇源文都没有的提示词示例专名。",
    ),
    CodeSpec(
        "under_translation",
        SEVERITY_REVIEW,
        "qc.definition.under_translation",
        "源文归一化 ≥10 字、译文 ≤2 字，疑似整句丢失。",
    ),
    CodeSpec(
        "invented_tool",
        SEVERITY_REVIEW,
        "qc.definition.invented_tool",
        "译文出现道具词，当前及前后 ±3 条源文都没有道具词。",
    ),
    CodeSpec(
        "organ_mismatch",
        SEVERITY_REVIEW,
        "qc.definition.organ_mismatch",
        "源文只有耳词、没有阴道类词，译文却出现阴道类词（耳→小穴错位）。",
    ),
    CodeSpec(
        "address_term",
        SEVERITY_REVIEW,
        "qc.definition.address_term",
        "源含旦那様/ご主人様，译文却译成先生/老师。",
    ),
    CodeSpec(
        "numeral_truncation",
        SEVERITY_REVIEW,
        "qc.definition.numeral_truncation",
        "源含 11–19 的十位数字，译文写成个位且无「十」。",
    ),
    CodeSpec(
        "duplicate_translation",
        SEVERITY_REVIEW,
        "qc.duplicate.index",
        "不同原文得到相同/高度相似的长译文（DuplicateIndex 0.75/0.80）。",
    ),
    CodeSpec(
        "short_duplicate",
        SEVERITY_REVIEW,
        "qc.duplicate.short",
        "短源文（≤11 字）得到完全相同的译文，且源文不是拟声/近义。",
    ),
    CodeSpec(
        "onomatopoeia_collapse",
        SEVERITY_INFO,
        "qc.stat.onomatopoeia_collapse",
        "同文件 ≥3 条拟声源 cue 落到同一译句；输出分组统计。",
    ),
    CodeSpec(
        "term_drift",
        SEVERITY_INFO,
        "qc.stat.term_drift",
        "同一术语条目在同文件出现 ≥2 种译法。",
    ),
    CodeSpec(
        "asr_term",
        SEVERITY_REVIEW,
        "qc.asr.term",
        "源文命中数据化 ASR 疑似错词表（term/依据/动作）。",
    ),
    CodeSpec(
        "honorific",
        SEVERITY_REVIEW,
        "qc.register.honorific",
        "亲密语域、非引语译文出现敬称「您」。",
    ),
    CodeSpec(
        "pronoun_mixed",
        SEVERITY_REVIEW,
        "qc.register.pronoun_mixed",
        "同一文件「您」与「你」并存且来回切换 ≥3 次（人称不统一）。",
    ),
    CodeSpec(
        "blank",
        SEVERITY_REVIEW,
        "qc.legacy.blank",
        "非空原文（非 ASR 幻觉）得到空译文。",
    ),
    CodeSpec(
        "kana",
        SEVERITY_REVIEW,
        "qc.legacy.kana",
        "译文残留日文假名。",
    ),
    CodeSpec(
        "latin",
        SEVERITY_REVIEW,
        "qc.legacy.latin",
        "译文含可疑拉丁词（SEXPAR/Track 等，ASMR/NG/BGM 除外）。",
    ),
    CodeSpec(
        "slash",
        SEVERITY_REVIEW,
        "qc.legacy.slash",
        "译文含中日文斜杠并列。",
    ),
    CodeSpec(
        "staging",
        SEVERITY_REVIEW,
        "qc.legacy.staging",
        "原文无括号，译文新增动作舞台指示。",
    ),
    CodeSpec(
        "length",
        SEVERITY_REVIEW,
        "qc.legacy.length",
        "译文长度超过 max(20, 4×源文归一化长度)。",
    ),
)

CODES: dict[str, CodeSpec] = {spec.code: spec for spec in CODE_SPECS}


def severity_of(code: str) -> str:
    """未知 code 一律按 info 处理，保证退出码不被新规则意外抬高。"""
    spec = CODES.get(code)
    return spec.severity if spec is not None else SEVERITY_INFO


def rule_of(code: str) -> str:
    spec = CODES.get(code)
    return spec.rule if spec is not None else f"qc.unknown.{code}"


@dataclass(frozen=True)
class Issue:
    """一条 QC 发现；``cue`` 为 None 表示文件级问题。"""

    severity: str
    code: str
    rule: str
    message: str
    cue: str | None = None
    source_text: str = ""
    translation_text: str = ""
    evidence: str = ""
    suggestion: str = ""
    level: str = ""

    def __post_init__(self) -> None:
        if not self.level:
            object.__setattr__(self, "level", LEVEL_BY_SEVERITY[self.severity])

    def as_dict(self) -> dict:
        data = {
            "level": self.level,
            "severity": self.severity,
            "code": self.code,
            "rule": self.rule,
            "message": self.message,
            "evidence": self.evidence,
            "source_text": self.source_text,
            "translation_text": self.translation_text,
            "suggestion": self.suggestion,
        }
        if self.cue is not None:
            data["cue"] = self.cue
        return data


@dataclass
class FileQC:
    """一个源/译文件对的检查结果；字段只增不减，旧键保持兼容。"""

    source: str = ""
    translated: str = ""
    cues: int = 0
    honorific: int = 0
    blank: int = 0
    kana: int = 0
    traditional: int = 0
    silent: dict[str, int] = field(
        default_factory=lambda: {"traditional": 0, "leading_punct": 0}
    )
    issues: list[Issue] = field(default_factory=list)
    register: str = "neutral"
    pronoun: dict[str, int] = field(default_factory=lambda: {"您": 0, "你": 0})
    fixpoint: dict[str, int] = field(
        default_factory=lambda: {"stable": 0, "total": 0, "drift": 0}
    )

    @property
    def errors(self) -> list[Issue]:
        return [item for item in self.issues if item.severity == SEVERITY_BLOCK]

    @property
    def warnings(self) -> list[Issue]:
        return [item for item in self.issues if item.severity == SEVERITY_REVIEW]

    @property
    def actionable(self) -> list[Issue]:
        return self.warnings

    @property
    def blocks(self) -> list[Issue]:
        return self.errors

    @property
    def reviews(self) -> list[Issue]:
        return self.warnings

    @property
    def infos(self) -> list[Issue]:
        return [item for item in self.issues if item.severity == SEVERITY_INFO]

    @property
    def honorific_review(self) -> int:
        return sum(1 for item in self.issues if item.code == "honorific")

    def severity_counts(self) -> dict[str, int]:
        counts = {severity: 0 for severity in SEVERITIES}
        for item in self.issues:
            counts[item.severity] = counts.get(item.severity, 0) + 1
        return counts

    def code_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.issues:
            counts[item.code] = counts.get(item.code, 0) + 1
        return counts

    def as_dict(self) -> dict:
        counts = self.severity_counts()
        return {
            "source": self.source,
            "translated": self.translated,
            "cues": self.cues,
            "honorific": self.honorific,
            "blank": self.blank,
            "kana": self.kana,
            "traditional": self.traditional,
            "silent": dict(self.silent),
            "register_summary": {
                "register": self.register,
                "nin": self.pronoun.get("您", 0),
                "ni": self.pronoun.get("你", 0),
                "honorific_review": self.honorific_review,
            },
            "pipeline_fixpoint": dict(self.fixpoint),
            "severity_counts": counts,
            "errors": counts[SEVERITY_BLOCK],
            "warnings": counts[SEVERITY_REVIEW],
            "actionable": counts[SEVERITY_REVIEW],
            "issues": [item.as_dict() for item in self.issues],
        }


@dataclass(frozen=True)
class ASRTerm:
    """``config/qc/asr_suspect.tsv`` 的一行：词、依据、建议动作。"""

    term: str
    reason: str = ""
    action: str = ""


@dataclass(frozen=True)
class QCOptions:
    """可调阈值；默认值就是标定集上的口径，bench 与 CLI 共用。"""

    only: frozenset[str] | None = None
    duplicate_source_ratio: float = 0.75
    duplicate_translation_ratio: float = 0.80
    duplicate_min_chars: int = 8
    short_min_chars: int = 6
    short_translation_ratio: float = 1.0
    short_max_source_chars: int = 11
    onomatopoeia_group_min: int = 3
    nearby_radius: int = 3

    def enabled(self, code: str) -> bool:
        return self.only is None or code in self.only


# 近义源文豁免：源文不同但语义等价时，同一译文不算复制。
# 这里只做“豁免”，不参与发现；阈值 0.60 沿用旧覆盖面，避免把
# 合法近义句推到 review。显式对子用于相似度算不出来的同义改写。
NEAR_SYNONYM_PAIRS: tuple[tuple[str, str], ...] = (
    ("気に入って", "お気に召して"),
    ("気に入った", "お気に召した"),
    ("挿入", "お挿入"),
)
# 同义问句/思考动词：短句同译在这些上下文里是合法改写，不是复制。
NEAR_SYNONYM_GROUPS: tuple[tuple[str, ...], ...] = (
    ("いいですか", "いかがでしょう", "いかがですか", "よろしいですか"),
    ("思う", "考える", "考え"),
)
NEAR_SYNONYM_RATIO = 0.60


def _looks_like_name_variant(left: str, right: str) -> bool:
    """专名两种 ASR 写法：共同前缀 ≥3 且只差一个字，后文继续重合。

    ``伊賀崎彩香`` / ``伊賀崎綾香`` 这类只换一个汉字、前后文一致的写法，
    是同一专名的归一问题，不是“不同原文复用译文”。
    """
    if len(left) < 6 or len(right) < 6:
        return False
    prefix = 0
    limit = min(len(left), len(right))
    while prefix < limit and left[prefix] == right[prefix]:
        prefix += 1
    if prefix < 3 or prefix + 1 >= limit:
        return False
    tail_left = left[prefix + 1 :]
    tail_right = right[prefix + 1 :]
    shared = 0
    while (
        shared < min(len(tail_left), len(tail_right))
        and tail_left[shared] == tail_right[shared]
    ):
        shared += 1
    return shared >= 2


def parse_asr_terms(text: str) -> tuple[ASRTerm, ...]:
    """解析 ASR 疑似错词表；空行与 ``#`` 注释跳过，坏行抛 ``ValueError``。

    纯文本解析，不碰文件系统；缺文件由调用方决定“不启用”。
    """
    terms: list[ASRTerm] = []
    for line_no, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = raw.split("\t")
        if len(parts) != 3:
            raise ValueError(
                f"ASR 可疑词表第 {line_no} 行必须为 3 列 term<TAB>依据<TAB>动作"
            )
        term, reason, action = (part.strip() for part in parts)
        if not term:
            raise ValueError(f"ASR 可疑词表第 {line_no} 行 term 不能为空")
        terms.append(ASRTerm(term=term, reason=reason, action=action))
    return tuple(terms)


def _clip(text: str, limit: int = 80) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)] + "…"


def _normalize(text: str, *, translate: bool = False) -> str:
    if translate:
        text = text.translate(TRADITIONAL_TO_SIMPLIFIED_MAP)
    return normalize_compare(text)


def _cue_number(cue: str | None) -> int:
    try:
        return int(cue) if cue is not None else -1
    except (TypeError, ValueError):
        return -1


def _sort_issues(issues: Sequence[Issue]) -> list[Issue]:
    return sorted(
        issues,
        key=lambda item: (
            _cue_number(item.cue),
            SEVERITY_ORDER.get(item.severity, 9),
            item.code,
            item.message,
        ),
    )


def _issue(
    severity: str,
    code: str,
    message: str,
    *,
    cue: str | None = None,
    source_text: str = "",
    translation_text: str = "",
    evidence: str = "",
    suggestion: str = "",
) -> Issue:
    return Issue(
        severity=severity,
        code=code,
        rule=rule_of(code),
        message=message,
        cue=cue,
        source_text=source_text,
        translation_text=translation_text,
        evidence=evidence,
        suggestion=suggestion,
    )


def _add(result: FileQC, issue: Issue, options: QCOptions) -> None:
    if options.enabled(issue.code):
        result.issues.append(issue)


def _audit_issue(
    audit,
    *,
    severity: str,
    rule: str,
    evidence: str,
    source_text: str,
    translation_text: str,
) -> Issue:
    return Issue(
        severity=severity,
        code=audit.code,
        rule=rule,
        message=audit.message,
        cue=str(audit.cue),
        source_text=source_text,
        translation_text=translation_text,
        evidence=evidence,
        suggestion=audit.hint,
    )


def _has_quote(text: str) -> bool:
    return any(char in QUOTE_CHARS for char in text)


def _sources_are_near_synonyms(left: str, right: str) -> bool:
    """两个源文是否可视为同一句话的合法改写（仅用于豁免）。"""
    left_norm = _normalize(left)
    right_norm = _normalize(right)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True
    if left_norm in right_norm or right_norm in left_norm:
        return min(len(left_norm), len(right_norm)) >= 4
    if any(
        (a in left_norm and b in right_norm) or (b in left_norm and a in right_norm)
        for a, b in NEAR_SYNONYM_PAIRS
    ):
        return True
    if any(
        any(term in left_norm for term in group)
        and any(term in right_norm for term in group)
        for group in NEAR_SYNONYM_GROUPS
    ):
        return True
    if _looks_like_name_variant(left_norm, right_norm):
        return True
    if min(len(left_norm), len(right_norm)) < 4:
        return False
    return (
        SequenceMatcher(None, left_norm, right_norm).ratio()
        >= NEAR_SYNONYM_RATIO
    )


KANA_ONLY_RE = re.compile(r"^[\u3040-\u30ff\u30fc]+$")
# 舞台指示的 QC 口径比生产端更宽：生产只删“耳边/轻抚”白名单（免得误删
# 「（笑）」这类语气注释），但人工复核要看到所有未授权括号，包括
# 「（轻轻探入）」「（脚步声）」「（腰部有规律地）」。
UNSOURCED_BRACKET_RE = re.compile(
    r"[（(【][^）)】]{0,20}[\u4e00-\u9fff][^）)】]{0,20}[）)】]"
)
# quality 的拟声判定把“全假名 + 促音”一律当拟声，ストップですわ 这类
# 带句尾助动词的短句会被误豁免；QC 只在这类句子上绕开该豁免。
SENTENCE_MARKERS = (
    "です",
    "ます",
    "ました",
    "ません",
    "ください",
    "でしょう",
    "ですか",
    "って",
    "たい",
    "ている",
    "てる",
)


def _is_sentence_like(text: str) -> bool:
    compact = _normalize(text)
    return any(marker in compact for marker in SENTENCE_MARKERS)
# quality 的拟声判定偏保守（表外片假名不算）；QC 统计分组需要覆盖
# 「シーコ」「この」这类 ≤10 字纯假名短句，否则拟声折叠看不见。
# 这只是 QC 的统计范围，不参与任何 block 判定。
KANA_ONO_MAX_CHARS = 10


def _is_onomatopoeia(text: str) -> bool:
    if is_onomatopoeia_or_reduplication(text):
        return True
    compact = _normalize(text)
    if not compact or len(compact) > KANA_ONO_MAX_CHARS:
        return False
    return bool(KANA_ONLY_RE.match(compact))


def _exempt_copy(source: str, other_source: str) -> bool:
    """拟声源文或近义源文不算“不同原文复用译文”。"""
    return (
        _is_onomatopoeia(source)
        or _is_onomatopoeia(other_source)
        or _sources_are_near_synonyms(source, other_source)
    )


KANJI_TEEN_RE = re.compile(r"(?<![一二三四五六七八九十百千])十([一二三四五六七八九])")
ARABIC_TEEN_RE = re.compile(r"(?<!\d)(1[1-9])(?!\d)")
KANJI_DIGITS = "一二三四五六七八九"


def _numeral_truncation(source: str, translation: str) -> list[tuple[str, str]]:
    """返回 (源数字, 译文数字) 列表；没有命中返回空。

    只认“十位数字被写成个位且译文没有「十」”：源文的 11–19 在中文里
    必须带「十」（或写阿拉伯数字），只写个位是明确的数位丢失。译文含
    阿拉伯原数时视为正确，避免把“11 个人”误报。
    """
    if not source or not translation:
        return []
    hits: list[tuple[str, str]] = []
    for match in KANJI_TEEN_RE.finditer(source):
        digit = match.group(1)
        source_number = f"十{digit}"
        if "十" in translation:
            continue
        if ARABIC_TEEN_RE.search(translation):
            continue
        if re.search(
            rf"(?<![{KANJI_DIGITS}十]){digit}(?![{KANJI_DIGITS}十])",
            translation,
        ):
            hits.append((source_number, digit))
    for match in ARABIC_TEEN_RE.finditer(source):
        number = match.group(1)
        unit = number[-1]
        digit = KANJI_DIGITS[int(unit) - 1]
        if number in translation or "十" in translation:
            continue
        if re.search(
            rf"(?<![{KANJI_DIGITS}十]){digit}(?![{KANJI_DIGITS}十])",
            translation,
        ):
            hits.append((number, digit))
    return hits


def _check_structural(
    result: FileQC,
    src_cues,
    zh_cues,
    options: QCOptions,
) -> None:
    if len(src_cues) != len(zh_cues):
        _add(
            result,
            _issue(
                SEVERITY_BLOCK,
                "cue_count",
                f"cue 数不一致：源 {len(src_cues)}，译 {len(zh_cues)}",
                evidence=f"源 {len(src_cues)} / 译 {len(zh_cues)}",
                suggestion="物理时间轴不得重切；请重跑或回退该文件",
            ),
            options,
        )
    for src, zh in zip(src_cues, zh_cues):
        if src.index != zh.index:
            _add(
                result,
                _issue(
                    SEVERITY_BLOCK,
                    "index",
                    f"序号不一致：{src.index} → {zh.index}",
                    cue=zh.index,
                    source_text=src.text,
                    translation_text=zh.text,
                    evidence=f"{src.index}→{zh.index}",
                    suggestion="序号必须与源文件逐条一致",
                ),
                options,
            )
        if src.timing != zh.timing:
            _add(
                result,
                _issue(
                    SEVERITY_BLOCK,
                    "timing",
                    f"时间轴不一致：{src.timing} → {zh.timing}",
                    cue=zh.index,
                    source_text=src.text,
                    translation_text=zh.text,
                    evidence=f"{src.timing}→{zh.timing}",
                    suggestion="物理时间轴不得漂移；请重跑或回退该文件",
                ),
                options,
            )
        residue = WINDOW_ID_ANY_RE.search(zh.text)
        if residue:
            _add(
                result,
                _issue(
                    SEVERITY_BLOCK,
                    "protocol_residue",
                    "译文正文泄漏协议编号残渣（含单边括号/全角/Markdown 变体）",
                    cue=zh.index,
                    source_text=src.text,
                    translation_text=zh.text,
                    evidence=_clip(residue.group(0), 32),
                    suggestion="删除编号残渣；重跑后处理或该窗口",
                ),
                options,
            )


def _check_legacy_per_cue(
    result: FileQC,
    src,
    zh,
    options: QCOptions,
) -> None:
    source_text = src.text
    translation_text = zh.text
    cue = zh.index

    if LEADING_INVALID_PUNCT_RE.search(translation_text):
        result.silent["leading_punct"] += 1
    if set(translation_text) & TRADITIONAL_CHARS:
        result.traditional += 1
        result.silent["traditional"] += 1

    if not translation_text.strip():
        if source_text.strip() and not ASR_HALLUCINATION_PATTERNS.search(source_text):
            result.blank += 1
            _add(
                result,
                _issue(
                    SEVERITY_REVIEW,
                    "blank",
                    "非空原文得到空译文",
                    cue=cue,
                    source_text=source_text,
                    translation_text=translation_text,
                    evidence="译文长度 0",
                    suggestion="确认是否漏译；ASR 片尾幻觉之外不应为空",
                ),
                options,
            )
    if UNTRANSLATED_KANA_RE.search(translation_text):
        result.kana += 1
        _add(
            result,
            _issue(
                SEVERITY_REVIEW,
                "kana",
                "译文残留日文假名",
                cue=cue,
                source_text=source_text,
                translation_text=translation_text,
                evidence=_clip(
                    "".join(UNTRANSLATED_KANA_RE.findall(translation_text)), 16
                ),
                suggestion="确认是否漏译或专名未本地化",
            ),
            options,
        )
    latin_tokens = [
        token
        for token in LATIN_TOKEN_RE.findall(translation_text)
        if token.upper() not in LATIN_ALLOW
    ]
    if latin_tokens:
        _add(
            result,
            _issue(
                SEVERITY_REVIEW,
                "latin",
                f"译文含可疑拉丁词：{'、'.join(latin_tokens[:3])}",
                cue=cue,
                source_text=source_text,
                translation_text=translation_text,
                evidence="、".join(latin_tokens[:3]),
                suggestion="确认是否模型臆造词或未本地化的 Track/SEXPAR",
            ),
            options,
        )
    if SLASH_ALTERNATIVE_RE.search(translation_text):
        _add(
            result,
            _issue(
                SEVERITY_REVIEW,
                "slash",
                "译文含中日文斜杠并列",
                cue=cue,
                source_text=source_text,
                translation_text=translation_text,
                evidence=_clip(SLASH_ALTERNATIVE_RE.search(translation_text).group(0), 16),
                suggestion="改成一个确定译名；不要用 A/B 或 先生/女士 并列",
            ),
            options,
        )
    brackets = list(UNSOURCED_BRACKET_RE.finditer(translation_text))
    if not SOURCE_HAS_BRACKETS_RE.search(source_text) and brackets:
        # 同一 cue 可能既有语气标注又有动作括号：只要存在一条未认可的，
        # 就按动作括号报 review，不能被前面的「（笑）」稀释掉。
        offending = next(
            (
                match
                for match in brackets
                if not TONE_ANNOTATION_BRACKETS_RE.match(match.group(0))
            ),
            None,
        )
        if offending is None:
            _add(
                result,
                _issue(
                    SEVERITY_INFO,
                    "staging",
                    "译文保留情绪/语音标注（已认可）",
                    cue=cue,
                    source_text=source_text,
                    translation_text=translation_text,
                    evidence=_clip(brackets[0].group(0), 24),
                    suggestion="无需处理；这不是无源舞台动作",
                ),
                options,
            )
        else:
            _add(
                result,
                _issue(
                    SEVERITY_REVIEW,
                    "staging",
                    "原文无括号，译文新增动作舞台指示",
                    cue=cue,
                    source_text=source_text,
                    translation_text=translation_text,
                    evidence=_clip(offending.group(0), 24),
                    suggestion="删除未授权动作括号；不要凭剧情脑补",
                ),
                options,
            )

    source_norm = _normalize(source_text)
    translation_norm = _normalize(translation_text, translate=True)
    if len(source_norm) >= 6 and len(translation_norm) >= max(
        20, len(source_norm) * 4
    ):
        _add(
            result,
            _issue(
                SEVERITY_REVIEW,
                "length",
                f"译文长度异常：源 {len(source_norm)} 字，译 {len(translation_norm)} 字",
                cue=cue,
                source_text=source_text,
                translation_text=translation_text,
                evidence=f"源 {len(source_norm)} → 译 {len(translation_norm)}",
                suggestion="检查是否合并相邻行、解释性扩写或加戏",
            ),
            options,
        )


def _check_garbled(result: FileQC, src, zh, options: QCOptions) -> None:
    match = REPLACEMENT_CHARS_RE.search(zh.text)
    if not match:
        return
    _add(
        result,
        _issue(
            SEVERITY_BLOCK,
            "garbled",
            "译文含 U+FFFD/U+FFFE/U+FFFF 乱码替换符",
            cue=zh.index,
            source_text=src.text,
            translation_text=zh.text,
            evidence=f"U+{ord(match.group(0)):04X} 第 {match.start() + 1} 字符",
            suggestion="重跑该窗口；替换符不得落盘",
        ),
        options,
    )


def _check_asr(
    result: FileQC,
    src,
    zh,
    asr_terms: Sequence[ASRTerm | str],
    options: QCOptions,
) -> None:
    for item in asr_terms:
        term = item.term if isinstance(item, ASRTerm) else str(item)
        if not term or term not in src.text:
            continue
        reason = item.reason if isinstance(item, ASRTerm) else ""
        action = item.action if isinstance(item, ASRTerm) else ""
        _add(
            result,
            _issue(
                SEVERITY_REVIEW,
                "asr_term",
                f"源文含疑似 ASR 错词：{term}",
                cue=zh.index,
                source_text=src.text,
                translation_text=zh.text,
                evidence=term + (f"（{reason}）" if reason else ""),
                suggestion=action or "听音确认原文；不要直接凭猜测改源",
            ),
            options,
        )


def _check_definition(
    result: FileQC,
    src,
    zh,
    document_text: str,
    nearby_sources: tuple[str, ...],
    options: QCOptions,
) -> None:
    if options.enabled("address_term"):
        for audit in audit_address_term(src.text, zh.text, zh.index):
            _add(
                result,
                _audit_issue(
                    audit,
                    severity=SEVERITY_REVIEW,
                    rule=rule_of("address_term"),
                    evidence="旦那様/ご主人様 → 先生/老师",
                    source_text=src.text,
                    translation_text=zh.text,
                ),
                options,
            )
    if options.enabled("invented_name"):
        for audit in audit_invented_name(
            src.text, zh.text, zh.index, document_text
        ):
            _add(
                result,
                _audit_issue(
                    audit,
                    severity=SEVERITY_REVIEW,
                    rule=rule_of("invented_name"),
                    evidence=_clip(
                        next(
                            (
                                item
                                for item in ("茜音", "赤音", "阿卡涅", "日和")
                                if item in zh.text
                            ),
                            audit.message,
                        ),
                        24,
                    ),
                    source_text=src.text,
                    translation_text=zh.text,
                ),
                options,
            )
    if options.enabled("invented_tool"):
        for audit in audit_invented_tool(
            src.text,
            zh.text,
            zh.index,
            nearby_sources=nearby_sources,
        ):
            _add(
                result,
                _audit_issue(
                    audit,
                    severity=SEVERITY_REVIEW,
                    rule=rule_of("invented_tool"),
                    evidence="±3 条内无道具词",
                    source_text=src.text,
                    translation_text=zh.text,
                ),
                options,
            )
    if options.enabled("under_translation"):
        # ASR 片尾幻觉被置空是预期行为，不能当成漏译；与 blank 同一豁免。
        if not ASR_HALLUCINATION_PATTERNS.search(src.text):
            for audit in audit_under_translation(src.text, zh.text, zh.index):
                source_len = len(_normalize(src.text))
                translation_len = len(_normalize(zh.text, translate=True))
                _add(
                    result,
                    _audit_issue(
                        audit,
                        severity=SEVERITY_REVIEW,
                        rule=rule_of("under_translation"),
                        evidence=f"源 {source_len} → 译 {translation_len}",
                        source_text=src.text,
                        translation_text=zh.text,
                    ),
                    options,
                )
    if options.enabled("numeral_truncation"):
        for source_number, translation_number in _numeral_truncation(
            src.text, zh.text
        ):
            _add(
                result,
                _issue(
                    SEVERITY_REVIEW,
                    "numeral_truncation",
                    f"数位丢失：{source_number} → {translation_number}",
                    cue=zh.index,
                    source_text=src.text,
                    translation_text=zh.text,
                    evidence=f"{source_number}→{translation_number}",
                    suggestion="11–19 必须保留「十」或写成阿拉伯数字",
                ),
                options,
            )


def _check_duplicates(
    result: FileQC,
    pairs,
    options: QCOptions,
) -> None:
    if not (options.enabled("duplicate_translation") or options.enabled("short_duplicate")):
        return
    duplicate_index = DuplicateIndex(
        source_ratio=options.duplicate_source_ratio,
        translation_ratio=options.duplicate_translation_ratio,
        min_chars=options.duplicate_min_chars,
    )
    short_index = DuplicateIndex(
        min_chars=options.short_min_chars,
        translation_ratio=options.short_translation_ratio,
    )
    for src, zh in pairs:
        if not zh.text.strip():
            continue
        match = duplicate_index.find_match(src.text, zh.text, zh.index)
        if match is not None:
            if not _exempt_copy(src.text, match.source):
                audits = audit_duplicate(
                    src.text, zh.text, zh.index, duplicate_index
                )
                message = (
                    audits[0].message
                    if audits
                    else f"译文与 {match.cue} 重复，但原文不同"
                )
                _add(
                    result,
                    _issue(
                        SEVERITY_REVIEW,
                        "duplicate_translation",
                        message,
                        cue=zh.index,
                        source_text=src.text,
                        translation_text=zh.text,
                        evidence=f"{match.cue}↔{zh.index}：{_clip(zh.text, 32)}",
                        suggestion="检查窗口错位/复制相邻行；按本条原文重译",
                    ),
                    options,
                )
        elif options.enabled("short_duplicate"):
            similar = short_index.find_similar(src.text, zh.text, zh.index)
            audits = audit_short_duplicate(
                src.text, zh.text, zh.index, short_index
            )
            # quality 把“全假名 + 促音”当拟声而豁免；当事句带
            # です/ます/ください 时它不是拟声，按复制处理。
            sentence_override = bool(
                similar is not None
                and _is_sentence_like(src.text)
                and _is_sentence_like(similar.source)
            )
            if (
                similar is not None
                and (audits or sentence_override)
                and not _sources_are_near_synonyms(src.text, similar.source)
            ):
                evidence = f"{similar.cue}↔{zh.index}：{_clip(zh.text, 32)}"
                if audits:
                    issue = _audit_issue(
                        audits[0],
                        severity=SEVERITY_REVIEW,
                        rule=rule_of("short_duplicate"),
                        evidence=evidence,
                        source_text=src.text,
                        translation_text=zh.text,
                    )
                else:
                    issue = _issue(
                        SEVERITY_REVIEW,
                        "short_duplicate",
                        f"短句译与第 {similar.cue} 条完全相同，但原文不同",
                        cue=zh.index,
                        source_text=src.text,
                        translation_text=zh.text,
                        evidence=evidence,
                        suggestion="按本条原文重译，禁止复用同一短译",
                    )
                _add(result, issue, options)
        duplicate_index.add(zh.index, src.text, zh.text)
        short_index.add(zh.index, src.text, zh.text)


def _check_glossary_per_cue(
    result: FileQC,
    src,
    zh,
    glossary: Glossary | None,
    options: QCOptions,
) -> None:
    if glossary is None or not options.enabled("glossary_forbidden"):
        return
    applied = glossary.apply(src.text, zh.text)
    if applied == zh.text:
        return
    evidence = "术语表收敛态不一致"
    for entry in glossary.entries:
        if entry.source not in src.text:
            continue
        for forbidden in entry.forbidden:
            if forbidden and forbidden in zh.text:
                evidence = f"{forbidden}→{entry.preferred}"
                break
        if evidence != "术语表收敛态不一致":
            break
    _add(
        result,
        _issue(
            SEVERITY_REVIEW,
            "glossary_forbidden",
            f"源文命中术语条目，译文仍含禁止译法（{evidence}）",
            cue=zh.index,
            source_text=src.text,
            translation_text=zh.text,
            evidence=evidence,
            suggestion="按术语表 preferred 收敛；不要保留 forbidden 形态",
        ),
        options,
    )


# 人称混用阈值：两种人称各 ≥2 条、切换 ≥3 次才报；单次场景切换不报。
PRONOUN_MIX_MIN_EACH = 2
PRONOUN_MIX_MIN_SWITCHES = 3


def _check_organ(
    result: FileQC,
    src,
    zh,
    options: QCOptions,
) -> None:
    """耳→阴道类词的器官错位；判定规则与生产端修复共用同一 guard。"""
    if not options.enabled("organ_mismatch"):
        return
    evidence = detect_organ_swap(src.text, zh.text)
    if evidence is None:
        return
    _add(
        result,
        _issue(
            SEVERITY_REVIEW,
            "organ_mismatch",
            f"器官错位：{evidence}",
            cue=zh.index,
            source_text=src.text,
            translation_text=zh.text,
            evidence=evidence,
            suggestion="按本条原文的器官直译（耳→耳朵），不要替换成另一处器官",
        ),
        options,
    )


def _check_register(
    result: FileQC,
    pairs,
    register: str,
    options: QCOptions,
) -> None:
    result.register = register
    result.pronoun["您"] = sum(zh.text.count("您") for _src, zh in pairs)
    result.pronoun["你"] = sum(zh.text.count("你") for _src, zh in pairs)
    if register == REGISTER_INTIMATE and options.enabled("honorific"):
        for src, zh in pairs:
            if "您" not in zh.text or _has_quote(zh.text):
                continue
            _add(
                result,
                _issue(
                    SEVERITY_REVIEW,
                    "honorific",
                    "亲密/非正式语境出现敬称“您”",
                    cue=zh.index,
                    source_text=src.text,
                    translation_text=zh.text,
                    evidence=f"您×{zh.text.count('您')} register=intimate",
                    suggestion="确认是否为正式/服务语境；亲昵耳语建议改为“你”",
                ),
                options,
            )
    _check_pronoun_mix(result, pairs, options)


def _check_pronoun_mix(result: FileQC, pairs, options: QCOptions) -> None:
    """同一文件 您/你 并存且来回切换时，在**首个切换 cue** 报一条 review。

    阈值 2 条 / 2 条 / 切换 ≥3 次：服务→亲密的一次性场景切换（1~2 次）
    属于合法语域转换，不算“混用”；来回翻烧饼才需要人工统一。
    """
    if not options.enabled("pronoun_mixed"):
        return
    sequence: list[tuple[str, str]] = []
    for _src, zh in pairs:
        if "您" in zh.text:
            sequence.append((zh.index, "您"))
        elif "你" in zh.text:
            sequence.append((zh.index, "你"))
    count_nin = sum(1 for _cue, pronoun in sequence if pronoun == "您")
    count_you = sum(1 for _cue, pronoun in sequence if pronoun == "你")
    if count_nin < PRONOUN_MIX_MIN_EACH or count_you < PRONOUN_MIX_MIN_EACH:
        return
    switches = 0
    anchor: str | None = None
    previous: str | None = None
    for cue, pronoun in sequence:
        if previous is not None and pronoun != previous:
            switches += 1
            if anchor is None:
                anchor = cue
        previous = pronoun
    if switches < PRONOUN_MIX_MIN_SWITCHES or anchor is None:
        return
    for src, zh in pairs:
        if zh.index != anchor:
            continue
        _add(
            result,
            _issue(
                SEVERITY_REVIEW,
                "pronoun_mixed",
                f"人称混用：您 {count_nin} 条 / 你 {count_you} 条，来回切换 {switches} 次",
                cue=anchor,
                source_text=src.text,
                translation_text=zh.text,
                evidence=f"您×{count_nin}/你×{count_you}，切换 {switches} 次，首切第 {anchor} 条",
                suggestion="同一文件统一用「你」；服务场景请整篇保留「您」，不要交替",
            ),
            options,
        )
        return


def _check_corpus_stats(
    result: FileQC,
    pairs,
    glossary: Glossary | None,
    options: QCOptions,
) -> None:
    if options.enabled("pipeline_drift"):
        total = len(pairs)
        drift: list[str] = []
        for src, zh in pairs:
            if postprocess_cue_text(zh.text, src.text) != zh.text:
                drift.append(zh.index)
        result.fixpoint = {
            "stable": total - len(drift),
            "total": total,
            "drift": len(drift),
        }
        if drift:
            sample = "、".join(drift[:3])
            _add(
                result,
                _issue(
                    SEVERITY_INFO,
                    "pipeline_drift",
                    f"{len(drift)}/{total} 条不是当前后处理的不动点",
                    evidence=f"{len(drift)}/{total}；示例 {sample}",
                    suggestion="对照 postprocess_cue_text 检查后处理是否已升级",
                ),
                options,
            )

    if options.enabled("onomatopoeia_collapse"):
        groups: dict[str, list[tuple[str, str, str]]] = {}
        for src, zh in pairs:
            if not zh.text.strip() or not _is_onomatopoeia(src.text):
                continue
            key = _normalize(zh.text, translate=True)
            if not key:
                continue
            groups.setdefault(key, []).append((zh.index, src.text, zh.text))
        for key, members in groups.items():
            if len(members) < options.onomatopoeia_group_min:
                continue
            distinct = {_normalize(source) for _cue, source, _zh in members}
            for cue, source, translation in members:
                _add(
                    result,
                    _issue(
                        SEVERITY_INFO,
                        "onomatopoeia_collapse",
                        f"拟声折叠：{len(members)} 条 / {len(distinct)} 个源词同译「{key}」",
                        cue=cue,
                        source_text=source,
                        translation_text=translation,
                        evidence=f"{len(members)} 条 / {len(distinct)} 个源词 → {key}",
                    ),
                    options,
                )

    if glossary is not None and options.enabled("term_drift"):
        for entry in glossary.entries:
            variants: dict[str, list[tuple[str, str, str]]] = {}
            for src, zh in pairs:
                if entry.source not in src.text:
                    continue
                variant = _normalize(zh.text, translate=True)
                if variant:
                    variants.setdefault(variant, []).append(
                        (zh.index, src.text, zh.text)
                    )
            if len(variants) < 2:
                continue
            summary = " / ".join(variants)
            for members in variants.values():
                for cue, source, translation in members:
                    _add(
                        result,
                        _issue(
                            SEVERITY_INFO,
                            "term_drift",
                            f"术语「{entry.source}」在同文件有 {len(variants)} 种译法",
                            cue=cue,
                            source_text=source,
                            translation_text=translation,
                            evidence=summary,
                            suggestion=f"统一为 preferred「{entry.preferred}」",
                        ),
                        options,
                    )


def check_pair(
    source_text: str,
    zh_text: str,
    *,
    glossary: Glossary | None = None,
    asr_terms: Sequence[ASRTerm | str] = (),
    options: QCOptions | None = None,
) -> FileQC:
    """检查一个源/译文本对；纯函数，不读写文件。

    参数
    ----
    source_text / zh_text
        完整的 SRT 文本（含序号与时间轴）。
    glossary
        ``quality.Glossary``；None 表示不启用术语检查。
    asr_terms
        ``ASRTerm`` 或纯词字符串；空序列表示不启用 ASR 术语检查。
    options
        阈值与 ``--only`` 过滤；None 用默认口径。
    """
    options = options or QCOptions()
    result = FileQC()

    src_cues = [
        sub
        for sub, _separator in parse_srt(source_text or "")
        if sub.index and sub.timing
    ]
    zh_cues = [
        sub
        for sub, _separator in parse_srt(zh_text or "")
        if sub.index and sub.timing
    ]
    if (source_text or "").strip() and not src_cues:
        result.issues.append(
            _issue(
                SEVERITY_BLOCK,
                "read",
                "源 SRT 解析失败：未找到合法 cue",
                evidence="parse_srt=0",
                suggestion="确认文件是 SRT（序号 + 时间轴 + 正文）",
            )
        )
        return result
    if (zh_text or "").strip() and not zh_cues:
        result.issues.append(
            _issue(
                SEVERITY_BLOCK,
                "read",
                "中文 SRT 解析失败：未找到合法 cue",
                evidence="parse_srt=0",
                suggestion="确认文件是 SRT（序号 + 时间轴 + 正文）",
            )
        )
        return result

    result.cues = len(zh_cues)
    result.honorific = sum(zh.text.count("您") for zh in zh_cues)
    document_text = "\n".join(src.text for src in src_cues)
    sources = [src.text for src in src_cues]
    pairs = list(zip(src_cues, zh_cues))
    register = detect_register(sources)

    _check_structural(result, src_cues, zh_cues, options)

    for position, (src, zh) in enumerate(pairs):
        _check_legacy_per_cue(result, src, zh, options)
        _check_garbled(result, src, zh, options)
        _check_asr(result, src, zh, asr_terms, options)
        start = max(0, position - options.nearby_radius)
        end = min(len(sources), position + options.nearby_radius + 1)
        nearby = tuple(
            text for index, text in enumerate(sources[start:end], start) if index != position
        )
        _check_definition(
            result, src, zh, document_text, nearby, options
        )
        _check_organ(result, src, zh, options)
        _check_glossary_per_cue(result, src, zh, glossary, options)

    _check_duplicates(result, pairs, options)
    _check_register(result, pairs, register, options)
    _check_corpus_stats(result, pairs, glossary, options)

    result.issues = _sort_issues(result.issues)
    return result


def _pair_parts(pair) -> tuple[str, str, str]:
    if len(pair) == 3:
        name, source_text, zh_text = pair
        return str(name), str(source_text), str(zh_text)
    if len(pair) == 2:
        source_text, zh_text = pair
        return "", str(source_text), str(zh_text)
    raise ValueError("check_corpus 的每一项必须是 (source, zh) 或 (name, source, zh)")


def check_corpus(
    pairs: Iterable,
    *,
    glossary: Glossary | None = None,
    asr_terms: Sequence[ASRTerm | str] = (),
    options: QCOptions | None = None,
) -> list[FileQC]:
    """批量检查；``pairs`` 每项是 (source_text, zh_text) 或 (name, 两者)。"""
    results: list[FileQC] = []
    for index, pair in enumerate(pairs, start=1):
        name, source_text, zh_text = _pair_parts(pair)
        result = check_pair(
            source_text,
            zh_text,
            glossary=glossary,
            asr_terms=asr_terms,
            options=options,
        )
        if name:
            stem = name[:-7] if name.endswith(".zh.srt") else name
            result.source = f"{stem}.srt"
            result.translated = name
        else:
            result.source = f"pair-{index}.srt"
            result.translated = f"pair-{index}.zh.srt"
        results.append(result)
    return results


def _render_issue(file_name: str, issue: Issue) -> list[str]:
    cue = f":{issue.cue}" if issue.cue else ""
    lines = [
        f"  [{file_name}{cue}] {issue.severity}/{issue.code}：{issue.message}"
    ]
    if issue.evidence:
        lines.append(f"  = 依据：{_clip(issue.evidence)}")
    if issue.source_text:
        lines.append(f"  - 原文：{_clip(issue.source_text)}")
    if issue.translation_text:
        lines.append(f"  + 译文：{_clip(issue.translation_text)}")
    if issue.suggestion:
        lines.append(f"  ? {issue.suggestion}")
    return lines


def sorted_issues(results: Sequence[FileQC]) -> list[tuple[FileQC, Issue]]:
    """按 (file, cue 号, severity, code) 稳定排序，供渲染与 JSON 复用。"""
    rows = [
        (result, issue) for result in results for issue in result.issues
    ]
    return sorted(
        rows,
        key=lambda row: (
            row[0].translated,
            _cue_number(row[1].cue),
            SEVERITY_ORDER.get(row[1].severity, 9),
            row[1].code,
            row[1].message,
        ),
    )


def render_report(results: Sequence[FileQC]) -> str:
    """人类可读报告；block 与 review 分节，info 只给总数。"""
    rows = sorted_issues(results)
    blocks = [(result.translated, issue) for result, issue in rows if issue.severity == SEVERITY_BLOCK]
    reviews = [(result.translated, issue) for result, issue in rows if issue.severity == SEVERITY_REVIEW]
    infos = [(result.translated, issue) for result, issue in rows if issue.severity == SEVERITY_INFO]
    cues = sum(result.cues for result in results)
    silent_traditional = sum(
        result.silent.get("traditional", 0) for result in results
    )
    silent_leading = sum(
        result.silent.get("leading_punct", 0) for result in results
    )

    marker = "✗" if blocks else "✓"
    lines = [
        f"{marker} 结构核验：{len(results)} 个文件 / {cues} cues，"
        f"block {len(blocks)} · review {len(reviews)} · info {len(infos)}"
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

    if blocks:
        lines.append("")
        lines.append(f"✗ 阻断问题 {len(blocks)} 处：")
        for file_name, issue in blocks:
            lines.extend(_render_issue(file_name, issue))
    if reviews:
        lines.append("")
        lines.append(f"⚠ 语义疑似点 {len(reviews)} 处：")
        for file_name, issue in reviews:
            lines.extend(_render_issue(file_name, issue))
    if infos:
        counts: dict[str, int] = {}
        for _file_name, issue in infos:
            counts[issue.code] = counts.get(issue.code, 0) + 1
        summary = "，".join(
            f"{code} {count}" for code, count in sorted(counts.items())
        )
        lines.append("")
        lines.append(f"· 统计信息 {len(infos)} 处（不进退出码）：{summary}")
        lines.append("  （逐条明细用 --json 的 files[].issues 查看）")
    if not blocks and not reviews:
        lines.append("✓ 未发现阻断问题或语义疑似点")
    return "\n".join(lines)
