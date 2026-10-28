"""批量翻译：.txt / .md / .srt。

- 文本按空行分段，段内超 chunk_chars 再按句末标点切，拼回时保留原空行结构；
- SRT 只翻译字幕文本，序号、时间轴、空行与换行风格原样保留；
- 任一段失败不中断，最终汇总失败段并由 CLI 返回非零退出码。
"""

from __future__ import annotations

import os
import re
import time
import unicodedata
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable

from . import prompts
from .client import LlamaClient
from .config import AppConfig, ModelSpec, TranslateSettings
from .errors import EngineError, OutputTruncated, TranslateError
from .metrics import (
    PROMPT_OVERHEAD,
    SegmentStat,
    estimate_tokens,
    translation_budget,
)

SENTENCE_ENDERS = "。！？!?；;"
BLOCK_SEP = re.compile(r"(\r?\n[ \t]*\r?\n)")
SRT_TIME = re.compile(r"^\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}")
WINDOW_ID_RE = re.compile(r"^\[#(\d+)\]\s*(.*)$")
WINDOW_ID_ANY_RE = re.compile(r"\s*\[#\d+\]\s*")
ASR_HALLUCINATION_PATTERNS = re.compile(
    r"(?:ご視聴ありがとうございました|チャンネル登録|高評価|ベルマーク|Thank you for watching|Please subscribe|Sous-titres)",
    re.IGNORECASE,
)
BOM = b"\xef\xbb\xbf"
FENCE_RE = re.compile(r"^```[^\n]*\n(.*?)\n```$", re.DOTALL)
QUOTE_PAIRS = {'"': '"', "'": "'", "“": "”", "‘": "’", "「": "」", "『": "』"}
QUOTE_CHARS = frozenset(QUOTE_PAIRS) | frozenset(QUOTE_PAIRS.values())

# --------------------------------------------------------------------------- #
# 确定性静默后处理（0 Token / 无模型调用）
# --------------------------------------------------------------------------- #

TRADITIONAL_TO_SIMPLIFIED = {
    "妳": "你", "話": "话", "會": "会", "個": "个", "這": "这",
    "們": "们", "來": "来", "時": "时", "說": "说", "現": "现",
    "實": "实", "體": "体", "為": "为", "與": "与", "從": "从",
    "對": "对", "開": "开", "關": "关", "頭": "头", "間": "间",
    "問": "问", "題": "题", "點": "点", "還": "还", "麼": "么",
    "樣": "样", "過": "过", "讓": "让", "給": "给", "帶": "带",
    "嗎": "吗", "讀": "读", "寫": "写", "聽": "听", "覺": "觉",
    "愛": "爱", "戀": "恋", "臉": "脸", "頰": "颊", "專": "专",
    "輯": "辑", "潔": "洁", "謝": "谢", "請": "请", "誰": "谁",
    "後": "后", "裡": "里", "裏": "里", "氣": "气", "溫": "温",
    "觸": "触", "撫": "抚", "動": "动", "變": "变", "極": "极",
    "樂": "乐", "響": "响", "臺": "台", "灣": "湾",
}
TRADITIONAL_TO_SIMPLIFIED_MAP = str.maketrans(TRADITIONAL_TO_SIMPLIFIED)
TRADITIONAL_CHARS = frozenset(TRADITIONAL_TO_SIMPLIFIED)

SOURCE_HAS_BRACKETS_RE = re.compile(r"[（(【\[]")
STAGING_BRACKETS_RE = re.compile(
    r"[（(](?:在)?(?:[左右双]?耳[边後后畔旁]|轻抚|抚摸|轻吻|深吻|轻舔|亲吻|"
    r"衣物摩擦声|呼吸声)[^）)]*[）)]"
)
REPEATED_PUNCT_RE = re.compile(r"([…—~～♡❤♪])。")
LEADING_INVALID_PUNCT_RE = re.compile(r"^[，、；：。！？!?\s]+")

FILENAME_PROMPT = (
    "翻译下面的文件名主体。[[K0]]、[[K1]] 等占位符必须原样保留。"
    "只输出结果，不要解释、不要加引号。\n"
    "文件名：{name}"
)
FILENAME_SYSTEM_PROMPT = (
    "你是文件名翻译器。只翻译文件名中的普通英文/日文词，已有中文保持原样。"
    "[[K0]]、[[K1]] 等占位符代表必须原样保留的编号或符号，输出时原样保留。"
    "示例：Hello_World → 你好_世界；Chapter_01 → 第01章；"
    "photos_2024 → 照片_2024；トラック① → 曲目①；"
    "report_临时 → 报告_临时；The Clock → 时钟。"
    "只输出文件名主体，不解释、不加引号。"
)
FILENAME_MAX_TOKENS = 96
FILENAME_MAX_RETRIES = 1
FILENAME_MAX_CHARS = 80
FILENAME_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
FILENAME_PREFIX = re.compile(
    r"^(?:(?:文件名|file name|翻译结果|翻译|译文|译名)\s*[:：]\s*)",
    re.IGNORECASE,
)
FILENAME_EXPLANATION = re.compile(
    r"(?:翻译|译文|译名|file name)\s*[:：]\s*", re.IGNORECASE
)
FILENAME_PLACEHOLDER = re.compile(r"\[\[K(\d+)\]\]")
FILENAME_KANA = re.compile(r"[\u3040-\u30ff]")
FILENAME_TOKEN = re.compile(r"[A-Za-z0-9+#&]+")
FILENAME_CODE_ID = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z]{1,4}\d{2,}(?![A-Za-z0-9])"
)
FILENAME_CODE_VERSION = re.compile(
    r"(?<![A-Za-z0-9])[vV]\d+(?![A-Za-z0-9])"
)
FILENAME_CODE_HYPHEN = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z]{1,4}[-_]\d+(?![A-Za-z0-9])"
)
FILENAME_CODE_NUMERIC = re.compile(r"(?<!\d)\d{2,}(?:[-_/.]\d{2,})+(?!\d)")
FILENAME_MARKER = re.compile(
    r"(?i)_(?:临时|final|draft|temp|copy|old|new|备份|草稿|保留)"
)
FILENAME_TECHNICAL_STEMS = frozenset(
    {"wav", "mp3", "mp4", "png", "jpg", "jpeg", "flac", "m4a", "txt"}
)
FILENAME_SYMBOL_CONNECTOR = re.compile(
    r"([♪♡❤💗✩★☆])(?:是|则|就)(?=[一-鿿])"
)


@dataclass
class TranslationFailure:
    source: str
    segment: str
    reason: str


@dataclass
class TranslationOutcome:
    source: Path
    output: Path | None
    segments: int
    failures: list[TranslationFailure] = field(default_factory=list)


@dataclass(frozen=True)
class TranslationSource:
    """一个待翻译文件；relative_parent 用于保留层级，translated_stem 为译名。"""

    path: Path
    relative_parent: Path = field(default_factory=Path)
    translated_stem: str | None = None


@dataclass(frozen=True)
class TranslationPlan:
    source: TranslationSource
    out_dir: Path | None
    target: Path

    @property
    def label(self) -> str:
        parent = self.source.relative_parent
        if parent in (Path(), Path(".")):
            return self.source.path.name
        return (parent / self.source.path.name).as_posix()


@dataclass(frozen=True)
class NameMask:
    text: str
    literals: list[str]


@dataclass(frozen=True)
class NameTranslation:
    name: str
    fallback_reason: str | None = None


@dataclass(frozen=True)
class NameTarget:
    path: Path
    is_dir: bool
    translated_name: str | None = None


@dataclass(frozen=True)
class NameRename:
    old: Path
    new: Path


@dataclass
class Subtitle:
    index: str
    timing: str
    text: str


def extract_content(result: dict) -> str:
    choices = result.get("choices") or []
    if not choices:
        raise EngineError("/v1/chat/completions 返回体缺少 choices")
    message = choices[0].get("message") or {}
    content = (message.get("content") or "").strip()
    if not content and message.get("reasoning_content"):
        raise EngineError("模型只返回了 reasoning_content，content 为空；思考未关闭？")
    if not content:
        raise EngineError("模型返回空内容")
    return content


@dataclass(frozen=True)
class TranslationOptions:
    n_ctx: int
    context_window: int
    context_summary_chars: int
    min_output_tokens: int
    output_ratio: float
    prompt_safety_tokens: int
    temperature: float
    max_split_depth: int


def _has_dialogue_quotes(text: str) -> bool:
    return any(char in QUOTE_CHARS for char in text)


def sanitize_translation_output(raw_text: str, source_text: str) -> str:
    """剥离模型额外加上的代码围栏/外层引号，同时保护原文对白引号。

    规则：
    - 先剥 Markdown 代码围栏（无论 source 是否有引号）；
    - source 含任何对白引号时，正文原样返回，不删任何引号；
    - source 无引号时，循环剥离成对的外层 ASCII/中文引号；不配对保留。
    - 不在这里删除 ``[#N]``：批次响应必须保留窗口编号，编号由调用方在
      cue 边界用 ``strip_window_id_prefix`` / 后处理清理。
    """
    text = raw_text.strip()
    if not text:
        return text
    for _ in range(3):
        match = FENCE_RE.match(text)
        if not match:
            break
        text = match.group(1).strip()
    text = text.lstrip("，、；：").lstrip()
    # 省略号/破折号已经承担句末语气，后面再追加句号属于重复标点。
    text = re.sub(r"(……|——)。", r"\1", text)
    if _has_dialogue_quotes(source_text):
        return text
    for _ in range(3):
        if len(text) < 2:
            break
        opening, closing = text[0], text[-1]
        if QUOTE_PAIRS.get(opening) != closing:
            break
        text = text[1:-1].strip()
    return text


def postprocess_cue_text(zh_text: str, src_text: str) -> str:
    """cue 级确定性清洗：简繁、行首标点、重复句号、未授权动作括号。

    只改字幕正文，不碰序号/时间轴；不调用模型。清洗结果为空时保留清洗前
    文本，避免把非空原文静默写成空字幕；这类残留交给 QC 提示。
    """
    original = zh_text.strip()
    if not original:
        return ""

    text = WINDOW_ID_ANY_RE.sub("", zh_text).strip()
    if not text:
        return original
    text = text.translate(TRADITIONAL_TO_SIMPLIFIED_MAP)
    if not SOURCE_HAS_BRACKETS_RE.search(src_text):
        stripped = STAGING_BRACKETS_RE.sub("", text)
        if stripped.strip():
            text = stripped

    lines = [
        LEADING_INVALID_PUNCT_RE.sub("", line, count=1)
        for line in text.splitlines()
    ]
    text = "\n".join(lines).strip()
    text = REPEATED_PUNCT_RE.sub(r"\1", text).strip()
    return text or original


def _is_mask_symbol(char: str) -> bool:
    """遮罩装饰符号，但保留下划线/点/连字符作为文件名分隔符。"""
    if "\u2460" <= char <= "\u2473":  # ①..⑳
        return True
    if char in "_.-" or char.isalnum() or char.isspace():
        return False
    category = unicodedata.category(char)
    return category.startswith("P") or category.startswith("S")


def _mask_name_literals(stem: str) -> NameMask:
    spans: list[tuple[int, int, str]] = []

    def add_span(start: int, end: int, text: str) -> None:
        if start < end:
            spans.append((start, end, text))

    for pattern in (
        FILENAME_CODE_ID,
        FILENAME_CODE_VERSION,
        FILENAME_CODE_HYPHEN,
        FILENAME_CODE_NUMERIC,
    ):
        for match in pattern.finditer(stem):
            add_span(match.start(), match.end(), match.group(0))
    for token_match in FILENAME_TOKEN.finditer(stem):
        token = token_match.group(0)
        letters = [char for char in token if char.isalpha()]
        uppercase = sum(1 for char in letters if char.isupper())
        has_internal_upper = any(char.isupper() for char in token[1:])
        if uppercase >= 2 or has_internal_upper or any(
            char in "+#&" for char in token
        ):
            add_span(token_match.start(), token_match.end(), token)
    for match in FILENAME_MARKER.finditer(stem):
        add_span(match.start(), match.end(), match.group(0))
    leading_dots = re.match(r"^\.+", stem)
    if leading_dots:
        add_span(0, leading_dots.end(), leading_dots.group(0))
    trailing_dots = re.search(r"\.+$", stem)
    if trailing_dots:
        add_span(trailing_dots.start(), trailing_dots.end(), trailing_dots.group(0))
    for index, char in enumerate(stem):
        if _is_mask_symbol(char):
            add_span(index, index + 1, char)

    spans.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    selected: list[tuple[int, int, str]] = []
    for start, end, text in spans:
        if selected and start < selected[-1][1]:
            continue
        selected.append((start, end, text))

    pieces: list[str] = []
    literals: list[str] = []
    cursor = 0
    for start, end, text in selected:
        pieces.append(stem[cursor:start])
        pieces.append(f"[[K{len(literals)}]]")
        literals.append(text)
        cursor = end
    pieces.append(stem[cursor:])
    return NameMask(text="".join(pieces), literals=literals)


def _has_translatable_text(masked: str) -> bool:
    remainder = FILENAME_PLACEHOLDER.sub(" ", masked)
    return bool(re.search(r"[A-Za-z]{2,}|[\u3040-\u30ff]", remainder))


def _placeholder_failure(text: str, literals: list[str]) -> str | None:
    expected = [f"[[K{index}]]" for index in range(len(literals))]
    for token in expected:
        if text.count(token) != 1:
            return f"占位符缺失或重复：{token}"
    found = FILENAME_PLACEHOLDER.findall(text)
    if len(found) != len(expected):
        return "出现未知占位符"
    return None


def _restore_literals(text: str, literals: list[str]) -> str:
    for index, literal in enumerate(literals):
        text = text.replace(f"[[K{index}]]", literal, 1)
    return text


def _strip_filename_explanation(raw: str) -> str:
    text = raw.strip()
    match = FILENAME_EXPLANATION.search(text)
    if match:
        after = text[match.end() :].strip()
        if after:
            return after
    return text


def sanitize_filename_stem(raw: str, source_suffix: str = "") -> str:
    """把模型输出收敛成安全的单段文件名（不含扩展名）。"""
    text = raw.strip()
    for _ in range(3):
        match = FENCE_RE.match(text)
        if not match:
            break
        text = match.group(1).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    text = _strip_filename_explanation(lines[0])
    text = FILENAME_PREFIX.sub("", text)
    if source_suffix and text.lower().endswith(source_suffix.lower()):
        text = text[: -len(source_suffix)]
    text = FILENAME_INVALID_CHARS.sub("_", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text or text in {".", ".."} or not text.strip("._- "):
        return ""
    return text


def _clip(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)] + "…"


# --------------------------------------------------------------------------- #
# 文档级实体倒排索引（无模型调用、无第三方依赖）
# --------------------------------------------------------------------------- #

KATAKANA_ENTITY_RE = re.compile(r"[\u30a1-\u30f6\u30fc]{2,8}")
HONORIFIC_NAME_RE = re.compile(
    r"([\u4e00-\u9fa5\u30a1-\u30f6\u30fc]{1,5})"
    r"(?:ちゃん|さん|くん|様|さま|先生|先輩)"
)
KANJI_CHUNK_RE = re.compile(r"[\u4e00-\u9fa5]{2,4}")
ENTITY_HONORIFIC_SUFFIXES = ("ちゃん", "さん", "くん", "様", "さま", "先生", "先輩")

# 噪声停用表：只过滤普通词/拟声词，不是译名词表。
ENTITY_STOPWORDS = frozenset(
    {
        "赤", "客", "お客", "姉", "兄", "皆", "奥", "旦那", "彼", "彼女",
        "子", "猫", "犬", "店員", "店長", "先生", "先輩", "友達", "家族",
        "母", "父", "妹", "弟", "おばあ", "おじい", "大丈", "本当", "今日",
        "明日", "昨日", "自分", "相手", "時間", "場所", "一緒", "最後",
        "最初", "全部", "全然", "普通", "最近", "今度", "今回", "来週",
        "先週", "今年", "去年", "毎日", "気持", "気分", "身体", "頭",
        "チュッ", "チュッチュ", "クチュ", "ジュル", "レロ", "コト", "コツン",
        "ドキドキ", "ゴクッ", "オイル", "ローション", "テスト", "カフェ",
        "ホテル", "マッサージ", "ラブラブ",
    }
)

ENTITY_MAX_CLUSTERS = 12
ENTITY_MAX_VARIANTS = 8
ENTITY_MAX_EXAMPLES = 2
ENTITY_EXAMPLE_CHARS = 80
ENTITY_CONTEXT_CHARS = 1200


def _common_prefix(values: set[str]) -> str:
    items = list(values)
    if not items:
        return ""
    prefix = items[0]
    for item in items[1:]:
        limit = min(len(prefix), len(item))
        index = 0
        while index < limit and prefix[index] == item[index]:
            index += 1
        prefix = prefix[:index]
        if not prefix:
            return ""
    return prefix


def _edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, 1):
        current = [i]
        for j, right_char in enumerate(right, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (0 if left_char == right_char else 1),
                )
            )
        previous = current
    return previous[-1]


def _seed_related(left: str, right: str) -> bool:
    if left == right or left in right or right in left:
        return True
    if left[:1] != right[:1]:
        return False
    if len(left) >= 3 and len(right) >= 3:
        if SequenceMatcher(None, left, right).ratio() >= 0.75:
            return True
    return abs(len(left) - len(right)) <= 1 and _edit_distance(left, right) <= 1


def _base_of_variant(variant: str) -> str:
    for suffix in ENTITY_HONORIFIC_SUFFIXES:
        if variant.endswith(suffix) and len(variant) > len(suffix):
            return variant[: -len(suffix)]
    return variant


class DynamicEntityEngine:
    """文档级实体索引：种子召回变体，首发锁定 + 前文示例注入。

    设计目标是“宁漏不错”：片假名/敬称名/共同前缀汉字变体才进入索引；
    锁定只接受“源行以实体开头且后接标点、译文前 2–4 字后接标点”的高置信
    场景，其余只保存前文示例。``enforce_locks`` 只清理未翻译的日文专名，
    不假装能确定性替换中文译名变体。
    """

    def __init__(self, raw_cues: list[str]) -> None:
        self.canonical_map: dict[str, str] = {}
        self.locked_translations: dict[str, str] = {}
        self.examples: dict[str, list[tuple[str, str]]] = {}
        self._variants: dict[str, set[str]] = {}
        self._canonical_order: list[str] = []
        self._build_index(raw_cues)

    # ------------------------------------------------------------------ #
    # 索引
    # ------------------------------------------------------------------ #

    def _build_index(self, raw_cues: list[str]) -> None:
        katakana_counts: Counter[str] = Counter()
        honorific_counts: Counter[str] = Counter()
        honorific_forms: dict[str, str] = {}
        kanji_counts: Counter[str] = Counter()
        all_counts: Counter[str] = Counter()
        first_seen: dict[str, int] = {}

        for index, text in enumerate(raw_cues):
            for match in KATAKANA_ENTITY_RE.finditer(text):
                term = match.group(0)
                katakana_counts[term] += 1
                all_counts[term] += 1
                first_seen.setdefault(term, index)
            for match in HONORIFIC_NAME_RE.finditer(text):
                name = match.group(1)
                full = match.group(0)
                if name in ENTITY_STOPWORDS:
                    continue
                honorific_counts[name] += 1
                honorific_forms[full] = name
                all_counts[name] += 1
                all_counts[full] += 1
                first_seen.setdefault(name, index)
                first_seen.setdefault(full, index)
            for match in KANJI_CHUNK_RE.finditer(text):
                term = match.group(0)
                kanji_counts[term] += 1
                all_counts[term] += 1
                first_seen.setdefault(term, index)

        seeds: dict[str, int] = {}
        for term, count in katakana_counts.items():
            if count >= 2 and term not in ENTITY_STOPWORDS:
                seeds[term] = max(seeds.get(term, 0), count)
        for term, count in honorific_counts.items():
            weighted = count * 2
            if weighted >= 2 and term not in ENTITY_STOPWORDS:
                seeds[term] = max(seeds.get(term, 0), weighted)

        prefix_groups: dict[str, set[str]] = {}
        for term in kanji_counts:
            if len(term) >= 3:
                prefix_groups.setdefault(term[:2], set()).add(term)
        for prefix, variants in prefix_groups.items():
            if len(variants) < 2 or prefix in ENTITY_STOPWORDS:
                continue
            common = _common_prefix(variants)
            if len(common) < 2 or common in ENTITY_STOPWORDS:
                continue
            seeds[common] = max(
                seeds.get(common, 0),
                sum(kanji_counts[item] for item in variants),
            )

        if not seeds:
            return

        clusters: list[set[str]] = []
        for term in sorted(seeds, key=lambda item: (-seeds[item], -len(item), item)):
            related = [
                cluster
                for cluster in clusters
                if any(_seed_related(term, other) for other in cluster)
            ]
            merged = {term}
            for cluster in related:
                merged |= cluster
                clusters.remove(cluster)
            clusters.append(merged)

        candidate_terms = (
            set(katakana_counts) | set(kanji_counts) | set(honorific_forms)
        )
        cluster_for: dict[str, str] = {}
        cluster_scores: dict[str, int] = {}
        for cluster in clusters:
            canonical = min(
                cluster,
                key=lambda item: (len(item), -seeds.get(item, 0), item),
            )
            variants = set(cluster)
            for term in candidate_terms:
                if term == canonical or canonical in term:
                    variants.add(term)
            for full, name in honorific_forms.items():
                if name == canonical:
                    variants.add(full)
            self._variants[canonical] = variants
            for term in variants:
                cluster_for[term] = canonical
            cluster_scores[canonical] = sum(
                all_counts[term] for term in variants
            )
            first_seen[canonical] = min(
                (first_seen.get(term, 10**9) for term in variants),
                default=10**9,
            )

        kept = sorted(
            cluster_scores,
            key=lambda item: (
                -cluster_scores[item],
                first_seen.get(item, 10**9),
                item,
            ),
        )[:ENTITY_MAX_CLUSTERS]
        kept_set = set(kept)
        self._variants = {
            canonical: variants
            for canonical, variants in self._variants.items()
            if canonical in kept_set
        }
        for term, canonical in cluster_for.items():
            if canonical in kept_set:
                self.canonical_map[term] = canonical
        self._canonical_order = kept

    # ------------------------------------------------------------------ #
    # 查询 / 捕捉 / 收敛
    # ------------------------------------------------------------------ #

    def get_query_context(self, window_text: str) -> str:
        """按当前窗口原文召回实体；无命中返回空串，由调用方叠加滑动前文。"""
        blocks: list[tuple[str, str, str]] = []
        for canonical in self._canonical_order:
            variants_set = self._variants.get(canonical)
            if not variants_set:
                continue
            if not any(variant in window_text for variant in variants_set):
                continue
            variants = sorted(
                variants_set, key=lambda item: (-len(item), item)
            )[:ENTITY_MAX_VARIANTS]
            locked = self.locked_translations.get(canonical, "")
            examples = self.examples.get(canonical, [])[:ENTITY_MAX_EXAMPLES]
            if len(variants) <= 1 and not locked and not examples:
                continue
            base = [f"- 实体「{canonical}」：{' / '.join(variants)}"]
            if locked:
                base.append(f"  已锁定译名：{locked}")
            with_examples = list(base)
            for source, translation in examples:
                with_examples.append(
                    f"  前文：{_clip(source, ENTITY_EXAMPLE_CHARS)} -> "
                    f"{_clip(translation, ENTITY_EXAMPLE_CHARS)}"
                )
            blocks.append(
                (canonical, "\n".join(base), "\n".join(with_examples))
            )

        if not blocks:
            return ""

        title = (
            "【本文实体与前文译法（仅供参考；同一实体必须统一译名，"
            "勿漂移，勿整句照抄）】"
        )
        for use_examples in (True, False):
            lines = [title]
            total = len(title) + 1
            for _canonical, without_examples, with_examples in blocks:
                block = with_examples if use_examples else without_examples
                if total + len(block) + 1 > ENTITY_CONTEXT_CHARS:
                    continue
                lines.append(block)
                total += len(block) + 1
            if len(lines) > 1:
                return "\n".join(lines)
        return ""

    def capture_translations(
        self, window_src: list[str], window_zh: list[str]
    ) -> None:
        """从完成窗口捕捉前文示例；仅高置信场景写入锁定译名。"""
        for source, translation in zip(window_src, window_zh):
            source = source.strip()
            translation = translation.strip()
            if not source or not translation:
                continue
            hits = [
                canonical
                for canonical in self._canonical_order
                if any(
                    variant in source
                    for variant in self._variants.get(canonical, ())
                )
            ]
            if not hits:
                continue
            for canonical in hits[:2]:
                self._add_example(canonical, source, translation)
            if len(hits) != 1:
                continue
            canonical = hits[0]
            if canonical in self.locked_translations:
                continue
            locked = self._extract_locked_name(source, translation, canonical)
            if locked:
                self.locked_translations[canonical] = locked

    def enforce_locks(self, zh_text: str) -> str:
        """把中文译文里未翻译的日文专名收敛到已锁定译名。"""
        if not zh_text or not self.locked_translations:
            return zh_text
        result = zh_text
        for canonical, locked in sorted(
            self.locked_translations.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            variants = sorted(
                self._variants.get(canonical, {canonical}),
                key=len,
                reverse=True,
            )
            for variant in variants:
                if variant and variant in result:
                    result = result.replace(variant, locked)
        return result

    def _add_example(
        self, canonical: str, source: str, translation: str
    ) -> None:
        items = self.examples.setdefault(canonical, [])
        pair = (source.strip(), translation.strip())
        if pair in items:
            return
        if len(items) < ENTITY_MAX_EXAMPLES:
            items.append(pair)

    def _extract_locked_name(
        self, source: str, translation: str, canonical: str
    ) -> str:
        stripped_source = source.lstrip()
        if not stripped_source or stripped_source.startswith(
            ("（", "(", "【", "[")
        ):
            return ""
        variants = sorted(
            self._variants.get(canonical, {canonical}),
            key=len,
            reverse=True,
        )
        matched = ""
        for variant in variants:
            if not stripped_source.startswith(variant):
                continue
            rest = stripped_source[len(variant):]
            if not rest or rest[0] in "、，,！!?？…":
                matched = variant
                break
        if not matched or _base_of_variant(matched) != canonical:
            return ""
        match = re.match(
            r"^([\u4e00-\u9fa5]{2,4})(?=[，。！？、；：…—~]|$)",
            translation,
        )
        if not match:
            return ""
        return match.group(1)


class TranslationEngine:
    """单段翻译管线：滑动上下文、预算门禁、截断重译与统计事件。"""

    def __init__(
        self,
        client: LlamaClient,
        prompt_dir: Path,
        system_text: str,
        *,
        options: TranslationOptions,
        on_stat: Callable[[SegmentStat], None] | None = None,
        task_template: str = "translate",
    ) -> None:
        self._client = client
        self._template = prompts.load_task(prompt_dir, task_template)
        self._system = system_text
        self._options = options
        window = max(0, int(self._options.context_window))
        self._history: deque[tuple[str, str]] = deque(maxlen=window)
        self._entity_engine: DynamicEntityEngine | None = None
        self._on_stat = on_stat

    def __call__(self, text: str) -> str:
        return self.translate(text)

    def translate(self, text: str) -> str:
        return self._translate_once(text, depth=0)

    def reset_history(self) -> None:
        """开始新文档时清空滑动上下文；文档级实体锁定表不受影响。"""
        self._history.clear()

    def set_entity_engine(self, engine: DynamicEntityEngine | None) -> None:
        """绑定文档级实体索引；由 translate_srt 在单文件翻译期间管理。"""
        self._entity_engine = engine

    def translate_filename(self, filename: str) -> str:
        """用翻译模型把文件名主体译成中文；不写入正文上下文。"""
        return self.translate_filename_result(filename).name

    def translate_filename_result(self, filename: str) -> NameTranslation:
        path = Path(filename)
        return self._translate_name_result(path.stem, path.suffix, filename)

    def translate_dirname(self, dirname: str) -> str:
        """翻译目录名整体，不把点号当扩展名剥离。"""
        return self.translate_dirname_result(dirname).name

    def translate_dirname_result(self, dirname: str) -> NameTranslation:
        return self._translate_name_result(dirname, "", dirname)

    def _translate_name_result(
        self, name: str, source_suffix: str, label: str
    ) -> NameTranslation:
        stem = name.strip()
        if not stem:
            raise TranslateError(f"名称为空，无法翻译：{label!r}")
        if stem.lower() in FILENAME_TECHNICAL_STEMS:
            return NameTranslation(stem)
        mask = _mask_name_literals(stem)
        if not _has_translatable_text(mask.text):
            return NameTranslation(stem)

        reason = ""
        for attempt in range(FILENAME_MAX_RETRIES + 1):
            result = self._call_name(mask.text, reason if attempt else "")
            choices = result.get("choices") or []
            finish_reason = ""
            if choices and isinstance(choices[0], dict):
                finish_reason = str(choices[0].get("finish_reason") or "")
            if finish_reason == "length":
                reason = "输出被截断"
                continue
            raw = extract_content(result)
            translated = sanitize_filename_stem(
                _strip_filename_explanation(raw), source_suffix
            )
            if not translated:
                reason = "结果为空"
                continue
            reason = _placeholder_failure(translated, mask.literals) or ""
            if reason:
                continue
            restored = _restore_literals(translated, mask.literals)
            restored = FILENAME_SYMBOL_CONNECTOR.sub(r"\1", restored)
            if len(restored) > FILENAME_MAX_CHARS:
                reason = "译名过长"
                continue
            if restored == stem and FILENAME_KANA.search(stem):
                reason = "未翻译日文假名"
                continue
            return NameTranslation(restored)
        return NameTranslation(stem, reason or "模型输出不可用")

    def _call_name(self, masked: str, correction: str) -> dict:
        user_content = FILENAME_PROMPT.format(name=masked)
        if correction:
            if "日文假名" in correction:
                user_content += (
                    "\n上一次结果没有翻译日文假名内容；"
                    "请把普通日文词翻译成中文，完整保留所有占位符，"
                    "只输出文件名主体。"
                )
            else:
                user_content += (
                    f"\n上一次结果不可用：{correction}。"
                    "请重新翻译，完整保留所有占位符，只输出文件名主体。"
                )
        messages: list[dict] = [
            {"role": "system", "content": FILENAME_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        return self._client.chat(
            messages,
            max_tokens=FILENAME_MAX_TOKENS,
            temperature=self._options.temperature,
        )

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #

    def _context_text(self) -> str:
        if not self._history:
            return "无前文，故事开端"
        limit = max(20, int(self._options.context_summary_chars))
        lines: list[str] = []
        for index, (source, translation) in enumerate(self._history, start=1):
            lines.append(f"[{index}] 原文：{_clip(source, limit)}")
            lines.append(f"    译文：{_clip(translation, limit)}")
        return "\n".join(lines)

    def _render(self, text: str) -> tuple[list[dict], int, int]:
        entity_context = ""
        if self._entity_engine is not None:
            entity_context = self._entity_engine.get_query_context(text)
        history = self._context_text()
        if entity_context and history == "无前文，故事开端":
            context = entity_context
        elif entity_context:
            context = entity_context + "\n\n【前文】\n" + history
        else:
            context = history
        prompt = prompts.render(
            self._template,
            {"input": text, "context_history": context},
        )
        messages: list[dict] = []
        if self._system.strip():
            messages.append({"role": "system", "content": self._system})
        messages.append({"role": "user", "content": prompt})
        prompt_tokens = (
            estimate_tokens(self._system)
            + estimate_tokens(prompt)
            + PROMPT_OVERHEAD
        )
        return messages, prompt_tokens, estimate_tokens(text)

    def _translate_once(self, text: str, *, depth: int) -> str:
        options = self._options
        messages, prompt_tokens, input_tokens = self._render(text)
        desired = max(
            int(options.min_output_tokens),
            int(int(input_tokens) * float(options.output_ratio)),
        )
        while (
            self._history
            and prompt_tokens + desired
            > int(options.n_ctx) - int(options.prompt_safety_tokens)
        ):
            self._history.popleft()
            messages, prompt_tokens, input_tokens = self._render(text)

        try:
            budget = translation_budget(
                options.n_ctx,
                prompt_tokens,
                input_tokens,
                min_output_tokens=options.min_output_tokens,
                output_ratio=options.output_ratio,
                safety_tokens=options.prompt_safety_tokens,
            )
        except ValueError as exc:
            raise TranslateError(str(exc)) from exc

        started = time.perf_counter()
        result = self._client.chat(
            messages,
            max_tokens=budget.max_tokens,
            temperature=options.temperature,
        )
        elapsed_s = time.perf_counter() - started
        choices = result.get("choices") or []
        finish_reason = ""
        if choices and isinstance(choices[0], dict):
            finish_reason = str(choices[0].get("finish_reason") or "")
        if finish_reason == "length":
            if depth >= options.max_split_depth or len(text) <= 1:
                raise OutputTruncated(
                    f"输出达到长度上限，拆分 {options.max_split_depth} 层仍未完成"
                )
            parts = _split_for_retry(text)
            if len(parts) <= 1:
                raise OutputTruncated("输出达到长度上限，无法继续拆分")
            return "".join(
                self._translate_once(part, depth=depth + 1) for part in parts
            )

        raw = extract_content(result)
        translated = sanitize_translation_output(raw, text)
        raw_usage = result.get("usage")
        usage = raw_usage if isinstance(raw_usage, dict) else {}
        raw_timings = result.get("timings")
        timings = raw_timings if isinstance(raw_timings, dict) else {}
        tokens = (
            int(usage.get("completion_tokens") or 0)
            or int(timings.get("predicted_n") or 0)
            or estimate_tokens(translated)
        )
        server_tps = float(timings.get("predicted_per_second") or 0.0)
        if server_tps > 0:
            tps = server_tps
            tps_source = "server.timings"
        else:
            tps = tokens / elapsed_s if elapsed_s > 0 else 0.0
            tps_source = "client.usage" if usage else "estimate"
        self._history.append((text, translated))
        if self._on_stat is not None:
            self._on_stat(
                SegmentStat(
                    source=text,
                    translation=translated,
                    elapsed_s=elapsed_s,
                    tokens=tokens,
                    tps=tps,
                    tps_source=tps_source,
                )
            )
        return translated


def make_engine_chat(
    client: LlamaClient,
    prompt_dir: Path,
    system_text: str,
    *,
    n_ctx: int,
    context_window: int,
    context_summary_chars: int,
    min_output_tokens: int,
    output_ratio: float,
    prompt_safety_tokens: int,
    temperature: float,
    max_split_depth: int,
    on_stat: Callable[[SegmentStat], None] | None = None,
    task_template: str = "translate",
) -> Callable[[str], str]:
    """兼容入口：返回可调用的单段翻译函数。"""
    options = TranslationOptions(
        n_ctx=n_ctx,
        context_window=context_window,
        context_summary_chars=context_summary_chars,
        min_output_tokens=min_output_tokens,
        output_ratio=output_ratio,
        prompt_safety_tokens=prompt_safety_tokens,
        temperature=temperature,
        max_split_depth=max_split_depth,
    )
    return TranslationEngine(
        client,
        prompt_dir,
        system_text,
        options=options,
        on_stat=on_stat,
        task_template=task_template,
    )


def chunk_text(text: str, limit: int) -> list[str]:
    if limit <= 0:
        raise TranslateError(f"chunk_chars 必须为正整数，实际为 {limit}")
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    rest = text
    while len(rest) > limit:
        window = rest[:limit]
        cut = 0
        for index in range(len(window) - 1, -1, -1):
            if window[index] in SENTENCE_ENDERS:
                cut = index + 1
                break
        if cut <= 0:
            space = window.rfind(" ")
            cut = space + 1 if space > 0 else limit
        chunks.append(rest[:cut])
        rest = rest[cut:]
    if rest:
        chunks.append(rest)
    return chunks


def _split_for_retry(text: str) -> list[str]:
    """优先在句末标点/空格二分；最差按中点切，保证所有字符不丢。"""
    if len(text) <= 1:
        return []
    parts = chunk_text(text, max(1, len(text) // 2))
    parts = [part for part in parts if part]
    if len(parts) >= 2:
        return parts
    middle = max(1, len(text) // 2)
    if middle >= len(text):
        return [text]
    return [text[:middle], text[middle:]]


def _translate_chunks(
    chat: Callable[[str], str],
    text: str,
    *,
    chunk_chars: int,
    source: str,
    label: str,
    failures: list[TranslationFailure],
) -> str | None:
    translated: list[str] = []
    for position, chunk in enumerate(chunk_text(text, chunk_chars), start=1):
        if not chunk.strip():
            translated.append(chunk)
            continue
        try:
            translated.append(chat(chunk))
        except (EngineError, TranslateError) as exc:
            failures.append(
                TranslationFailure(source=source, segment=f"{label}#{position}", reason=str(exc))
            )
            return None
    return "".join(translated)


def parse_srt(text: str, eol: str = "\n") -> list[tuple[Subtitle, str]]:
    """解析 SRT 字幕块：使用 splitlines() 兼容任意混合换行（CRLF/LF/CR），彻底阻断 0 段静默写回。"""
    blocks: list[tuple[Subtitle, str]] = []
    pieces = BLOCK_SEP.split(text)
    index = 0
    while index < len(pieces):
        content = pieces[index]
        separator = pieces[index + 1] if index + 1 < len(pieces) else ""
        index += 2
        if not content.strip():
            if blocks and separator:
                previous, previous_sep = blocks[-1]
                blocks[-1] = (previous, previous_sep + content + separator)
            continue

        # 核心修复点：使用 splitlines() 处理单块内部的混合换行，不再使用脆弱的 content.split(eol)
        raw_lines = [line.strip("\r\n") for line in content.splitlines()]
        if (
            len(raw_lines) >= 2
            and raw_lines[0].strip().isdigit()
            and SRT_TIME.match(raw_lines[1].strip())
        ):
            blocks.append(
                (
                    Subtitle(
                        index=raw_lines[0].strip(),
                        timing=raw_lines[1].strip(),
                        text=eol.join(raw_lines[2:]),
                    ),
                    separator,
                )
            )
        else:
            blocks.append((Subtitle(index="", timing="", text=content), separator))
    return blocks


def _normalize_eol(text: str, eol: str) -> str:
    """把任意换行统一为目标 eol，避免重组后残留混合行尾。"""
    target = eol or "\n"
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", target)


def rebuild_srt(blocks: list[tuple[Subtitle, str]], eol: str) -> str:
    target = eol or "\n"
    output: list[str] = []
    for subtitle, separator in blocks:
        if subtitle.index and subtitle.timing:
            text = _normalize_eol(subtitle.text, target)
            if text.strip():
                output.append(target.join([subtitle.index, subtitle.timing, text]))
            else:
                # 空正文不能再拼末尾换行，否则会与块分隔符叠加成三连换行，
                # 导致下一块被 parse_srt 误判为未识别内容。
                output.append(target.join([subtitle.index, subtitle.timing]))
        else:
            output.append(_normalize_eol(subtitle.text, target))
        output.append(_normalize_eol(separator, target))
    return "".join(output)


def split_prose(text: str) -> list[tuple[str, str]]:
    """返回 [(段落, 段落间分隔符)]；分隔符原样保留。"""
    pieces = BLOCK_SEP.split(text)
    paragraphs: list[tuple[str, str]] = []
    index = 0
    while index < len(pieces):
        content = pieces[index]
        separator = pieces[index + 1] if index + 1 < len(pieces) else ""
        index += 2
        paragraphs.append((content, separator))
    return paragraphs


def _parse_window_items(
    raw_text: str,
) -> tuple[dict[str, list[str]], list[str], bool]:
    translations: dict[str, list[str]] = {}
    order: list[str] = []
    current_id: str | None = None
    duplicated = False
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = WINDOW_ID_RE.match(stripped)
        if match:
            current_id = match.group(1)
            if current_id in translations:
                duplicated = True
                continue
            translations[current_id] = [match.group(2).strip()]
            order.append(current_id)
        elif current_id is not None:
            translations[current_id].append(stripped)
    return translations, order, duplicated


def parse_window_response(
    raw_text: str, expected_ids: list[str]
) -> tuple[bool, dict[str, str], list[str]]:
    """解析带有 [#序号] 前缀的模型多行输出。

    返回：(是否完全对齐, {序号: 译文}, 缺失序号列表)。
    支持同一序号下多行字幕的自动合并（续行逻辑）。
    重复序号、额外序号与乱序都判为失败，交给调用方整体回退。
    """
    translations, order, duplicated = _parse_window_items(raw_text)
    missing = [cid for cid in expected_ids if cid not in translations]
    if duplicated or missing or order != expected_ids:
        return False, {}, missing

    aligned_dict = {
        cid: "\n".join(translations[cid]).strip() for cid in expected_ids
    }
    return True, aligned_dict, []


def recover_window_response(
    raw_text: str, expected_ids: list[str]
) -> tuple[bool, dict[str, str], list[str]]:
    """恢复“模型每窗从 [#1] 重新编号”的已知失败模式。

    仅在输出编号恰好为 1..K、无重复且数量与 expected_ids 一致时，按位置重映射；
    乱序、缺项、多翻、重复一律不恢复，继续走逐句回退。
    """
    translations, order, duplicated = _parse_window_items(raw_text)
    if (
        duplicated
        or len(order) != len(expected_ids)
        or len(translations) != len(expected_ids)
    ):
        return False, {}, []
    if order == expected_ids:
        return parse_window_response(raw_text, expected_ids)
    expected_count = len(expected_ids)
    if order != [str(index) for index in range(1, expected_count + 1)]:
        return False, {}, []
    aligned_dict = {
        expected_id: "\n".join(translations[actual_id]).strip()
        for expected_id, actual_id in zip(expected_ids, order)
    }
    return True, aligned_dict, []


def strip_window_id_prefix(text: str) -> str:
    """删除译文任意位置误带的 [#N]；编号不应进入正文。"""
    return WINDOW_ID_ANY_RE.sub("", text.strip()).strip()


def _normalize_for_compare(text: str) -> str:
    """去掉空白与常见断句符号，用于检测“不同原文、相同译文”的窗口错位。"""
    return re.sub(r"[\s。！？，、…—]+", "", text)


def _source_texts_are_similar(left: str, right: str) -> bool:
    """短句/同义改写不触发重复检测，避免把正常近义句误判为错位。"""
    if not left or not right:
        return False
    return SequenceMatcher(None, left, right).ratio() >= 0.75


def _window_translation_is_suspicious(
    window: list[tuple[int, Subtitle]],
    aligned: dict[str, str],
) -> bool:
    """检测结构完整但语义明显错位的窗口，避免把相邻行译文写进当前 cue。

    三类信号：
    1. 正文残留 [#N]；
    2. 非空原文对应空译文；
    3. 不同原文得到完全相同且足够长的译文（合法重复短句不触发）。
    """
    pairs: list[tuple[str, str]] = []
    for _index, subtitle in window:
        translation = aligned.get(subtitle.index, "")
        if WINDOW_ID_ANY_RE.search(translation):
            return True
        if subtitle.text.strip() and not translation.strip():
            return True
        pairs.append((_normalize_for_compare(subtitle.text), translation))

    for left in range(len(pairs)):
        source_left, translation_left = pairs[left]
        normalized_left = _normalize_for_compare(translation_left)
        if len(normalized_left) < 8 or len(source_left) < 4:
            continue
        for right in range(left + 1, len(pairs)):
            source_right, translation_right = pairs[right]
            if source_left == source_right or len(source_right) < 4:
                continue
            if _source_texts_are_similar(source_left, source_right):
                continue
            normalized_right = _normalize_for_compare(translation_right)
            if normalized_left == normalized_right:
                return True
            # 相邻行部分重叠：一个译文完整包含另一个，或两者高度相似，通常是把下一行内容并进来了。
            if len(normalized_left) >= 8 and len(normalized_right) >= 8 and (
                normalized_left in normalized_right
                or normalized_right in normalized_left
                or SequenceMatcher(None, normalized_left, normalized_right).ratio()
                >= 0.8
            ):
                return True
    return False


def _translate_single_cue(
    chat: Callable[[str], str],
    subtitle: Subtitle,
    *,
    source: str,
    failures: list[TranslationFailure],
) -> str | None:
    """窗口回退时按“单编号窗口”重译，避免无编号裸文本被模型续写多行。"""
    prompt = f"[#{subtitle.index}] {subtitle.text}"
    try:
        raw = chat(prompt)
    except (EngineError, TranslateError) as exc:
        failures.append(
            TranslationFailure(
                source=source,
                segment=f"line {subtitle.index}",
                reason=str(exc),
            )
        )
        return None

    ok, aligned, _missing = parse_window_response(raw, [subtitle.index])
    if not ok:
        ok, aligned, _missing = recover_window_response(raw, [subtitle.index])
    if ok:
        return strip_window_id_prefix(aligned[subtitle.index])

    cleaned = strip_window_id_prefix(raw)
    if not cleaned:
        failures.append(
            TranslationFailure(
                source=source,
                segment=f"line {subtitle.index}",
                reason="单句回退返回空译文",
            )
        )
        return None
    # 没有编号却返回多个编号/多行，说明模型仍在续写相邻行；宁可保留原文，不写入错位内容。
    source_lines = len([line for line in subtitle.text.splitlines() if line.strip()])
    leaked_ids = WINDOW_ID_ANY_RE.findall(raw)
    if len(leaked_ids) > 1 or (
        source_lines <= 1 and len([line for line in cleaned.splitlines() if line.strip()]) > 2
    ):
        failures.append(
            TranslationFailure(
                source=source,
                segment=f"line {subtitle.index}",
                reason="单句回退疑似续写相邻行，已保留原文",
            )
        )
        return None
    return cleaned


def translate_srt(
    chat: Callable[[str], str],
    text: str,
    *,
    chunk_chars: int,
    source: str,
    eol: str,
    failures: list[TranslationFailure],
    on_segment: Callable[[int, int], None] | None = None,
    window_size: int = 1,
    max_window_chars: int = 250,
) -> str:
    blocks = parse_srt(text, eol)
    valid_items = [
        (index, block)
        for index, (block, _separator) in enumerate(blocks)
        if block.index and block.text.strip()
    ]
    total = len(valid_items)
    translatable: list[tuple[int, Subtitle]] = []
    filtered = 0
    for item in valid_items:
        subtitle = item[1]
        if ASR_HALLUCINATION_PATTERNS.search(subtitle.text):
            subtitle.text = ""
            filtered += 1
        else:
            translatable.append(item)
    done = 0

    if window_size <= 1:
        clear_entity = getattr(chat, "set_entity_engine", None)
        if callable(clear_entity):
            clear_entity(None)
        for _block_index, subtitle in valid_items:
            if not subtitle.text.strip():
                # ASR 幻觉已置空：不调用模型，但仍计入已完成段，保持进度与 segments 一致
                done += 1
                if on_segment is not None:
                    on_segment(done, total)
                continue
            result = _translate_chunks(
                chat,
                subtitle.text,
                chunk_chars=chunk_chars,
                source=source,
                label=f"line {subtitle.index}",
                failures=failures,
            )
            if result is not None:
                subtitle.text = postprocess_cue_text(result, subtitle.text)
            # 失败时保留原文，继续下一条
            done += 1
            if on_segment is not None:
                on_segment(done, total)
        return rebuild_srt(blocks, eol)

    entity_engine = DynamicEntityEngine(
        [subtitle.text for _index, subtitle in translatable]
    )
    setter = getattr(chat, "set_entity_engine", None)
    if callable(setter):
        setter(entity_engine)
    try:
        windows: list[list[tuple[int, Subtitle]]] = []
        current: list[tuple[int, Subtitle]] = []
        current_chars = 0
        for item in translatable:
            text_len = len(item[1].text)
            if current and (
                len(current) >= window_size
                or current_chars + text_len > max_window_chars
            ):
                windows.append(current)
                current = []
                current_chars = 0
            current.append(item)
            current_chars += text_len
            if len(current) >= window_size:
                windows.append(current)
                current = []
                current_chars = 0
        if current:
            windows.append(current)

        for window in windows:
            expected_ids = [item[1].index for item in window]
            fallback = len(set(expected_ids)) != len(expected_ids)
            if not fallback:
                batch_prompt = "\n".join(
                    f"[#{item[1].index}] {item[1].text}" for item in window
                )
                try:
                    batch_output = chat(batch_prompt)
                    ok, aligned, _missing = parse_window_response(
                        batch_output, expected_ids
                    )
                    if not ok:
                        ok, aligned, _missing = recover_window_response(
                            batch_output, expected_ids
                        )
                    if ok and _window_translation_is_suspicious(window, aligned):
                        ok = False
                    if ok:
                        window_sources = [item[1].text for item in window]
                        staging: list[tuple[Subtitle, str]] = []
                        for _block_index, subtitle in window:
                            text = strip_window_id_prefix(
                                aligned[subtitle.index]
                            )
                            text = postprocess_cue_text(text, subtitle.text)
                            text = entity_engine.enforce_locks(text)
                            staging.append((subtitle, text))
                        entity_engine.capture_translations(
                            window_sources,
                            [text for _subtitle, text in staging],
                        )
                        for subtitle, text in staging:
                            subtitle.text = entity_engine.enforce_locks(text)
                    else:
                        fallback = True
                except (EngineError, TranslateError):
                    fallback = True

            if fallback:
                # 硬熔断：该窗口原地回退为逐句串行，绝不采用部分对齐结果
                reset_history = getattr(chat, "reset_history", None)
                for _block_index, subtitle in window:
                    # 单句回退不继承上一句的滑动上下文，避免模型续写相邻行。
                    if callable(reset_history):
                        reset_history()
                    result = _translate_single_cue(
                        chat,
                        subtitle,
                        source=source,
                        failures=failures,
                    )
                    if result is not None:
                        result = postprocess_cue_text(result, subtitle.text)
                        result = entity_engine.enforce_locks(result)
                        entity_engine.capture_translations(
                            [subtitle.text], [result]
                        )
                        result = entity_engine.enforce_locks(result)
                        subtitle.text = result

            done += len(window)
            if on_segment is not None:
                on_segment(done, total)

        if filtered:
            # 被 ASR 硬清洗的空块不发请求，但要在末尾补齐进度，保证 done==total
            done += filtered
            if on_segment is not None:
                on_segment(done, total)

        return rebuild_srt(blocks, eol)
    finally:
        if callable(setter):
            setter(None)


def translate_prose(
    chat: Callable[[str], str],
    text: str,
    *,
    chunk_chars: int,
    source: str,
    failures: list[TranslationFailure],
    on_segment: Callable[[int, int], None] | None = None,
) -> str:
    paragraphs = split_prose(text)
    total = sum(1 for paragraph, _ in paragraphs if paragraph.strip())
    done = 0
    output: list[str] = []
    for position, (paragraph, separator) in enumerate(paragraphs, start=1):
        match = re.match(r"^(\s*)(.*?)(\s*)$", paragraph, re.DOTALL)
        if match is None:
            # re 已保证可匹配；这里只是防御，绝不吞掉内容
            output.append(paragraph)
            output.append(separator)
            continue
        leading, core, trailing = match.groups()
        if not core:
            output.append(paragraph)
            output.append(separator)
            continue  # 空白段落不计入进度
        result = _translate_chunks(
            chat,
            core,
            chunk_chars=chunk_chars,
            source=source,
            label=f"block {position}",
            failures=failures,
        )
        output.append(leading + (result if result is not None else core) + trailing)
        output.append(separator)
        done += 1
        if on_segment is not None:
            on_segment(done, total)
    return "".join(output)


def _detect_eol(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def _read_text(path: Path) -> tuple[str, bool, str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise TranslateError(f"读取失败：{path}；{exc.strerror or exc}") from exc
    has_bom = raw.startswith(BOM)
    try:
        text = raw[len(BOM):].decode("utf-8") if has_bom else raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TranslateError(
            f"文件不是 UTF-8：{path}；第 {exc.start} 字节，请先转码"
        ) from exc
    return text, has_bom, _detect_eol(text)


def output_path(
    source: Path,
    out_dir: Path | None,
    suffix: str,
    stem: str | None = None,
) -> Path:
    if stem is not None:
        base = sanitize_filename_stem(stem, source.suffix) or source.stem
    else:
        base = source.stem
    name = f"{base}{suffix}{source.suffix}"
    return (out_dir / name) if out_dir is not None else (source.parent / name)


def _path_key(path: Path) -> str:
    """跨 Windows 大小写/符号链接比较路径；目标文件通常尚不存在。"""
    return os.path.normcase(os.path.realpath(os.path.abspath(str(path))))


def _is_generated_output(path: Path, output_suffix: str) -> bool:
    suffix = (output_suffix or "").lower()
    return bool(suffix) and path.stem.lower().endswith(suffix)


def _is_excluded_dir(candidate: Path, excluded_key: str | None) -> bool:
    """仅剪掉 --out-dir 本身，避免把上次产物再次当输入。"""
    if not excluded_key:
        return False
    return _path_key(candidate) == excluded_key


def collect_translate_inputs(
    inputs: list[Path],
    settings: TranslateSettings,
    *,
    exclude_dir: Path | None = None,
) -> list[TranslationSource]:
    """把文件/目录输入展开为去重后的文件列表。

    - 目录递归遍历，只收 settings.source_suffixes，显式文件不按后缀过滤；
    - 目录中的 ``*.zh.md`` 等已有产物跳过，避免自我递归；
    - 同名文件按真实路径去重，首次出现的目录层级决定输出层级；
    - 空目录与不可遍历目录在启动模型前报错；显式文件仍由 translate_file 判定。
    """
    allowed = frozenset(item.lower() for item in settings.source_suffixes)
    excluded_key = _path_key(exclude_dir) if exclude_dir is not None else None
    collected: list[TranslationSource] = []
    seen: set[str] = set()

    def add(path: Path, relative_parent: Path) -> bool:
        key = _path_key(path)
        if key in seen:
            return False
        seen.add(key)
        collected.append(TranslationSource(path=path, relative_parent=relative_parent))
        return True

    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            matched = 0
            errors: list[OSError] = []

            def on_error(exc: OSError) -> None:
                errors.append(exc)

            for root, dirs, files in os.walk(
                path, topdown=True, onerror=on_error, followlinks=False
            ):
                root_path = Path(root)
                dirs[:] = sorted(
                    (
                        name
                        for name in dirs
                        if not _is_excluded_dir(root_path / name, excluded_key)
                    ),
                    key=str.lower,
                )
                files.sort(key=str.lower)
                for name in files:
                    candidate = root_path / name
                    if candidate.suffix.lower() not in allowed:
                        continue
                    if _is_generated_output(candidate, settings.output_suffix):
                        continue
                    if not candidate.is_file():
                        continue
                    matched += 1
                    add(candidate, candidate.parent.relative_to(path))
            if errors:
                first = errors[0]
                detail = first.strerror or str(first)
                raise TranslateError(
                    f"遍历目录失败：{first.filename or path}；{detail}"
                ) from first
            if matched == 0:
                suffixes = "、".join(settings.source_suffixes)
                raise TranslateError(
                    f"目录中没有可翻译文件：{path}；支持：{suffixes}"
                )
        else:
            # 显式文件保持旧契约：存在性、后缀与可读性仍由 translate_file 判定。
            # 这里不能提前拒绝，否则调用方替换/延迟写盘的场景会被误伤。
            add(path, Path())

    if not collected:
        raise TranslateError("没有可翻译的输入")
    return collected


def collect_name_targets(inputs: list[Path]) -> list[NameTarget]:
    """收集只改名模式的目标：显式文件本身 + 目录下的文件/子目录。

    传入的根目录本身不重命名，避免把用户当前目录或仓库根整体改掉；
    隐藏文件和隐藏目录不进入递归，防止误改 ``.git`` 等。
    """
    targets: list[NameTarget] = []
    seen: set[str] = set()

    def add(path: Path, is_dir: bool) -> None:
        key = _path_key(path)
        if key in seen:
            return
        seen.add(key)
        targets.append(NameTarget(path=path, is_dir=is_dir))

    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            found = 0
            errors: list[OSError] = []

            def on_error(exc: OSError) -> None:
                errors.append(exc)

            for root, dirs, files in os.walk(
                path, topdown=True, onerror=on_error, followlinks=False
            ):
                root_path = Path(root)
                dirs[:] = sorted(
                    (name for name in dirs if not name.startswith(".")),
                    key=str.lower,
                )
                files = sorted(
                    (name for name in files if not name.startswith(".")),
                    key=str.lower,
                )
                for name in dirs:
                    add(root_path / name, True)
                    found += 1
                for name in files:
                    child = root_path / name
                    if child.is_file() or child.is_symlink():
                        add(child, False)
                        found += 1
            if errors:
                first = errors[0]
                detail = first.strerror or str(first)
                raise TranslateError(
                    f"遍历目录失败：{first.filename or path}；{detail}"
                ) from first
            if found == 0:
                raise TranslateError(f"目录中没有可重命名的文件或子目录：{path}")
        elif path.is_file() or path.is_symlink():
            add(path, False)
        else:
            raise TranslateError(f"路径不存在或不是文件/目录：{path}")

    if not targets:
        raise TranslateError("没有可重命名的文件或目录")
    return targets


def plan_name_renames(targets: list[NameTarget]) -> list[NameRename]:
    """生成并校验只改名计划；不触碰文件内容。"""
    renames: list[NameRename] = []
    new_keys: dict[str, NameRename] = {}
    old_keys = {_path_key(target.path) for target in targets}
    moved_old: set[str] = set()

    for target in targets:
        if target.translated_name is None:
            raise TranslateError(f"缺少译名：{target.path}")
        old = target.path
        if target.is_dir:
            new_name = sanitize_filename_stem(target.translated_name, "")
        else:
            base = sanitize_filename_stem(target.translated_name, old.suffix)
            new_name = f"{base}{old.suffix}" if base else ""
        if not new_name:
            raise TranslateError(f"译名为空或不可用：{old}")
        if new_name == old.name:
            continue
        new_path = old.with_name(new_name)
        key = _path_key(new_path)
        if key in new_keys:
            raise TranslateError(
                f"译名冲突：{new_keys[key].old} 与 {old} 都会变成 {new_path}"
            )
        rename = NameRename(old=old, new=new_path)
        new_keys[key] = rename
        moved_old.add(_path_key(old))
        renames.append(rename)

    for rename in renames:
        key = _path_key(rename.new)
        if key in old_keys and key not in moved_old:
            raise TranslateError(
                f"译名会占用未移动的源名称：{rename.old} → {rename.new}；"
                "请调整译名或先移动同名目标"
            )
        if rename.new.exists() and key not in moved_old:
            raise TranslateError(
                f"目标名称已存在：{rename.new}；请调整译名或先移动同名目标"
            )
    return renames


def _rename_temp(path: Path) -> Path:
    for _ in range(10):
        candidate = path.with_name(f".vanitas-rename-{uuid.uuid4().hex}")
        if not candidate.exists():
            return candidate
    raise TranslateError(f"无法生成临时文件名：{path}")


def apply_name_renames(renames: list[NameRename]) -> None:
    """按深度从深到浅、每层两阶段重命名，支持交换与纯大小写变化。"""
    groups: dict[int, list[NameRename]] = {}
    for rename in renames:
        groups.setdefault(len(rename.old.parts), []).append(rename)

    for depth in sorted(groups, reverse=True):
        group = sorted(groups[depth], key=lambda item: str(item.old).lower())
        staged: list[tuple[NameRename, Path]] = []
        try:
            for rename in group:
                temp = _rename_temp(rename.old)
                rename.old.rename(temp)
                staged.append((rename, temp))
        except OSError as exc:
            for rename, temp in reversed(staged):
                try:
                    temp.rename(rename.old)
                except OSError:
                    pass
            raise TranslateError(
                f"重命名失败：{exc.filename or exc}；已尝试回滚当前层"
            ) from exc

        completed: list[tuple[NameRename, Path]] = []
        try:
            for rename, temp in staged:
                temp.rename(rename.new)
                completed.append((rename, temp))
        except OSError as exc:
            for rename, temp in reversed(completed):
                try:
                    rename.new.rename(temp)
                except OSError:
                    pass
            for rename, temp in reversed(staged):
                try:
                    if temp.exists():
                        temp.rename(rename.old)
                except OSError:
                    pass
            raise TranslateError(
                f"重命名失败：{exc.filename or exc}；已尝试回滚当前层"
            ) from exc


def plan_translation_outputs(
    sources: list[TranslationSource],
    settings: TranslateSettings,
    out_dir: Path | None,
    *,
    output_suffix: str | None = None,
) -> list[TranslationPlan]:
    """计算输出路径并在写盘前拦截冲突，避免静默覆盖。

    --out-dir 存在时，目录输入保留相对层级；直接指定的文件仍平铺在输出根。
    output_suffix 为 None 时使用 settings.output_suffix；传空串可用于“译名不加 .zh”。
    """
    suffix = settings.output_suffix if output_suffix is None else output_suffix
    plans: list[TranslationPlan] = []
    targets: dict[str, TranslationSource] = {}
    source_keys = {_path_key(item.path) for item in sources}

    for source in sources:
        target_dir = out_dir / source.relative_parent if out_dir is not None else None
        target = output_path(
            source.path,
            target_dir,
            suffix,
            source.translated_stem,
        )
        key = _path_key(target)
        previous = targets.get(key)
        if previous is not None:
            raise TranslateError(
                f"输出路径冲突：{previous.path} 与 {source.path} 都会写到 {target}；"
                "请调整 --out-dir 或输入目录"
            )
        if key in source_keys:
            raise TranslateError(
                f"输出会覆盖输入文件：{source.path} → {target}；"
                "请更换 --out-dir 或输出后缀"
            )
        targets[key] = source
        plans.append(
            TranslationPlan(source=source, out_dir=target_dir, target=target)
        )
    return plans


def translate_file(
    chat: Callable[[str], str],
    source: Path,
    settings,
    *,
    out_dir: Path | None = None,
    output_stem: str | None = None,
    output_suffix: str | None = None,
    on_segment: Callable[[int, int], None] | None = None,
    window_size: int = 1,
    max_window_chars: int = 250,
) -> TranslationOutcome:
    source = Path(source)
    if not source.is_file():
        raise TranslateError(f"文件不存在：{source}")
    suffix = source.suffix.lower()
    if suffix not in settings.source_suffixes:
        raise TranslateError(
            f"不支持的扩展名 {source.suffix!r}：{source}；"
            f"允许：{'、'.join(settings.source_suffixes)}"
        )
    text, has_bom, eol = _read_text(source)
    failures: list[TranslationFailure] = []
    segments = 0
    if suffix == ".srt":
        blocks = parse_srt(text, eol)
        segments = sum(1 for subtitle, _ in blocks if subtitle.index and subtitle.text.strip())
        translated = translate_srt(
            chat,
            text,
            chunk_chars=settings.chunk_chars,
            source=str(source),
            eol=eol,
            failures=failures,
            on_segment=on_segment,
            window_size=window_size,
            max_window_chars=max_window_chars,
        )
    else:
        segments = sum(1 for paragraph, _ in split_prose(text) if paragraph.strip())
        translated = translate_prose(
            chat,
            text,
            chunk_chars=settings.chunk_chars,
            source=str(source),
            failures=failures,
            on_segment=on_segment,
        )

    target = output_path(
        source,
        out_dir,
        settings.output_suffix if output_suffix is None else output_suffix,
        output_stem,
    )
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        data = translated.encode("utf-8")
        target.write_bytes(BOM + data if has_bom else data)
    except OSError as exc:
        raise TranslateError(f"写入失败：{target}；{exc.strerror or exc}") from exc

    outcome = TranslationOutcome(
        source=source, output=target, segments=segments, failures=failures
    )
    if failures:
        # 输出已写出（失败段保留原文），由调用方汇总并返回非零
        return outcome
    return outcome


def build_chat_for(
    config: AppConfig,
    model: ModelSpec,
    port: int,
    prompt_dir: Path,
    *,
    on_stat: Callable[[SegmentStat], None] | None = None,
) -> Callable[[str], str]:
    client = LlamaClient.for_model(config, model, port)
    profile = config.resolve_profile(model)
    settings = config.settings.translate
    return make_engine_chat(
        client,
        prompt_dir,
        prompts.system_prompt(prompt_dir),
        n_ctx=profile.ctx,
        context_window=settings.context_window,
        context_summary_chars=settings.context_summary_chars,
        min_output_tokens=settings.min_output_tokens,
        output_ratio=settings.output_ratio,
        prompt_safety_tokens=settings.prompt_safety_tokens,
        temperature=settings.temperature,
        max_split_depth=settings.max_split_depth,
        on_stat=on_stat,
    )
