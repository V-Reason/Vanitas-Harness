"""确定性翻译质量守卫：术语后处理、语域/人称、文档级重复检测与审计。

本模块刻意只依赖标准库与 ``vanitas.errors``，不 import ``vanitas.translate``：
翻译管线负责“怎么译”，这里只负责“译完怎么确定性地查/收敛”，避免两边
互相引用。所有函数都是纯函数或无模型调用的状态机，不产生请求、不写盘。

边界（P1/P2a/T1）：
- 重复检测只覆盖“不同原文却复用同一译文”的文档级复制，不判断语义对错；
- 语域检测只按固定词表计数，不推断人物关系；
- 术语表只替换 forbidden，不保证 preferred 一定出现，也不触发重译；
- P2a 的五类语义审计都是“高置信窄规则”：源文与译文同时命中固定模式
  才报，空源/空译直接放行；不做开放式语义判断，未命中仍交人工抽检；
- T1 的五个 code 同样只报不强制替换：short_duplicate 只认“短源文 + 长同译”，
  address_term 只认闭集称谓，invented_tool 只认闭集道具词，
  invented_name 只认提示词示例专名，under_translation 只认“长源短译”；
  是否重译由 ``translate.pipeline`` 的 ASMR 门控与预算决定。
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

from .errors import TranslateError

# --------------------------------------------------------------------------- #
# 数据驱动上下文替换规则
# --------------------------------------------------------------------------- #

# 规则表随仓库分发；调用方显式传 rules 时不读盘，缺文件时退化为空表。
DEFAULT_RULES_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "glossary" / "rules.tsv"
)
RULE_COLUMNS = 4
RULE_TYPES = frozenset(
    {
        "organ_guard",
        "context_sub",
        "leak_guard",
        "wasei_clean",
        "sfx_repair",
        "anatomy_guard",
    }
)


@dataclass(frozen=True)
class ContextRule:
    """一行上下文替换规则：源文触发正则、译文错误形态正则与收敛目标。

    ``rule_type`` 决定触发极性（见 ``apply_context_rules``），模式本身全部
    来自配置文件——代码不认识任何具体器官词、专名或固定台词。
    """

    rule_type: str
    source_re: re.Pattern[str]
    bad_re: re.Pattern[str]
    replacement: str


def _compile_rule(
    path: Path, line_no: int, rule_type: str, source: str, bad: str, repl: str
) -> ContextRule:
    try:
        source_re = re.compile(source)
        bad_re = re.compile(bad)
    except re.error as exc:
        raise TranslateError(f"{path}:{line_no}: 规则正则非法：{exc}") from exc
    return ContextRule(rule_type, source_re, bad_re, repl)


def load_context_rules(path: Path) -> tuple[ContextRule, ...]:
    """读取 4 列规则表；缺文件返回空表，坏行带 ``文件:行号`` 抛错。

    失败关闭是刻意的：一条写坏的正则如果被静默跳过，对应的确定性守卫就
    整体失效，而译文仍然"看起来正常"，比直接报错更难发现。
    """
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ()
    except OSError as exc:
        raise TranslateError(
            f"{path}: 读取上下文规则表失败：{exc.strerror or exc}"
        ) from exc

    rules: list[ContextRule] = []
    for line_no, line in enumerate(raw.splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = [part.strip() for part in line.split("\t")]
        if len(parts) != RULE_COLUMNS:
            raise TranslateError(
                f"{path}:{line_no}: 规则表必须为 4 列 "
                "rule_type<TAB>source_pattern<TAB>bad_pattern<TAB>replacement"
            )
        rule_type, source, bad, repl = parts
        if rule_type not in RULE_TYPES:
            raise TranslateError(
                f"{path}:{line_no}: 未知规则类型 {rule_type!r}；"
                f"允许：{'、'.join(sorted(RULE_TYPES))}"
            )
        if not source or not bad or not repl:
            raise TranslateError(f"{path}:{line_no}: 规则列不得为空")
        rules.append(_compile_rule(path, line_no, rule_type, source, bad, repl))
    return tuple(rules)


@lru_cache(maxsize=1)
def default_context_rules() -> tuple[ContextRule, ...]:
    """仓库自带规则表；缺失只告警一次并返回空表，坏表照常抛错。"""
    if not DEFAULT_RULES_PATH.is_file():
        warnings.warn(
            f"上下文规则表缺失：{DEFAULT_RULES_PATH}；"
            "器官/人称/示例专名收敛将不生效",
            RuntimeWarning,
            stacklevel=2,
        )
        return ()
    return load_context_rules(DEFAULT_RULES_PATH)

# --------------------------------------------------------------------------- #
# 术语表
# --------------------------------------------------------------------------- #

GLOSSARY_COLUMNS = 3


@dataclass(frozen=True)
class GlossaryEntry:
    """一行术语：源词、首选译法、禁止译法（可多个）。"""

    source: str
    preferred: str
    forbidden: tuple[str, ...]


class Glossary:
    """ASMR 术语表：只在源文命中时替换译文里的 forbidden 词。

    术语表不存在或为空都视为空表，翻译流程不需要额外分支；坏行必须带
    ``文件:行号`` 抛 ``TranslateError``，避免静默丢词。
    """

    def __init__(self, entries: tuple[GlossaryEntry, ...] = ()) -> None:
        self._entries = entries

    @property
    def entries(self) -> tuple[GlossaryEntry, ...]:
        return self._entries

    @classmethod
    def load(cls, path: Path) -> Glossary:
        path = Path(path)
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return cls()
        except OSError as exc:
            raise TranslateError(
                f"{path}: 读取术语表失败：{exc.strerror or exc}"
            ) from exc

        entries: list[GlossaryEntry] = []
        for line_no, line in enumerate(raw.splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != GLOSSARY_COLUMNS:
                raise TranslateError(
                    f"{path}:{line_no}: 术语表必须为 3 列 "
                    f"source<TAB>preferred<TAB>forbidden"
                )
            source, preferred, forbidden_text = (
                part.strip() for part in parts
            )
            if not source or not preferred:
                raise TranslateError(
                    f"{path}:{line_no}: source 与 preferred 不能为空"
                )
            forbidden = tuple(
                item.strip()
                for item in forbidden_text.split("|")
                if item.strip()
            )
            entries.append(
                GlossaryEntry(
                    source=source,
                    preferred=preferred,
                    forbidden=forbidden,
                )
            )
        return cls(tuple(entries))

    def apply(self, source: str, text: str) -> str:
        """把命中源词条目的 forbidden 译法收敛成 preferred。

        最长 forbidden 优先，避免“业余水平”先被“业余”截断；preferred
        没出现也不报错，更不触发重译——术语表只做确定性后处理。
        """
        if not source or not text:
            return text
        replacements: list[tuple[str, str]] = []
        for entry in self._entries:
            if not entry.source or entry.source not in source:
                continue
            for forbidden in entry.forbidden:
                if forbidden:
                    replacements.append((forbidden, entry.preferred))
        for forbidden, preferred in sorted(
            replacements, key=lambda item: len(item[0]), reverse=True
        ):
            text = text.replace(forbidden, preferred)
        return text

    def context(self, source: str) -> str:
        """可选的简短术语提示；无命中返回空串。"""
        if not source:
            return ""
        hints = [
            f"{entry.source}→{entry.preferred}"
            for entry in self._entries
            if entry.source and entry.source in source
        ]
        return "；".join(hints)


# --------------------------------------------------------------------------- #
# 语域与人称
# --------------------------------------------------------------------------- #

SERVICE_WORDS = (
    "いらっしゃいませ",
    "ございます",
    "お客様",
    "くださいませ",
    "施術",
    "予約",
)
INTIMATE_WORDS = (
    "先輩",
    "お姉ちゃん",
    "好き",
    "ねぇ",
    "膝枕",
    "添い寝",
    "気持ちいい",
)
# 明确性行为词：出现即可定性，但仍让位于服务语域。提示词的口径是“纯服务
# 场景可保留敬称，一旦有身体互动全篇用你”，所以只加这一条强证据通道，
# 不改“单句撒娇不足以定性”的阈值（见 detect_register）。
ACT_WORDS = (
    "射精",
    "勃起",
    "肉棒",
    "おっぱい",
    "おまんこ",
    "まんこ",
    "おちんちん",
    "ちんちん",
    "ちんぽ",
    "亀頭",
    "膣",
    "絶頂",
    "中出し",
    "フェラ",
    "手コキ",
    "パイズリ",
    "イッちゃう",
    "SEX",
    "えっち",
)
REGISTER_SERVICE = "service"
REGISTER_INTIMATE = "intimate"
REGISTER_NEUTRAL = "neutral"


@dataclass(frozen=True)
class TranslationProfile:
    """文档级语域与人称策略；pronoun 为 ``你`` 时才强制去敬称。"""

    register: str
    pronoun: str


def detect_register(sources) -> str:
    """按固定词表统计文档语域；证据不足或势均力敌时回落到 neutral。

    “命中的不同词”各计 1，同一词重复出现不累积；总分 < 2、同分或差值
    不足 1 都判 neutral，避免单句敬语/单句撒娇把整篇带偏。

    强证据通道：出现明确性行为词、且整篇没有任何服务词时判 intimate。
    这类文本的敬称口径由提示词硬性规定为“你”，不能因为“只有一个 distinct
    情感词”就落回 neutral（S2/V2 的 ``02``/``01_`` 正是这样漏掉的）；
    带 お客様/ご奉仕 的服务文件仍走原判定，敬称豁免不被抹平。
    """
    texts = [str(text) for text in sources]
    service = sum(
        1 for word in SERVICE_WORDS if any(word in text for text in texts)
    )
    intimate = sum(
        1 for word in INTIMATE_WORDS if any(word in text for text in texts)
    )
    act = sum(1 for word in ACT_WORDS if any(word in text for text in texts))
    if act and not service:
        return REGISTER_INTIMATE
    if service + intimate < 2:
        return REGISTER_NEUTRAL
    if service == intimate or abs(service - intimate) < 1:
        return REGISTER_NEUTRAL
    return REGISTER_SERVICE if service > intimate else REGISTER_INTIMATE


def apply_pronoun_policy(text: str, profile: TranslationProfile) -> str:
    """仅当 profile.pronoun 明确为 ``你`` 时把 ``您`` 统一成 ``你``。

    service/neutral 的 pronoun 是 ``auto``，完全不改：服务语境保留敬称
    是刻意的，不能由本函数“顺手”抹平。
    """
    if profile.pronoun != "你" or not text:
        return text
    return text.replace("您", "你")


# --------------------------------------------------------------------------- #
# 器官错位（源文只有耳部词 → 译文却出现下体词）
# --------------------------------------------------------------------------- #

# 规则与词表全部在 config/glossary/rules.tsv；这里只保留展示用标签。
# 新增第二类器官守卫时，须把标签改成通用形态（当前只登记了耳部守卫）。
ORGAN_GUARD_EVIDENCE = "源=耳"


def _rules_of_type(
    rule_type: str, rules: Sequence[ContextRule] | None
) -> tuple[ContextRule, ...]:
    chosen = default_context_rules() if rules is None else tuple(rules)
    return tuple(rule for rule in chosen if rule.rule_type == rule_type)


def detect_organ_swap(
    source: str,
    translation: str,
    rules: Sequence[ContextRule] | None = None,
) -> str | None:
    """源文命中器官守卫、译文却出现该守卫的坏词时返回证据串。"""
    text = str(translation or "")
    src = str(source or "")
    if not text or not src:
        return None
    for rule in _rules_of_type("organ_guard", rules):
        if not rule.source_re.search(src):
            continue
        match = rule.bad_re.search(text)
        if match is None:
            continue
        return f"{ORGAN_GUARD_EVIDENCE} 译={match.group(0)}"
    return None


def repair_organ_swap(
    source: str,
    translation: str,
    rules: Sequence[ContextRule] | None = None,
) -> str:
    """把器官守卫命中的坏词收敛成规则目标；守卫不成立时原样返回。"""
    if detect_organ_swap(source, translation, rules) is None:
        return translation
    return apply_context_rules(
        source, translation, rules, rule_types=("organ_guard",)
    )


# --------------------------------------------------------------------------- #
# 文档级重复检测
# --------------------------------------------------------------------------- #

COMPARE_STRIP_RE = re.compile(r"[\s。！？，、…—]+")
TRIGRAM_SIZE = 3
# 窄重复补充：源文明显不同（规范化后 ≤0.5）但译文共享长开头时，译文
# 整体相似度可能低于 0.80 而漏报；只认 ≥6 字共同前缀，不调低既有全局
# 阈值。边界取到 0.5 是因为 #80/#2 的源文规范化后正好 0.5。
TRANSLATION_PREFIX_MIN_CHARS = 6
PREFIX_DUPLICATE_SOURCE_RATIO = 0.5


@dataclass(frozen=True)
class DuplicateItem:
    """已接受的 cue 译文；cue 保留调用方类型（SRT 序号可为 str/int）。"""

    cue: int | str
    source: str
    translation: str


@dataclass(frozen=True)
class DuplicateMatch:
    """命中的历史条目；``cue`` 保留登记时的类型（SRT 序号可为 str/int）。"""

    cue: int | str
    source: str
    translation: str


def _cue_order(cue: int | str) -> int:
    """排序用的数值序号；非数字 cue 退化为 0，仅保证排序稳定。"""
    try:
        return int(cue)
    except (TypeError, ValueError):
        return 0


def _common_prefix_length(left: str, right: str) -> int:
    """两个规范化译文从头开始的共同字符数。"""
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


@dataclass(frozen=True)
class AuditIssue:
    """质量审计发现；hint 直接作为重译纠偏提示。"""

    code: str
    cue: int | str
    message: str
    hint: str


def normalize_compare(text: str) -> str:
    """去空白与常见断句符号；不做简繁转换，也不抹促音/长音。"""
    return COMPARE_STRIP_RE.sub("", text or "")


class DuplicateIndex:
    """文档级重复索引：精确哈希短路 + 3-gram 倒排召回。

    只索引“最终接受”的译文；短句可以 add 但 find 会跳过，避免语气词
    误报。候选按共享 trigram 数降序、同分按 cue 序号，只比较前
    ``max_candidates`` 个，长文档下查询成本可控。另有一条窄补充：
    源文规范化相似度 ≤0.5 且译文共同前缀 ≥6 字也算复制，用于长共同
    开头但整体相似度不足的场景，不改 0.75/0.80 全局阈值。
    """

    def __init__(
        self,
        *,
        source_ratio: float = 0.75,
        translation_ratio: float = 0.80,
        min_chars: int = 12,
        max_candidates: int = 32,
    ) -> None:
        self.source_ratio = float(source_ratio)
        self.translation_ratio = float(translation_ratio)
        self.min_chars = int(min_chars)
        self.max_candidates = int(max_candidates)
        self._items: list[DuplicateItem] = []
        self._orders: list[int] = []
        self._exact: dict[str, list[int]] = {}
        self._grams: dict[str, list[int]] = {}

    def __len__(self) -> int:
        return len(self._items)

    def reset(self) -> None:
        self._items.clear()
        self._orders.clear()
        self._exact.clear()
        self._grams.clear()

    def _trigrams(self, normalized: str) -> set[str]:
        if len(normalized) < TRIGRAM_SIZE:
            return set()
        return {
            normalized[start : start + TRIGRAM_SIZE]
            for start in range(len(normalized) - TRIGRAM_SIZE + 1)
        }

    def _same_source(self, left: str, right: str) -> bool:
        if left == right:
            return True
        if not left or not right:
            return False
        return SequenceMatcher(None, left, right).ratio() >= self.source_ratio

    def find_match(
        self,
        source: str,
        translation: str,
        cue: int | str | None = None,
        *,
        strict: bool = False,
    ) -> DuplicateMatch | None:
        """返回与当前译文重复的历史条目；无命中返回 None。

        源文相似（同一句话的合法重复）不算复制；译文完全相同、相似度
        达到阈值或长句互相包含才命中。阈值、候选排序与并列规则与旧
        ``find()`` 完全一致，返回值保留登记时的 cue 类型。

        ``strict=True`` 只认“规范化译文完全相同或相似度 ≥ translation_ratio”：
        长共同前缀与互相包含两条窄规则不参与，供 T1 short_duplicate 使用
        （其契约明确要求相同/≥0.85，不能被 <0.85 的窄规则带出命中）。
        """
        normalized = normalize_compare(translation)
        if len(normalized) < self.min_chars:
            return None
        source_norm = normalize_compare(source)

        # 精确命中短路：先检查完全相同的规范化译文，再走倒排召回。
        for item_index in self._exact.get(normalized, ()):
            item = self._items[item_index]
            if cue is not None and item.cue == cue:
                continue
            if self._same_source(source_norm, normalize_compare(item.source)):
                continue
            return DuplicateMatch(item.cue, item.source, item.translation)

        counts: dict[int, int] = {}
        for gram in self._trigrams(normalized):
            for item_index in self._grams.get(gram, ()):
                counts[item_index] = counts.get(item_index, 0) + 1
        ranked = sorted(
            counts,
            key=lambda item_index: (
                -counts[item_index],
                self._orders[item_index],
                item_index,
            ),
        )

        for item_index in ranked[: self.max_candidates]:
            item = self._items[item_index]
            if cue is not None and item.cue == cue:
                continue
            candidate_source = normalize_compare(item.source)
            if self._same_source(source_norm, candidate_source):
                continue
            previous = normalize_compare(item.translation)
            if len(previous) < self.min_chars:
                continue
            if previous == normalized:
                return DuplicateMatch(item.cue, item.source, item.translation)
            if (
                SequenceMatcher(None, normalized, previous).ratio()
                >= self.translation_ratio
            ):
                return DuplicateMatch(item.cue, item.source, item.translation)
            if strict:
                continue
            if previous in normalized or normalized in previous:
                return DuplicateMatch(item.cue, item.source, item.translation)
            if (
                _common_prefix_length(normalized, previous)
                >= TRANSLATION_PREFIX_MIN_CHARS
                and SequenceMatcher(None, source_norm, candidate_source).ratio()
                <= PREFIX_DUPLICATE_SOURCE_RATIO
            ):
                return DuplicateMatch(item.cue, item.source, item.translation)
        return None

    def find_similar(
        self,
        source: str,
        translation: str,
        cue: int | str | None = None,
    ) -> DuplicateMatch | None:
        """严格相似查找：只认译文相同/≥ translation_ratio，供短重复守卫用。"""
        return self.find_match(source, translation, cue, strict=True)

    def find(
        self,
        source: str,
        translation: str,
        cue: int | str | None = None,
    ) -> int | str | None:
        """返回与当前译文重复的历史 cue 序号；无命中返回 None。"""
        match = self.find_match(source, translation, cue)
        return match.cue if match is not None else None

    def add(
        self, cue: int | str, source: str, translation: str
    ) -> None:
        """登记最终接受的译文；失败/保留原文的 cue 不得进入索引。"""
        item = DuplicateItem(
            cue=cue, source=str(source), translation=str(translation)
        )
        item_index = len(self._items)
        self._items.append(item)
        self._orders.append(_cue_order(cue))
        normalized = normalize_compare(item.translation)
        self._exact.setdefault(normalized, []).append(item_index)
        for gram in self._trigrams(normalized):
            self._grams.setdefault(gram, []).append(item_index)


def _audit_snippet(text: str, limit: int = 24) -> str:
    """把提示里的原文压成单行片段；超过上限才截断并加省略号。"""
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "…"


def audit_duplicate(
    source: str,
    translation: str,
    cue: int | str,
    index: DuplicateIndex,
) -> list[AuditIssue]:
    """当前 cue 是否复用了历史译文；命中给出可执行的纠偏提示。"""
    match = index.find_match(source, translation, cue)
    if match is None:
        return []
    return [
        AuditIssue(
            code="duplicate_translation",
            cue=cue,
            message=f"译文与第 {match.cue} 条重复，但原文不同",
            hint=(
                f"与第 {match.cue} 条重复；"
                f"该条原文「{_audit_snippet(match.source)}」，"
                f"本条原文「{_audit_snippet(source)}」；"
                "禁止复用，只按本条原文重译"
            ),
        )
    ]


# --------------------------------------------------------------------------- #
# T1：ASMR 确定性守卫（只报不替换；重译/预算由 translate.pipeline 统一管）
# --------------------------------------------------------------------------- #

# 五个 code 是 T1 的唯一闭集：translate.pipeline 用它们做 ASMR 门控、每 cue 一次
# 重译与每文件预算上限；新 code 必须同时进 candidate_ok.semantic_codes
# （pipeline 用 ``*ASMR_T1_CODES`` 展开，加到这里即自动生效）。
ASMR_T1_CODES = frozenset(
    {
        "short_duplicate",
        "address_term",
        "invented_tool",
        "invented_name",
        "under_translation",
    }
)

# short_duplicate：只抓“短源文却得到同一个长译文”的跨 cue 复用。
# 上限取 11：真实语料里 9~11 字的短句仍可能整句复用；继续放大只会把
# 合法近义对判成重复。
SHORT_DUPLICATE_MAX_SOURCE_CHARS = 11

# 纯拟声/叠词的常见汉字。只用于 T1 的反例放行，不参与翻译或替换。
# 来源：提示词 sfx 分节列出的“沙沙/簌簌/咕噜/哧溜/嗯姆”等拟声词与
# 中文叠词习惯；命中即视为“两个不同短源文同译”可能合法，不重译。
ONOMATOPOEIA_CHARS = frozenset(
    "沙唰哗滴嗒咚砰啪滋咕噜哼哈呵嘻嘿噗嗤哧溜嗯姆啊呀哦噢喔呜喵汪咩"
    "嗷吼嗡嗖咻呼啦嘶嚓咔铛叮铃啵唧啾嘎呱嚯唔呃哇嘤啧咂嘘嚅喃哔哞"
    "咿咝锵锒吱咯喳呖呶呲咧啸"
)

# 全假名 + 长音不一定都是拟声：コーヒー/ケーキ 是正常名词。只有带促音
# 「ッ」或下列闭集里的呼吸/拟声长音才按拟声放行；不再用“任意全假名+ー”
# 的宽条件，避免 short_duplicate 对正常片假名词漏报。
KANA_ONO_EXEMPT = frozenset(
    {
        "じゅるー",
        "ちゅるー",
        "ぐちゅー",
        "くちゅー",
        "にゅるー",
        "ぬるー",
        "はぁ",
        "ふぅ",
        "ふー",
        "はー",
        "あーん",
        "んっ",
        "ちゅっ",
        "チュッ",
    }
)

# address_term：源文闭集 = 旦那様/ご主人様；只查译文是否误用“先生/老师”。
# 不列“老公/主人/老爷”封闭译名，也不做 glossary/强制替换（B8 §1.8/§2.1）。
ADDRESS_SOURCE_TERMS = ("旦那様", "ご主人様")
ADDRESS_BAD_TRANSLATIONS = ("先生", "老师")

# invented_tool：道具词闭集来自提示词领域词典与真实采耳道具。
# ホルホール 是采耳刷别名（源文有真实道具时不得误报为“臆造”）；耳ブラシ 是
# 具体耳刷复合词，不能放宽成通用「ブラシ」，否则 歯ブラシ/ヘアブラシ 会漏报。
TOOL_SOURCE_TERMS = (
    "梵天",
    "ぼんてん",
    "耳掻き",
    "耳かき",
    "綿棒",
    "タワシ",
    "ホルホール",
    "耳ブラシ",
)
TOOL_TRANSLATION_TERMS = ("梵天", "掏耳勺", "棉签", "采耳刷", "棕刷")

KANA_CHAR_RE = re.compile(r"[\u3040-\u309f\u30a0-\u30ff]")


def _is_kana_char(char: str) -> bool:
    return bool(KANA_CHAR_RE.match(char))


def is_onomatopoeia_or_reduplication(text: str) -> bool:
    """判断文本是否“纯拟声/叠词”，供 short_duplicate 放行反例。

    覆盖四类：
    1. 短单元重复覆盖整串（沙沙、はむはむ、コショコショ、沙沙沙沙）；
    2. 中文 ABB 叠词（软绵绵）；
    3. 全部由常见拟声汉字组成（沙、哗、咕噜……）；
    4. 全假名且带促音（ッ，如カリッ），或落在呼吸/长音拟声闭集
       （はぁ/ふぅ/じゅるー……）。普通长音片假名词（コーヒー/ケーキ）
       不在此列，避免 short_duplicate 漏报。
    普通名词（ミルク、好き、だめ）不会被判为拟声。
    """
    compact = normalize_compare(text)
    if len(compact) < 2:
        return True
    for unit_len in (1, 2, 3):
        if len(compact) < unit_len * 2:
            continue
        if len(compact) % unit_len:
            continue
        unit = compact[:unit_len]
        if unit * (len(compact) // unit_len) == compact:
            return True
    if len(compact) == 3 and compact[1] == compact[2]:
        return True
    if all(char in ONOMATOPOEIA_CHARS for char in compact):
        return True
    if all(_is_kana_char(char) or char == "ー" for char in compact):
        if "ッ" in compact or compact in KANA_ONO_EXEMPT:
            return True
    return False


def audit_short_duplicate(
    source: str,
    translation: str,
    cue: int | str,
    index: DuplicateIndex,
    *,
    max_source_chars: int = SHORT_DUPLICATE_MAX_SOURCE_CHARS,
) -> list[AuditIssue]:
    """两个短源文得到同一长译文时报告；调用方只做一次定点重译。

    触发（AND）：译文规范化 ≥8 字且相同/≥0.85（由 ``DuplicateIndex(
    min_chars=8, translation_ratio=0.85)`` + ``find_similar`` 保证）；较短
    源文 ≤``max_source_chars``（默认 11）；两侧源文不同（由索引的
    source_ratio 保证）；源文/译文均非纯拟声或叠词。
    """
    if not str(source).strip() or not str(translation).strip():
        return []
    source_norm = normalize_compare(source)
    if not source_norm:
        return []
    match = index.find_similar(source, translation, cue)
    if match is None:
        return []
    previous_norm = normalize_compare(match.source)
    if min(len(source_norm), len(previous_norm)) > max_source_chars:
        return []
    if (
        is_onomatopoeia_or_reduplication(source)
        or is_onomatopoeia_or_reduplication(match.source)
        or is_onomatopoeia_or_reduplication(translation)
    ):
        return []
    return [
        AuditIssue(
            code="short_duplicate",
            cue=cue,
            message=f"短句译文与第 {match.cue} 条重复，但原文不同",
            hint=(
                f"与第 {match.cue} 条短句译文重复；"
                f"该条原文「{_audit_snippet(match.source, 16)}」，"
                f"本条原文「{_audit_snippet(source, 16)}」；"
                "按本条原文重译，禁止复用同一短译"
            ),
        )
    ]


def audit_address_term(
    source: str, translation: str, cue: int | str
) -> list[AuditIssue]:
    """源含旦那様/ご主人様、译文却译成“先生/老师”时报告。

    只做 audit-only：不写 glossary、不强制替换、不写死单一译名；hint 也
    不列“老公/主人/老爷”封闭表。姓+様（南様/梅地様）与お客様不在闭集。
    """
    if not str(source).strip() or not str(translation).strip():
        return []
    if not any(term in source for term in ADDRESS_SOURCE_TERMS):
        return []
    if not any(term in translation for term in ADDRESS_BAD_TRANSLATIONS):
        return []
    return [
        AuditIssue(
            code="address_term",
            cue=cue,
            message="称谓被译成“先生/老师”",
            hint="称谓按人物关系译，不译“先生/老师”，不写死单一译名",
        )
    ]


def audit_invented_tool(
    source: str,
    translation: str,
    cue: int | str,
    nearby_sources: tuple[str, ...] = (),
) -> list[AuditIssue]:
    """译文出现道具词、但源文 ±3 cue 内都没有道具词时报告。

    ``nearby_sources`` 是同一文件内当前 cue 前后各 3 条的其他源文（不含
    当前条）；本函数会连同当前 ``source`` 一起查闭集。跨 cue 正确候选
    （如 ±3 内有「綿棒で」）必须 0 命中；KU100/ASMR/NG/BGM 不在表内。
    """
    if not str(source).strip() or not str(translation).strip():
        return []
    used = [term for term in TOOL_TRANSLATION_TERMS if term in translation]
    if not used:
        return []
    sources = (str(source), *(str(item) for item in nearby_sources))
    if any(term in text for text in sources for term in TOOL_SOURCE_TERMS):
        return []
    return [
        AuditIssue(
            code="invented_tool",
            cue=cue,
            message=f"译文出现源文没有的道具词“{used[0]}”",
            hint=(
                "原文及前后 ±3 条都没有道具词；"
                "不要凭场景补“掏耳勺/棉签/梵天”等道具，按原文直译"
            ),
        )
    ]


# 提示词示例专名会渗漏到译文：模型把示例当成“本作已知人名”，在原文根本
# 没有名字的拟声/语气位上补出来。规则（含一切错误写法）登记在
# config/glossary/rules.tsv 的 leak_guard 行；代码只负责“全篇都没有才降级”。

# 片假名折叠成平假名：源文里的片假名写法与规则表登记的平假名写法是同一个
# 人名，直接比对会漏配。
_KATAKANA_TO_HIRAGANA = {
    code: code - 0x60 for code in range(0x30A1, 0x30F7)
}


def _fold_kana(text: str) -> str:
    return str(text).translate(_KATAKANA_TO_HIRAGANA)


# 只剩标点/符号（没有汉字、假名、字母、数字）时视为“占位”，可以整条降级成省略号。
_PUNCT_ONLY_RE = re.compile(r"^[^\u4e00-\u9fff\u3040-\u30ffA-Za-z0-9]*$")


def apply_context_rules(
    source: str,
    translation: str,
    rules: Sequence[ContextRule] | None = None,
    *,
    document_text: str = "",
    rule_types: Collection[str] | None = None,
) -> str:
    """按规则类型执行确定性上下文修正；纯函数、0 Token。

    触发极性：
    - ``organ_guard`` / ``context_sub`` / ``wasei_clean`` / ``sfx_repair`` /
      ``anatomy_guard``：源文命中 ``source_re`` 才替换；
    - ``leak_guard``：源文与全篇**都没有**该写法时才处理，且只有“删完只剩
      标点”才整条降级为 ``replacement``——名字嵌在正常句子里的原样返回，
      交给 ``audit_invented_name`` 做一次定点重译，避免把句子删残。
    """
    if not translation:
        return translation
    chosen = default_context_rules() if rules is None else tuple(rules)
    if not chosen:
        return translation
    src = str(source or "")
    text = str(translation)
    leak_text = text
    leak_replacement = ""
    leak_removed = False
    for rule in chosen:
        if rule_types is not None and rule.rule_type not in rule_types:
            continue
        if rule.rule_type == "leak_guard":
            if not src.strip():
                continue
            haystack = _fold_kana(f"{src}\n{document_text}")
            if rule.source_re.search(haystack):
                continue
            if rule.bad_re.search(leak_text):
                leak_text = rule.bad_re.sub("", leak_text)
                leak_removed = True
                leak_replacement = rule.replacement
            continue
        if not src:
            continue
        if rule.source_re.search(src):
            text = rule.bad_re.sub(rule.replacement, text)
    if leak_removed and _PUNCT_ONLY_RE.match(leak_text):
        return leak_replacement
    return text


def strip_example_names(
    translation: str,
    source: str,
    document_text: str = "",
    rules: Sequence[ContextRule] | None = None,
) -> str:
    """把“整条只剩提示词示例专名”的译文确定性降级成省略号。

    模型会把提示词示例当成“本作已知人名”，在拟声/气息位补出来；既然整篇
    源文根本没有这个名字，删掉它是确定性的，不需要再问模型。只在“删完只剩
    标点”时生效——那正是拟声位被填空的形态。名字嵌在正常句子里的原样返回，
    交给 ``audit_invented_name`` 做一次定点重译，避免把句子删残。
    """
    if not translation or not str(source or "").strip():
        return translation
    return apply_context_rules(
        source,
        translation,
        rules,
        document_text=document_text,
        rule_types=("leak_guard",),
    )


def audit_invented_name(
    source: str,
    translation: str,
    cue: int | str,
    document_text: str = "",
    rules: Sequence[ContextRule] | None = None,
) -> list[AuditIssue]:
    """译文出现提示词示例专名、而整篇源文没有该名字的任何写法时报告。

    ``document_text`` 是同一文件全部原始源文的拼接（不含译文）。空串时退化为
    只查本条源文，单句路径没有文档表时仍能拦住最明显的渗漏。比对前先把片假名
    折成平假名，同一人名的两种写法视为一致。
    """
    if not str(source).strip() or not str(translation).strip():
        return []
    haystack = _fold_kana(f"{source}\n{document_text}")
    for rule in _rules_of_type("leak_guard", rules):
        if rule.source_re.search(haystack):
            continue
        match = rule.bad_re.search(str(translation))
        if match is None:
            continue
        used = match.group(0)
        return [
            AuditIssue(
                code="invented_name",
                cue=cue,
                message=f"译文出现源文没有的提示词示例专名“{used}”",
                hint=(
                    "原文没有出现任何人物名；"
                    "拟声、语气、指示位都不要填人名，只按本条原文重译"
                ),
            )
        ]
    return []


def enforce_gender_anatomy_invariant(
    source: str,
    translation: str,
    rules: Sequence[ContextRule] | None = None,
) -> str:
    """女声对男听众音频的生理互斥断路器；判定词表全部在规则表。

    只在源文没有任何女性器官词、译文却出现女性器官词时生效；源文有该词
    时原样返回，避免改写合法的女性视角内容。具体替换档位（肉棒/前端/这里）
    由 ``anatomy_guard`` 数据行决定，代码不感知领域词。
    """
    if not str(source or "").strip():
        return translation
    return apply_context_rules(
        source, translation, rules, rule_types=("anatomy_guard",)
    )


def _clean_after_removal(text: str) -> str:
    """删除专名/道具词后的标点与空白收敛；只做确定性清理。"""
    cleaned = re.sub(r"[，、；：]{2,}", "，", text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip(" ，、；：")


def _strip_leaked_names(
    translation: str,
    source: str,
    document_text: str = "",
    rules: Sequence[ContextRule] | None = None,
) -> str:
    """强制删除“源文与全篇都没有”的 ``leak_guard`` 专名（含句中形态）。

    与 ``strip_example_names`` 的分工：后者只在“整条只剩专名”时降级，避免
    删残正常句子；本函数是 T1 重译被拒后的最后兜底，允许删句中专名，并在
    删除后收敛标点；结果为空/纯标点时退化为省略号。
    """
    text = str(translation or "")
    if not text or not str(source or "").strip():
        return text
    haystack = _fold_kana(f"{source}\n{document_text}")
    result = text
    for rule in _rules_of_type("leak_guard", rules):
        if rule.source_re.search(haystack):
            continue
        result = rule.bad_re.sub("", result)
    result = _clean_after_removal(result)
    if not result or _PUNCT_ONLY_RE.match(result):
        return "……"
    return result


def salvage_t1_rejection(
    current_text: str,
    source_text: str,
    t1_codes: set[str] | frozenset[str],
    document_text: str = "",
    rules: Sequence[ContextRule] | None = None,
) -> str:
    """T1 重译失败/被拒后的确定性兜底；绝不让坏译原样落盘。

    只处理三类可确定性消解的 T1 code：示例专名渗漏、臆造道具、短句复制。
    称谓与漏译不在此处强行改写（语义风险高，保留原重译路径）。返回值保证
    非空：清理后为空或纯标点时统一退化为省略号。
    """
    text = str(current_text or "")
    if not text:
        return "……"
    changed = False
    if "invented_name" in t1_codes:
        stripped = strip_example_names(
            text, source_text, document_text, rules
        )
        if stripped != text:
            text = stripped
            changed = True
        forced = _strip_leaked_names(
            text, source_text, document_text, rules
        )
        if forced != text:
            text = forced
            changed = True
    if "invented_tool" in t1_codes:
        for term in TOOL_TRANSLATION_TERMS:
            pattern = rf"{re.escape(term)}(?=(?:老师|小姐|先生|酱))"
            replaced = re.sub(pattern, "", text)
            if replaced != text:
                text = replaced
                changed = True
            replaced = text.replace(term, "它")
            if replaced != text:
                text = replaced
                changed = True
        text = _clean_after_removal(text)
        if not text or _PUNCT_ONLY_RE.match(text) or text == "它":
            text = "……"
            changed = True
    if (
        "short_duplicate" in t1_codes
        and not changed
        and len(normalize_compare(text)) > 8
        and len(normalize_compare(source_text)) <= 6
    ):
        text = "……"
        changed = True
    text = text.strip()
    if not text or _PUNCT_ONLY_RE.match(text):
        return "……"
    return text


# 漏译阈值：源文归一化 ≥10 字却只译出 ≤2 字——几乎整句丢失。
# **刻意不做“译文/源文比例”判定**：日译中的合法压缩与真漏译在比例上不可分，
# 低阈值会在真实语料里误报；比例类漏译只能靠人工/语义复审，宁漏不错。
UNDER_TRANSLATION_MIN_SOURCE = 10
UNDER_TRANSLATION_MAX_TRANSLATION = 2


def audit_under_translation(
    source: str, translation: str, cue: int | str
) -> list[AuditIssue]:
    """源文够长却几乎没译出来时报告（吞句、整句丢失）。"""
    if not str(source).strip() or not str(translation).strip():
        return []
    if is_onomatopoeia_or_reduplication(source):
        return []
    source_length = len(normalize_compare(source))
    if source_length < UNDER_TRANSLATION_MIN_SOURCE:
        return []
    translation_length = len(normalize_compare(translation))
    if translation_length > UNDER_TRANSLATION_MAX_TRANSLATION:
        return []
    return [
        AuditIssue(
            code="under_translation",
            cue=cue,
            message=(
                f"源文 {source_length} 字只译出 {translation_length} 字，疑似漏译"
            ),
            hint="按本条原文完整译出，不要只留一个叹词，也不要把后半句并到上一条",
        )
    ]


CUSTOMER_REVERSED_RE = re.compile(
    r"(?:我们|我)[^。！？]{0,10}(?:触碰|摸|碰|接触)[^。！？]{0,8}(?:客人|您)"
)
CUSTOMER_OBJECT_RE = re.compile(
    r"(?:触碰|接触|摸|碰)[^。！？]{0,8}(?:客人|您)"
)
QUOTE_PRIVATE_BAD_RE = re.compile(r"让我叫|我的名字|让你叫|叫我叫")
QUOTE_PRIVATE_THIRD_RE = re.compile(r"叫她|叫他|称她|称他")
FIRST_PERSON_SOURCE_RE = re.compile(r"私|僕|俺|わたし|あたし|自分")
FIRST_PERSON_CUE_RE = re.compile(r"名前|呼ぶ|呼んで|あげて")
FIRST_PERSON_TRANSLATION_RE = re.compile(r"我的名字|让我|叫我|用我")
TAG_QUESTION_SOURCE_RE = re.compile(
    r"ね|でしょ|でしょう|だろう|確認|だろ|じゃん|っけ|っしょ"
)
TAG_QUESTION_TRANSLATIONS = ("对吧", "是吧", "对不对", "好不好")
# 否定疑问「〜ていませんか」问的是“你不是…吗/你不…吗”，中文若写成
# 肯定句会把语义方向反过来。只认持续体否定疑问，避免误伤「〜ませんか」
# 这种邀请句（如「一緒に行きませんか」= 要不要一起去）。
NEGATIVE_QUESTION_SOURCE_RE = re.compile(r"ていませんか|ではありませんか")
NEGATIVE_TRANSLATION_RE = re.compile(r"不|没|没有|难道|不是|未")


def audit_customer_direction(
    source: str, translation: str, cue: int | str
) -> list[AuditIssue]:
    """源文是“禁止客人触碰我们”，译文却写成“我们触碰客人”时报告。"""
    if not source.strip() or not translation.strip():
        return []
    if not (
        any(word in source for word in ("お客様", "客"))
        and any(word in source for word in ("私たち", "私", "僕", "俺"))
        and "触れ" in source
        and any(word in source for word in ("禁止", "だめ", "いけない", "ない"))
    ):
        return []

    reversed_direction = bool(CUSTOMER_REVERSED_RE.search(translation))
    if not reversed_direction:
        # 无主语“禁止触碰客人”也要覆盖；动词前 8 字内已出现客人/您时，
        # 说明是“客人不能触碰我们”这类正确方向，不报。
        for match in CUSTOMER_OBJECT_RE.finditer(translation):
            leading = translation[max(0, match.start() - 8) : match.start()]
            if not any(word in leading for word in ("客人", "您")):
                reversed_direction = True
                break
    if not reversed_direction:
        return []
    return [
        AuditIssue(
            code="customer_direction",
            cue=cue,
            message="译文把禁止方向译反了（客人→我们）",
            hint=(
                "原文是客人→我们：禁止客人触碰我们，"
                "不要译成我们触碰客人"
            ),
        )
    ]


def audit_quote_private(
    source: str, translation: str, cue: int | str
) -> list[AuditIssue]:
    """``私のことは…って`` 是引用他人，不能被译成“让我叫/我的名字/让你叫/叫我叫”。"""
    if not source.strip() or not translation.strip():
        return []
    if "私のことは" not in source:
        return []
    if not source.rstrip("。！？!?… ").endswith("って"):
        return []
    if QUOTE_PRIVATE_BAD_RE.search(translation):
        return [
            AuditIssue(
                code="quote_private",
                cue=cue,
                message="引用他人被译成了第一人称",
                hint=(
                    "这是引用他人（…って）；不要译成“让我叫/我的名字/让你叫/叫我叫”，"
                    "保持第三人称转述；单独“叫我”不算坏译"
                ),
            )
        ]
    # “叫她/称他”是明确的第三人称转述；显式放行，避免未来把目标人称
    # 误并进坏模式。单独“叫我”不报，因为它在语境中可能正确。
    if QUOTE_PRIVATE_THIRD_RE.search(translation):
        return []
    return []


def repair_quote_private(source: str, text: str) -> str:
    """引用句里的叠词口误 ``叫我叫`` 收敛成 ``叫我``。

    只用于源文是 ``私のことは…って`` 的引用场景（与 ``audit_quote_private``
    同一触发条件），且译文确实含 ``叫我叫`` 时才替换。普通“叫我”在语境中
    可能正确，不在此处改动；条件不满足原样返回。
    """
    if not text or "叫我叫" not in text:
        return text
    if "私のことは" not in source:
        return text
    if not source.rstrip("。！？!?… ").endswith("って"):
        return text
    return text.replace("叫我叫", "叫我")


def audit_negative_question(
    source: str, translation: str, cue: int | str
) -> list[AuditIssue]:
    """源文是「〜ていませんか」否定疑问，译文却写成肯定句时报告。

    例：「私のこと、覚えていませんか?」= 你不记得我了吗；译成“你还记得我
    吗”会把方向反转。只匹配持续体否定疑问，不碰「〜ませんか」邀请句。
    """
    if not source.strip() or not translation.strip():
        return []
    if not NEGATIVE_QUESTION_SOURCE_RE.search(source):
        return []
    if NEGATIVE_TRANSLATION_RE.search(translation):
        return []
    return [
        AuditIssue(
            code="negative_question",
            cue=cue,
            message="否定疑问被译成了肯定句",
            hint=(
                "原文是「〜ていませんか」否定疑问；"
                "译文要保留否定（如“你不记得我了吗”），不要译成肯定"
            ),
        )
    ]


def audit_first_person(
    source: str, translation: str, cue: int | str
) -> list[AuditIssue]:
    """源文没有第一人称，译文却用“我的名字/让我/叫我/用我”。"""
    if not source.strip() or not translation.strip():
        return []
    if FIRST_PERSON_SOURCE_RE.search(source):
        return []
    if not FIRST_PERSON_CUE_RE.search(source):
        return []
    if not FIRST_PERSON_TRANSLATION_RE.search(translation):
        return []
    return [
        AuditIssue(
            code="first_person",
            cue=cue,
            message="原文没有第一人称，译文却用了“我”",
            hint=(
                "原文不是第一人称（没有私/僕/俺/わたし/あたし/自分）；"
                "不要补成我的名字/让我/叫我/用我"
            ),
        )
    ]


def repair_first_person_name(
    source: str,
    text: str,
    rules: Sequence[ContextRule] | None = None,
) -> str:
    """源文命中 ``context_sub`` 时收敛第一人称归属；不命中时原样返回。

    触发短语与替换对全部在 ``config/glossary/rules.tsv``；“源文已出现第一
    人称就不改”的守卫也写在规则的 ``source_pattern`` 负向前瞻里。这样固定
    源句不再出现在代码中，换语料只改数据，不需要改 Python。
    """
    if not text:
        return text
    return apply_context_rules(
        source, text, rules, rule_types=("context_sub",)
    )


def audit_tag_question(
    source: str, translation: str, cue: int | str
) -> list[AuditIssue]:
    """源文没有标签问，译文首/尾却补“对吧/是吧”时报告。"""
    if not source.strip() or not translation.strip():
        return []
    if TAG_QUESTION_SOURCE_RE.search(source):
        return []
    stripped = translation.strip().rstrip("。！？!?… ")
    if not stripped:
        return []
    if not (
        stripped.startswith(TAG_QUESTION_TRANSLATIONS)
        or stripped.endswith(TAG_QUESTION_TRANSLATIONS)
    ):
        return []
    return [
        AuditIssue(
            code="tag_question",
            cue=cue,
            message="原文不是标签问，译文却补了“对吧/是吧”",
            hint=(
                "原文没有ね/でしょ/だろう等标签问；"
                "不要补“对吧/是吧/对不对/好不好”，按陈述句译"
            ),
        )
    ]


def combine_hints(issues, limit: int = 2) -> str:
    """按出现顺序去重回译提示，最多拼接 ``limit`` 条。"""
    if limit <= 0:
        return ""
    hints: list[str] = []
    for issue in issues:
        hint = (getattr(issue, "hint", "") or "").strip()
        if not hint or hint in hints:
            continue
        hints.append(hint)
        if len(hints) >= limit:
            break
    return "；".join(hints)


def audit_translation(
    source: str,
    translation: str,
    cue: int | str,
    *,
    index: DuplicateIndex,
    profile: TranslationProfile,
    short_index: DuplicateIndex | None = None,
    window_index: DuplicateIndex | None = None,
    nearby_sources: tuple[str, ...] = (),
    asmr_guard: bool = False,
    document_text: str = "",
) -> list[AuditIssue]:
    """按固定优先级汇总文档级、P2a 与 T1 审计；不调用模型。

    固定顺序：duplicate → short_duplicate → customer_direction →
    quote_private → negative_question → first_person → tag_question →
    address_term → invented_name → invented_tool → under_translation →
    honorific。T1 五个 code 只在 ``asmr_guard=True`` 时参与；
    ``short_duplicate`` 需要调用方提供每文件独立的 ``short_index``
    （min_chars=8 / translation_ratio=0.85），合流批次还可传 ``window_index``
    覆盖当前窗口内已接受 cue；两者最多报一条，避免同 cue 重复计费。
    ``document_text`` 是该文件全部原始源文的拼接，供 invented_name 判断
    “整篇到底有没有这个名字”；不传时退化为只查本条源文。
    """
    issues = [*audit_duplicate(source, translation, cue, index)]
    if asmr_guard:
        for candidate_index in (short_index, window_index):
            if candidate_index is None:
                continue
            short_issues = audit_short_duplicate(
                source, translation, cue, candidate_index
            )
            if short_issues:
                issues.extend(short_issues)
                break
    issues.extend(audit_customer_direction(source, translation, cue))
    issues.extend(audit_quote_private(source, translation, cue))
    issues.extend(audit_negative_question(source, translation, cue))
    issues.extend(audit_first_person(source, translation, cue))
    issues.extend(audit_tag_question(source, translation, cue))
    if asmr_guard:
        issues.extend(audit_address_term(source, translation, cue))
        issues.extend(
            audit_invented_name(source, translation, cue, document_text)
        )
        issues.extend(
            audit_invented_tool(
                source, translation, cue, nearby_sources=nearby_sources
            )
        )
        issues.extend(audit_under_translation(source, translation, cue))
    issues.extend(audit_honorific(translation, profile, cue))
    return issues


def audit_honorific(
    text: str, profile: TranslationProfile, cue: int | str
) -> list[AuditIssue]:
    """亲密语域仍出现敬称“您”时提示统一用“你”。"""
    if profile.register != REGISTER_INTIMATE or "您" not in text:
        return []
    return [
        AuditIssue(
            code="honorific",
            cue=cue,
            message="亲密语域出现敬称“您”",
            hint="本作亲密关系，统一用你",
        )
    ]
