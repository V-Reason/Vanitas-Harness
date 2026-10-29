"""确定性翻译质量守卫：术语后处理、语域/人称、文档级重复检测与审计。

本模块刻意只依赖标准库与 ``vanitas.errors``，不 import ``translate.py``：
翻译管线负责“怎么译”，这里只负责“译完怎么确定性地查/收敛”，避免两边
互相引用。所有函数都是纯函数或无模型调用的状态机，不产生请求、不写盘。

边界（P1/P2a）：
- 重复检测只覆盖“不同原文却复用同一译文”的文档级复制，不判断语义对错；
- 语域检测只按固定词表计数，不推断人物关系；
- 术语表只替换 forbidden，不保证 preferred 一定出现，也不触发重译；
- P2a 的四类语义审计都是“高置信窄规则”：源文与译文同时命中固定模式
  才报，空源/空译直接放行；不做开放式语义判断，未命中仍交人工抽检。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from .errors import TranslateError

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
    """
    texts = [str(text) for text in sources]
    service = sum(
        1 for word in SERVICE_WORDS if any(word in text for text in texts)
    )
    intimate = sum(
        1 for word in INTIMATE_WORDS if any(word in text for text in texts)
    )
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
    ) -> DuplicateMatch | None:
        """返回与当前译文重复的历史条目；无命中返回 None。

        源文相似（同一句话的合法重复）不算复制；译文完全相同、相似度
        达到阈值或长句互相包含才命中。阈值、候选排序与并列规则与旧
        ``find()`` 完全一致，返回值保留登记时的 cue 类型。
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


# 主客方向：只处理源文明确出现“客人/我们/触碰/禁止”四个条件的句子。
CUSTOMER_REVERSED_RE = re.compile(
    r"(?:我们|我)[^。！？]{0,10}(?:触碰|摸|碰|接触)[^。！？]{0,8}(?:客人|您)"
)
CUSTOMER_OBJECT_RE = re.compile(
    r"(?:触碰|接触|摸|碰)[^。！？]{0,8}(?:客人|您)"
)
QUOTE_PRIVATE_BAD_RE = re.compile(r"让我叫|我的名字|让你叫")
QUOTE_PRIVATE_THIRD_RE = re.compile(r"叫她|叫他|称她|称他")
FIRST_PERSON_SOURCE_RE = re.compile(r"私|僕|俺|わたし|あたし|自分")
FIRST_PERSON_CUE_RE = re.compile(r"名前|呼ぶ|呼んで|あげて")
FIRST_PERSON_TRANSLATION_RE = re.compile(r"我的名字|让我|叫我|用我")
FIRST_PERSON_NAME_SOURCE = "下の名前で呼んであげて"
FIRST_PERSON_NAME_REPLACEMENTS = (
    ("我的名字", "她的名字"),
    ("让我", "让她"),
    ("叫我", "叫她"),
)
TAG_QUESTION_SOURCE_RE = re.compile(
    r"ね|でしょ|でしょう|だろう|確認|だろ|じゃん|っけ|っしょ"
)
TAG_QUESTION_TRANSLATIONS = ("对吧", "是吧", "对不对", "好不好")


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
    """``私のことは…って`` 是引用他人，不能被译成“让我叫/我的名字/让你叫”。"""
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
                    "这是引用他人（…って）；不要译成“让我叫/我的名字/让你叫”，"
                    "保持第三人称转述；单独“叫我”不算坏译"
                ),
            )
        ]
    # “叫她/称他”是明确的第三人称转述；显式放行，避免未来把目标人称
    # 误并进坏模式。单独“叫我”不报，因为它在语境中可能正确。
    if QUOTE_PRIVATE_THIRD_RE.search(translation):
        return []
    return []


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


def repair_first_person_name(source: str, text: str) -> str:
    """固定源短语下的第一人称收敛：把“我的名字/让我/叫我”改成第三人称。

    只用于源文明确要求“用下名呼称”（``下の名前で呼んであげて``）且源文
    没有第一人称的场景。模型会在这里固执地输出“请用我的名字好好叫我哦”，
    重译提示不跟；确定性替换比继续加 hint 更可靠。不命中固定源短语时原样
    返回，避免误改普通自我介绍。
    """
    if not text or FIRST_PERSON_NAME_SOURCE not in source:
        return text
    if FIRST_PERSON_SOURCE_RE.search(source):
        return text
    for old, new in FIRST_PERSON_NAME_REPLACEMENTS:
        text = text.replace(old, new)
    return text


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
) -> list[AuditIssue]:
    """按固定优先级汇总文档级与 P2a 语义审计；不调用模型。"""
    return [
        *audit_duplicate(source, translation, cue, index),
        *audit_customer_direction(source, translation, cue),
        *audit_quote_private(source, translation, cue),
        *audit_first_person(source, translation, cue),
        *audit_tag_question(source, translation, cue),
        *audit_honorific(translation, profile, cue),
    ]


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
