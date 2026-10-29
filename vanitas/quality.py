"""确定性翻译质量守卫：术语后处理、语域/人称、文档级重复检测与审计。

本模块刻意只依赖标准库与 ``vanitas.errors``，不 import ``translate.py``：
翻译管线负责“怎么译”，这里只负责“译完怎么确定性地查/收敛”，避免两边
互相引用。所有函数都是纯函数或无模型调用的状态机，不产生请求、不写盘。

边界（P1）：
- 重复检测只覆盖“不同原文却复用同一译文”的文档级复制，不判断语义对错；
- 语域检测只按固定词表计数，不推断人物关系；
- 术语表只替换 forbidden，不保证 preferred 一定出现，也不触发重译。
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


@dataclass(frozen=True)
class DuplicateItem:
    """已接受的 cue 译文；cue 保留调用方类型（SRT 序号可为 str/int）。"""

    cue: int | str
    source: str
    translation: str


def _cue_order(cue: int | str) -> int:
    """排序用的数值序号；非数字 cue 退化为 0，仅保证排序稳定。"""
    try:
        return int(cue)
    except (TypeError, ValueError):
        return 0


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
    ``max_candidates`` 个，长文档下查询成本可控。
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

    def find(
        self,
        source: str,
        translation: str,
        cue: int | str | None = None,
    ) -> int | str | None:
        """返回与当前译文重复的历史 cue 序号；无命中返回 None。

        源文相似（同一句话的合法重复）不算复制；译文完全相同、相似度
        达到阈值或长句互相包含才命中。返回值保留登记时的 cue 类型。
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
            return item.cue

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
            if self._same_source(source_norm, normalize_compare(item.source)):
                continue
            previous = normalize_compare(item.translation)
            if len(previous) < self.min_chars:
                continue
            if previous == normalized:
                return item.cue
            if (
                SequenceMatcher(None, normalized, previous).ratio()
                >= self.translation_ratio
            ):
                return item.cue
            if previous in normalized or normalized in previous:
                return item.cue
        return None

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


def audit_duplicate(
    source: str,
    translation: str,
    cue: int | str,
    index: DuplicateIndex,
) -> list[AuditIssue]:
    """当前 cue 是否复用了历史译文；命中给出可执行的纠偏提示。"""
    previous = index.find(source, translation, cue)
    if previous is None:
        return []
    return [
        AuditIssue(
            code="duplicate_translation",
            cue=cue,
            message=f"译文与第 {previous} 条重复，但原文不同",
            hint=f"当前译文与第 {previous} 条重复，但原文不同；禁止复用",
        )
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
