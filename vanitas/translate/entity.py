"""文档级实体倒排索引（无模型调用、无第三方依赖）。

职责边界：只回答两个问题——“本窗口提到了哪些已登记实体”、
“已锁定译名/前文示例是什么”；不改写整句、不做通用文本清洗。
锁定判定所需的写回硬校验从 ``cleaner`` 取，保证与最终写回同一条规则。
"""

from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher

from .cleaner import _clip, validate_cue_translation

KATAKANA_TERM_RE = re.compile(r"[\u30a1-\u30f6\u30fc]{2,10}")
HIRAGANA_TERM_RE = re.compile(r"[\u3041-\u3096]{3,10}")
KANJI_KATAKANA_NAME_CHAR = r"[\u4e00-\u9fa5\u30a1-\u30f6\u30fc]"
HIRAGANA_NAME_CHAR = r"[\u3041-\u3096]"
NAME_CHAR_CLASS = rf"(?:{KANJI_KATAKANA_NAME_CHAR}|{HIRAGANA_NAME_CHAR})"
NAME_BOUNDARY = r"(?:^|(?<=[、。！？\s「『（(]))"
# 汉字/片假名名字：不允许混入平假名，避免把“他のお客”“綿棒よりも”整段吞进来。
# 不加行首/标点边界，因为“描いてくださった梅地様”这类名字前可直接接动词。
KANJI_KATAKANA_HONORIFIC_RE = re.compile(
    rf"({KANJI_KATAKANA_NAME_CHAR}{{1,8}})"
    r"(?:ちゃん|さん|くん|様|さま|先生|先輩)"
)
# 平假名名字单独处理：只允许纯平假名 base，避免“いの皆”这类混合词。
HIRAGANA_HONORIFIC_RE = re.compile(
    rf"{NAME_BOUNDARY}({HIRAGANA_NAME_CHAR}{{2,5}})"
    r"(?:ちゃん|さん|くん|様|さま|先生|先輩)"
)
# 片假名昵称单独处理；平假名不加“ちん”，避免把普通词/拟声词误当名字。
KATAKANA_NICKNAME_RE = re.compile(r"([\u30a1-\u30f6\u30fc]{2,8})ちん")
EXPLICIT_NAME_RE = re.compile(rf"({NAME_CHAR_CLASS}{{2,10}})(?=という名|役)")
KANJI_CHUNK_RE = re.compile(r"[\u4e00-\u9fa5]{2,4}")
ENTITY_HONORIFIC_SUFFIXES = (
    "ちゃん", "さん", "くん", "様", "さま", "先生", "先輩", "ちん"
)

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
        # 本轮真实语料中暴露的普通名词/拟声词；不得因为它们重复出现就锁成专名。
        "トラック", "セックス", "バイバイ", "バイバーイ", "キャ", "チン",
        "コショコショ", "ピコ", "カリ", "ヌルヌル", "オノマトペ", "ループ",
        "ダメ", "エッチ", "キッズ", "イラスト", "ジェル", "マジ", "結構",
        "恋人", "耳掃除", "美容師", "変態", "サイト", "オススメ",
        "アイドル", "突然", "お姉", "客", "聞", "マゾ", "アンスリウム",
    }
)

ENTITY_MAX_CLUSTERS = 12
ENTITY_MAX_VARIANTS = 8
ENTITY_MAX_EXAMPLES = 2
ENTITY_EXAMPLE_CHARS = 80
ENTITY_CONTEXT_CHARS = 1200


def _kana_key(text: str) -> str:
    """平假名→片假名键，仅用于把同一姓名的两种写法对齐。"""
    return "".join(
        chr(ord(char) + 0x60) if "\u3041" <= char <= "\u3096" else char
        for char in text
    )


def _explicit_alias(
    text: str, explicit_keys: dict[str, str]
) -> tuple[str, str] | None:
    """从连续平假名串里找出命中明确姓名的子串，优先最长匹配。"""
    best: tuple[str, str] | None = None
    for start in range(len(text)):
        for end in range(start + 3, len(text) + 1):
            candidate = text[start:end]
            canonical = explicit_keys.get(_kana_key(candidate))
            if canonical and (best is None or len(candidate) > len(best[1])):
                best = (canonical, candidate)
    return best


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
    if left == right:
        return True
    # 首字符不同就不是同一名字；禁止“チン 包含于 ノアチン”式跨名合并。
    if left[:1] != right[:1]:
        return False
    if left in right or right in left:
        return True
    if len(left) >= 3 and len(right) >= 3:
        if SequenceMatcher(None, left, right).ratio() >= 0.75:
            return True
    return abs(len(left) - len(right)) <= 1 and _edit_distance(left, right) <= 1


def _is_name_variant(canonical: str, term: str) -> bool:
    """只允许同一名字的前缀变体，不再接受任意包含关系。"""
    if term == canonical:
        return True
    if term in ENTITY_STOPWORDS or not term.startswith(canonical):
        return False
    suffix = term[len(canonical) :]
    return suffix in ENTITY_HONORIFIC_SUFFIXES or len(suffix) <= 4


def _base_of_variant(variant: str) -> str:
    for suffix in ENTITY_HONORIFIC_SUFFIXES:
        if variant.endswith(suffix) and len(variant) > len(suffix):
            return variant[: -len(suffix)]
    return variant


class DynamicEntityEngine:
    """文档级实体索引：种子召回变体，首发锁定 + 前文示例注入。

    设计目标是“宁漏不错”：只有敬称名、明确姓名上下文和共同前缀汉字姓名
    才进入索引；裸片假名/拟声词/普通名词不建实体。锁定只接受“源行以实体
    开头且后接标点、译文前 2–4 字后接标点”的高置信场景，其余只保存前文
    示例。``enforce_locks`` 只清理未翻译的日文专名。
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
        honorific_counts: Counter[str] = Counter()
        honorific_forms: dict[str, str] = {}
        explicit_counts: Counter[str] = Counter()
        katakana_terms: Counter[str] = Counter()
        hiragana_terms: Counter[str] = Counter()
        kanji_counts: Counter[str] = Counter()
        all_counts: Counter[str] = Counter()
        first_seen: dict[str, int] = {}

        def add_name(name: str, full: str, index: int, weight: int = 1) -> None:
            if not name or name in ENTITY_STOPWORDS:
                return
            honorific_counts[name] += weight
            honorific_forms[full] = name
            all_counts[name] += weight
            all_counts[full] += weight
            first_seen.setdefault(name, index)
            first_seen.setdefault(full, index)

        for index, text in enumerate(raw_cues):
            for pattern in (KANJI_KATAKANA_HONORIFIC_RE, HIRAGANA_HONORIFIC_RE):
                for match in pattern.finditer(text):
                    add_name(match.group(1), match.group(0), index)
            for match in KATAKANA_NICKNAME_RE.finditer(text):
                add_name(match.group(1), match.group(0), index)
            for match in EXPLICIT_NAME_RE.finditer(text):
                name = match.group(1)
                if name in ENTITY_STOPWORDS:
                    continue
                explicit_counts[name] += 1
                all_counts[name] += 1
                first_seen.setdefault(name, index)
            for match in KATAKANA_TERM_RE.finditer(text):
                term = match.group(0)
                katakana_terms[term] += 1
                all_counts[term] += 1
                first_seen.setdefault(term, index)
            for match in HIRAGANA_TERM_RE.finditer(text):
                term = match.group(0)
                hiragana_terms[term] += 1
                all_counts[term] += 1
                first_seen.setdefault(term, index)
            for match in KANJI_CHUNK_RE.finditer(text):
                term = match.group(0)
                kanji_counts[term] += 1
                all_counts[term] += 1
                first_seen.setdefault(term, index)

        seeds: dict[str, int] = {}
        for term, count in honorific_counts.items():
            if term not in ENTITY_STOPWORDS:
                seeds[term] = max(seeds.get(term, 0), count * 2)
        for term, count in explicit_counts.items():
            if term not in ENTITY_STOPWORDS:
                seeds[term] = max(seeds.get(term, 0), count * 2)

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

        # 平假名写法（みさきひより）只有在规范键命中明确姓名（ミサキヒヨリ）时才并入，
        # 不做无证据的昵称猜测。
        explicit_keys = {_kana_key(name): name for name in explicit_counts}
        extra_variants: dict[str, set[str]] = {}
        for term in hiragana_terms:
            alias = _explicit_alias(term, explicit_keys)
            if alias:
                canonical, matched = alias
                extra_variants.setdefault(canonical, set()).add(matched)

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
            set(kanji_counts)
            | set(katakana_terms)
            | set(honorific_forms)
            | set(honorific_counts)
            | set(explicit_counts)
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
                if _is_name_variant(canonical, term):
                    variants.add(term)
            for full, name in honorific_forms.items():
                if name == canonical:
                    variants.add(full)
            variants |= extra_variants.get(canonical, set())
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
            # 没有锁定名/前文示例时不要只注入“实体变体”：模型会把它当专名
            # 原样保留，反而造成假名残留；首次出现让模型自然译成中文。
            if not locked and not examples:
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
            # 坏译不能进入“前文示例”，否则实体上下文会把错误固定下来。
            if validate_cue_translation(translation) is not None:
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
        """把中文译文里未翻译的日文专名收敛到已锁定译名。

        两层防护，缺一不可：
        1. 先把**已经是锁定译名**的片段用哨兵挡住——否则当锁定名本身包含
           日文基名（梅地先生 ⊃ 梅地）时，正确的中文名会被再替换一次；
        2. 每个实体只做一遍替换（正则交替、长变体优先）——逐个
           ``str.replace`` 会让长变体替换出的中文名再次命中短变体，
           得到「梅地先生先生」。
        """
        if not zh_text or not self.locked_translations:
            return zh_text
        result = zh_text
        for index, (canonical, locked) in enumerate(
            sorted(
                self.locked_translations.items(),
                key=lambda item: len(item[0]),
                reverse=True,
            )
        ):
            if not locked:
                continue
            variants = sorted(
                (
                    variant
                    for variant in self._variants.get(canonical, {canonical})
                    if variant
                ),
                key=len,
                reverse=True,
            )
            if not variants:
                continue
            pattern = re.compile("|".join(re.escape(item) for item in variants))
            # 哨兵用私用区字符 + 序号，正常译文不会出现。
            marker = f"\ue000{index}\ue001"
            masked = locked in result
            if masked:
                result = result.replace(locked, marker)
            if pattern.search(result):
                result = pattern.sub(lambda _match, value=locked: value, result)
            if masked:
                result = result.replace(marker, locked)
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
        if canonical in ENTITY_STOPWORDS:
            return ""
        stripped_source = source.lstrip()
        if not stripped_source or stripped_source.startswith(
            ("（", "(", "【", "[")
        ):
            return ""
        if validate_cue_translation(translation) is not None:
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
