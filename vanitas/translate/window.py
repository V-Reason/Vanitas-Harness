"""微批次合流协议：[#N] 编号解析、重编号恢复、结构/语义错位熔断。

职责边界：只判断“模型这一窗的返回能不能用”，不负责重译、不负责
写回、不持有预算。调用方（``pipeline``）拿到 ``ok=False`` 后自行回退
逐句路径。

``WINDOW_ID_RE`` / ``WINDOW_ID_ANY_RE`` 的实际定义在 ``cleaner``（低层），
这里 import 后重新暴露，使“编号协议”对外仍以本模块为入口，且全包只有
一处 pattern 定义。
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from .cleaner import WINDOW_ID_ANY_RE, WINDOW_ID_RE  # noqa: F401  (re-export)
from .types import Subtitle

__all__ = [
    "WINDOW_ID_RE",
    "WINDOW_ID_ANY_RE",
    "_parse_window_items",
    "parse_window_response",
    "recover_window_response",
    "strip_window_id_prefix",
    "_normalize_for_compare",
    "_source_texts_are_similar",
    "_copies_history",
    "_translation_is_too_short",
    "_window_translation_is_suspicious",
]


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


def _copies_history(
    source_text: str,
    translation: str,
    history: tuple[tuple[str, str], ...],
) -> bool:
    """当前译文是否整段复用了历史前文。

    只抓“不同源文却出现长句复制”的高置信场景；源文相似或译文过短时跳过，
    避免把合法重复句/语气词误判成错位。
    """
    normalized = _normalize_for_compare(translation)
    if len(normalized) < 12 or not history:
        return False
    source_norm = _normalize_for_compare(source_text)
    for history_source, history_translation in history:
        previous = _normalize_for_compare(history_translation)
        if len(previous) < 8:
            continue
        if _source_texts_are_similar(
            source_norm, _normalize_for_compare(history_source)
        ):
            continue
        if previous in normalized or normalized in previous:
            return True
        match = SequenceMatcher(None, normalized, previous).find_longest_match(
            0, len(normalized), 0, len(previous)
        )
        if match.size >= 10:
            return True
    return False


def _translation_is_too_short(source_text: str, translation: str) -> bool:
    """非空原文对应极短译文时判可疑；只拦明显漏译，不追求语义判断。

    QC 目前只查“超长”，没有“过短=漏译”。批量窗口一旦把邻近行/拟声词
    写进当前 cue，可能整句丢失但仍通过硬校验；这里只拦“长源文 + 极短
    译文”的明显漏译，触发整窗回退让逐句路径重译。阈值刻意保守：源文
    至少 30 字、译文不超过 6 字，避免把简短中文口语/语气词误判。
    """
    source_norm = _normalize_for_compare(source_text)
    translation_norm = _normalize_for_compare(translation)
    if len(source_norm) < 30 or not translation_norm:
        return False
    return len(translation_norm) <= 6


def _window_translation_is_suspicious(
    window: list[tuple[int, Subtitle]],
    aligned: dict[str, str],
    history: tuple[tuple[str, str], ...] = (),
) -> bool:
    """检测结构完整但语义明显错位的窗口，避免把相邻行译文写进当前 cue。

    五类信号：
    1. 正文残留 [#N]；
    2. 非空原文对应空译文；
    3. 非空原文对应极短译文（明显漏译）；
    4. 不同原文得到完全相同且足够长的译文（合法重复短句不触发）；
    5. 当前译文整段复用历史前文（跨窗复制，例如 396 -> 411）。
    """
    pairs: list[tuple[str, str]] = []
    for _index, subtitle in window:
        translation = aligned.get(subtitle.index, "")
        if WINDOW_ID_ANY_RE.search(translation):
            return True
        if subtitle.text.strip() and not translation.strip():
            return True
        if _translation_is_too_short(subtitle.text, translation):
            return True
        if _copies_history(subtitle.text, translation, history):
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
