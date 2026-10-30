"""跨 cue 断词右吸附：把左 cue 末尾的半截词搬到右 cue 开头。

背景：Whisper 的 ``--max_line_width`` 会把一个词切在两条 cue 之间，例如
``カ`` / ``ウンター``。合并 cue 会改变 cue 数与时间轴（ASR 修复路径），
但翻译层不应改动字幕结构。这里只在原 cue 数量、序号与时间轴完全不变的
前提下，把断词片段从左侧 cue 的末尾“吸附”到右侧 cue 的开头：左 cue 去掉
``tail``，右 cue 用 ``replacement`` 替换其 ``head``。

约束：
- 纯标准库，不 import ``translate.py``；规则表与 timing 解析放在本模块，
  由 ``asr_repair`` 复用，避免两份实现漂移。
- 表仍为 3 列 TSV（``left_tail<TAB>right_head<TAB>replacement``），
  注释/空行跳过，坏行与重复键带 ``文件:行号`` 抛 ``TranslateError``。
- 守卫全部是硬条件；任一不满足只 skip，正文与 cue 结构原样返回，不抛错。
- 从右到左处理，同一 cue 在一次调用中最多移动一次；成功移动后左右两侧
  不再命中同一条规则，二次运行必须 0 变更。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .errors import TranslateError

JOIN_COLUMNS = 3

TIMING_RE = re.compile(
    r"^(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
    r"\s*-->\s*"
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
)

# 单字助词属于左句语法，搬到右 cue 会把句子拆坏。
PARTICLE_TAILS = frozenset("をがのはにへとでもかねよな")

# 左前缀去尾后的合法结尾：句末标点表示断词发生在两个句子之间；连接形/
# 助词说明断词落在语法单位中间。其余情况多半是 ASR 的巧合，宁可不动。
PREFIX_END_PUNCT = "、。！？…"
PREFIX_END_SUFFIXES = (
    "たら",
    "れば",
    "けど",
    "から",
    "ので",
    "って",
    "て",
    "で",
    "と",
    "が",
    "は",
    "を",
    "に",
    "へ",
    "も",
    "の",
    "ね",
    "よ",
    "か",
    "な",
)

MAX_TAIL_CHARS = 2
MAX_HEAD_CHARS = 4


@dataclass(frozen=True)
class JoinRule:
    """一行合并规则；只在两个 cue 的边界同时命中时使用。"""

    left_tail: str
    right_head: str
    replacement: str


def load_joins(path: Path) -> tuple[JoinRule, ...]:
    """读取 ``left_tail<TAB>right_head<TAB>replacement`` 合并表。

    ``#`` 注释与空行跳过；列数不对、空字段、重复键都带 ``文件:行号``
    抛 ``TranslateError``，不允许静默丢规则。
    """
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise TranslateError(f"{path}: 合并表不存在") from exc
    except OSError as exc:
        raise TranslateError(
            f"{path}: 读取合并表失败：{exc.strerror or exc}"
        ) from exc

    rules: list[JoinRule] = []
    seen: dict[tuple[str, str], int] = {}
    for line_no, line in enumerate(raw.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != JOIN_COLUMNS:
            raise TranslateError(
                f"{path}:{line_no}: 合并表必须为 3 列 "
                "left_tail<TAB>right_head<TAB>replacement"
            )
        left_tail, right_head, replacement = (
            part.strip() for part in parts
        )
        if not left_tail or not right_head or not replacement:
            raise TranslateError(
                f"{path}:{line_no}: left_tail/right_head/replacement 不能为空"
            )
        key = (left_tail, right_head)
        if key in seen:
            raise TranslateError(
                f"{path}:{line_no}: 重复键：{left_tail}<TAB>{right_head}"
                f"（首次见第 {seen[key]} 行）"
            )
        seen[key] = line_no
        rules.append(
            JoinRule(
                left_tail=left_tail,
                right_head=right_head,
                replacement=replacement,
            )
        )
    return tuple(rules)


def parse_timing(
    timing: str,
) -> tuple[int, int, str, str] | None:
    """解析 ``HH:MM:SS,mmm --> HH:MM:SS,mmm``。

    返回 ``(start_ms, end_ms, start_text, end_text)``；无法解析返回 None。
    毫秒按右补零解释（`,1` 等价 100ms），与常见 SRT 工具一致。
    """
    match = TIMING_RE.match(timing.strip())
    if match is None:
        return None
    start_h, start_m, start_s, start_frac = match.groups()[:4]
    end_h, end_m, end_s, end_frac = match.groups()[4:]
    start_ms = (
        int(start_h) * 3_600_000
        + int(start_m) * 60_000
        + int(start_s) * 1000
        + int(start_frac.ljust(3, "0"))
    )
    end_ms = (
        int(end_h) * 3_600_000
        + int(end_m) * 60_000
        + int(end_s) * 1000
        + int(end_frac.ljust(3, "0"))
    )
    start_text = f"{int(start_h):02d}:{start_m}:{start_s},{start_frac.ljust(3, '0')}"
    end_text = f"{int(end_h):02d}:{end_m}:{end_s},{end_frac.ljust(3, '0')}"
    return start_ms, end_ms, start_text, end_text


def _match_rule(
    left: str, right: str, rules: tuple[JoinRule, ...]
) -> JoinRule | None:
    for rule in rules:
        if left.endswith(rule.left_tail) and right.startswith(rule.right_head):
            return rule
    return None


def _has_newline(text: str) -> bool:
    return "\n" in text or "\r" in text


def _is_only_punctuation_or_space(text: str) -> bool:
    """前缀去尾后若只剩标点/空白，说明左 cue 没有可保留的正文。"""
    for char in text:
        if char.isspace():
            continue
        if unicodedata.category(char)[0] not in ("P", "Z"):
            return False
    return True


def _prefix_is_valid(prefix: str) -> bool:
    if not prefix.strip():
        return False
    if _is_only_punctuation_or_space(prefix):
        return False
    return (
        prefix[-1] in PREFIX_END_PUNCT
        or prefix.endswith(PREFIX_END_SUFFIXES)
    )


def _rule_is_usable(rule: JoinRule) -> bool:
    """规则本身的形状守卫；不依赖左右 cue 内容。"""
    tail, head = rule.left_tail, rule.right_head
    return (
        rule.replacement == tail + head
        and not _has_newline(rule.replacement)
        and 0 < len(tail) <= MAX_TAIL_CHARS
        and 0 < len(head) <= MAX_HEAD_CHARS
        and tail not in PARTICLE_TAILS
    )


def transfer_texts(
    texts: list[str],
    timings: list[str],
    rules: tuple[JoinRule, ...],
    *,
    max_gap_ms: int = 500,
) -> tuple[list[str], int]:
    """把左 cue 末尾的断词片段右吸附到下一 cue，保持 cue 数量不变。

    返回 ``(新文本列表, 移动次数)``；入参不会被就地修改。任一守卫不满足
    的边界直接跳过，不抛错；规则表本身的坏行由 :func:`load_joins` 抛出。
    """
    result = list(texts)
    if len(result) < 2 or len(timings) != len(result) or not rules:
        return result, 0

    moved: set[int] = set()
    changes = 0
    # 从右到左：右侧边界先定型，同一 cue 不会在一次调用里被移动两次。
    for position in range(len(result) - 2, -1, -1):
        if position in moved or position + 1 in moved:
            continue
        left = result[position]
        right = result[position + 1]
        if not left or not right:
            continue
        # 源 SRT 正文可能多行（Whisper 的 20×2）。只要 tail/head 仍落在
        # 文本边界，换行本身不影响吸附；保留原换行，不做无谓压行。
        rule = _match_rule(left, right, rules)
        if rule is None or not _rule_is_usable(rule):
            continue
        tail, head = rule.left_tail, rule.right_head
        prefix = left[: -len(tail)]
        if not _prefix_is_valid(prefix):
            continue
        left_timing = parse_timing(timings[position])
        right_timing = parse_timing(timings[position + 1])
        if left_timing is None or right_timing is None:
            continue
        if right_timing[0] < left_timing[1]:
            continue
        gap = right_timing[0] - left_timing[1]
        if gap > max_gap_ms:
            continue
        new_left = prefix
        new_right = rule.replacement + right[len(head):]
        # 若移动后同一条规则仍能命中（tail 恰是前缀结尾，或 head 恰是
        # replacement 的前缀），该表本身有歧义，二次运行会继续搬；跳过。
        if new_left.endswith(tail) or new_right.startswith(head):
            continue
        result[position] = new_left
        result[position + 1] = new_right
        moved.update((position, position + 1))
        changes += 1
    return result, changes
