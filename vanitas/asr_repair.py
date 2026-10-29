"""ASR 跨 cue 断词修复：表驱动、贪心合并、默认不写盘。

背景：Whisper 的 ``--max_line_width`` 会把一个词切成相邻两条 cue
（如 ``カ`` / ``ウンター``），翻译层会把两半当成两个句子。这里用一张
显式表只在“左 cue 以 tail 结尾、右 cue 以 head 开头”时合并，避免用
通用中文/日文规则猜测断词。

约束：
- 只处理 SRT 正文，不改源 ASR 文件；CLI 必须写到独立 out-dir；
- 守卫全部是硬条件（前置文本非空、时间轴可解析、无重叠、间隔 ≤1s、
  合并后 ≤8s），任一不满足只记 skip，不动 cue；
- 二次运行必须 0 变更（幂等）：合并后 tail/head 已被 replacement 取代。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .cue_boundary import (
    JoinRule,
    _match_rule,
    load_joins,
    parse_timing,
)
from .errors import TranslateError
from .translate import parse_srt

MAX_GAP_MS = 1000
MAX_MERGED_DURATION_MS = 8000


@dataclass(frozen=True)
class AsrCue:
    """单条字幕；正文已压成单行（日文按无空格拼接）。"""

    index: str
    timing: str
    text: str


@dataclass(frozen=True)
class Skip:
    """一次“本可合并但被守卫拒绝”的记录，用于 dry-run 报告。"""

    left: str
    right: str
    reason: str


@dataclass
class RepairResult:
    cues: list[AsrCue]
    merged: int = 0
    skipped: list[Skip] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.merged > 0


def parse_cues(text: str) -> list[AsrCue]:
    """解析 SRT；每个 cue 的多行正文按无空格拼接并压成单行。"""
    raw = text.lstrip("\ufeff")
    cues: list[AsrCue] = []
    for subtitle, _separator in parse_srt(raw, "\n"):
        if not subtitle.index or not subtitle.timing:
            continue
        body = "".join(subtitle.text.splitlines()).strip()
        cues.append(
            AsrCue(
                index=subtitle.index,
                timing=subtitle.timing,
                text=body,
            )
        )
    return cues


def render_srt(cues: list[AsrCue]) -> str:
    """按 1..N 重排序号，输出单行正文、LF 分隔的标准 SRT。"""
    blocks = [
        f"{position}\n{cue.timing}\n{cue.text}"
        for position, cue in enumerate(cues, start=1)
    ]
    if not blocks:
        return ""
    return "\n\n".join(blocks) + "\n"


def _timing_is_monotonic(cues: list[AsrCue]) -> bool:
    previous_start: int | None = None
    for cue in cues:
        parsed = parse_timing(cue.timing)
        if parsed is None:
            return False
        start_ms, end_ms = parsed[0], parsed[1]
        if end_ms < start_ms:
            return False
        if previous_start is not None and start_ms < previous_start:
            return False
        previous_start = start_ms
    return True


def repair_cues(
    cues: list[AsrCue],
    rules: tuple[JoinRule, ...],
    *,
    max_gap_ms: int = MAX_GAP_MS,
    max_duration_ms: int = MAX_MERGED_DURATION_MS,
) -> RepairResult:
    """按规则贪心合并相邻 cue，直到不动点；返回重排后的结果。"""
    work = list(cues)
    skipped: list[Skip] = []
    skipped_keys: set[tuple[str, str, str]] = set()
    merged = 0

    def record_skip(left: AsrCue, right: AsrCue, reason: str) -> None:
        key = (left.index, right.index, reason)
        if key in skipped_keys:
            return
        skipped_keys.add(key)
        skipped.append(Skip(left=left.index, right=right.index, reason=reason))

    # 单次扫描内合并后留在原地，天然支持向右链式合并；外层不动点循环
    # 只作为兜底，上限为初始 cue 数，避免异常规则导致无限循环。
    max_passes = max(1, len(work))
    for _ in range(max_passes):
        changed = False
        position = 0
        while position < len(work) - 1:
            left = work[position]
            right = work[position + 1]
            rule = _match_rule(left.text, right.text, rules)
            if rule is None:
                position += 1
                continue
            left_prefix = left.text[: -len(rule.left_tail)]
            if not left_prefix.strip():
                record_skip(left, right, "left 前置为空")
                position += 1
                continue
            left_timing = parse_timing(left.timing)
            right_timing = parse_timing(right.timing)
            if left_timing is None or right_timing is None:
                record_skip(left, right, "时间轴无法解析")
                position += 1
                continue
            left_start, left_end, left_start_text, _left_end_text = left_timing
            right_start, right_end, _right_start_text, right_end_text = right_timing
            if right_start < left_end:
                record_skip(left, right, "right.start < left.end")
                position += 1
                continue
            gap = right_start - left_end
            if gap > max_gap_ms:
                record_skip(
                    left,
                    right,
                    f"间隔 {gap}ms 超过 {max_gap_ms}ms",
                )
                position += 1
                continue
            duration = right_end - left_start
            if duration > max_duration_ms:
                record_skip(
                    left,
                    right,
                    f"合并后时长 {duration}ms 超过 {max_duration_ms}ms",
                )
                position += 1
                continue
            merged_text = (
                left_prefix
                + rule.replacement
                + right.text[len(rule.right_head) :]
            )
            work[position] = AsrCue(
                index=left.index,
                timing=f"{left_start_text} --> {right_end_text}",
                text=merged_text,
            )
            del work[position + 1]
            merged += 1
            changed = True
            # 不前进：合并结果可能继续与下一条形成链式断词。
        if not changed:
            break

    renumbered = [
        AsrCue(index=str(position), timing=cue.timing, text=cue.text)
        for position, cue in enumerate(work, start=1)
    ]
    if not _timing_is_monotonic(renumbered):
        raise TranslateError("合并后时间轴不单调；请人工检查该 SRT")
    return RepairResult(cues=renumbered, merged=merged, skipped=skipped)


def repair_text(
    text: str,
    rules: tuple[JoinRule, ...],
    *,
    max_gap_ms: int = MAX_GAP_MS,
    max_duration_ms: int = MAX_MERGED_DURATION_MS,
) -> RepairResult:
    """解析 + 合并的便捷入口；不写盘，渲染用 ``render_srt()``。"""
    result = repair_cues(
        parse_cues(text),
        rules,
        max_gap_ms=max_gap_ms,
        max_duration_ms=max_duration_ms,
    )
    return result
