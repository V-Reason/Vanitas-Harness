"""停滞检测：连续 / 震荡 / 重复 / 批次重复 / 批次震荡。

策略化设计：每个 Detector 独立判定一类停滞，Tracker 遍历返回第一个命中原因。
返回值集合与原 evaluate_stagnation 严格一致：
{"连续", "震荡", "重复", "批次重复", "批次震荡"}。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class StagnationView:
    """检测器需要的窗口快照。"""

    recent: tuple[tuple[str, str], ...]
    rounds: tuple[tuple[str, ...], ...]
    stagnation_limit: int


class StagnationDetector(Protocol):
    def detect(self, view: StagnationView) -> str | None:
        ...


class ContinuousDetector:
    """连续 N 次相同动作（默认 stagnation_limit 次）。"""

    def detect(self, view: StagnationView) -> str | None:
        recent = view.recent
        limit = view.stagnation_limit
        if len(recent) >= limit and len(set(recent[-limit:])) == 1:
            return "连续"
        return None


class PingPongDetector:
    """A-B-A-B（4 步）或 A-B-C-A-B-C（6 步）乒乓。"""

    def detect(self, view: StagnationView) -> str | None:
        recent = view.recent
        if len(recent) >= 4:
            if (
                recent[-1] == recent[-3]
                and recent[-2] == recent[-4]
                and recent[-1] != recent[-2]
            ):
                return "震荡"
        if len(recent) >= 6:
            if (
                recent[-1] == recent[-4]
                and recent[-2] == recent[-5]
                and recent[-3] == recent[-6]
            ):
                return "震荡"
        return None


class RecentRepeatDetector:
    """6 步窗口内某签名 ≥3 次。"""

    def detect(self, view: StagnationView) -> str | None:
        window = view.recent[-6:]
        if not window:
            return None
        if max(Counter(window).values()) >= 3:
            return "重复"
        return None


class BatchRepeatDetector:
    """最近 3 轮批次签名完全一致。"""

    def detect(self, view: StagnationView) -> str | None:
        rounds = view.rounds
        if len(rounds) >= 3 and len(set(rounds[-3:])) == 1:
            return "批次重复"
        return None


class BatchPingPongDetector:
    """批次 A-B-A-B（4 轮）或 A-B-C-A-B-C（6 轮）乒乓。"""

    def detect(self, view: StagnationView) -> str | None:
        rounds = view.rounds
        if len(rounds) >= 4:
            if (
                rounds[-1] == rounds[-3]
                and rounds[-2] == rounds[-4]
                and rounds[-1] != rounds[-2]
            ):
                return "批次震荡"
        if len(rounds) >= 6:
            if (
                rounds[-1] == rounds[-4]
                and rounds[-2] == rounds[-5]
                and rounds[-3] == rounds[-6]
            ):
                return "批次震荡"
        return None


DEFAULT_DETECTORS: tuple[StagnationDetector, ...] = (
    ContinuousDetector(),
    PingPongDetector(),
    RecentRepeatDetector(),
    BatchRepeatDetector(),
    BatchPingPongDetector(),
)


def evaluate_stagnation(
    recent: list[tuple[str, str]],
    rounds: list[tuple[str, ...]],
    stagnation_limit: int,
    *,
    detectors: tuple[StagnationDetector, ...] = DEFAULT_DETECTORS,
) -> str | None:
    """遍历检测器，返回第一个命中的原因字符串或 None。"""
    view = StagnationView(
        recent=tuple(recent),
        rounds=tuple(rounds),
        stagnation_limit=max(2, int(stagnation_limit)),
    )
    for detector in detectors:
        reason = detector.detect(view)
        if reason is not None:
            return reason
    return None


__all__ = [
    "StagnationView",
    "StagnationDetector",
    "ContinuousDetector",
    "PingPongDetector",
    "RecentRepeatDetector",
    "BatchRepeatDetector",
    "BatchPingPongDetector",
    "DEFAULT_DETECTORS",
    "evaluate_stagnation",
]
