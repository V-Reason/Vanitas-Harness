"""统一度量：token 估算、上下文/输出预算、汉字计数、耗时与速率。

本模块是纯函数与纯数据，不 import 任何业务模块，也不做 I/O：
- `estimate_tokens` 只做粗略估算，不冒充服务端 tokenizer；
- 聊天与翻译的预算公式集中在此，避免 repl/translate 各写一套；
- `SegmentStat` 是 translate → monitor 的只读事件，UI 层不反向依赖业务层。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

# CJK 统一表意文字、扩展 A、兼容表意文字，外加日文假名与韩文音节：
# 估算 token 时这些字符都按「接近 1 token」处理。
CJK_RANGES = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0x3040, 0x30FF),
    (0xAC00, 0xD7AF),
)
# 只统计“汉字”时使用：不把假名/韩文/标点算进“累计汉字”。
HAN_RANGES = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
)
CJK_PUNCT = set("，。！？；：、（）《》“”‘’【】")
MESSAGE_OVERHEAD = 4
PROMPT_OVERHEAD = 8


_CACHE_CHAR_LIMIT = 1024


def _estimate_tokens_uncached(text: str) -> int:
    """粗略估算：CJK 约 1 字 1 token，其它约 4 字符 1 token。"""
    if not text:
        return 0
    cjk = 0
    for char in text:
        code = ord(char)
        if char in CJK_PUNCT or any(start <= code <= end for start, end in CJK_RANGES):
            cjk += 1
    other = len(text) - cjk
    return max(1, cjk + (other + 3) // 4)


@lru_cache(maxsize=2048)
def _estimate_tokens_cached(text: str) -> int:
    return _estimate_tokens_uncached(text)


def estimate_tokens(text: str) -> int:
    if len(text) <= _CACHE_CHAR_LIMIT:
        return _estimate_tokens_cached(text)
    return _estimate_tokens_uncached(text)


def context_budget(ctx: int, max_context_tokens: int, reserve: int = 512) -> int:
    """历史消息允许占用的 token 上限。"""
    return max(256, min(ctx, max_context_tokens) - reserve)


def chat_max_tokens(
    ctx: int,
    history_tokens: int,
    reserve: int = 512,
    min_output: int = 256,
) -> int:
    """单轮生成预算：优先保证输出空间，最低 min_output（默认 256）。"""
    return max(int(min_output), int(ctx) - int(history_tokens) - int(reserve))


def count_cjk(text: str) -> int:
    """统计汉字数量，不含标点、假名、韩文与时间戳。"""
    count = 0
    for char in text:
        code = ord(char)
        if any(start <= code <= end for start, end in HAN_RANGES):
            count += 1
    return count


def format_duration(seconds: float) -> str:
    """把秒格式化为 `MM分SS秒`；超过一小时用 `HH时MM分SS秒`。"""
    total = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}时{minutes:02d}分{secs:02d}秒"
    return f"{minutes:02d}分{secs:02d}秒"


def compute_rate(units: float, seconds: float) -> float:
    """units / seconds；0 或负耗时返回 0.0，避免除零。"""
    elapsed = float(seconds)
    if elapsed <= 0:
        return 0.0
    return float(units) / elapsed


@dataclass(frozen=True)
class TranslationBudget:
    max_tokens: int
    desired_tokens: int
    available_tokens: int


def translation_budget(
    n_ctx: int,
    prompt_tokens: int,
    input_tokens: int,
    *,
    min_output_tokens: int = 512,
    output_ratio: float = 2.5,
    safety_tokens: int = 64,
) -> TranslationBudget:
    """翻译输出预算。

    期望值 = max(最小输出, 输入 token * 比例)，再被
    `n_ctx - prompt - safety` 截断；若可用预算连最小输出都不够，
    抛 ValueError 由翻译层转成可读的 TranslateError，绝不静默半句输出。
    """
    ctx = int(n_ctx)
    prompt = int(prompt_tokens)
    if ctx <= 0:
        raise ValueError(f"n_ctx 必须为正整数，实际为 {n_ctx!r}")
    if prompt < 0:
        raise ValueError(f"prompt_tokens 不能为负，实际为 {prompt}")
    desired = max(int(min_output_tokens), int(int(input_tokens) * float(output_ratio)))
    available = ctx - prompt - int(safety_tokens)
    if available < int(min_output_tokens):
        raise ValueError(
            f"翻译提示词预算不足：n_ctx={ctx}，prompt={prompt}，"
            f"safety={safety_tokens}，可用 {available} < 最小输出 {min_output_tokens}；"
            "请减小 chunk_chars、缩短上下文窗口或提高档位 ctx"
        )
    return TranslationBudget(
        max_tokens=min(desired, available),
        desired_tokens=desired,
        available_tokens=available,
    )


@dataclass(frozen=True)
class SegmentStat:
    """一次翻译请求完成后的只读统计，供 monitor 与测试消费。"""

    source: str
    translation: str
    elapsed_s: float
    tokens: int
    tps: float
    tps_source: str = "estimate"
