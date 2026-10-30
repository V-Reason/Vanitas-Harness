"""P3 两阶段定向反思：风险分流 → 预算门禁 → 隔离复审 → 硬校验验收。

职责边界：本模块只回答“这一条要不要复审、复审结果能不能采纳”，
不认识 SRT、不认识文件、不写滑动上下文。真正的写回与审计顺序由
``pipeline`` 决定。

依赖方向：``critique → {types, cleaner, quality}``。**不 import pipeline**：
``finalize`` 与 ``candidate_ok`` 是 ``translate_srt`` 的闭包（依赖 entity
engine / glossary / profile / 文档级索引），因此以可调用参数注入；这也是
唯一能同时满足“单向依赖”和“复用同一套验收口径”的做法。
"""

from __future__ import annotations

from typing import Callable, Collection, Sequence

from .cleaner import _clip, sanitize_translation_output, validate_cue_translation
from ..errors import EngineError, TranslateError
from ..quality import (
    DuplicateIndex,
    audit_customer_direction,
    audit_first_person,
    audit_negative_question,
    is_onomatopoeia_or_reduplication,
    normalize_compare,
)
from .types import CritiqueBudget

# 复审模板名（config/prompts/task_translate_critique.md，只认 {input}）。
CRITIQUE_TEMPLATE = "translate_critique"
# 只输出一句修正或【保持】，96 token 足够；超出说明模型在写解释，按截断处理。
CRITIQUE_MAX_TOKENS = 96
KEEP_MARKERS = ("【保持】",)
KEEP_WORDS = ("保持", "KEEP")
# 以接续助词收尾却没说完的句子：初译若强行收句，说明可能臆造了结尾。
HANGING_ENDINGS = ("から", "けど", "けれど", "のに", "ので", "って")
TERMINAL_CHARS = "。！？!?…"
FAST_PASS_MAX_CHARS = 4
HANGING_DRAFT_MAX_CHARS = 24
CRITIQUE_LENGTH_LIMIT = 1.8

RISK_HINTS = {
    "customer_direction": "谁对谁做（授受方向）是否反了",
    "negative_question": "否定反诘是否被译成肯定句",
    "hanging_conjunction": "句子没说完时是否被强行补成完整句",
    "first_person": "原文无第一人称却出现「我/我的/让我」",
    "weak_duplicate": "是否复用了前文的句式模板",
}


def is_fast_pass(source: str) -> bool:
    """确定性直通：纯拟声/叠词与极短寒暄永远不进复审。

    这类行没有语义可校准，复审只会引入“润色噪声”，而它们的数量在
    ASMR 语料里恰恰最多——放它们进来会把 10% 预算瞬间烧光。
    """
    compact = normalize_compare(source)
    if len(compact) <= FAST_PASS_MAX_CHARS:
        return True
    return is_onomatopoeia_or_reduplication(source)


def _is_hanging(source: str, draft: str, nearby: Sequence[str]) -> bool:
    """以接续助词收尾、且初译强行收句：跨行话没说完却当成了独立句。"""
    if not nearby or not any(text.strip() for text in nearby):
        return False
    body = source.strip().rstrip(TERMINAL_CHARS + " 　")
    if not body or not body.endswith(HANGING_ENDINGS):
        return False
    tail = draft.strip()
    if not tail or tail[-1] not in TERMINAL_CHARS:
        return False
    return len(normalize_compare(tail)) <= HANGING_DRAFT_MAX_CHARS


def triage_cue_risk(
    source: str,
    draft: str,
    nearby_sources: tuple[str, ...] = (),
    duplicate_index: DuplicateIndex | None = None,
) -> set[str]:
    """纯 CPU、0 Token 的风险分流：返回命中的风险 code 集合。

    空集合表示直通 Stage 1 初译，不产生任何额外请求。五类风险与
    ``quality`` 的确定性审计同源，避免“分流器认为有病、审计认为没病”
    两套口径打架。
    """
    if not source.strip() or not draft.strip():
        return set()
    if is_fast_pass(source):
        return set()

    codes: set[str] = set()
    if audit_customer_direction(source, draft, 0):
        codes.add("customer_direction")
    if audit_negative_question(source, draft, 0):
        codes.add("negative_question")
    if audit_first_person(source, draft, 0):
        codes.add("first_person")
    if _is_hanging(source, draft, nearby_sources):
        codes.add("hanging_conjunction")
    if duplicate_index is not None and duplicate_index.find_match(
        source, draft
    ) is not None:
        codes.add("weak_duplicate")
    return codes


def build_critique_input(
    source: str,
    draft: str,
    context_lines: Sequence[tuple[str, str]] = (),
    risk_codes: Collection[str] = (),
) -> str:
    """把前文/原文/草稿/焦点风险压成一段结构化文本，灌进模板的 ``{input}``。

    模板只认 ``{input}``（``prompts.ALLOWED`` 白名单），所以这里做格式化，
    不给底座增加占位符解析复杂度。
    """
    lines: list[str] = ["【前文参考】"]
    if context_lines:
        for index, (previous_source, previous_zh) in enumerate(
            context_lines, start=1
        ):
            lines.append(f"{index}. 原文：{_clip(previous_source, 60)}")
            lines.append(f"   译文：{_clip(previous_zh, 60)}")
    else:
        lines.append("无（本文件开头）")
    lines += [
        "",
        "【当前日文】",
        source.strip(),
        "",
        "【初译草稿】",
        draft.strip(),
        "",
        "【焦点风险】",
    ]
    for code in sorted(risk_codes):
        lines.append(f"- {code}：{RISK_HINTS.get(code, code)}")
    return "\n".join(lines)


def parse_critique_response(
    raw_output: str, original_draft: str
) -> tuple[bool, str]:
    """解析复审输出；返回 ``(是否改判, 最终文本)``。

    只有明确给出不同译文才算“改判”。模型输出【保持】、空串，或与草稿
    规范化后完全相同（纯标点差异）都算未改判——这类“零变更放行”是防止
    复审变成无意义润色的第一道闸。
    """
    text = (raw_output or "").strip()
    if not text:
        return False, original_draft
    if any(marker in text for marker in KEEP_MARKERS):
        return False, original_draft
    head = text.strip("。！？!?… \t\r\n")
    if head in KEEP_WORDS or head.upper() in KEEP_WORDS:
        return False, original_draft

    cleaned = sanitize_translation_output(text, original_draft)
    if not cleaned.strip():
        return False, original_draft
    if normalize_compare(cleaned) == normalize_compare(original_draft):
        return False, original_draft
    return True, cleaned


def execute_critique_repair(
    chat: Callable[..., str],
    *,
    cue: int | str,
    source: str,
    draft: str,
    context: Sequence[tuple[str, str]] = (),
    risk_codes: Collection[str] = (),
    critique_budget: CritiqueBudget,
    finalize: Callable[[str, str], str],
    accept: Callable[[str, str, str, Collection[str]], bool],
    max_tokens: int = CRITIQUE_MAX_TOKENS,
) -> str:
    """对单条 cue 做一次隔离复审；任何失败都返回初译草稿，不上抛。

    隔离策略：调用前 ``history_snapshot()``，调用前 ``reset_history()``，
    调用后 ``finally`` 里 ``restore_history(base)``。即使将来 ``chat.critique``
    被改成会写历史，主滑动上下文也不会被污染。
    """
    codes = "/".join(sorted(risk_codes)) or "none"
    if not critique_budget.can_critique():
        critique_budget.record_event(cue, codes, "budget", "budget_exhausted")
        return draft

    critique_budget.record_attempt()
    prompt = build_critique_input(source, draft, context, risk_codes)
    snapshot = getattr(chat, "history_snapshot", None)
    reset_history = getattr(chat, "reset_history", None)
    restore_history = getattr(chat, "restore_history", None)
    critique = getattr(chat, "critique", None)
    base = snapshot() if callable(snapshot) else ()

    try:
        if callable(reset_history):
            reset_history()
        if not callable(critique):
            critique_budget.record_event(
                cue, codes, "error", "no_critique_channel"
            )
            return draft
        raw = critique(prompt, max_tokens=max_tokens)
    except (EngineError, TranslateError) as exc:
        critique_budget.record_event(cue, codes, "error", str(exc)[:120])
        return draft
    finally:
        if callable(restore_history):
            restore_history(base)

    changed, candidate = parse_critique_response(raw, draft)
    if not changed:
        critique_budget.record_event(cue, codes, "kept", "model_kept")
        return draft

    text = finalize(candidate, source)
    if validate_cue_translation(text) is not None:
        critique_budget.record_event(cue, codes, "rejected", "hard_validation")
        return draft
    source_len = len(normalize_compare(source))
    if source_len and len(normalize_compare(text)) > (
        CRITIQUE_LENGTH_LIMIT * source_len
    ):
        # 复审写成解释性长句时宁可不要；candidate_ok 的 1.6× 源文上限更严，
        # 这里保留 1.8× 作为 accept 注入被削弱时的独立兜底。
        critique_budget.record_event(cue, codes, "rejected", "too_long")
        return draft
    if not accept(source, draft, text, risk_codes):
        critique_budget.record_event(
            cue, codes, "rejected", "candidate_rejected"
        )
        return draft

    critique_budget.record_event(cue, codes, "repaired", "critique_repaired")
    return text
