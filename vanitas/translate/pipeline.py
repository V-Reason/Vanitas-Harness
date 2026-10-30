"""端到端业务管线：translate_file / translate_srt / translate_prose 的串联编排。

职责边界：本层是唯一同时认识“文件、字幕结构、窗口协议、审计与预算”的地方，
负责把下层原子能力按固定顺序装配成一次完整翻译；它自身不实现清洗规则、
不实现编号解析、不实现路径规划。依赖方向严格单向：pipeline → 下层各模块。
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Callable

from .. import prompts
from .cleaner import (
    _clip,
    _collapse_blank_lines,
    postprocess_cue_text,
    salvage_slash_translation,
    validate_cue_translation,
)
from ..cue_boundary import JoinRule, transfer_texts
from .entity import DynamicEntityEngine
from ..errors import EngineError, TranslateError
from .format import chunk_text, parse_srt, rebuild_srt, split_prose
from .io import _read_text, output_path
from ..quality import (
    ASMR_T1_CODES,
    DuplicateIndex,
    TranslationProfile,
    apply_pronoun_policy,
    audit_translation,
    combine_hints,
    detect_register,
    normalize_compare,
    repair_first_person_name,
    repair_quote_private,
)
from .types import (
    ASR_HALLUCINATION_PATTERNS,
    BOM,
    Subtitle,
    T1Budget,
    TranslationFailure,
    TranslationOutcome,
)
from .window import (
    WINDOW_ID_ANY_RE,
    _window_translation_is_suspicious,
    parse_window_response,
    recover_window_response,
    strip_window_id_prefix,
)

# T1 预算按候选级：整包最多 5 次、单文件最多 2 次（514 口径 5/514=0.97%）；
# 由 CLI 创建一个 T1Budget 跨文件共享，translate_srt 每次文件 begin_file。
T1_NEARBY_RADIUS = 3

SINGLE_RETRY_HINT = (
    "（上一次输出仍残留日文假名；请把包括人名、称谓、助词、片假名名词在内的"
    "所有日文假名都译成中文。人名中的“ちゃん/さん/君/様”必须译成"
    "“酱/先生/君/大人”等中文称谓，片假名名词按语义翻译；"
    "不得保留任何假名或整句日文。）"
)


def _translate_chunks(
    chat: Callable[[str], str],
    text: str,
    *,
    chunk_chars: int,
    source: str,
    label: str,
    failures: list[TranslationFailure],
) -> str | None:
    """按 chunk 顺序翻译并拼接；任一分块失败即记 failure 并返回 None。

    逐块失败不中断整篇：已翻译分块保留在本地列表里，但仍整体判失败，
    由调用方决定“保留原文”还是“整段回退”，避免半译半原的混合写回。
    """
    translated: list[str] = []
    for position, chunk in enumerate(chunk_text(text, chunk_chars), start=1):
        if not chunk.strip():
            translated.append(chunk)
            continue
        try:
            translated.append(chat(chunk))
        except (EngineError, TranslateError) as exc:
            failures.append(
                TranslationFailure(source=source, segment=f"{label}#{position}", reason=str(exc))
            )
            return None
    return "".join(translated)


def _translate_single_cue(
    chat: Callable[[str], str],
    subtitle: Subtitle,
    *,
    source: str,
    failures: list[TranslationFailure],
    correction: str = "",
    record_failure: bool = True,
) -> str | None:
    """窗口回退时按“单编号窗口”重译，避免无编号裸文本被模型续写多行。"""
    prompt = f"[#{subtitle.index}] {subtitle.text}"
    if correction:
        prompt = f"{prompt}\n{correction}"
    try:
        raw = chat(prompt)
    except (EngineError, TranslateError) as exc:
        if record_failure:
            failures.append(
                TranslationFailure(
                    source=source,
                    segment=f"line {subtitle.index}",
                    reason=str(exc),
                )
            )
        return None

    ok, aligned, _missing = parse_window_response(raw, [subtitle.index])
    if not ok:
        ok, aligned, _missing = recover_window_response(raw, [subtitle.index])
    if ok:
        return strip_window_id_prefix(aligned[subtitle.index])

    # 裸文本出口不经过 parse_window_response，必须自己压空行：`A\n\nB` 写回会把
    # 单条 cue 拆成两个 SRT block。折叠后再做续写行数守卫。
    cleaned = _collapse_blank_lines(strip_window_id_prefix(raw))
    if not cleaned:
        if record_failure:
            failures.append(
                TranslationFailure(
                    source=source,
                    segment=f"line {subtitle.index}",
                    reason="单句回退返回空译文",
                )
            )
        return None
    # 没有编号却返回多个编号/多行，说明模型仍在续写相邻行；宁可保留原文，不写入错位内容。
    source_lines = len([line for line in subtitle.text.splitlines() if line.strip()])
    leaked_ids = WINDOW_ID_ANY_RE.findall(raw)
    if len(leaked_ids) > 1 or (
        source_lines <= 1 and len([line for line in cleaned.splitlines() if line.strip()]) > 2
    ):
        if record_failure:
            failures.append(
                TranslationFailure(
                    source=source,
                    segment=f"line {subtitle.index}",
                    reason="单句回退疑似续写相邻行，已保留原文",
                )
            )
        return None
    return cleaned


def translate_srt(
    chat: Callable[[str], str],
    text: str,
    *,
    chunk_chars: int,
    source: str,
    eol: str,
    failures: list[TranslationFailure],
    on_segment: Callable[[int, int], None] | None = None,
    window_size: int = 1,
    max_window_chars: int = 250,
    join_rules: tuple[JoinRule, ...] = (),
    lookahead_cues: int = 0,
    asmr_guard: bool = False,
    t1_budget: T1Budget | None = None,
    file_cues: int | None = None,
) -> str:
    """翻译 SRT；ASMR 守卫开启时必须传入调用方共享的 ``t1_budget``。"""
    if asmr_guard and t1_budget is None:
        # ASMR 预算必须由调用方跨文件共享，禁止按文件静默新建。
        raise TranslateError(
            "ASMR 守卫需要调用方显式传入共享 T1Budget；"
            "不要按文件静默新建预算"
        )
    blocks = parse_srt(text, eol)
    valid_items = [
        (index, block)
        for index, (block, _separator) in enumerate(blocks)
        if block.index and block.text.strip()
    ]
    total = len(valid_items)
    translatable: list[tuple[int, Subtitle]] = []
    filtered = 0
    for item in valid_items:
        subtitle = item[1]
        if ASR_HALLUCINATION_PATTERNS.search(subtitle.text):
            subtitle.text = ""
            filtered += 1
        else:
            translatable.append(item)
    # 文档级快照前先做跨 cue 断词右吸附：只改正文，不改 cue 数/序号/时间轴。
    if join_rules and translatable:
        texts = [subtitle.text for _index, subtitle in translatable]
        timings = [subtitle.timing for _index, subtitle in translatable]
        texts, _ = transfer_texts(texts, timings, join_rules)
        for (_index, subtitle), joined in zip(translatable, texts):
            subtitle.text = joined
    source_snapshot = [subtitle.text for _index, subtitle in translatable]
    done = 0

    # 快照已排除 ASR 幻觉；这里仍过滤一次，防止调用方直接复用快照时漏清洗。
    register = detect_register(
        [
            text
            for text in source_snapshot
            if not ASR_HALLUCINATION_PATTERNS.search(text)
        ]
    )
    profile = TranslationProfile(
        register=register,
        pronoun="你" if register == "intimate" else "auto",
    )
    index = DuplicateIndex()
    # T1 只对 ASMR 模板启用：生产/实验模板都算 ASMR；没有
    # task_template 属性的 duck-typed chat 不启用。
    t1_budget = t1_budget if t1_budget is not None else T1Budget()
    if asmr_guard:
        t1_budget.begin_file(source, file_cues)
    else:
        t1_budget.begin_file(source)
    engine_template = getattr(chat, "task_template", None)
    asmr_enabled = bool(asmr_guard) and prompts.is_asmr_task(engine_template)
    short_index = (
        DuplicateIndex(min_chars=8, translation_ratio=0.85)
        if asmr_enabled
        else None
    )
    glossary = getattr(chat, "glossary", None)
    snapshot_history = getattr(chat, "history_snapshot", None)
    reset_history = getattr(chat, "reset_history", None)
    restore_history = getattr(chat, "restore_history", None)

    def finalize(
        zh_text: str,
        source_text: str,
        entity: DynamicEntityEngine | None = None,
    ) -> str:
        """统一的 cue 级后处理顺序：清洗 → 锁定名 → 术语 → 人称 → 窄修复。"""
        text_out = postprocess_cue_text(zh_text, source_text)
        if entity is not None:
            text_out = entity.enforce_locks(text_out)
        if glossary is not None:
            text_out = glossary.apply(source_text, text_out)
        text_out = apply_pronoun_policy(text_out, profile)
        # 仅“下の名前で呼んであげて”场景：模型固执输出“我的名字/让我/叫我”，
        # 在 validate 前做确定性第三人称收敛；其余源文原样返回。
        text_out = repair_first_person_name(source_text, text_out)
        # 仅“私のことは…って”引用场景：叠词口误“叫我叫”收敛成“叫我”，
        # 放在人称收敛之后，避免前一步替换出的文本被再次改写。
        return repair_quote_private(source_text, text_out)

    def cue_of(subtitle: Subtitle) -> int:
        return int(subtitle.index) if subtitle.index.isdigit() else 0

    def history_now() -> tuple[tuple[str, str], ...]:
        if callable(snapshot_history):
            return snapshot_history()
        return ()

    def restore(snapshot: tuple[tuple[str, str], ...]) -> None:
        if callable(restore_history):
            restore_history(snapshot)
        elif callable(reset_history):
            reset_history()

    # invented_tool 的 ±3 cue 视野：按文档原始 cue 顺序建表（ASR 幻觉置空
    # 的 cue 仍占位，避免把距离算近）。nearby 不含当前 cue，当前源文由
    # audit_invented_tool 自己查。
    all_cues = [(cue_of(subtitle), subtitle.text) for _i, subtitle in valid_items]
    cue_positions = {
        cue: position for position, (cue, _text) in enumerate(all_cues)
    }

    def nearby_sources_for(cue: int) -> tuple[str, ...]:
        position = cue_positions.get(cue)
        if position is None:
            return ()
        start = max(0, position - T1_NEARBY_RADIUS)
        end = min(len(all_cues), position + T1_NEARBY_RADIUS + 1)
        window = all_cues[start:end]
        return tuple(
            text
            for offset, (_cue, text) in enumerate(window, start=start)
            if offset != position
        )

    def audit(
        source_text: str,
        translation: str,
        cue: int,
        nearby_sources: tuple[str, ...] = (),
        window_index: DuplicateIndex | None = None,
    ) -> list:
        return audit_translation(
            source_text,
            translation,
            cue,
            index=index,
            profile=profile,
            short_index=short_index,
            window_index=window_index,
            nearby_sources=nearby_sources,
            asmr_guard=asmr_enabled,
        )

    def candidate_ok(
        source_text: str,
        original: str,
        candidate: str,
        cue: int,
        trigger_codes: set[str],
        nearby_sources: tuple[str, ...] = (),
        window_index: DuplicateIndex | None = None,
    ) -> bool:
        """审计重译候选的统一验收：硬校验、长度/换行与残余问题。"""
        if validate_cue_translation(candidate) is not None:
            return False
        if not candidate.strip():
            return False
        if "\n" in candidate and "\n" not in source_text:
            return False
        cn = normalize_compare(candidate)
        sn = normalize_compare(source_text)
        if len(cn) > max(12, int(1.6 * len(sn))):
            return False
        # 标签问/引用/主客方向与 T1 守卫本来就会显著改变句长（短重复重译
        # 可能远短于原译），不做原文长度比例限制；其余场景防止“修掉一个
        # 问题却扩写成另一句”。
        semantic_codes = {
            "tag_question",
            "quote_private",
            "customer_direction",
            "first_person",
            *ASMR_T1_CODES,
        }
        if not (trigger_codes & semantic_codes):
            on = normalize_compare(original)
            if len(on) >= 6 and not (0.5 <= len(cn) / len(on) <= 3.0):
                return False
        codes = {
            issue.code
            for issue in audit(
                source_text, candidate, cue, nearby_sources, window_index
            )
        }
        if codes & (
            set(trigger_codes) | {"duplicate_translation"} | ASMR_T1_CODES
        ):
            return False
        return True

    def retry_for_audit(
        subtitle: Subtitle,
        source_text: str,
        hint: str,
        base: tuple[tuple[str, str], ...],
    ) -> str | None:
        """审计命中的一次定点重译；异常只保留原译，不写 failure。"""
        if callable(reset_history):
            reset_history()
        try:
            return _translate_single_cue(
                chat,
                subtitle,
                source=source,
                failures=failures,
                correction=hint,
                record_failure=False,
            )
        except (EngineError, TranslateError):
            return None
        finally:
            restore(base)

    t1_budget_warned = False

    def warn_t1(
        cue: int, trigger_codes: set[str], reason: str
    ) -> None:
        """T1 重译失败/候选被拒时写 RuntimeWarning，不写 failure。"""
        if not asmr_enabled:
            return
        t1_codes = trigger_codes & ASMR_T1_CODES
        if not t1_codes:
            return
        names = "/".join(sorted(t1_codes))
        warnings.warn(
            f"ASMR 守卫 {names} 在 cue {cue} 命中后{reason}，保留原译",
            RuntimeWarning,
            stacklevel=4,
        )

    def limit_t1_issues(cue: int, issues: list) -> list:
        """候选级 T1 预算耗尽后只保留非 T1 问题，避免继续产生调用。"""
        nonlocal t1_budget_warned
        if not asmr_enabled:
            return issues
        t1_codes = {
            issue.code for issue in issues if issue.code in ASMR_T1_CODES
        }
        if not t1_codes:
            return issues
        if t1_budget.can_retry():
            return issues
        scope = (
            "per_file"
            if t1_budget.file_total >= t1_budget.per_file
            else "total"
        )
        t1_budget.record_event(cue, t1_codes, "budget", scope)
        if not t1_budget_warned:
            warnings.warn(
                "ASMR 守卫达到候选级预算"
                f"（整包 {t1_budget.max_total} / 单文件 {t1_budget.per_file}）"
                f"，cue {cue} 起后续命中保留原译",
                RuntimeWarning,
                stacklevel=4,
            )
            t1_budget_warned = True
        return [
            issue for issue in issues if issue.code not in ASMR_T1_CODES
        ]

    def audit_retry(
        subtitle: Subtitle,
        source_text: str,
        current_text: str,
        cue: int,
        entity: DynamicEntityEngine | None,
        base: tuple[tuple[str, str], ...],
        window_index: DuplicateIndex | None = None,
    ) -> str:
        """命中审计时做且仅做一次定点重译；失败/拒绝保留原译并告警。"""
        nearby = nearby_sources_for(cue)
        issues = audit(
            source_text, current_text, cue, nearby, window_index
        )
        if not issues:
            return current_text
        issues = limit_t1_issues(cue, issues)
        if not issues:
            return current_text
        trigger_codes = {issue.code for issue in issues}
        t1_codes = (
            trigger_codes & ASMR_T1_CODES if asmr_enabled else set()
        )
        if t1_codes:
            t1_budget.record_attempt()
        retry = retry_for_audit(
            subtitle, source_text, combine_hints(issues), base
        )
        if retry is None:
            if t1_codes:
                t1_budget.record_event(
                    cue, t1_codes, "error", "retry_failed"
                )
            warn_t1(cue, trigger_codes, "重译失败")
            return current_text
        candidate = finalize(retry, source_text, entity)
        if candidate_ok(
            source_text,
            current_text,
            candidate,
            cue,
            trigger_codes,
            nearby,
            window_index,
        ):
            if t1_codes:
                t1_budget.record_event(
                    cue, t1_codes, "accepted", "retranslated"
                )
            return candidate
        if t1_codes:
            t1_budget.record_event(
                cue, t1_codes, "rejected", "candidate_rejected"
            )
        warn_t1(cue, trigger_codes, "候选未通过")
        return current_text

    def add_indexes(cue: int, source_text: str, translation: str) -> None:
        """只登记最终接受的译文；short_index 仅 ASMR 门控开启时存在。"""
        index.add(cue, source_text, translation)
        if short_index is not None:
            short_index.add(cue, source_text, translation)

    if window_size <= 1:
        clear_entity = getattr(chat, "set_entity_engine", None)
        if callable(clear_entity):
            clear_entity(None)
        for _block_index, subtitle in valid_items:
            if not subtitle.text.strip():
                # ASR 幻觉已置空：不调用模型，但仍计入已完成段，保持进度与 segments 一致
                done += 1
                if on_segment is not None:
                    on_segment(done, total)
                continue
            source_text = subtitle.text
            cue = cue_of(subtitle)
            base = history_now()
            accepted_text = ""
            result = _translate_chunks(
                chat,
                source_text,
                chunk_chars=chunk_chars,
                source=source,
                label=f"line {subtitle.index}",
                failures=failures,
            )
            if result is not None:
                result = finalize(result, source_text)
                reason = validate_cue_translation(result)
                if reason:
                    salvaged = salvage_slash_translation(result)
                    if validate_cue_translation(salvaged) is None:
                        result = salvaged
                        reason = None
                if reason:
                    failures.append(
                        TranslationFailure(
                            source=source,
                            segment=f"line {subtitle.index}",
                            reason=reason,
                        )
                    )
                else:
                    result = audit_retry(
                        subtitle, source_text, result, cue, None, base
                    )
                    subtitle.text = result
                    accepted_text = result
                    add_indexes(cue, source_text, result)
            # 失败时保留原文，并把失败的调用从滑动上下文里剔除。
            if accepted_text:
                restore(base + ((source_text, accepted_text),))
            else:
                restore(base)
            done += 1
            if on_segment is not None:
                on_segment(done, total)
        return rebuild_srt(blocks, eol)

    entity_engine = DynamicEntityEngine(
        [subtitle.text for _index, subtitle in translatable]
    )
    setter = getattr(chat, "set_entity_engine", None)
    if callable(setter):
        setter(entity_engine)
    try:
        windows: list[list[tuple[int, Subtitle]]] = []
        current: list[tuple[int, Subtitle]] = []
        current_chars = 0
        for item in translatable:
            text_len = len(item[1].text)
            if current and (
                len(current) >= window_size
                or current_chars + text_len > max_window_chars
            ):
                windows.append(current)
                current = []
                current_chars = 0
            current.append(item)
            current_chars += text_len
            if len(current) >= window_size:
                windows.append(current)
                current = []
                current_chars = 0
        if current:
            windows.append(current)

        # 后文参考只取紧邻的下一窗原文，按 cue 截断到 120 字；仅用于理解，
        # 不进入 input_batch、不写历史、不参与写回。超长单条至少保留一条。
        window_lookaheads: list[str] = []
        if lookahead_cues > 0:
            flat = [item for window in windows for item in window]
            cursor = 0
            for window in windows:
                cursor += len(window)
                parts: list[str] = []
                used = 0
                for item in flat[cursor : cursor + lookahead_cues]:
                    # 后文参考不带 [#N]：带编号会诱导模型把参考项当输入翻译，
                    # 导致编号多翻/乱序并整窗回退；这里只给裸原文。
                    line = f"- {item[1].text}"
                    if parts and used + len(line) > 120:
                        break
                    parts.append(line)
                    used += len(line)
                lookahead = "\n".join(parts)
                if len(lookahead) > 120:
                    lookahead = lookahead[:120].rstrip() + "…"
                window_lookaheads.append(lookahead)
        else:
            window_lookaheads = [""] * len(windows)

        set_lookahead = getattr(chat, "set_lookahead", None)
        clear_lookahead = getattr(chat, "clear_lookahead", None)
        for window_index, window in enumerate(windows):
            expected_ids = [item[1].index for item in window]
            window_sources = [item[1].text for item in window]
            window_cues = [cue_of(item[1]) for item in window]
            base_history = history_now()
            fallback = len(set(expected_ids)) != len(expected_ids)
            accepted: list[tuple[str, str]] = []

            if not fallback:
                batch_prompt = "\n".join(
                    f"[#{item[1].index}] {item[1].text}" for item in window
                )
                lookahead = window_lookaheads[window_index]
                if lookahead and callable(set_lookahead):
                    set_lookahead(lookahead)
                try:
                    batch_output = chat(batch_prompt)
                finally:
                    if callable(clear_lookahead):
                        clear_lookahead()
                try:
                    ok, aligned, _missing = parse_window_response(
                        batch_output, expected_ids
                    )
                    if not ok:
                        ok, aligned, _missing = recover_window_response(
                            batch_output, expected_ids
                        )
                    if ok and _window_translation_is_suspicious(
                        window, aligned, base_history
                    ):
                        ok = False
                    if not ok:
                        fallback = True
                    else:
                        staging: list[str] = []
                        for _block_index, subtitle in window:
                            text = strip_window_id_prefix(
                                aligned[subtitle.index]
                            )
                            text = finalize(text, subtitle.text, entity_engine)
                            staging.append(text)
                        bad = next(
                            (
                                reason
                                for text in staging
                                if (reason := validate_cue_translation(text))
                            ),
                            None,
                        )
                        if bad:
                            fallback = True
                        else:
                            # 审计+定点重译：每个问题 cue 最多一次，且不写 failure。
                            # 窗口内已接受项进 window_index，使同窗短重复也能在
                            # 本窗命中；窗口最终接受后才并入全局 short_index。
                            window_short_index = (
                                DuplicateIndex(
                                    min_chars=8, translation_ratio=0.85
                                )
                                if asmr_enabled
                                else None
                            )
                            for position, (_block_index, subtitle) in (
                                enumerate(window)
                            ):
                                source_text = window_sources[position]
                                cue = window_cues[position]
                                staging[position] = audit_retry(
                                    subtitle,
                                    source_text,
                                    staging[position],
                                    cue,
                                    entity_engine,
                                    base_history,
                                    window_short_index,
                                )
                                if window_short_index is not None:
                                    window_short_index.add(
                                        cue, source_text, staging[position]
                                    )
                            entity_engine.capture_translations(
                                window_sources, staging
                            )
                            # 无条件恢复批前上下文：干净窗口也要回到 batch 前语义，
                            # 否则整窗记录会把 context_window 的 cue 语义放大。
                            restore(
                                base_history
                                + tuple(zip(window_sources, staging))
                            )
                            for position in range(len(window)):
                                add_indexes(
                                    window_cues[position],
                                    window_sources[position],
                                    staging[position],
                                )
                            for position, (_block_index, subtitle) in (
                                enumerate(window)
                            ):
                                subtitle.text = staging[position]
                except (EngineError, TranslateError):
                    fallback = True

            if fallback:
                # 硬熔断：该窗口原地回退为逐句串行，绝不采用部分对齐结果
                for position, (_block_index, subtitle) in enumerate(window):
                    previous = ""
                    if position > 0:
                        previous = window[position - 1][1].text
                    elif base_history:
                        previous = base_history[-1][1]
                    # 单句回退不继承上一句的滑动上下文，避免模型续写相邻行。
                    if callable(reset_history):
                        reset_history()
                    result = _translate_single_cue(
                        chat,
                        subtitle,
                        source=source,
                        failures=failures,
                    )
                    if result is not None:
                        result = finalize(result, subtitle.text, entity_engine)
                        reason = validate_cue_translation(result)
                        if reason:
                            # 给自造词/未本地化专名一次纠偏机会；仍不合格则保留原文。
                            if callable(reset_history):
                                reset_history()
                            retry_hint = SINGLE_RETRY_HINT
                            if previous and validate_cue_translation(previous) is None:
                                retry_hint += f" 上一句译文：{_clip(previous, 80)}"
                            retry = _translate_single_cue(
                                chat,
                                subtitle,
                                source=source,
                                failures=failures,
                                correction=retry_hint,
                                record_failure=False,
                            )
                            if retry is not None:
                                retry = finalize(
                                    retry, subtitle.text, entity_engine
                                )
                                if validate_cue_translation(retry) is None:
                                    result = retry
                                    reason = None
                        if reason:
                            salvaged = salvage_slash_translation(result)
                            if validate_cue_translation(salvaged) is None:
                                result = salvaged
                                reason = None
                        if reason:
                            failures.append(
                                TranslationFailure(
                                    source=source,
                                    segment=f"line {subtitle.index}",
                                    reason=reason,
                                )
                            )
                        else:
                            # 逐句回退也要做文档级审计：批次失败不代表复制/敬称
                            # 风险消失，命中时同样只做一次定点重译。
                            cue = window_cues[position]
                            source_text = window_sources[position]
                            result = audit_retry(
                                subtitle,
                                source_text,
                                result,
                                cue,
                                entity_engine,
                                base_history,
                            )
                            entity_engine.capture_translations(
                                [source_text], [result]
                            )
                            add_indexes(cue, source_text, result)
                            subtitle.text = result
                            accepted.append((source_text, result))
                restore(base_history + tuple(accepted))

            done += len(window)
            if on_segment is not None:
                on_segment(done, total)

        if filtered:
            # 被 ASR 硬清洗的空块不发请求，但要在末尾补齐进度，保证 done==total
            done += filtered
            if on_segment is not None:
                on_segment(done, total)

        return rebuild_srt(blocks, eol)
    finally:
        if callable(setter):
            setter(None)


def translate_prose(
    chat: Callable[[str], str],
    text: str,
    *,
    chunk_chars: int,
    source: str,
    failures: list[TranslationFailure],
    on_segment: Callable[[int, int], None] | None = None,
) -> str:
    paragraphs = split_prose(text)
    total = sum(1 for paragraph, _ in paragraphs if paragraph.strip())
    done = 0
    output: list[str] = []
    for position, (paragraph, separator) in enumerate(paragraphs, start=1):
        match = re.match(r"^(\s*)(.*?)(\s*)$", paragraph, re.DOTALL)
        if match is None:
            # re 已保证可匹配；这里只是防御，绝不吞掉内容
            output.append(paragraph)
            output.append(separator)
            continue
        leading, core, trailing = match.groups()
        if not core:
            output.append(paragraph)
            output.append(separator)
            continue  # 空白段落不计入进度
        result = _translate_chunks(
            chat,
            core,
            chunk_chars=chunk_chars,
            source=source,
            label=f"block {position}",
            failures=failures,
        )
        output.append(leading + (result if result is not None else core) + trailing)
        output.append(separator)
        done += 1
        if on_segment is not None:
            on_segment(done, total)
    return "".join(output)


def translate_file(
    chat: Callable[[str], str],
    source: Path,
    settings,
    *,
    out_dir: Path | None = None,
    output_stem: str | None = None,
    output_suffix: str | None = None,
    on_segment: Callable[[int, int], None] | None = None,
    window_size: int = 1,
    max_window_chars: int = 250,
    join_rules: tuple[JoinRule, ...] = (),
    lookahead_cues: int = 0,
    asmr_guard: bool = False,
    t1_budget: T1Budget | None = None,
    file_cues: int | None = None,
) -> TranslationOutcome:
    """翻译单文件；ASMR 守卫开启时必须传入跨文件共享的 ``t1_budget``。"""
    source = Path(source)
    if not source.is_file():
        raise TranslateError(f"文件不存在：{source}")
    suffix = source.suffix.lower()
    if suffix not in settings.source_suffixes:
        raise TranslateError(
            f"不支持的扩展名 {source.suffix!r}：{source}；"
            f"允许：{'、'.join(settings.source_suffixes)}"
        )
    text, has_bom, eol = _read_text(source)
    if asmr_guard and t1_budget is None:
        # ASR 预算必须由调用方跨文件共享，禁止按文件静默新建。
        raise TranslateError(
            "ASMR 守卫需要调用方显式传入共享 T1Budget；"
            "不要按文件静默新建预算"
        )
    t1_budget = t1_budget if t1_budget is not None else T1Budget()
    t1_budget.begin_file(str(source), file_cues)
    failures: list[TranslationFailure] = []
    segments = 0
    if suffix == ".srt":
        blocks = parse_srt(text, eol)
        segments = sum(1 for subtitle, _ in blocks if subtitle.index and subtitle.text.strip())
        translated = translate_srt(
            chat,
            text,
            chunk_chars=settings.chunk_chars,
            source=str(source),
            eol=eol,
            failures=failures,
            on_segment=on_segment,
            window_size=window_size,
            max_window_chars=max_window_chars,
            join_rules=join_rules,
            lookahead_cues=lookahead_cues,
            asmr_guard=asmr_guard,
            t1_budget=t1_budget,
            file_cues=file_cues,
        )
    else:
        segments = sum(1 for paragraph, _ in split_prose(text) if paragraph.strip())
        translated = translate_prose(
            chat,
            text,
            chunk_chars=settings.chunk_chars,
            source=str(source),
            failures=failures,
            on_segment=on_segment,
        )

    target = output_path(
        source,
        out_dir,
        settings.output_suffix if output_suffix is None else output_suffix,
        output_stem,
    )
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        data = translated.encode("utf-8")
        target.write_bytes(BOM + data if has_bom else data)
    except OSError as exc:
        raise TranslateError(f"写入失败：{target}；{exc.strerror or exc}") from exc

    outcome = TranslationOutcome(
        source=source,
        output=target,
        segments=segments,
        failures=failures,
        t1_events=tuple(t1_budget.file_events),
    )
    if failures:
        # 输出已写出（失败段保留原文），由调用方汇总并返回非零
        return outcome
    return outcome
