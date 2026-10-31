"""端到端业务管线：translate_file / translate_srt / translate_prose 的串联编排。

职责边界：本层是唯一同时认识“文件、字幕结构、窗口协议、审计与预算”的地方，
负责把下层原子能力按固定顺序装配成一次完整翻译；它自身不实现清洗规则、
不实现编号解析、不实现路径规划。依赖方向严格单向：pipeline → 下层各模块。
"""

from __future__ import annotations

import os
import re
import warnings
from collections.abc import Sequence
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
from .io import _read_text, extract_metadata_entity_seeds, output_path
from ..quality import (
    ASMR_T1_CODES,
    ContextRule,
    DuplicateIndex,
    TranslationProfile,
    apply_context_rules,
    apply_pronoun_policy,
    audit_translation,
    combine_hints,
    default_context_rules,
    detect_register,
    enforce_gender_anatomy_invariant,
    normalize_compare,
    repair_quote_private,
    salvage_t1_rejection,
    strip_example_names,
)
from .types import (
    ASR_HALLUCINATION_PATTERNS,
    BOM,
    CritiqueBudget,
    Subtitle,
    T1Budget,
    TranslationFailure,
    TranslationOutcome,
)
from .window import (
    WINDOW_ID_ANY_RE,
    WindowAlignment,
    align_window_response,
    identify_suspicious_cues,
    parse_window_response,
    recover_window_response,
    strip_window_id_prefix,
)

# T1 预算按候选级：整包 max(5, round(total_cues*1%))、单文件
# max(2, ceil(file_cues*1%))；由 CLI 创建一个 T1Budget 跨文件共享，
# translate_srt 每次文件 begin_file。
T1_NEARBY_RADIUS = 3

SINGLE_RETRY_HINT = (
    "（上一次输出仍残留日文假名；请把包括人名、称谓、助词、片假名名词在内的"
    "所有日文假名都译成中文。人名中的“ちゃん/さん/君/様”必须译成"
    "“酱/先生/君/大人”等中文称谓，片假名名词按语义翻译；"
    "不得保留任何假名或整句日文。）"
)

# 失败原因回灌：把**本次**硬校验的具体原因临时拼进重译请求，让模型知道错在哪，
# 而不是只给它一句泛泛的"译得不对"。这条提示只进当前这一次请求（调用方先
# reset_history、事后 restore），不进 `_history`，也不污染后续窗口。
REASON_RETRY_HINT = (
    "（上一版译文未通过写回校验：{reason}。"
    "请针对该问题重译这一句，只输出中文译文，不要解释。）"
)


def _reason_retry_hint(reason: str, previous: str = "") -> str:
    """拼装一次性的重译提示：通用假名提示 + 本次失败原因（+ 上一句译文）。"""
    hint = f"{SINGLE_RETRY_HINT}\n{REASON_RETRY_HINT.format(reason=reason)}"
    if previous and validate_cue_translation(previous) is None:
        hint += f" 上一句译文：{_clip(previous, 80)}"
    return hint

# 断点快照：显式可见的 <output>.part.srt，只收已接受条目，完成后由调用方删除。
PARTIAL_SUFFIX = ".part.srt"


def partial_snapshot_path(target: Path) -> Path:
    """``<output>.part.srt``：与目标同目录，便于用户直接看到中断进度。"""
    return target.with_name(target.name + PARTIAL_SUFFIX)


def _load_partial(
    path: Path, translatable: list[tuple[int, Subtitle]]
) -> dict[str, str]:
    """读取断点快照；序号与时间轴逐条对齐才采纳，否则整体丢弃。

    宁可从头重译，也不把快照错位写进别的 cue：时间轴是源文件的物理
    指纹，对不上就说明快照不属于本次输入（换文件或源被改过）。
    """
    if not path.is_file():
        return {}
    try:
        raw = path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return {}
    expected = {
        subtitle.index: subtitle.timing for _index, subtitle in translatable
    }
    completed: dict[str, str] = {}
    for block, _separator in parse_srt(raw, "\n"):
        if not block.index or not block.timing:
            return {}
        if expected.get(block.index) != block.timing:
            return {}
        completed[block.index] = block.text
    return completed


def _write_partial(
    path: Path, blocks: list[tuple[Subtitle, str]], eol: str, completed: set[str]
) -> None:
    """原子刷新快照：先写 ``.part_tmp`` 再 ``os.replace``，杜绝半截脏数据。

    磁盘写失败只告警不中断翻译：让一次临时 I/O 故障废掉十几分钟已完成的
    推理不值得，但也不能静默——告警保证用户知道续传能力已失效。
    """
    subset = [
        (block, separator)
        for block, separator in blocks
        if block.index in completed
    ]
    data = rebuild_srt(subset, eol).encode("utf-8")
    tmp = path.with_name(path.name + ".part_tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    except OSError as exc:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        warnings.warn(
            f"断点快照写入失败：{path}；{exc.strerror or exc}；本次运行不支持续传",
            RuntimeWarning,
            stacklevel=3,
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
    由调用方决定"丢弃该段"还是"整段回退"，避免半译半原的混合写回。
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
    # 没有编号却返回多个编号/多行，说明模型仍在续写相邻行；宁可丢弃该句，不写入错位内容。
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
                    reason="单句回退疑似续写相邻行，已回退为原文（无可用候选）",
                )
            )
        return None
    return cleaned


def _fill_failed_cue(subtitle: Subtitle, candidate: str) -> None:
    """失败兜底：把这次尝试的文本填回去，**绝不输出空行、也不默认搬日文原文**。

    优先填模型自己产出的候选：它没过硬校验（可能带 kana / SEXPRA 之类），但至少
    是目标语言的成品雏形，QC 会按 `kana` / `latin` 把它点名，观众也不会读到整句
    日语。只有在连候选都没有（引擎异常、一字未出）时，`candidate` 会退回到原
    正文——那种情况下没有任何可用文本，空行只会把失败藏起来。
    """
    if candidate:
        subtitle.text = candidate


class SrtTranslationSession:
    """单文件 SRT 翻译会话：把原先嵌套在 ``translate_srt`` 里的闭包收拢成方法。

    状态全部是实例属性，每个 Stage 只依赖 ``self``：可以直接构造会话并单独
    调用预处理、单句/窗口推进、局部回退等方法，不需要完整的 ``translate_srt``
    作用域。``execute()`` 是单次生命周期，会原地改写 ``Subtitle`` 正文。
    """

    def __init__(
        self,
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
        critique_budget: CritiqueBudget | None = None,
        checkpoint_path: Path | None = None,
        resume: bool = False,
        strip_trailing_periods: bool = True,
        context_rules: Sequence[ContextRule] | None = None,
    ) -> None:
        self.chat = chat
        self.text = text
        self.chunk_chars = chunk_chars
        self.source = source
        self.eol = eol
        self.failures = failures
        self.on_segment = on_segment
        self.window_size = window_size
        self.max_window_chars = max_window_chars
        self.join_rules = join_rules
        self.lookahead_cues = lookahead_cues
        self.asmr_guard = asmr_guard
        self.t1_budget = t1_budget if t1_budget is not None else T1Budget()
        self.file_cues = file_cues
        self.critique_budget = critique_budget
        self.checkpoint_path = checkpoint_path
        self.resume = resume
        self.strip_trailing_periods = strip_trailing_periods
        self._injected_rules = context_rules

        self.blocks: list[tuple[Subtitle, str]] = []
        self.valid_items: list[tuple[int, Subtitle]] = []
        self.translatable: list[tuple[int, Subtitle]] = []
        self.source_snapshot: list[str] = []
        self.total = 0
        self.filtered = 0
        self.done = 0
        self.profile = TranslationProfile(register="neutral", pronoun="auto")
        self.index = DuplicateIndex()
        self.short_index: DuplicateIndex | None = None
        self.asmr_enabled = False
        self.glossary = None
        self.rules: tuple[ContextRule, ...] = ()
        self.document_text = ""
        self.all_cues: list[tuple[int, str]] = []
        self.cue_positions: dict[int, int] = {}
        self.completed_indexes: set[str] = set()
        self.resumed_pairs: list[tuple[str, str]] = []
        self.entity_engine: DynamicEntityEngine | None = None
        self.t1_budget_warned = False
        self._snapshot_cb = None
        self._reset_cb = None
        self._restore_cb = None

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #

    def execute(self) -> str:
        """跑完整生命周期；单句路径与合流窗口路径二选一。"""
        self._preprocess()
        self._resume_from_checkpoint()
        if self.window_size <= 1:
            return self._run_single()
        return self._run_windows()

    def _preprocess(self) -> None:
        """解析 SRT、清洗 ASR 幻觉、断词吸附，并建立文档级索引与语域。"""
        self.blocks = parse_srt(self.text, self.eol)
        self.valid_items = [
            (index, block)
            for index, (block, _separator) in enumerate(self.blocks)
            if block.index and block.text.strip()
        ]
        self.total = len(self.valid_items)
        translatable: list[tuple[int, Subtitle]] = []
        for item in self.valid_items:
            subtitle = item[1]
            if ASR_HALLUCINATION_PATTERNS.search(subtitle.text):
                subtitle.text = ""
                self.filtered += 1
            else:
                translatable.append(item)
        self.translatable = translatable
        # 文档级快照前先做跨 cue 断词右吸附：只改正文，不改 cue 数/序号/时间轴。
        if self.join_rules and translatable:
            texts = [subtitle.text for _index, subtitle in translatable]
            timings = [subtitle.timing for _index, subtitle in translatable]
            texts, _ = transfer_texts(texts, timings, self.join_rules)
            for (_index, subtitle), joined in zip(translatable, texts):
                subtitle.text = joined
        self.source_snapshot = [
            subtitle.text for _index, subtitle in translatable
        ]
        # 快照已排除 ASR 幻觉；这里仍过滤一次，防止调用方直接复用快照时漏清洗。
        register = detect_register(
            [
                text
                for text in self.source_snapshot
                if not ASR_HALLUCINATION_PATTERNS.search(text)
            ]
        )
        self.profile = TranslationProfile(
            register=register,
            pronoun="你" if register == "intimate" else "auto",
        )
        self.index = DuplicateIndex()
        # T1 只对 ASMR 模板启用：生产/实验模板都算 ASMR；没有
        # task_template 属性的 duck-typed chat 不启用。
        if self.asmr_guard:
            self.t1_budget.begin_file(self.source, self.file_cues)
        else:
            self.t1_budget.begin_file(self.source)
        engine_template = getattr(self.chat, "task_template", None)
        self.asmr_enabled = bool(self.asmr_guard) and prompts.is_asmr_task(
            engine_template
        )
        self.short_index = (
            DuplicateIndex(min_chars=8, translation_ratio=0.85)
            if self.asmr_enabled
            else None
        )
        self.glossary = getattr(self.chat, "glossary", None)
        rules = self._injected_rules
        if rules is None:
            rules = getattr(self.chat, "context_rules", None)
        if rules is None:
            # 自定义 --config-dir 没有 rules.tsv 时回落仓库默认表：器官/人称
            # 收敛是安全网，不能因为配置目录精简就静默消失。
            rules = default_context_rules()
        self.rules = tuple(rules)
        self._snapshot_cb = getattr(self.chat, "history_snapshot", None)
        self._reset_cb = getattr(self.chat, "reset_history", None)
        self._restore_cb = getattr(self.chat, "restore_history", None)
        # invented_tool 的 ±3 cue 视野：按文档原始 cue 顺序建表（ASR 幻觉置空
        # 的 cue 仍占位，避免把距离算近）。nearby 不含当前 cue，当前源文由
        # audit_invented_tool 自己查。必须在续传回填之前建表：回填会把已完成
        # cue 的 subtitle.text 换成中文，之后建表会让 nearby_sources 混入译文。
        self.all_cues = [
            (self._cue_of(subtitle), subtitle.text)
            for _i, subtitle in self.valid_items
        ]
        self.cue_positions = {
            cue: position
            for position, (cue, _text) in enumerate(self.all_cues)
        }
        # 整篇源文拼一份给审计用：invented_name 要判断“这个专名整篇到底有没有
        # 出现过”，只查当前 cue 会把“别的 cue 确实提到过人名”误判成漏译。
        self.document_text = "\n".join(text for _cue, text in self.all_cues)

    def _resume_from_checkpoint(self) -> None:
        """把快照里已完成的 cue 回填进 blocks/索引/历史，后续只跑剩余项。"""
        completed: dict[str, str] = {}
        if self.resume and self.checkpoint_path is not None:
            completed = _load_partial(self.checkpoint_path, self.translatable)
        for _block_index, subtitle in self.translatable:
            done_text = completed.get(subtitle.index)
            if done_text is None:
                continue
            source_text = subtitle.text
            subtitle.text = done_text
            self.completed_indexes.add(subtitle.index)
            self.resumed_pairs.append((source_text, done_text))
            cue = self._cue_of(subtitle)
            self.index.add(cue, source_text, done_text)
            if self.short_index is not None:
                self.short_index.add(cue, source_text, done_text)
        if self.resumed_pairs:
            # 回填顺序必须与文档顺序一致，否则历史“最近 N 条”的语义会错位。
            self._restore_history(tuple(self.resumed_pairs))
            self.done = len(self.resumed_pairs)

    # ------------------------------------------------------------------ #
    # 基础设施
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cue_of(subtitle: Subtitle) -> int:
        return int(subtitle.index) if subtitle.index.isdigit() else 0

    def _history_now(self) -> tuple[tuple[str, str], ...]:
        if callable(self._snapshot_cb):
            return self._snapshot_cb()
        return ()

    def _restore_history(
        self, snapshot: tuple[tuple[str, str], ...]
    ) -> None:
        if callable(self._restore_cb):
            self._restore_cb(snapshot)
        elif callable(self._reset_cb):
            self._reset_cb()

    def _reset_history(self) -> None:
        if callable(self._reset_cb):
            self._reset_cb()

    def _set_entity(self, engine: DynamicEntityEngine | None) -> None:
        setter = getattr(self.chat, "set_entity_engine", None)
        if callable(setter):
            setter(engine)

    def _set_lookahead(self, text: str) -> None:
        setter = getattr(self.chat, "set_lookahead", None)
        if callable(setter):
            setter(text)

    def _clear_lookahead(self) -> None:
        clearer = getattr(self.chat, "clear_lookahead", None)
        if callable(clearer):
            clearer()

    def _report_progress(self) -> None:
        if self.on_segment is not None:
            self.on_segment(self.done, self.total)

    # ------------------------------------------------------------------ #
    # cue 级后处理与审计
    # ------------------------------------------------------------------ #

    def _finalize_cue(
        self,
        zh_text: str,
        source_text: str,
        entity: DynamicEntityEngine | None = None,
    ) -> str:
        """统一的 cue 级后处理：清洗 → 示例专名 → 锁定名 → 术语 → 人称 → 规则。"""
        text_out = postprocess_cue_text(
            zh_text,
            source_text,
            strip_trailing_period=self.strip_trailing_periods,
        )
        if self.asmr_enabled:
            # 整条只有“提示词示例专名”时降级成省略号（拟声位被填人名的形态）；
            # 必须早于审计：这一步是确定性的，不该再花一次重译去问模型。
            text_out = strip_example_names(
                text_out, source_text, self.document_text, self.rules
            )
        if entity is not None:
            text_out = entity.enforce_locks(text_out)
        if self.glossary is not None:
            text_out = self.glossary.apply(source_text, text_out)
        text_out = apply_pronoun_policy(text_out, self.profile)
        # 器官错位、人称归属、和制词与坏拟声全部由 rules.tsv 驱动；后两项只对
        # ASMR 生效，避免改变 subtitle/prose/general 的既有产物。
        rule_types = ["organ_guard", "context_sub"]
        if self.asmr_enabled:
            rule_types += ["wasei_clean", "sfx_repair"]
        text_out = apply_context_rules(
            source_text,
            text_out,
            self.rules,
            rule_types=tuple(rule_types),
        )
        if self.asmr_enabled:
            # 放在 organ_guard 之后：源文有耳词时先收敛成“耳朵”，
            # 不能再被生理互斥误改成“这里/肉棒”。
            text_out = enforce_gender_anatomy_invariant(
                source_text, text_out, self.rules
            )
        # 仅“私のことは…って”引用场景：叠词口误“叫我叫”收敛成“叫我”，
        # 放在人称收敛之后，避免前一步替换出的文本被再次改写。
        return repair_quote_private(source_text, text_out)

    def _nearby_sources_for(self, cue: int) -> tuple[str, ...]:
        position = self.cue_positions.get(cue)
        if position is None:
            return ()
        start = max(0, position - T1_NEARBY_RADIUS)
        end = min(len(self.all_cues), position + T1_NEARBY_RADIUS + 1)
        window = self.all_cues[start:end]
        return tuple(
            text
            for offset, (_cue, text) in enumerate(window, start=start)
            if offset != position
        )

    def _audit(
        self,
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
            index=self.index,
            profile=self.profile,
            short_index=self.short_index,
            window_index=window_index,
            nearby_sources=nearby_sources,
            asmr_guard=self.asmr_enabled,
            document_text=self.document_text,
        )

    def _candidate_ok(
        self,
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
            for issue in self._audit(
                source_text, candidate, cue, nearby_sources, window_index
            )
        }
        if codes & (
            set(trigger_codes) | {"duplicate_translation"} | ASMR_T1_CODES
        ):
            return False
        return True

    def _retry_for_audit(
        self,
        subtitle: Subtitle,
        source_text: str,
        hint: str,
        base: tuple[tuple[str, str], ...],
    ) -> str | None:
        """审计命中的一次定点重译；异常只保留原译，不写 failure。"""
        self._reset_history()
        try:
            return _translate_single_cue(
                self.chat,
                subtitle,
                source=self.source,
                failures=self.failures,
                correction=hint,
                record_failure=False,
            )
        except (EngineError, TranslateError):
            return None
        finally:
            self._restore_history(base)

    def _warn_t1(
        self, cue: int, trigger_codes: set[str], reason: str
    ) -> None:
        """T1 重译失败/候选被拒时写 RuntimeWarning，不写 failure。"""
        if not self.asmr_enabled:
            return
        t1_codes = trigger_codes & ASMR_T1_CODES
        if not t1_codes:
            return
        names = "/".join(sorted(t1_codes))
        warnings.warn(
            f"ASMR 守卫 {names} 在 cue {cue} 命中后{reason}，"
            "保留重译前文本（可确定性消解的坏词已降级）",
            RuntimeWarning,
            stacklevel=3,
        )

    def _limit_t1_issues(self, cue: int, issues: list) -> list:
        """候选级 T1 预算耗尽后只保留非 T1 问题，避免继续产生调用。"""
        if not self.asmr_enabled:
            return issues
        t1_codes = {
            issue.code for issue in issues if issue.code in ASMR_T1_CODES
        }
        if not t1_codes:
            return issues
        if self.t1_budget.can_retry():
            return issues
        scope = (
            "per_file"
            if self.t1_budget.file_total >= self.t1_budget.per_file
            else "total"
        )
        self.t1_budget.record_event(cue, t1_codes, "budget", scope)
        if not self.t1_budget_warned:
            warnings.warn(
                "ASMR 守卫达到候选级预算"
                f"（整包 {self.t1_budget.max_total}"
                f" / 单文件 {self.t1_budget.per_file}）"
                f"，cue {cue} 起后续命中保留原译",
                RuntimeWarning,
                stacklevel=3,
            )
            self.t1_budget_warned = True
        return [
            issue for issue in issues if issue.code not in ASMR_T1_CODES
        ]

    def _audit_retry(
        self,
        subtitle: Subtitle,
        source_text: str,
        current_text: str,
        cue: int,
        entity: DynamicEntityEngine | None,
        base: tuple[tuple[str, str], ...],
        window_index: DuplicateIndex | None = None,
    ) -> str:
        """命中审计时做且仅做一次定点重译；失败/拒绝保留原译并告警。"""
        nearby = self._nearby_sources_for(cue)
        issues = self._audit(
            source_text, current_text, cue, nearby, window_index
        )
        if not issues:
            return current_text
        raw_issues = issues
        issues = self._limit_t1_issues(cue, issues)
        if not issues:
            # 预算耗尽时不再重译，但已知的 T1 坏词仍要确定性消解，
            # 不能因为“没预算”把已识别的坏译原样落盘。
            budget_codes = {
                issue.code for issue in raw_issues
            } & ASMR_T1_CODES
            if self.asmr_enabled and budget_codes:
                return self._salvage(
                    current_text, source_text, budget_codes
                )
            return current_text
        trigger_codes = {issue.code for issue in issues}
        t1_codes = (
            trigger_codes & ASMR_T1_CODES if self.asmr_enabled else set()
        )
        if t1_codes:
            self.t1_budget.record_attempt()
        retry = self._retry_for_audit(
            subtitle, source_text, combine_hints(issues), base
        )
        if retry is None:
            if t1_codes:
                self.t1_budget.record_event(
                    cue, t1_codes, "error", "retry_failed"
                )
            self._warn_t1(cue, trigger_codes, "重译失败")
            if t1_codes:
                return self._salvage(current_text, source_text, t1_codes)
            return current_text
        candidate = self._finalize_cue(retry, source_text, entity)
        if self._candidate_ok(
            source_text,
            current_text,
            candidate,
            cue,
            trigger_codes,
            nearby,
            window_index,
        ):
            if t1_codes:
                self.t1_budget.record_event(
                    cue, t1_codes, "accepted", "retranslated"
                )
            return candidate
        if t1_codes:
            self.t1_budget.record_event(
                cue, t1_codes, "rejected", "candidate_rejected"
            )
        self._warn_t1(cue, trigger_codes, "候选未通过")
        if t1_codes:
            return self._salvage(current_text, source_text, t1_codes)
        return current_text

    def _salvage(
        self, current_text: str, source_text: str, t1_codes: set[str]
    ) -> str:
        """T1 重译失败/被拒后的确定性兜底；失败则退化为省略号。"""
        text = salvage_t1_rejection(
            current_text,
            source_text,
            t1_codes,
            self.document_text,
            self.rules,
        )
        return text if validate_cue_translation(text) is None else "……"

    def _add_indexes(
        self, cue: int, source_text: str, translation: str
    ) -> None:
        """只登记最终接受的译文；short_index 仅 ASMR 门控开启时存在。"""
        self.index.add(cue, source_text, translation)
        if self.short_index is not None:
            self.short_index.add(cue, source_text, translation)

    def _fill_failure(self, subtitle: Subtitle, reason: str) -> None:
        self.failures.append(
            TranslationFailure(
                source=self.source,
                segment=f"line {subtitle.index}",
                reason=reason,
            )
        )

    # ------------------------------------------------------------------ #
    # 单句路径
    # ------------------------------------------------------------------ #

    def _run_single(self) -> str:
        """window_size<=1：逐 cue 串行翻译，失败按三级兜底。"""
        self._set_entity(None)
        for _block_index, subtitle in self.valid_items:
            if subtitle.index in self.completed_indexes:
                # 断点续传：已完成的 cue 不重复请求，进度计数在回填时已加过。
                continue
            if not subtitle.text.strip():
                # ASR 幻觉已置空：不调用模型，但仍计入进度，保持 done==total。
                self.done += 1
                self._report_progress()
                continue
            source_text = subtitle.text
            cue = self._cue_of(subtitle)
            base = self._history_now()
            accepted_text = ""
            result = _translate_chunks(
                self.chat,
                source_text,
                chunk_chars=self.chunk_chars,
                source=self.source,
                label=f"line {subtitle.index}",
                failures=self.failures,
            )
            if result is not None:
                result = self._finalize_cue(result, source_text)
                reason = validate_cue_translation(result)
                if reason:
                    salvaged = salvage_slash_translation(result)
                    if validate_cue_translation(salvaged) is None:
                        result = salvaged
                        reason = None
                if reason:
                    # 把**本次**的具体失败原因临时回灌，隔离重译一次；前提是
                    # 调用方不把这次提示写进历史（reset + 末尾 restore）。
                    self._reset_history()
                    retry = _translate_single_cue(
                        self.chat,
                        subtitle,
                        source=self.source,
                        failures=self.failures,
                        correction=_reason_retry_hint(reason),
                        record_failure=False,
                    )
                    if retry is not None:
                        retry = self._finalize_cue(retry, source_text)
                        if validate_cue_translation(retry) is None:
                            result = retry
                            reason = None
                if reason:
                    # 重译仍不合格：填回候选（非空、非日文原文），并记失败。
                    _fill_failed_cue(subtitle, result)
                    self._fill_failure(subtitle, reason)
                else:
                    result = self._audit_retry(
                        subtitle, source_text, result, cue, None, base
                    )
                    subtitle.text = result
                    accepted_text = result
                    self._add_indexes(cue, source_text, result)
            # 失败段保持非空（填候选）或原文（无候选），并把失败的调用从上下文剔除。
            if accepted_text:
                self._restore_history(
                    base + ((source_text, accepted_text),)
                )
                self.completed_indexes.add(subtitle.index)
            else:
                self._restore_history(base)
            if self.checkpoint_path is not None:
                _write_partial(
                    self.checkpoint_path,
                    self.blocks,
                    self.eol,
                    self.completed_indexes,
                )
            self.done += 1
            self._report_progress()
        return rebuild_srt(self.blocks, self.eol)

    # ------------------------------------------------------------------ #
    # 合流窗口路径
    # ------------------------------------------------------------------ #

    def _build_entity_engine(self) -> None:
        """实体索引必须从**源文**建：续传后 subtitle.text 已是译文。"""
        self.entity_engine = DynamicEntityEngine(
            list(self.source_snapshot)
        )
        # 文件名元数据先验：作品名里的主角名比 ASR 的汉字写法可靠。
        prior_seeds = extract_metadata_entity_seeds(Path(self.source))
        if prior_seeds:
            self.entity_engine.seed_prior_entities(prior_seeds)
        if self.resumed_pairs:
            # 让续传后的专名锁定接上前半段，避免恢复后译名漂移。
            self.entity_engine.capture_translations(
                [pair[0] for pair in self.resumed_pairs],
                [pair[1] for pair in self.resumed_pairs],
            )
        self._set_entity(self.entity_engine)

    def _build_windows(self) -> list[list[tuple[int, Subtitle]]]:
        """按 window_size 与 max_window_chars 切窗，跳过已完成/已清洗项。"""
        windows: list[list[tuple[int, Subtitle]]] = []
        current: list[tuple[int, Subtitle]] = []
        current_chars = 0
        for item in self.translatable:
            if item[1].index in self.completed_indexes:
                continue
            text_len = len(item[1].text)
            if current and (
                len(current) >= self.window_size
                or current_chars + text_len > self.max_window_chars
            ):
                windows.append(current)
                current = []
                current_chars = 0
            current.append(item)
            current_chars += text_len
            if len(current) >= self.window_size:
                windows.append(current)
                current = []
                current_chars = 0
        if current:
            windows.append(current)
        return windows

    def _window_lookaheads(
        self, windows: list[list[tuple[int, Subtitle]]]
    ) -> list[str]:
        """后文参考只取紧邻的下一窗原文，按 cue 截断到 120 字。

        后文参考只用于理解，不进入 input_batch、不写历史、不参与写回；
        超长单条至少保留一条。
        """
        if self.lookahead_cues <= 0:
            return [""] * len(windows)
        flat = [item for window in windows for item in window]
        lookaheads: list[str] = []
        cursor = 0
        for window in windows:
            cursor += len(window)
            parts: list[str] = []
            used = 0
            for item in flat[cursor : cursor + self.lookahead_cues]:
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
            lookaheads.append(lookahead)
        return lookaheads

    def _partial_window_staging(
        self,
        window: list[tuple[int, Subtitle]],
        alignment: WindowAlignment,
        base: tuple[tuple[str, str], ...],
    ) -> list[str] | None:
        """局部回退：只对损坏 cue 发单编号请求；任一修不回来就整窗熔断。

        返回按 ``window`` 原始顺序排好的译文；返回 None 表示调用方必须走
        整窗逐句回退。损坏项来自三处：编号缺失、语义可疑（泄漏/空译/漏译/
        跨窗复制/同窗成对重复）、以及后处理仍过不了写回硬校验。
        """
        aligned = alignment.aligned
        corrupted = set(alignment.missing_ids)
        corrupted |= identify_suspicious_cues(window, aligned, base)
        served: dict[str, str] = {}
        for _block_index, subtitle in window:
            raw = aligned.get(subtitle.index)
            if raw is None:
                continue
            text = self._finalize_cue(
                strip_window_id_prefix(raw),
                subtitle.text,
                self.entity_engine,
            )
            if validate_cue_translation(text) is not None:
                corrupted.add(subtitle.index)
                continue
            served[subtitle.index] = text
        if not corrupted:
            return [served[subtitle.index] for _b, subtitle in window]
        if len(corrupted) * 2 >= len(window):
            # 半数及以上损坏 ⇒ 这一批整体不可信：模型退化通常整窗一致，只是
            # 只有少数行能被确定性检出（长句被压成拟声词时，同窗短句照样过
            # 硬校验）。此时保留任何“看起来干净”的批次译文都会放过脏文本，
            # 因此整窗回退；只有损坏项在少数时，其余条目才值得采信。
            return None
        # 只补损坏项：先丢掉整窗调用留下的滑动上下文痕迹，再逐条隔离重译，
        # 避免上一句单编号译文被当成“前文”影响下一句，也避免整窗记录被放大。
        self._restore_history(base)
        for _block_index, subtitle in window:
            if subtitle.index not in corrupted:
                continue
            self._reset_history()
            recovered = _translate_single_cue(
                self.chat,
                subtitle,
                source=self.source,
                failures=self.failures,
                record_failure=False,
            )
            if recovered is None:
                return None
            text = self._finalize_cue(
                recovered, subtitle.text, self.entity_engine
            )
            if validate_cue_translation(text) is not None:
                # 单句补救也做一次斜杠兜底；仍不合格就整窗熔断。
                salvaged = salvage_slash_translation(text)
                if validate_cue_translation(salvaged) is not None:
                    return None
                text = salvaged
            served[subtitle.index] = text
        self._restore_history(base)
        if any(
            subtitle.index not in served for _b, subtitle in window
        ):
            return None
        return [served[subtitle.index] for _b, subtitle in window]

    def _process_window(
        self,
        window: list[tuple[int, Subtitle]],
        lookahead: str,
        base_history: tuple[tuple[str, str], ...],
    ) -> list[str]:
        """翻译单个合流窗口；返回本窗已完成的 SRT 序号列表。"""
        expected_ids = [item[1].index for item in window]
        window_sources = [item[1].text for item in window]
        window_cues = [self._cue_of(item[1]) for item in window]
        fallback = len(set(expected_ids)) != len(expected_ids)
        accepted: list[tuple[str, str]] = []
        window_completed: list[str] = []
        if not fallback:
            batch_prompt = "\n".join(
                f"[#{item[1].index}] {item[1].text}" for item in window
            )
            if lookahead:
                self._set_lookahead(lookahead)
            try:
                try:
                    batch_output = self.chat(batch_prompt)
                finally:
                    self._clear_lookahead()
                alignment = align_window_response(batch_output, expected_ids)
                staging = (
                    None
                    if alignment.is_catastrophic
                    else self._partial_window_staging(
                        window, alignment, base_history
                    )
                )
                if staging is None:
                    # 灾难性结构损坏，或损坏项单句补救失败：整窗熔断回退。
                    fallback = True
                else:
                    # 审计+定点重译：每个问题 cue 最多一次，且不写 failure。
                    # 窗口内已接受项进 window_index，使同窗短重复也能在本窗
                    # 命中；窗口最终接受后才并入全局 short_index。
                    window_short_index = (
                        DuplicateIndex(
                            min_chars=8, translation_ratio=0.85
                        )
                        if self.asmr_enabled
                        else None
                    )
                    for position, (_block_index, subtitle) in enumerate(
                        window
                    ):
                        source_text = window_sources[position]
                        cue = window_cues[position]
                        staging[position] = self._audit_retry(
                            subtitle,
                            source_text,
                            staging[position],
                            cue,
                            self.entity_engine,
                            base_history,
                            window_short_index,
                        )
                        if window_short_index is not None:
                            window_short_index.add(
                                cue, source_text, staging[position]
                            )
                    self.entity_engine.capture_translations(
                        window_sources, staging
                    )
                    # 无条件恢复批前上下文：干净窗口也要回到 batch 前语义，
                    # 否则整窗记录会把 context_window 的 cue 语义放大。
                    self._restore_history(
                        base_history + tuple(zip(window_sources, staging))
                    )
                    for position in range(len(window)):
                        self._add_indexes(
                            window_cues[position],
                            window_sources[position],
                            staging[position],
                        )
                    for position, (_block_index, subtitle) in enumerate(
                        window
                    ):
                        subtitle.text = staging[position]
                    window_completed = list(expected_ids)
            except (EngineError, TranslateError):
                fallback = True
        if fallback:
            self._fallback_window(
                window,
                window_sources,
                window_cues,
                base_history,
                accepted,
                window_completed,
            )
            self._restore_history(base_history + tuple(accepted))
        return window_completed

    def _fallback_window(
        self,
        window: list[tuple[int, Subtitle]],
        window_sources: list[str],
        window_cues: list[int],
        base_history: tuple[tuple[str, str], ...],
        accepted: list[tuple[str, str]],
        window_completed: list[str],
    ) -> None:
        """硬熔断：该窗口原地回退为逐句串行，绝不采用部分对齐结果。"""
        for position, (_block_index, subtitle) in enumerate(window):
            previous = ""
            if position > 0:
                previous = window[position - 1][1].text
            elif base_history:
                previous = base_history[-1][1]
            # 单句回退不继承上一句的滑动上下文，避免模型续写相邻行。
            self._reset_history()
            result = _translate_single_cue(
                self.chat,
                subtitle,
                source=self.source,
                failures=self.failures,
            )
            if result is None:
                # 一字未出（引擎异常/疑似续写相邻行）：没有可用候选，
                # 只能保留原文；失败已由 _translate_single_cue 记录。
                continue
            result = self._finalize_cue(
                result, subtitle.text, self.entity_engine
            )
            reason = validate_cue_translation(result)
            if reason:
                # 给自造词/未本地化专名一次纠偏机会；原因临时回灌。
                self._reset_history()
                retry_hint = _reason_retry_hint(reason, previous)
                retry = _translate_single_cue(
                    self.chat,
                    subtitle,
                    source=self.source,
                    failures=self.failures,
                    correction=retry_hint,
                    record_failure=False,
                )
                if retry is not None:
                    retry = self._finalize_cue(
                        retry, subtitle.text, self.entity_engine
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
                # 重译仍不合格：填回候选（非空、非日文原文），并记失败。
                _fill_failed_cue(subtitle, result)
                self._fill_failure(subtitle, reason)
            else:
                # 逐句回退也要做文档级审计：批次失败不代表复制/敬称风险消失。
                cue = window_cues[position]
                source_text = window_sources[position]
                result = self._audit_retry(
                    subtitle,
                    source_text,
                    result,
                    cue,
                    self.entity_engine,
                    base_history,
                )
                self.entity_engine.capture_translations([source_text], [result])
                self._add_indexes(cue, source_text, result)
                subtitle.text = result
                accepted.append((source_text, result))
                window_completed.append(subtitle.index)

    def _run_windows(self) -> str:
        """合流窗口主循环；实体引擎在 finally 里无条件卸载。"""
        self._build_entity_engine()
        setter = getattr(self.chat, "set_entity_engine", None)
        try:
            windows = self._build_windows()
            lookaheads = self._window_lookaheads(windows)
            for window_index, window in enumerate(windows):
                base_history = self._history_now()
                completed = self._process_window(
                    window, lookaheads[window_index], base_history
                )
                self.completed_indexes.update(completed)
                if self.checkpoint_path is not None:
                    # 快照按窗口刷新：只有“整窗已收尾”才值得落盘。
                    _write_partial(
                        self.checkpoint_path,
                        self.blocks,
                        self.eol,
                        self.completed_indexes,
                    )
                self.done += len(window)
                self._report_progress()
            if self.filtered:
                # 被 ASR 硬清洗的空块不发请求，但要在末尾补齐进度。
                self.done += self.filtered
                self._report_progress()
            return rebuild_srt(self.blocks, self.eol)
        finally:
            if callable(setter):
                setter(None)


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
    critique_budget: CritiqueBudget | None = None,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    strip_trailing_periods: bool = True,
) -> str:
    """翻译 SRT；ASMR 守卫开启时必须传入调用方共享的 ``t1_budget``。

    复审已退役（``critique_budget`` 仅兼容），``resume`` 用 ``.part.srt`` 续传。
    """
    if asmr_guard and t1_budget is None:
        # ASMR 预算必须由调用方跨文件共享，禁止按文件静默新建。
        raise TranslateError(
            "ASMR 守卫需要调用方显式传入共享 T1Budget；"
            "不要按文件静默新建预算"
        )
    if critique_budget is not None:
        # 复审链路已退役：保留参数只为不破坏外部调用方，绝不再花模型请求。
        warnings.warn(
            "P3 复审已退役：运行时不再调用复审模型；参数仅为兼容保留",
            RuntimeWarning,
            stacklevel=2,
        )
    return SrtTranslationSession(
        chat, text, chunk_chars=chunk_chars, source=source, eol=eol,
        failures=failures, on_segment=on_segment, window_size=window_size,
        max_window_chars=max_window_chars, join_rules=join_rules,
        lookahead_cues=lookahead_cues, asmr_guard=asmr_guard,
        t1_budget=t1_budget, file_cues=file_cues,
        critique_budget=critique_budget, checkpoint_path=checkpoint_path,
        resume=resume, strip_trailing_periods=strip_trailing_periods,
    ).execute()

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
        # 散文块没有"候选文本"可填（分块调用直接返回 None），失败块保留原文：
        # 空段会在文档里挖洞，且散文没有 cue 级的 blank 审计通道。
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
    critique_budget: CritiqueBudget | None = None,
    resume: bool = False,
) -> TranslationOutcome:
    """翻译单文件；ASMR 守卫开启时必须传入跨文件共享的 ``t1_budget``。

    ``critique_budget`` 为兼容参数：P3 复审已退役，传入只会触发一次告警，
    ``critique_events`` 恒为空。``resume=True`` 时若存在与源时间轴对齐的
    ``<output>.part.srt``，已完成 cue 不再请求模型。
    """
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

    target = output_path(
        source,
        out_dir,
        settings.output_suffix if output_suffix is None else output_suffix,
        output_stem,
    )
    # 断点快照只对 SRT 有意义（散文没有 cue/时间轴指纹，无法安全校验）。
    snapshot = partial_snapshot_path(target) if suffix == ".srt" else None
    try:
        # 快照在翻译**过程中**就要落盘，目录必须提前建好；否则 --out-dir 场景
        # 会因为父目录不存在把续传能力静默降级成告警。
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise TranslateError(
            f"无法创建输出目录：{target.parent}；{exc.strerror or exc}"
        ) from exc
    if snapshot is not None and not resume:
        # 非续传模式先清旧快照：否则后续 --resume 会把上一次的残片当成进度。
        if snapshot.is_file() and snapshot.stat().st_size > 0:
            # 有进度却没带 --resume：这是用户最容易踩的坑，必须说出来。
            warnings.warn(
                f"发现未完成的断点快照 {snapshot}，未加 --resume，已丢弃并从头发起；"
                "如需续传请重跑并加 --resume",
                RuntimeWarning,
                stacklevel=2,
            )
        try:
            snapshot.unlink(missing_ok=True)
        except OSError as exc:
            warnings.warn(
                f"旧断点快照无法清理：{snapshot}；{exc.strerror or exc}",
                RuntimeWarning,
                stacklevel=2,
            )

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
            critique_budget=critique_budget,
            checkpoint_path=snapshot,
            resume=resume,
            strip_trailing_periods=getattr(
                settings, "strip_trailing_periods", True
            ),
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

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        data = translated.encode("utf-8")
        target.write_bytes(BOM + data if has_bom else data)
    except OSError as exc:
        raise TranslateError(f"写入失败：{target}；{exc.strerror or exc}") from exc

    if snapshot is not None:
        # 终态已落地，快照失去意义；删除失败只告警，不影响已完成的产物。
        try:
            snapshot.unlink(missing_ok=True)
        except OSError as exc:
            warnings.warn(
                f"断点快照清理失败：{snapshot}；{exc.strerror or exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    outcome = TranslationOutcome(
        source=source,
        output=target,
        segments=segments,
        failures=failures,
        t1_events=tuple(t1_budget.file_events),
        # P3 复审已退役：字段保留以维持数据契约，值恒为空。
        critique_events=(),
    )
    if failures:
        # 输出已写出（失败段已置空、不回填原文），由调用方汇总并返回非零
        return outcome
    return outcome
