"""核心上下文引擎：滑动视窗、预算门禁、截断二分重译、文件名翻译。

职责边界：一次 ``translate()`` 只负责“一段文本 → 一段译文”，
不认识 SRT、不认识文件、不做 cue 级后处理与审计（那些在 ``pipeline``）。
上下文是文档级可变状态，但所有权归单个 ``TranslationEngine`` 实例。
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Callable

from .. import prompts
from .cleaner import _clip, sanitize_translation_output
from ..client import LlamaClient
from ..config import AppConfig, ModelSpec
from ..cue_boundary import JoinRule, load_joins
from .entity import DynamicEntityEngine
from ..errors import EngineError, OutputTruncated, TranslateError
from .format import _split_for_retry
from .io import (
    FILENAME_KANA,
    FILENAME_MAX_CHARS,
    FILENAME_MAX_RETRIES,
    FILENAME_MAX_TOKENS,
    FILENAME_PROMPT,
    FILENAME_SYMBOL_CONNECTOR,
    FILENAME_SYSTEM_PROMPT,
    FILENAME_TECHNICAL_STEMS,
    _has_translatable_text,
    _mask_name_literals,
    _placeholder_failure,
    _restore_literals,
    _strip_filename_explanation,
    sanitize_filename_stem,
)
from ..metrics import (
    PROMPT_OVERHEAD,
    SegmentStat,
    estimate_tokens,
    translation_budget,
)
from ..quality import Glossary
from .types import NameTranslation, TranslationOptions


def extract_content(result: dict) -> str:
    choices = result.get("choices") or []
    if not choices:
        raise EngineError("/v1/chat/completions 返回体缺少 choices")
    message = choices[0].get("message") or {}
    content = (message.get("content") or "").strip()
    if not content and message.get("reasoning_content"):
        raise EngineError("模型只返回了 reasoning_content，content 为空；思考未关闭？")
    if not content:
        raise EngineError("模型返回空内容")
    return content


class TranslationEngine:
    """单段翻译管线：滑动上下文、预算门禁、截断重译与统计事件。"""

    def __init__(
        self,
        client: LlamaClient,
        prompt_dir: Path,
        system_text: str,
        *,
        options: TranslationOptions,
        on_stat: Callable[[SegmentStat], None] | None = None,
        task_template: str = "translate",
    ) -> None:
        self._client = client
        self._template = prompts.load_task(prompt_dir, task_template)
        self._task_template = task_template
        self._system = system_text
        self._options = options
        window = max(0, int(self._options.context_window))
        self._history: deque[tuple[str, str]] = deque(maxlen=window)
        self._entity_engine: DynamicEntityEngine | None = None
        self._lookahead = ""
        self._on_stat = on_stat
        # 术语表只在 ASMR 任务加载（生产/实验模板都算）；其余任务保持空表。
        self._glossary = Glossary()
        if prompts.is_asmr_task(task_template):
            self._glossary = Glossary.load(
                prompt_dir.parent / "glossary" / "asmr.tsv"
            )
        # 合并表与术语表同级：仅 ASMR 任务且表存在时启用，缺失时保持旧行为。
        self._join_rules: tuple[JoinRule, ...] = ()
        if prompts.is_asmr_task(task_template):
            joins_path = prompt_dir.parent / "asr_joins.tsv"
            if joins_path.exists():
                self._join_rules = load_joins(joins_path)

    @property
    def glossary(self) -> Glossary:
        """只读术语表；translate_srt 通过它做 cue 级确定性后处理。"""
        return self._glossary

    @property
    def task_template(self) -> str:
        """只读任务模板名；T1 守卫用它识别生产/实验 ASMR 模板。"""
        return self._task_template

    @property
    def join_rules(self) -> tuple[JoinRule, ...]:
        """只读跨 cue 断词规则；非 ASMR 任务或表缺失时为空元组。"""
        return self._join_rules

    def __call__(self, text: str) -> str:
        return self.translate(text)

    def translate(self, text: str) -> str:
        return self._translate_once(text, depth=0)

    def reset_history(self) -> None:
        """开始新文档时清空滑动上下文；文档级实体锁定表不受影响。"""
        self._history.clear()

    def record_window_history(self, pairs: list[tuple[str, str]]) -> None:
        """把合流窗口的自动整窗记录替换为逐 cue 记录。

        ``_translate_once`` 每次 ``chat`` 调用都会 append 一条“整窗”记录；
        合流窗口有 6 条 cue 时，直接把 12 条 maxlen 当成 72 条。这里弹出
        刚追加的整窗条目，再按 cue 写入，恢复 context_window 的 cue 语义。
        """
        if not pairs:
            return
        if self._history:
            self._history.pop()
        for source, translation in pairs:
            self._history.append((source, translation))

    def history_snapshot(self) -> tuple[tuple[str, str], ...]:
        """批次调用前的只读快照，供跨窗复制检查使用。"""
        return tuple(self._history)

    def restore_history(
        self, snapshot: tuple[tuple[str, str], ...]
    ) -> None:
        """把滑动上下文恢复到给定快照；保留 deque 的 maxlen 语义。

        审计重译前会 ``reset_history()``，结束后必须回到批次前语义，否则
        单句 ``[#N]`` 提示会污染后续窗口的参考上下文。``context_window=0``
        时只清空，不尝试 append（maxlen=0 的 deque 不该承载任何历史）。
        """
        maxlen = self._history.maxlen
        self._history.clear()
        if maxlen == 0:
            return
        for source, translation in snapshot:
            self._history.append((source, translation))

    def set_entity_engine(self, engine: DynamicEntityEngine | None) -> None:
        """绑定文档级实体索引；由 translate_srt 在单文件翻译期间管理。"""
        self._entity_engine = engine

    def set_lookahead(self, text: str) -> None:
        """设置下一窗口的只读后文参考；仅供当前一次批量调用理解上下文。

        后文不是待翻译输入，调用方必须在批次返回后立即 ``clear_lookahead``，
        否则会被带入下一窗甚至写进滑动历史。
        """
        self._lookahead = text.strip()

    def clear_lookahead(self) -> None:
        """清除后文参考；与 ``set_lookahead`` 成对使用。"""
        self._lookahead = ""

    def translate_filename(self, filename: str) -> str:
        """用翻译模型把文件名主体译成中文；不写入正文上下文。"""
        return self.translate_filename_result(filename).name

    def translate_filename_result(self, filename: str) -> NameTranslation:
        path = Path(filename)
        return self._translate_name_result(path.stem, path.suffix, filename)

    def translate_dirname(self, dirname: str) -> str:
        """翻译目录名整体，不把点号当扩展名剥离。"""
        return self.translate_dirname_result(dirname).name

    def translate_dirname_result(self, dirname: str) -> NameTranslation:
        return self._translate_name_result(dirname, "", dirname)

    def _translate_name_result(
        self, name: str, source_suffix: str, label: str
    ) -> NameTranslation:
        stem = name.strip()
        if not stem:
            raise TranslateError(f"名称为空，无法翻译：{label!r}")
        if stem.lower() in FILENAME_TECHNICAL_STEMS:
            return NameTranslation(stem)
        mask = _mask_name_literals(stem)
        if not _has_translatable_text(mask.text):
            return NameTranslation(stem)

        reason = ""
        for attempt in range(FILENAME_MAX_RETRIES + 1):
            result = self._call_name(mask.text, reason if attempt else "")
            choices = result.get("choices") or []
            finish_reason = ""
            if choices and isinstance(choices[0], dict):
                finish_reason = str(choices[0].get("finish_reason") or "")
            if finish_reason == "length":
                reason = "输出被截断"
                continue
            raw = extract_content(result)
            translated = sanitize_filename_stem(
                _strip_filename_explanation(raw), source_suffix
            )
            if not translated:
                reason = "结果为空"
                continue
            reason = _placeholder_failure(translated, mask.literals) or ""
            if reason:
                continue
            restored = _restore_literals(translated, mask.literals)
            restored = FILENAME_SYMBOL_CONNECTOR.sub(r"\1", restored)
            if len(restored) > FILENAME_MAX_CHARS:
                reason = "译名过长"
                continue
            if restored == stem and FILENAME_KANA.search(stem):
                reason = "未翻译日文假名"
                continue
            return NameTranslation(restored)
        return NameTranslation(stem, reason or "模型输出不可用")

    def _call_name(self, masked: str, correction: str) -> dict:
        user_content = FILENAME_PROMPT.format(name=masked)
        if correction:
            if "日文假名" in correction:
                user_content += (
                    "\n上一次结果没有翻译日文假名内容；"
                    "请把普通日文词翻译成中文，完整保留所有占位符，"
                    "只输出文件名主体。"
                )
            else:
                user_content += (
                    f"\n上一次结果不可用：{correction}。"
                    "请重新翻译，完整保留所有占位符，只输出文件名主体。"
                )
        messages: list[dict] = [
            {"role": "system", "content": FILENAME_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        return self._client.chat(
            messages,
            max_tokens=FILENAME_MAX_TOKENS,
            temperature=self._options.temperature,
        )

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #

    def _context_text(self) -> str:
        if not self._history:
            return "无前文，故事开端"
        limit = max(20, int(self._options.context_summary_chars))
        lines: list[str] = []
        for index, (source, translation) in enumerate(self._history, start=1):
            lines.append(f"[{index}] 原文：{_clip(source, limit)}")
            lines.append(f"    译文：{_clip(translation, limit)}")
        return "\n".join(lines)

    def _render(self, text: str) -> tuple[list[dict], int, int]:
        entity_context = ""
        if self._entity_engine is not None:
            entity_context = self._entity_engine.get_query_context(text)
        history = self._context_text()
        if entity_context and history == "无前文，故事开端":
            context = entity_context
        elif entity_context:
            context = entity_context + "\n\n【前文】\n" + history
        else:
            context = history
        if self._lookahead:
            context += (
                "\n\n【后文参考（仅用于理解指代/省略/语气，"
                "禁止翻译或写入当前编号）】\n"
                + self._lookahead
            )
        prompt = prompts.render(
            self._template,
            {"input": text, "context_history": context},
        )
        messages: list[dict] = []
        if self._system.strip():
            messages.append({"role": "system", "content": self._system})
        messages.append({"role": "user", "content": prompt})
        prompt_tokens = (
            estimate_tokens(self._system)
            + estimate_tokens(prompt)
            + PROMPT_OVERHEAD
        )
        return messages, prompt_tokens, estimate_tokens(text)

    def _translate_once(self, text: str, *, depth: int) -> str:
        options = self._options
        messages, prompt_tokens, input_tokens = self._render(text)
        desired = max(
            int(options.min_output_tokens),
            int(int(input_tokens) * float(options.output_ratio)),
        )
        while (
            self._history
            and prompt_tokens + desired
            > int(options.n_ctx) - int(options.prompt_safety_tokens)
        ):
            self._history.popleft()
            messages, prompt_tokens, input_tokens = self._render(text)

        try:
            budget = translation_budget(
                options.n_ctx,
                prompt_tokens,
                input_tokens,
                min_output_tokens=options.min_output_tokens,
                output_ratio=options.output_ratio,
                safety_tokens=options.prompt_safety_tokens,
            )
        except ValueError as exc:
            raise TranslateError(str(exc)) from exc

        started = time.perf_counter()
        result = self._client.chat(
            messages,
            max_tokens=budget.max_tokens,
            temperature=options.temperature,
        )
        elapsed_s = time.perf_counter() - started
        choices = result.get("choices") or []
        finish_reason = ""
        if choices and isinstance(choices[0], dict):
            finish_reason = str(choices[0].get("finish_reason") or "")
        if finish_reason == "length":
            if depth >= options.max_split_depth or len(text) <= 1:
                raise OutputTruncated(
                    f"输出达到长度上限，拆分 {options.max_split_depth} 层仍未完成"
                )
            parts = _split_for_retry(text)
            if len(parts) <= 1:
                raise OutputTruncated("输出达到长度上限，无法继续拆分")
            return "".join(
                self._translate_once(part, depth=depth + 1) for part in parts
            )

        raw = extract_content(result)
        translated = sanitize_translation_output(raw, text)
        raw_usage = result.get("usage")
        usage = raw_usage if isinstance(raw_usage, dict) else {}
        raw_timings = result.get("timings")
        timings = raw_timings if isinstance(raw_timings, dict) else {}
        tokens = (
            int(usage.get("completion_tokens") or 0)
            or int(timings.get("predicted_n") or 0)
            or estimate_tokens(translated)
        )
        server_tps = float(timings.get("predicted_per_second") or 0.0)
        if server_tps > 0:
            tps = server_tps
            tps_source = "server.timings"
        else:
            tps = tokens / elapsed_s if elapsed_s > 0 else 0.0
            tps_source = "client.usage" if usage else "estimate"
        self._history.append((text, translated))
        if self._on_stat is not None:
            self._on_stat(
                SegmentStat(
                    source=text,
                    translation=translated,
                    elapsed_s=elapsed_s,
                    tokens=tokens,
                    tps=tps,
                    tps_source=tps_source,
                )
            )
        return translated


def make_engine_chat(
    client: LlamaClient,
    prompt_dir: Path,
    system_text: str,
    *,
    n_ctx: int,
    context_window: int,
    context_summary_chars: int,
    min_output_tokens: int,
    output_ratio: float,
    prompt_safety_tokens: int,
    temperature: float,
    max_split_depth: int,
    on_stat: Callable[[SegmentStat], None] | None = None,
    task_template: str = "translate",
) -> Callable[[str], str]:
    """兼容入口：返回可调用的单段翻译函数。"""
    options = TranslationOptions(
        n_ctx=n_ctx,
        context_window=context_window,
        context_summary_chars=context_summary_chars,
        min_output_tokens=min_output_tokens,
        output_ratio=output_ratio,
        prompt_safety_tokens=prompt_safety_tokens,
        temperature=temperature,
        max_split_depth=max_split_depth,
    )
    return TranslationEngine(
        client,
        prompt_dir,
        system_text,
        options=options,
        on_stat=on_stat,
        task_template=task_template,
    )


def build_chat_for(
    config: AppConfig,
    model: ModelSpec,
    port: int,
    prompt_dir: Path,
    *,
    on_stat: Callable[[SegmentStat], None] | None = None,
) -> Callable[[str], str]:
    client = LlamaClient.for_model(config, model, port)
    profile = config.resolve_profile(model)
    settings = config.settings.translate
    return make_engine_chat(
        client,
        prompt_dir,
        prompts.system_prompt(prompt_dir),
        n_ctx=profile.ctx,
        context_window=settings.context_window,
        context_summary_chars=settings.context_summary_chars,
        min_output_tokens=settings.min_output_tokens,
        output_ratio=settings.output_ratio,
        prompt_safety_tokens=settings.prompt_safety_tokens,
        temperature=settings.temperature,
        max_split_depth=settings.max_split_depth,
        on_stat=on_stat,
    )
