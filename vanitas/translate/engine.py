"""核心上下文引擎：滑动视窗、预算门禁、截断二分重译、文件名翻译。

职责边界：一次 ``translate()`` 只负责“一段文本 → 一段译文”，
不认识 SRT、不认识文件、不做 cue 级后处理与审计（那些在 ``pipeline``）。
上下文是文档级可变状态，但所有权归单个 ``TranslationEngine`` 实例。
"""

from __future__ import annotations

import inspect
import time
from collections import deque
from pathlib import Path
from typing import Callable

from .. import prompts
from .cleaner import WINDOW_ID_NUMBER_RE, _clip, sanitize_translation_output
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
from ..quality import ContextRule, Glossary, load_context_rules
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


def next_window_id(text: str) -> int | None:
    """从带协议编号的输入推断“下一条”编号；无编号返回 ``None``。

    必须用窗口内**最大编号 + 1**，不能用编号个数：合流窗口的编号是文档级
    连续序号（第二窗从 7 开始），按个数会算出 7 并掐断合法首行。
    """
    numbers = [
        int(match.group(1))
        for match in WINDOW_ID_NUMBER_RE.finditer(str(text or ""))
    ]
    if not numbers:
        return None
    return max(numbers) + 1


# 连续空行是任何窗口协议下都不可能合法的形态，作为基础截断符始终挂载。
BASE_STOP_TOKENS = ("\n\n\n\n",)


def window_stop_tokens(text: str) -> list[str]:
    """构造服务端物理截断符；越界编号按“最大编号 + 1”动态生成。

    覆盖模型常见的协议变形（缺左括号、全角括号）；由推理端在 C++ 层停机，
    客户端就不会收到越界标签，也就不需要靠异常熔断整窗。
    """
    next_id = next_window_id(text)
    if next_id is None:
        return list(BASE_STOP_TOKENS)
    return [
        f"[#{next_id}]",
        f"#{next_id}]",
        f"【#{next_id}】",
        *BASE_STOP_TOKENS,
    ]


class StreamingAbortDetector:
    """针对微批次输出流实施物理早停，提前掐断连接释放 GPU。

    两条路径必须分开：
    - **硬熔断**（空行洪水/短语死锁/字符失控）：已产出的前缀确定不可用，
      抛 ``OutputTruncated``，由调用方整窗回退；
    - **软停机**（模型吐出“下一句”编号）：前缀本身可能已经完全正确，只把
      越界标记之后的尾巴截掉。服务端 stop 正常时这一步不会触发。

    所有判据都基于已收到的前缀，不预测未来，也不看语义——宁可漏停，
    不可误停（误停会白白多一次回退请求）。
    """

    def __init__(
        self, expected_next_id: int | None, max_chars: int = 4000
    ) -> None:
        self.expected_next_id = (
            int(expected_next_id) if expected_next_id is not None else None
        )
        self.max_chars = max(100, int(max_chars))
        self.buffer = ""
        self.aborted = False
        self.abort_reason = ""
        self.soft_stop = False
        self.soft_offset = 0

    def feed(self, delta: str) -> bool:
        """处理新 token 片段；返回 True 表示继续，False 表示应立刻断开。"""
        self.buffer += delta
        if len(self.buffer) > self.max_chars:
            return self._abort(f"输出字符数超出上限 {self.max_chars}")

        if self.expected_next_id is not None:
            limit_tag = f"[#{self.expected_next_id}]"
            position = self.buffer.find(limit_tag)
            if position >= 0:
                self.soft_stop = True
                self.soft_offset = position
                return False

        if "\n\n\n\n" in self.buffer or "\r\n\r\n\r\n\r\n" in self.buffer:
            return self._abort("检测到异常空行连续溢出")

        tail = self.buffer[-120:]
        for phrase_len in (12, 16, 20, 24):
            if len(tail) < phrase_len * 3:
                continue
            unit = tail[-phrase_len:]
            if tail.endswith(unit * 3):
                return self._abort(f"检测到短语死锁循环：{unit[:10]}...")
        return True

    def _abort(self, reason: str) -> bool:
        self.aborted = True
        self.abort_reason = reason
        return False


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
        self._prompt_dir = prompt_dir
        self._template = prompts.load_task(prompt_dir, task_template)
        self._task_template = task_template
        self._system = system_text
        self._options = options
        window = max(0, int(self._options.context_window))
        self._history: deque[tuple[str, str]] = deque(maxlen=window)
        self._entity_engine: DynamicEntityEngine | None = None
        self._lookahead = ""
        self._on_stat = on_stat
        # 客户端方法是否接受某个关键字（如 stop）的探测缓存：鸭子类型替身
        # 只实现 chat()，不能因为多传一个 stop 就把整条管线打穿。
        self._kwargs_support: dict[tuple[str, str], bool] = {}
        # P3 复审模板按需加载：默认关闭复审时不读文件，也不影响既有配置目录
        # （只带生产模板的隔离目录仍能正常跑翻译）。
        self._critique_template: str | None = None
        # 术语表只在 ASMR 任务加载（生产/实验模板都算）；其余任务保持空表。
        self._glossary = Glossary()
        if prompts.is_asmr_task(task_template):
            self._glossary = Glossary.load(
                prompt_dir.parent / "glossary" / "asmr.tsv"
            )
        # 上下文替换规则与模板目录同级；None 表示本配置目录没有规则表，
        # 由管线回落到仓库自带的默认表（自定义 --config-dir 仍能跑）。
        self._context_rules: tuple[ContextRule, ...] | None = None
        rules_path = prompt_dir.parent / "glossary" / "rules.tsv"
        if rules_path.exists():
            self._context_rules = load_context_rules(rules_path)
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
    def context_rules(self) -> tuple[ContextRule, ...] | None:
        """只读上下文规则；None 表示配置目录没有 rules.tsv，由管线用默认表。"""
        return self._context_rules

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

    def critique(self, text: str, *, max_tokens: int = 96) -> str:
        """P3 复审通道：独立模板、独立 token 预算，**绝不写滑动上下文**。

        与 ``translate()`` 的关键区别：这里不经过 ``_translate_once``，
        因此不 append ``_history``、不做上下文预算裁剪、也不被截断二分
        重译接管。模型输出为空或触顶时抛 ``EngineError``，由调用方降级
        保留初译。
        """
        template = self._critique_text()
        prompt = prompts.render(template, {"input": text})
        messages: list[dict] = []
        if self._system.strip():
            messages.append({"role": "system", "content": self._system})
        messages.append({"role": "user", "content": prompt})
        result = self._client.chat(
            messages,
            max_tokens=max_tokens,
            temperature=self._options.temperature,
        )
        choices = result.get("choices") or []
        if choices and isinstance(choices[0], dict):
            if str(choices[0].get("finish_reason") or "") == "length":
                raise OutputTruncated("复审输出达到长度上限，已保留初译")
        return extract_content(result)

    def _critique_text(self) -> str:
        """按需加载复审模板并缓存；与翻译模板分开，互不污染。"""
        if self._critique_template is None:
            self._critique_template = prompts.load_task(
                self._prompt_dir, "translate_critique"
            )
        return self._critique_template

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
        stop_tokens = (
            window_stop_tokens(text)
            if options.enable_server_stop_tokens
            else None
        )
        top_p = options.top_p
        stream_chat = getattr(self._client, "stream_chat", None)
        if callable(stream_chat):
            (
                raw,
                finish_reason,
                elapsed_s,
                tokens,
                tps,
                tps_source,
            ) = self._generate_stream(
                messages, budget, text, stop_tokens, top_p
            )
        else:
            # 鸭子类型的测试替身只有 chat()；保留阻塞路径，生产客户端恒走流式。
            (
                raw,
                finish_reason,
                elapsed_s,
                tokens,
                tps,
                tps_source,
            ) = self._generate_blocking(
                messages, budget, text, started, stop_tokens, top_p
            )

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

        translated = sanitize_translation_output(raw, text)
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

    def _supports_keyword(self, method_name: str, keyword: str) -> bool:
        """客户端方法是否显式接受关键字（或有 ``**kwargs``）；按方法缓存。"""
        cache_key = (method_name, keyword)
        cached = self._kwargs_support.get(cache_key)
        if cached is not None:
            return cached
        method = getattr(self._client, method_name, None)
        supported = False
        if callable(method):
            try:
                parameters = inspect.signature(method).parameters
            except (TypeError, ValueError):
                parameters = {}
            supported = keyword in parameters or any(
                item.kind is inspect.Parameter.VAR_KEYWORD
                for item in parameters.values()
            )
        self._kwargs_support[cache_key] = supported
        return supported

    def _stop_kwargs(
        self, method_name: str, stop: list[str] | None
    ) -> dict[str, list[str]]:
        """只在客户端真的支持时才带 stop，避免打穿鸭子类型替身。"""
        if not stop or not self._supports_keyword(method_name, "stop"):
            return {}
        return {"stop": list(stop)}

    def _top_p_kwargs(
        self, method_name: str, top_p: float | None
    ) -> dict[str, float]:
        """只在客户端支持且 top_p<1.0 时透传；1.0 表示不干预。"""
        if (
            top_p is None
            or top_p >= 1.0
            or not self._supports_keyword(method_name, "top_p")
        ):
            return {}
        return {"top_p": float(top_p)}

    def _generate_blocking(
        self,
        messages: list[dict],
        budget,
        text: str,
        started: float,
        stop: list[str] | None = None,
        top_p: float | None = None,
    ) -> tuple[str, str, float, int, float, str]:
        """阻塞路径：兼容只有 ``chat()`` 的鸭子类型客户端。"""
        result = self._client.chat(
            messages,
            max_tokens=budget.max_tokens,
            temperature=self._options.temperature,
            **self._stop_kwargs("chat", stop),
            **self._top_p_kwargs("chat", top_p),
        )
        elapsed_s = time.perf_counter() - started
        choices = result.get("choices") or []
        finish_reason = ""
        if choices and isinstance(choices[0], dict):
            finish_reason = str(choices[0].get("finish_reason") or "")
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
        return raw, finish_reason, elapsed_s, tokens, tps, tps_source

    def _generate_stream(
        self,
        messages: list[dict],
        budget,
        text: str,
        stop: list[str] | None = None,
        top_p: float | None = None,
    ) -> tuple[str, str, float, int, float, str]:
        """流式路径：边收边判，命中畸形立即掐断 Socket 释放 GPU。

        ``stream.close()`` 会把 ``GeneratorExit`` 抛回 ``chat_stream`` 的
        ``finally``，由底层 ``response.close()`` 真正断开 HTTP 连接；这是
        早停能省下算力的唯一原因，不能只 break 循环。越界编号是软停机：
        截掉越界标记之后的尾巴，保留前面已经合法的译文。
        """
        detector = StreamingAbortDetector(
            next_window_id(text),
            max_chars=max(4000, len(text) * 3),
        )
        started = time.perf_counter()
        parts: list[str] = []
        done_event: dict | None = None
        soft_stopped = False
        stream = self._client.stream_chat(
            messages,
            max_tokens=budget.max_tokens,
            temperature=self._options.temperature,
            **self._stop_kwargs("stream_chat", stop),
            **self._top_p_kwargs("stream_chat", top_p),
        )
        try:
            for event in stream:
                kind = event.get("type")
                if kind == "delta":
                    chunk = event.get("content") or ""
                    if not chunk:
                        continue
                    parts.append(chunk)
                    if not detector.feed(chunk):
                        if detector.soft_stop:
                            soft_stopped = True
                            break
                        raise OutputTruncated(
                            f"流式早停：{detector.abort_reason}"
                        )
                elif kind == "done":
                    done_event = event
        finally:
            closer = getattr(stream, "close", None)
            if callable(closer):
                closer()
        elapsed_s = time.perf_counter() - started
        raw = "".join(parts)
        if soft_stopped:
            raw = detector.buffer[: detector.soft_offset]
        if not raw.strip():
            raise EngineError("模型返回空内容")
        done = done_event or {}
        if soft_stopped:
            finish_reason = "stop"
        else:
            finish_reason = str(done.get("finish_reason") or "")
        tokens = int(done.get("tokens") or 0) or estimate_tokens(raw)
        tps = float(done.get("tps") or 0.0)
        tps_source = str(done.get("tps_source") or "estimate")
        return raw, finish_reason, elapsed_s, tokens, tps, tps_source



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
    enable_server_stop_tokens: bool = True,
    top_p: float = 0.85,
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
        enable_server_stop_tokens=enable_server_stop_tokens,
        top_p=top_p,
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
        enable_server_stop_tokens=settings.enable_server_stop_tokens,
        top_p=settings.top_p,
    )
