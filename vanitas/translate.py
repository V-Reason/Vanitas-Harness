"""批量翻译：.txt / .md / .srt。

- 文本按空行分段，段内超 chunk_chars 再按句末标点切，拼回时保留原空行结构；
- SRT 只翻译字幕文本，序号、时间轴、空行与换行风格原样保留；
- 任一段失败不中断，最终汇总失败段并由 CLI 返回非零退出码。
"""

from __future__ import annotations

import os
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import prompts
from .client import LlamaClient
from .config import AppConfig, ModelSpec, TranslateSettings
from .errors import EngineError, OutputTruncated, TranslateError
from .metrics import (
    PROMPT_OVERHEAD,
    SegmentStat,
    estimate_tokens,
    translation_budget,
)

SENTENCE_ENDERS = "。！？!?；;"
BLOCK_SEP = re.compile(r"(\r?\n[ \t]*\r?\n)")
SRT_TIME = re.compile(r"^\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}")
WINDOW_ID_RE = re.compile(r"^\[#(\d+)\]\s*(.*)$")
ASR_HALLUCINATION_PATTERNS = re.compile(
    r"(?:ご視聴ありがとうございました|チャンネル登録|高評価|ベルマーク|Thank you for watching|Please subscribe|Sous-titres)",
    re.IGNORECASE,
)
BOM = b"\xef\xbb\xbf"
FENCE_RE = re.compile(r"^```[^\n]*\n(.*?)\n```$", re.DOTALL)
QUOTE_PAIRS = {'"': '"', "'": "'", "“": "”", "‘": "’", "「": "」", "『": "』"}
QUOTE_CHARS = frozenset(QUOTE_PAIRS) | frozenset(QUOTE_PAIRS.values())
FILENAME_PROMPT = (
    "翻译下面的文件名主体。只翻译普通英文/日文词；中文、专有名词、缩写、"
    "型号和符号原样保留。不要解释、不要加引号、不要保留扩展名、不要换行。"
    "只输出结果。\n文件名：{name}"
)
FILENAME_SYSTEM_PROMPT = (
    "你是文件名翻译器，只输出译名主体。"
    "中文文件名必须原样照搬，不润色、不改写、不扩写；"
    "英文专有名词、缩写、型号、符号（如 AI、PDF、v2、NYC、C++、&、+、#、[]、【】）"
    "必须原样保留，不翻译、不展开、不音译；"
    "只翻译确实需要翻译的英文/日文普通词，禁止添加原文没有的词语或解释。"
    "示例：AI简报_v2 → AI简报_v2；NYC_2024 → NYC_2024；The Clock → 时钟。"
    "宁可保留原文，也不要改写专名或缩写。"
)
FILENAME_PROTECTED_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9+#&._-]*")
FILENAME_MAX_TOKENS = 96
FILENAME_MAX_CHARS = 80
FILENAME_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
FILENAME_PREFIX = re.compile(r"^(?:文件名|译名|翻译结果)\s*[:：]\s*")


@dataclass
class TranslationFailure:
    source: str
    segment: str
    reason: str


@dataclass
class TranslationOutcome:
    source: Path
    output: Path | None
    segments: int
    failures: list[TranslationFailure] = field(default_factory=list)


@dataclass(frozen=True)
class TranslationSource:
    """一个待翻译文件；relative_parent 用于保留层级，translated_stem 为译名。"""

    path: Path
    relative_parent: Path = field(default_factory=Path)
    translated_stem: str | None = None


@dataclass(frozen=True)
class TranslationPlan:
    source: TranslationSource
    out_dir: Path | None
    target: Path

    @property
    def label(self) -> str:
        parent = self.source.relative_parent
        if parent in (Path(), Path(".")):
            return self.source.path.name
        return (parent / self.source.path.name).as_posix()


@dataclass(frozen=True)
class NameTarget:
    path: Path
    is_dir: bool
    translated_name: str | None = None


@dataclass(frozen=True)
class NameRename:
    old: Path
    new: Path


@dataclass
class Subtitle:
    index: str
    timing: str
    text: str


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


@dataclass(frozen=True)
class TranslationOptions:
    n_ctx: int
    context_window: int
    context_summary_chars: int
    min_output_tokens: int
    output_ratio: float
    prompt_safety_tokens: int
    temperature: float
    max_split_depth: int


def _has_dialogue_quotes(text: str) -> bool:
    return any(char in QUOTE_CHARS for char in text)


def sanitize_translation_output(raw_text: str, source_text: str) -> str:
    """剥离模型额外加上的代码围栏/外层引号，同时保护原文对白引号。

    规则：
    - 先剥 Markdown 代码围栏（无论 source 是否有引号）；
    - source 含任何对白引号时，正文原样返回，不删任何引号；
    - source 无引号时，循环剥离成对的外层 ASCII/中文引号；不配对保留。
    """
    text = raw_text.strip()
    if not text:
        return text
    for _ in range(3):
        match = FENCE_RE.match(text)
        if not match:
            break
        text = match.group(1).strip()
    # 省略号/破折号已经承担句末语气，后面再追加句号属于重复标点。
    text = re.sub(r"(……|——)。", r"\1", text)
    if _has_dialogue_quotes(source_text):
        return text
    for _ in range(3):
        if len(text) < 2:
            break
        opening, closing = text[0], text[-1]
        if QUOTE_PAIRS.get(opening) != closing:
            break
        text = text[1:-1].strip()
    return text


def _protected_name_tokens(stem: str) -> list[str]:
    """找出必须逐字保留的缩写/型号/符号片段，供模型不听话时兜底。"""
    protected: list[str] = []
    for token in FILENAME_PROTECTED_TOKEN.findall(stem):
        letters = [char for char in token if char.isalpha()]
        uppercase = sum(1 for char in letters if char.isupper())
        has_internal_upper = any(char.isupper() for char in token[1:])
        if (
            any(char.isdigit() for char in token)
            or any(char in "+#&" for char in token)
            or uppercase >= 2
            or has_internal_upper
        ):
            protected.append(token)
    return protected


def sanitize_filename_stem(raw: str, source_suffix: str = "") -> str:
    """把模型输出收敛成安全的单段文件名（不含扩展名）。"""
    text = sanitize_translation_output(raw, "")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    text = FILENAME_PREFIX.sub("", lines[0])
    text = text.strip(" \t\"'“”‘’「」『』《》")
    if source_suffix and text.lower().endswith(source_suffix.lower()):
        text = text[: -len(source_suffix)]
        text = text.strip(" \t\"'“”‘’「」『』《》")
    text = FILENAME_INVALID_CHARS.sub("_", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"_+", "_", text)
    text = text.strip(" ._。！？!?；;，,、")
    if len(text) > FILENAME_MAX_CHARS:
        text = text[:FILENAME_MAX_CHARS].rstrip(" ._")
    return text


def _clip(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)] + "…"


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
        self._system = system_text
        self._options = options
        window = max(0, int(self._options.context_window))
        self._history: deque[tuple[str, str]] = deque(maxlen=window)
        self._on_stat = on_stat

    def __call__(self, text: str) -> str:
        return self.translate(text)

    def translate(self, text: str) -> str:
        return self._translate_once(text, depth=0)

    def reset_history(self) -> None:
        """开始新文档时清空滑动上下文，避免跨文件标题/人称串染。"""
        self._history.clear()

    def translate_filename(self, filename: str) -> str:
        """用翻译模型把文件名主体译成中文；不写入正文上下文。"""
        path = Path(filename)
        return self._translate_name(path.stem, path.suffix, filename)

    def translate_dirname(self, dirname: str) -> str:
        """翻译目录名整体，不把点号当扩展名剥离。"""
        return self._translate_name(dirname, "", dirname)

    def _translate_name(self, name: str, source_suffix: str, label: str) -> str:
        stem = name.strip()
        if not stem:
            raise TranslateError(f"名称为空，无法翻译：{label!r}")
        messages: list[dict] = [
            {"role": "system", "content": FILENAME_SYSTEM_PROMPT}
        ]
        messages.append(
            {
                "role": "user",
                "content": FILENAME_PROMPT.format(name=stem),
            }
        )
        result = self._client.chat(
            messages,
            max_tokens=FILENAME_MAX_TOKENS,
            temperature=self._options.temperature,
        )
        choices = result.get("choices") or []
        finish_reason = ""
        if choices and isinstance(choices[0], dict):
            finish_reason = str(choices[0].get("finish_reason") or "")
        if finish_reason == "length":
            raise TranslateError(f"名称翻译输出被截断：{label}")
        translated = sanitize_filename_stem(
            extract_content(result), source_suffix
        )
        missing = [
            token
            for token in _protected_name_tokens(stem)
            if token not in translated
        ]
        if missing:
            # 模型常把 NYC 音译成“纽约市”或把 AI 展开成“人工智能”；
            # 宁可不改，也不能产出错误专名。
            translated = sanitize_filename_stem(stem, source_suffix)
        if not translated:
            raise TranslateError(f"名称翻译结果为空：{label}")
        return translated

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
        prompt = prompts.render(
            self._template,
            {"input": text, "context_history": self._context_text()},
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


def chunk_text(text: str, limit: int) -> list[str]:
    if limit <= 0:
        raise TranslateError(f"chunk_chars 必须为正整数，实际为 {limit}")
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    rest = text
    while len(rest) > limit:
        window = rest[:limit]
        cut = 0
        for index in range(len(window) - 1, -1, -1):
            if window[index] in SENTENCE_ENDERS:
                cut = index + 1
                break
        if cut <= 0:
            space = window.rfind(" ")
            cut = space + 1 if space > 0 else limit
        chunks.append(rest[:cut])
        rest = rest[cut:]
    if rest:
        chunks.append(rest)
    return chunks


def _split_for_retry(text: str) -> list[str]:
    """优先在句末标点/空格二分；最差按中点切，保证所有字符不丢。"""
    if len(text) <= 1:
        return []
    parts = chunk_text(text, max(1, len(text) // 2))
    parts = [part for part in parts if part]
    if len(parts) >= 2:
        return parts
    middle = max(1, len(text) // 2)
    if middle >= len(text):
        return [text]
    return [text[:middle], text[middle:]]


def _translate_chunks(
    chat: Callable[[str], str],
    text: str,
    *,
    chunk_chars: int,
    source: str,
    label: str,
    failures: list[TranslationFailure],
) -> str | None:
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


def parse_srt(text: str, eol: str = "\n") -> list[tuple[Subtitle, str]]:
    """解析 SRT 字幕块：使用 splitlines() 兼容任意混合换行（CRLF/LF/CR），彻底阻断 0 段静默写回。"""
    blocks: list[tuple[Subtitle, str]] = []
    pieces = BLOCK_SEP.split(text)
    index = 0
    while index < len(pieces):
        content = pieces[index]
        separator = pieces[index + 1] if index + 1 < len(pieces) else ""
        index += 2
        if not content.strip():
            if blocks and separator:
                previous, previous_sep = blocks[-1]
                blocks[-1] = (previous, previous_sep + content + separator)
            continue

        # 核心修复点：使用 splitlines() 处理单块内部的混合换行，不再使用脆弱的 content.split(eol)
        raw_lines = [line.strip("\r\n") for line in content.splitlines()]
        if (
            len(raw_lines) >= 2
            and raw_lines[0].strip().isdigit()
            and SRT_TIME.match(raw_lines[1].strip())
        ):
            blocks.append(
                (
                    Subtitle(
                        index=raw_lines[0].strip(),
                        timing=raw_lines[1].strip(),
                        text=eol.join(raw_lines[2:]),
                    ),
                    separator,
                )
            )
        else:
            blocks.append((Subtitle(index="", timing="", text=content), separator))
    return blocks


def _normalize_eol(text: str, eol: str) -> str:
    """把任意换行统一为目标 eol，避免重组后残留混合行尾。"""
    target = eol or "\n"
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", target)


def rebuild_srt(blocks: list[tuple[Subtitle, str]], eol: str) -> str:
    target = eol or "\n"
    output: list[str] = []
    for subtitle, separator in blocks:
        if subtitle.index and subtitle.timing:
            text = _normalize_eol(subtitle.text, target)
            if text.strip():
                output.append(target.join([subtitle.index, subtitle.timing, text]))
            else:
                # 空正文不能再拼末尾换行，否则会与块分隔符叠加成三连换行，
                # 导致下一块被 parse_srt 误判为未识别内容。
                output.append(target.join([subtitle.index, subtitle.timing]))
        else:
            output.append(_normalize_eol(subtitle.text, target))
        output.append(_normalize_eol(separator, target))
    return "".join(output)


def split_prose(text: str) -> list[tuple[str, str]]:
    """返回 [(段落, 段落间分隔符)]；分隔符原样保留。"""
    pieces = BLOCK_SEP.split(text)
    paragraphs: list[tuple[str, str]] = []
    index = 0
    while index < len(pieces):
        content = pieces[index]
        separator = pieces[index + 1] if index + 1 < len(pieces) else ""
        index += 2
        paragraphs.append((content, separator))
    return paragraphs


def parse_window_response(
    raw_text: str, expected_ids: list[str]
) -> tuple[bool, dict[str, str], list[str]]:
    """解析带有 [#序号] 前缀的模型多行输出。

    返回：(是否完全对齐, {序号: 译文}, 缺失序号列表)。
    支持同一序号下多行字幕的自动合并（续行逻辑）。
    重复序号、额外序号与乱序都判为失败，交给调用方整体回退。
    """
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

    missing = [cid for cid in expected_ids if cid not in translations]
    if duplicated or missing or order != expected_ids:
        return False, {}, missing

    aligned_dict = {
        cid: "\n".join(translations[cid]).strip() for cid in expected_ids
    }
    return True, aligned_dict, []


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
) -> str:
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
    done = 0

    if window_size <= 1:
        for _block_index, subtitle in valid_items:
            if not subtitle.text.strip():
                # ASR 幻觉已置空：不调用模型，但仍计入已完成段，保持进度与 segments 一致
                done += 1
                if on_segment is not None:
                    on_segment(done, total)
                continue
            result = _translate_chunks(
                chat,
                subtitle.text,
                chunk_chars=chunk_chars,
                source=source,
                label=f"line {subtitle.index}",
                failures=failures,
            )
            if result is not None:
                subtitle.text = result
            # 失败时保留原文，继续下一条
            done += 1
            if on_segment is not None:
                on_segment(done, total)
        return rebuild_srt(blocks, eol)

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

    for window in windows:
        expected_ids = [item[1].index for item in window]
        fallback = len(set(expected_ids)) != len(expected_ids)
        if not fallback:
            batch_prompt = "\n".join(
                f"[#{item[1].index}] {item[1].text}" for item in window
            )
            try:
                batch_output = chat(batch_prompt)
                ok, aligned, _missing = parse_window_response(
                    batch_output, expected_ids
                )
                if ok:
                    for _block_index, subtitle in window:
                        subtitle.text = aligned[subtitle.index]
                else:
                    fallback = True
            except (EngineError, TranslateError):
                fallback = True

        if fallback:
            # 硬熔断：该窗口原地回退为逐句串行，绝不采用部分对齐结果
            for _block_index, subtitle in window:
                result = _translate_chunks(
                    chat,
                    subtitle.text,
                    chunk_chars=chunk_chars,
                    source=source,
                    label=f"line {subtitle.index}",
                    failures=failures,
                )
                if result is not None:
                    subtitle.text = result

        done += len(window)
        if on_segment is not None:
            on_segment(done, total)

    if filtered:
        # 被 ASR 硬清洗的空块不发请求，但要在末尾补齐进度，保证 done==total
        done += filtered
        if on_segment is not None:
            on_segment(done, total)

    return rebuild_srt(blocks, eol)


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


def _detect_eol(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def _read_text(path: Path) -> tuple[str, bool, str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise TranslateError(f"读取失败：{path}；{exc.strerror or exc}") from exc
    has_bom = raw.startswith(BOM)
    try:
        text = raw[len(BOM):].decode("utf-8") if has_bom else raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TranslateError(
            f"文件不是 UTF-8：{path}；第 {exc.start} 字节，请先转码"
        ) from exc
    return text, has_bom, _detect_eol(text)


def output_path(
    source: Path,
    out_dir: Path | None,
    suffix: str,
    stem: str | None = None,
) -> Path:
    if stem is not None:
        base = sanitize_filename_stem(stem, source.suffix) or source.stem
    else:
        base = source.stem
    name = f"{base}{suffix}{source.suffix}"
    return (out_dir / name) if out_dir is not None else (source.parent / name)


def _path_key(path: Path) -> str:
    """跨 Windows 大小写/符号链接比较路径；目标文件通常尚不存在。"""
    return os.path.normcase(os.path.realpath(os.path.abspath(str(path))))


def _is_generated_output(path: Path, output_suffix: str) -> bool:
    suffix = (output_suffix or "").lower()
    return bool(suffix) and path.stem.lower().endswith(suffix)


def _is_excluded_dir(candidate: Path, excluded_key: str | None) -> bool:
    """仅剪掉 --out-dir 本身，避免把上次产物再次当输入。"""
    if not excluded_key:
        return False
    return _path_key(candidate) == excluded_key


def collect_translate_inputs(
    inputs: list[Path],
    settings: TranslateSettings,
    *,
    exclude_dir: Path | None = None,
) -> list[TranslationSource]:
    """把文件/目录输入展开为去重后的文件列表。

    - 目录递归遍历，只收 settings.source_suffixes，显式文件不按后缀过滤；
    - 目录中的 ``*.zh.md`` 等已有产物跳过，避免自我递归；
    - 同名文件按真实路径去重，首次出现的目录层级决定输出层级；
    - 空目录与不可遍历目录在启动模型前报错；显式文件仍由 translate_file 判定。
    """
    allowed = frozenset(item.lower() for item in settings.source_suffixes)
    excluded_key = _path_key(exclude_dir) if exclude_dir is not None else None
    collected: list[TranslationSource] = []
    seen: set[str] = set()

    def add(path: Path, relative_parent: Path) -> bool:
        key = _path_key(path)
        if key in seen:
            return False
        seen.add(key)
        collected.append(TranslationSource(path=path, relative_parent=relative_parent))
        return True

    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            matched = 0
            errors: list[OSError] = []

            def on_error(exc: OSError) -> None:
                errors.append(exc)

            for root, dirs, files in os.walk(
                path, topdown=True, onerror=on_error, followlinks=False
            ):
                root_path = Path(root)
                dirs[:] = sorted(
                    (
                        name
                        for name in dirs
                        if not _is_excluded_dir(root_path / name, excluded_key)
                    ),
                    key=str.lower,
                )
                files.sort(key=str.lower)
                for name in files:
                    candidate = root_path / name
                    if candidate.suffix.lower() not in allowed:
                        continue
                    if _is_generated_output(candidate, settings.output_suffix):
                        continue
                    if not candidate.is_file():
                        continue
                    matched += 1
                    add(candidate, candidate.parent.relative_to(path))
            if errors:
                first = errors[0]
                detail = first.strerror or str(first)
                raise TranslateError(
                    f"遍历目录失败：{first.filename or path}；{detail}"
                ) from first
            if matched == 0:
                suffixes = "、".join(settings.source_suffixes)
                raise TranslateError(
                    f"目录中没有可翻译文件：{path}；支持：{suffixes}"
                )
        else:
            # 显式文件保持旧契约：存在性、后缀与可读性仍由 translate_file 判定。
            # 这里不能提前拒绝，否则调用方替换/延迟写盘的场景会被误伤。
            add(path, Path())

    if not collected:
        raise TranslateError("没有可翻译的输入")
    return collected


def collect_name_targets(inputs: list[Path]) -> list[NameTarget]:
    """收集只改名模式的目标：显式文件本身 + 目录下的文件/子目录。

    传入的根目录本身不重命名，避免把用户当前目录或仓库根整体改掉；
    隐藏文件和隐藏目录不进入递归，防止误改 ``.git`` 等。
    """
    targets: list[NameTarget] = []
    seen: set[str] = set()

    def add(path: Path, is_dir: bool) -> None:
        key = _path_key(path)
        if key in seen:
            return
        seen.add(key)
        targets.append(NameTarget(path=path, is_dir=is_dir))

    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            found = 0
            errors: list[OSError] = []

            def on_error(exc: OSError) -> None:
                errors.append(exc)

            for root, dirs, files in os.walk(
                path, topdown=True, onerror=on_error, followlinks=False
            ):
                root_path = Path(root)
                dirs[:] = sorted(
                    (name for name in dirs if not name.startswith(".")),
                    key=str.lower,
                )
                files = sorted(
                    (name for name in files if not name.startswith(".")),
                    key=str.lower,
                )
                for name in dirs:
                    add(root_path / name, True)
                    found += 1
                for name in files:
                    child = root_path / name
                    if child.is_file() or child.is_symlink():
                        add(child, False)
                        found += 1
            if errors:
                first = errors[0]
                detail = first.strerror or str(first)
                raise TranslateError(
                    f"遍历目录失败：{first.filename or path}；{detail}"
                ) from first
            if found == 0:
                raise TranslateError(f"目录中没有可重命名的文件或子目录：{path}")
        elif path.is_file() or path.is_symlink():
            add(path, False)
        else:
            raise TranslateError(f"路径不存在或不是文件/目录：{path}")

    if not targets:
        raise TranslateError("没有可重命名的文件或目录")
    return targets


def plan_name_renames(targets: list[NameTarget]) -> list[NameRename]:
    """生成并校验只改名计划；不触碰文件内容。"""
    renames: list[NameRename] = []
    new_keys: dict[str, NameRename] = {}
    old_keys = {_path_key(target.path) for target in targets}
    moved_old: set[str] = set()

    for target in targets:
        if target.translated_name is None:
            raise TranslateError(f"缺少译名：{target.path}")
        old = target.path
        if target.is_dir:
            new_name = sanitize_filename_stem(target.translated_name, "")
        else:
            base = sanitize_filename_stem(target.translated_name, old.suffix)
            new_name = f"{base}{old.suffix}" if base else ""
        if not new_name:
            raise TranslateError(f"译名为空或不可用：{old}")
        if new_name == old.name:
            continue
        new_path = old.with_name(new_name)
        key = _path_key(new_path)
        if key in new_keys:
            raise TranslateError(
                f"译名冲突：{new_keys[key].old} 与 {old} 都会变成 {new_path}"
            )
        rename = NameRename(old=old, new=new_path)
        new_keys[key] = rename
        moved_old.add(_path_key(old))
        renames.append(rename)

    for rename in renames:
        key = _path_key(rename.new)
        if key in old_keys and key not in moved_old:
            raise TranslateError(
                f"译名会占用未移动的源名称：{rename.old} → {rename.new}；"
                "请调整译名或先移动同名目标"
            )
        if rename.new.exists() and key not in moved_old:
            raise TranslateError(
                f"目标名称已存在：{rename.new}；请调整译名或先移动同名目标"
            )
    return renames


def _rename_temp(path: Path) -> Path:
    for _ in range(10):
        candidate = path.with_name(f".vanitas-rename-{uuid.uuid4().hex}")
        if not candidate.exists():
            return candidate
    raise TranslateError(f"无法生成临时文件名：{path}")


def apply_name_renames(renames: list[NameRename]) -> None:
    """按深度从深到浅、每层两阶段重命名，支持交换与纯大小写变化。"""
    groups: dict[int, list[NameRename]] = {}
    for rename in renames:
        groups.setdefault(len(rename.old.parts), []).append(rename)

    for depth in sorted(groups, reverse=True):
        group = sorted(groups[depth], key=lambda item: str(item.old).lower())
        staged: list[tuple[NameRename, Path]] = []
        try:
            for rename in group:
                temp = _rename_temp(rename.old)
                rename.old.rename(temp)
                staged.append((rename, temp))
        except OSError as exc:
            for rename, temp in reversed(staged):
                try:
                    temp.rename(rename.old)
                except OSError:
                    pass
            raise TranslateError(
                f"重命名失败：{exc.filename or exc}；已尝试回滚当前层"
            ) from exc

        completed: list[tuple[NameRename, Path]] = []
        try:
            for rename, temp in staged:
                temp.rename(rename.new)
                completed.append((rename, temp))
        except OSError as exc:
            for rename, temp in reversed(completed):
                try:
                    rename.new.rename(temp)
                except OSError:
                    pass
            for rename, temp in reversed(staged):
                try:
                    if temp.exists():
                        temp.rename(rename.old)
                except OSError:
                    pass
            raise TranslateError(
                f"重命名失败：{exc.filename or exc}；已尝试回滚当前层"
            ) from exc


def plan_translation_outputs(
    sources: list[TranslationSource],
    settings: TranslateSettings,
    out_dir: Path | None,
    *,
    output_suffix: str | None = None,
) -> list[TranslationPlan]:
    """计算输出路径并在写盘前拦截冲突，避免静默覆盖。

    --out-dir 存在时，目录输入保留相对层级；直接指定的文件仍平铺在输出根。
    output_suffix 为 None 时使用 settings.output_suffix；传空串可用于“译名不加 .zh”。
    """
    suffix = settings.output_suffix if output_suffix is None else output_suffix
    plans: list[TranslationPlan] = []
    targets: dict[str, TranslationSource] = {}
    source_keys = {_path_key(item.path) for item in sources}

    for source in sources:
        target_dir = out_dir / source.relative_parent if out_dir is not None else None
        target = output_path(
            source.path,
            target_dir,
            suffix,
            source.translated_stem,
        )
        key = _path_key(target)
        previous = targets.get(key)
        if previous is not None:
            raise TranslateError(
                f"输出路径冲突：{previous.path} 与 {source.path} 都会写到 {target}；"
                "请调整 --out-dir 或输入目录"
            )
        if key in source_keys:
            raise TranslateError(
                f"输出会覆盖输入文件：{source.path} → {target}；"
                "请更换 --out-dir 或输出后缀"
            )
        targets[key] = source
        plans.append(
            TranslationPlan(source=source, out_dir=target_dir, target=target)
        )
    return plans


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
) -> TranslationOutcome:
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
        source=source, output=target, segments=segments, failures=failures
    )
    if failures:
        # 输出已写出（失败段保留原文），由调用方汇总并返回非零
        return outcome
    return outcome


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
