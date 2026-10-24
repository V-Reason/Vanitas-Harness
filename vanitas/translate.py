"""批量翻译：.txt / .md / .srt。

- 文本按空行分段，段内超 chunk_chars 再按句末标点切，拼回时保留原空行结构；
- SRT 只翻译字幕文本，序号、时间轴、空行与换行风格原样保留；
- 任一段失败不中断，最终汇总失败段并由 CLI 返回非零退出码。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import prompts
from .config import AppConfig, ModelSpec
from .engine import EngineClient
from .errors import EngineError, TranslateError

SENTENCE_ENDERS = "。！？!?；;"
BLOCK_SEP = re.compile(r"(\r?\n[ \t]*\r?\n)")
SRT_TIME = re.compile(r"^\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}")
BOM = b"\xef\xbb\xbf"


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
        raise EngineError("模型只返回了 reasoning_content，content 为空（思考未关闭？）")
    if not content:
        raise EngineError("模型返回空内容")
    return content


def make_engine_chat(
    client: EngineClient, prompt_dir: Path, system_text: str
) -> Callable[[str], str]:
    """把「一段文本」映射为「一段译文」；超时/重试由 EngineClient 负责。"""
    template = prompts.load_task(prompt_dir, "translate")

    def chat(text: str) -> str:
        prompt = prompts.render(template, {"input": text})
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": prompt},
        ]
        max_tokens = max(256, min(2048, len(text) * 2))
        return extract_content(
            client.chat(messages, max_tokens=max_tokens, temperature=0.2)
        )

    return chat


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


def parse_srt(text: str, eol: str) -> list[tuple[Subtitle, str]]:
    """返回 [(字幕块, 块后分隔符)]；无法识别的块按原文保留。"""
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
        lines = content.split(eol)
        if len(lines) >= 2 and lines[0].strip().isdigit() and SRT_TIME.match(lines[1].strip()):
            blocks.append(
                (
                    Subtitle(
                        index=lines[0].strip(),
                        timing=lines[1],
                        text=eol.join(lines[2:]),
                    ),
                    separator,
                )
            )
        else:
            blocks.append((Subtitle(index="", timing="", text=content), separator))
    return blocks


def rebuild_srt(blocks: list[tuple[Subtitle, str]], eol: str) -> str:
    output: list[str] = []
    for subtitle, separator in blocks:
        if subtitle.index and subtitle.timing:
            text = subtitle.text.replace("\r\n", "\n").replace("\n", eol)
            output.append(eol.join([subtitle.index, subtitle.timing, text]))
        else:
            output.append(subtitle.text)
        output.append(separator)
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


def translate_srt(
    chat: Callable[[str], str],
    text: str,
    *,
    chunk_chars: int,
    source: str,
    eol: str,
    failures: list[TranslationFailure],
) -> str:
    blocks = parse_srt(text, eol)
    for position, (subtitle, _separator) in enumerate(blocks, start=1):
        if not subtitle.text.strip() or not subtitle.index:
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
    return rebuild_srt(blocks, eol)


def translate_prose(
    chat: Callable[[str], str],
    text: str,
    *,
    chunk_chars: int,
    source: str,
    failures: list[TranslationFailure],
) -> str:
    output: list[str] = []
    for position, (paragraph, separator) in enumerate(split_prose(text), start=1):
        match = re.match(r"^(\s*)(.*?)(\s*)$", paragraph, re.DOTALL)
        assert match is not None
        leading, core, trailing = match.groups()
        if not core:
            output.append(paragraph)
        else:
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
    return "".join(output)


def _detect_eol(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def _read_text(path: Path) -> tuple[str, bool, str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise TranslateError(f"读取失败：{path}（{exc.strerror or exc}）") from exc
    has_bom = raw.startswith(BOM)
    try:
        text = raw[len(BOM):].decode("utf-8") if has_bom else raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TranslateError(
            f"文件不是 UTF-8：{path}（第 {exc.start} 字节；请先转码）"
        ) from exc
    return text, has_bom, _detect_eol(text)


def output_path(source: Path, out_dir: Path | None, suffix: str) -> Path:
    name = f"{source.stem}{suffix}{source.suffix}"
    return (out_dir / name) if out_dir is not None else (source.parent / name)


def translate_file(
    chat: Callable[[str], str],
    source: Path,
    settings,
    *,
    out_dir: Path | None = None,
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
        segments = sum(1 for subtitle, _ in blocks if subtitle.index)
        translated = translate_srt(
            chat,
            text,
            chunk_chars=settings.chunk_chars,
            source=str(source),
            eol=eol,
            failures=failures,
        )
    else:
        segments = sum(1 for paragraph, _ in split_prose(text) if paragraph.strip())
        translated = translate_prose(
            chat,
            text,
            chunk_chars=settings.chunk_chars,
            source=str(source),
            failures=failures,
        )

    target = output_path(source, out_dir, settings.output_suffix)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        data = translated.encode("utf-8")
        target.write_bytes(BOM + data if has_bom else data)
    except OSError as exc:
        raise TranslateError(f"写入失败：{target}（{exc.strerror or exc}）") from exc

    outcome = TranslationOutcome(
        source=source, output=target, segments=segments, failures=failures
    )
    if failures:
        # 输出已写出（失败段保留原文），由调用方汇总并返回非零
        return outcome
    return outcome


def build_chat_for(
    config: AppConfig, model: ModelSpec, port: int, prompt_dir: Path
) -> Callable[[str], str]:
    client = EngineClient.for_model(config, model, port)
    return make_engine_chat(client, prompt_dir, prompts.system_prompt(prompt_dir))
