"""文本格式协议：SRT 结构解析/重建、EOL 归一化、纯文本分段与分块。

职责边界：只处理“字节解码后的字符串 ↔ 结构”这一段确定性转换，
不做任何质量判断、不调用模型、不碰文件系统。
"""

from __future__ import annotations

import re

from ..errors import TranslateError
from .types import Subtitle

SENTENCE_ENDERS = "。！？!?；;"
BLOCK_SEP = re.compile(r"(\r?\n[ \t]*\r?\n)")
SRT_TIME = re.compile(r"^\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}")


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


def _detect_eol(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


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
