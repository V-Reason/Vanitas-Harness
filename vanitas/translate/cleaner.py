"""确定性文本清洗与写回硬校验（0 Token，不调用模型）。

职责边界：这里只做“给定源文/译文 → 收敛后的译文”的纯函数变换，
不认识文件、不解析 SRT 结构、不持有任何状态。凡是需要模型或需要
窗口上下文的判断，都不属于本模块。

``WINDOW_ID_RE`` / ``WINDOW_ID_ANY_RE`` 定义在本层而非 ``window.py``：
``postprocess_cue_text`` 也必须剥 ``[#N]``，而 ``window.py`` 反过来要用
它们解析编号；定义在较低层可保证两个使用者共享同一份 pattern，避免
两处规则漂移。
"""

from __future__ import annotations

import re

# 连续空行（兼容 CRLF/CR）：“换行 + 空白 + 换行”才算空行。SRT 用空行分块，
# 正文里的空行会被重新解析成额外 block，直接破坏 cue 数/序号/时间轴；
# 折叠与写回校验共用同一 pattern，避免两处规则漂移。
BLANK_LINE_RUN_RE = re.compile(r"(?:\r\n|\r|\n)(?:[^\S\r\n]*(?:\r\n|\r|\n))+")

WINDOW_ID_RE = re.compile(r"^\[#(\d+)\]\s*(.*)$")
WINDOW_ID_ANY_RE = re.compile(r"\s*\[#\d+\]\s*")

FENCE_RE = re.compile(r"^```[^\n]*\n(.*?)\n```$", re.DOTALL)
QUOTE_PAIRS = {'"': '"', "'": "'", "“": "”", "‘": "’", "「": "」", "『": "』"}
QUOTE_CHARS = frozenset(QUOTE_PAIRS) | frozenset(QUOTE_PAIRS.values())

TRADITIONAL_TO_SIMPLIFIED = {
    "妳": "你", "話": "话", "會": "会", "個": "个", "這": "这",
    "們": "们", "來": "来", "時": "时", "說": "说", "現": "现",
    "實": "实", "體": "体", "為": "为", "與": "与", "從": "从",
    "對": "对", "開": "开", "關": "关", "頭": "头", "間": "间",
    "問": "问", "題": "题", "點": "点", "還": "还", "麼": "么",
    "樣": "样", "過": "过", "讓": "让", "給": "给", "帶": "带",
    "嗎": "吗", "讀": "读", "寫": "写", "聽": "听", "覺": "觉",
    "愛": "爱", "戀": "恋", "臉": "脸", "頰": "颊", "專": "专",
    "輯": "辑", "潔": "洁", "謝": "谢", "請": "请", "誰": "谁",
    "後": "后", "裡": "里", "裏": "里", "氣": "气", "溫": "温",
    "觸": "触", "撫": "抚", "動": "动", "變": "变", "極": "极",
    "樂": "乐", "響": "响", "臺": "台", "灣": "湾",
}
TRADITIONAL_TO_SIMPLIFIED_MAP = str.maketrans(TRADITIONAL_TO_SIMPLIFIED)
TRADITIONAL_CHARS = frozenset(TRADITIONAL_TO_SIMPLIFIED)

SOURCE_HAS_BRACKETS_RE = re.compile(r"[（(【\[]")
STAGING_BRACKETS_RE = re.compile(
    r"[（(](?:在)?(?:[左右双]?耳[边後后畔旁]|轻抚|抚摸|轻吻|深吻|轻舔|亲吻|"
    r"衣物摩擦声|呼吸声)[^）)]*[）)]"
)
REPEATED_PUNCT_RE = re.compile(r"([…—~～♡❤♪])。")
LEADING_INVALID_PUNCT_RE = re.compile(r"^[，、；：。！？!?\s]+")

UNTRANSLATED_KANA_RE = re.compile(r"[\u3040-\u30ff]")
ARTIFACT_LATIN_RE = re.compile(r"(?:SEXPAR|SEXXPAR|TRACK)", re.IGNORECASE)
SLASH_ALTERNATIVE_RE = re.compile(
    r"[\u4e00-\u9fffA-Za-z]/[\u4e00-\u9fffA-Za-z]"
)
LATIN_TOKEN_RE = re.compile(r"[A-Za-z]{3,}")
LATIN_ALLOW = frozenset({"ASMR", "NG", "BGM"})


def _has_dialogue_quotes(text: str) -> bool:
    return any(char in QUOTE_CHARS for char in text)


def sanitize_translation_output(raw_text: str, source_text: str) -> str:
    """剥离模型额外加上的代码围栏/外层引号，同时保护原文对白引号。

    规则：
    - 先剥 Markdown 代码围栏（无论 source 是否有引号）；
    - source 含任何对白引号时，正文原样返回，不删任何引号；
    - source 无引号时，循环剥离成对的外层 ASCII/中文引号；不配对保留。
    - 不在这里删除 ``[#N]``：批次响应必须保留窗口编号，编号由调用方在
      cue 边界用 ``strip_window_id_prefix`` / 后处理清理。
    """
    text = raw_text.strip()
    if not text:
        return text
    for _ in range(3):
        match = FENCE_RE.match(text)
        if not match:
            break
        text = match.group(1).strip()
    text = text.lstrip("，、；：").lstrip()
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


def repair_slash_alternatives(text: str, *, fallback: bool = False) -> str:
    """把“日文/中文”或“Track/中文”式斜杠并列收敛到唯一中文项。

    两边都是中文（先生/女士）或都是普通拉丁词时默认不猜，交给写回校验/QC。
    ``fallback=True`` 只用于硬校验失败后的兜底：仍无唯一中文项时取第一个
    非空项，避免把整句原样写回。正常后处理仍保持保守行为。
    """
    if "/" not in text:
        return text
    parts = [part.strip() for part in text.split("/")]
    if len(parts) < 2 or not all(parts):
        return text
    clean = [
        part
        for part in parts
        if not UNTRANSLATED_KANA_RE.search(part)
        and not ARTIFACT_LATIN_RE.search(part)
    ]
    if len(clean) == 1:
        return clean[0]
    if fallback and clean:
        return clean[0]
    return text


def _collapse_blank_lines(text: str) -> str:
    """把连续空行折叠成单个换行；多行字幕的合法单换行原样保留。"""
    return BLANK_LINE_RUN_RE.sub("\n", text)


def postprocess_cue_text(zh_text: str, src_text: str) -> str:
    """cue 级确定性清洗：空行折叠、简繁、行首标点、重复句号、未授权动作括号。

    只改字幕正文，不碰序号/时间轴；不调用模型。空行折叠在去 ``[#N]`` 之后、
    逐行处理之前完成：模型常输出 ``A\\n\\nB``，而 SRT 以空行分块，写回会多出
    一个正文块；多行字幕内部的单换行必须保留。清洗结果为空时保留清洗前
    文本，避免把非空原文静默写成空字幕；这类残留交给 QC 提示。
    """
    original = zh_text.strip()
    if not original:
        return ""

    text = _collapse_blank_lines(WINDOW_ID_ANY_RE.sub("", zh_text)).strip()
    if not text:
        return original
    text = text.translate(TRADITIONAL_TO_SIMPLIFIED_MAP)
    if not SOURCE_HAS_BRACKETS_RE.search(src_text):
        stripped = STAGING_BRACKETS_RE.sub("", text)
        if stripped.strip():
            text = stripped

    lines = [
        LEADING_INVALID_PUNCT_RE.sub("", line, count=1)
        for line in text.splitlines()
    ]
    text = "\n".join(lines).strip()
    text = REPEATED_PUNCT_RE.sub(r"\1", text).strip()
    text = repair_slash_alternatives(text)
    return text or original


def validate_cue_translation(text: str) -> str | None:
    """写回前的硬校验；返回原因，None 表示可写回。

    只拒绝确定会破坏中文字幕的伪影：空行、假名、SEXPAR 类臆造词、中日文斜杠
    并列。空行在 ``postprocess_cue_text`` 之后仍要再拦一次：任何绕过后处理的
    写回路径都会把单条 cue 拆成两个 SRT block，破坏 cue 数/序号/时间轴。
    其他拉丁词（曲名/人名）交给 QC 的 actionable 检查，避免误杀。
    """
    if BLANK_LINE_RUN_RE.search(text):
        return "译文含空行"
    if UNTRANSLATED_KANA_RE.search(text):
        return "译文残留日文假名"
    if ARTIFACT_LATIN_RE.search(text):
        return "译文含臆造拉丁词"
    if SLASH_ALTERNATIVE_RE.search(text):
        return "译文含中日文斜杠并列"
    return None


def salvage_slash_translation(text: str) -> str:
    """硬校验失败后的斜杠兜底：优先唯一中文项，否则取第一个非空项。

    只在 ``validate_cue_translation`` 已判定失败后调用；正常路径仍走
    ``postprocess_cue_text`` 的保守规则，不改变“先生/女士不猜”的契约。
    """
    if "/" not in text:
        return text
    return repair_slash_alternatives(text, fallback=True)


def _clip(text: str, limit: int) -> str:
    """把一段文本压成单行并截断到 limit；超长以省略号收尾。"""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)] + "…"
