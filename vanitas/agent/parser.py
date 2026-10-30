"""工具调用解析：从模型输出提取 `<tool_call>` / 裸 JSON / `<function>` 方言。

本模块只负责字符串 → 动作 dict，不做审批、不执行、不感知 UI。
"""
from __future__ import annotations

import json
import re

from .actions import AgentAction, _action_from_payload


_JSON_PATH_VALUE_RE = re.compile(
    r'("(?:path|src|dst|source|dest|destination|file|files|command)"\s*:\s*")'
    r'((?:[^"\\]|\\.)*)(")',
    re.DOTALL,
)
_TOOL_CALL_RE = re.compile(
    r"<tool_call\s*>\s*(.*?)\s*</tool_call\s*>", re.DOTALL | re.IGNORECASE
)
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
FUNCTION_OPEN = "<function"
FUNCTION_CLOSE = "</function>"
FUNCTION_RESULTS_OPEN = "<function_results"
FUNCTION_RESULTS_CLOSE = "</function_results>"
_FUNCTION_XML_RE = re.compile(
    r'<function\s+name="(?P<name>[^"]+)"\s*>(?P<body>.*?)</function>',
    re.DOTALL | re.IGNORECASE,
)
_PARAM_TAG_RE = re.compile(
    r'<param\s+name="(?P<pname>[^"]+)"\s*>(?:<!\[CDATA\[(?P<cdata>.*?)\]\]>|(?P<plain>.*?))</param>',
    re.DOTALL | re.IGNORECASE,
)
_JSON_DECODER = json.JSONDecoder(strict=False)
_HEX = frozenset("0123456789abcdefABCDEF")
_VALID_JSON_ESCAPES = frozenset('"\\/bfnrt')
def _escape_path_backslashes(value: str) -> str:
    """把路径字段里的 Windows 单反斜杠补成合法 JSON 转义。

    模型经常输出 `T:\\TMP\\test`（源码里只有一个反斜杠），而 JSON 会把
    `\\T` 判为非法；`\\t` 又会被误解析成制表符。路径字段按“字面反斜杠”
    处理：奇数个反斜杠补一个，偶数个保持不变（已经转义过的路径不重复加倍）。
    """
    pieces: list[str] = []
    index = 0
    while index < len(value):
        if value[index] != "\\":
            pieces.append(value[index])
            index += 1
            continue
        start = index
        while index < len(value) and value[index] == "\\":
            index += 1
        run = index - start
        following = value[index] if index < len(value) else ""
        if following == '"':
            # 保留 \"，避免把路径/命令里的合法引号转义拆坏。
            pieces.append("\\" * run)
        else:
            pieces.append("\\" * (run if run % 2 == 0 else run + 1))
    return "".join(pieces)


def _repair_path_escapes(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        return (
            match.group(1)
            + _escape_path_backslashes(match.group(2))
            + match.group(3)
        )

    return _JSON_PATH_VALUE_RE.sub(replace, text)


def _repair_invalid_json_escapes(text: str) -> str:
    """兜底：把字符串中仍非法的单反斜杠补齐；不改变对象结构。"""
    pieces: list[str] = []
    index = 0
    in_string = False
    while index < len(text):
        char = text[index]
        if char == '"':
            pieces.append(char)
            in_string = not in_string
            index += 1
            continue
        if not in_string or char != "\\":
            pieces.append(char)
            index += 1
            continue

        start = index
        while index < len(text) and text[index] == "\\":
            index += 1
        run = index - start
        following = text[index] if index < len(text) else ""
        valid_unicode = (
            following == "u"
            and index + 4 < len(text)
            and all(item in _HEX for item in text[index + 1:index + 5])
        )
        if following == '"' and run % 2 == 1:
            # 最后一个反斜杠转义引号，跳过引号，避免把字符串提前闭合。
            pieces.append("\\" * run + '"')
            index += 1
        elif following in _VALID_JSON_ESCAPES or valid_unicode:
            pieces.append("\\" * run)
        else:
            pieces.append("\\" * (run if run % 2 == 0 else run + 1))
    return "".join(pieces)


def _json_loads(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 模型有时把 content 中的换行直接写进 JSON 字符串；允许控制字符，
        # 仍由后续强类型校验约束字段。
        return json.loads(text, strict=False)


def _json_variants(text: str) -> list[str]:
    """按“原样 → 路径字段修复 → 全局转义兜底”的顺序给出候选文本。"""
    variants = [text]
    repaired = _repair_path_escapes(text)
    if repaired not in variants:
        variants.append(repaired)
    repaired_all = _repair_invalid_json_escapes(repaired)
    if repaired_all not in variants:
        variants.append(repaired_all)
    return variants


def _load_action_json(text: str):
    """先按标准 JSON 解析，失败后逐级修复路径字段与非法转义。"""
    first_error: json.JSONDecodeError | None = None
    for candidate in _json_variants(text):
        try:
            return _json_loads(candidate)
        except json.JSONDecodeError as exc:
            if first_error is None:
                first_error = exc
    if first_error is None:  # pragma: no cover - _json_variants 永远非空
        raise json.JSONDecodeError("无法解析 JSON", text, 0)
    raise first_error


def _payload_calls(payload: object) -> list[dict] | None:
    """把顶层 payload 规范成动作 dict 列表；普通 JSON 返回 None。"""
    if isinstance(payload, dict):
        return [payload] if "action" in payload else None
    if isinstance(payload, list):
        if not payload:
            return []
        if not all(isinstance(item, dict) for item in payload):
            return None
        if not any("action" in item for item in payload):
            return None
        return list(payload)
    return None


def _balanced_json_blocks(text: str) -> list[str]:
    """用栈扫描字符串外的成对 `{}`/`[]`，返回每个顶层块。"""
    blocks: list[str] = []
    stack: list[str] = []
    start = 0
    in_string = False
    escaped = False
    pairs = {"}": "{", "]": "["}
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char in "[{":
            if not stack:
                start = index
            stack.append(char)
            continue
        if char not in "]}":
            continue
        if not stack:
            continue
        if stack[-1] != pairs[char]:
            # 损坏的嵌套不拖累后续独立块；从下一个起始括号重新开始。
            stack.clear()
            continue
        stack.pop()
        if not stack:
            blocks.append(text[start : index + 1])
    return blocks


def _raw_decode_action(text: str) -> object | None:
    """从任意 `{`/`[` 起点用 raw_decode 提取前面正确的 JSON。"""
    for candidate in _json_variants(text):
        for start, char in enumerate(candidate):
            if char not in "[{":
                continue
            try:
                payload, _end = _JSON_DECODER.raw_decode(candidate[start:])
            except json.JSONDecodeError:
                continue
            if _payload_calls(payload) is None:
                continue
            return payload
    return None


def _payloads_from_block(block: str) -> list[dict] | None:
    try:
        payload = _load_action_json(block)
    except json.JSONDecodeError:
        return None
    return _payload_calls(payload)


def _payloads_from_text(text: str) -> list[dict] | None:
    """按 direct → raw_decode → 栈平衡的顺序提取一个文本片段中的调用。"""
    stripped = text.strip()
    if not stripped:
        return None
    try:
        payload = _load_action_json(stripped)
    except json.JSONDecodeError:
        pass
    else:
        calls = _payload_calls(payload)
        if calls is not None:
            return calls

    decoded = _raw_decode_action(text)
    if decoded is not None:
        calls = _payload_calls(decoded)
        if calls is not None:
            if not isinstance(decoded, dict):
                return calls
            # 多对象文本重新按块解析，避免反斜杠修复改变 raw_decode 的偏移。
            blocks = _balanced_json_blocks(text)
            if len(blocks) > 1:
                recovered: list[dict] = []
                for block in blocks:
                    extra = _payloads_from_block(block)
                    if extra:
                        recovered.extend(extra)
                if recovered:
                    return recovered
            return calls

    calls = []
    recognized = False
    for block in _balanced_json_blocks(text):
        extra = _payloads_from_block(block)
        if extra is None:
            continue
        recognized = True
        calls.extend(extra)
    return calls if recognized else None


def _payloads_from_function_xml(text: str) -> list[dict] | None:
    """解析 MiniCPM 原生 <function name="..."> + <param> / CDATA 工具方言。"""
    calls: list[dict] = []
    for match in _FUNCTION_XML_RE.finditer(text):
        name = match.group("name").strip()
        if not name:
            continue
        params: dict[str, str] = {}
        for pmatch in _PARAM_TAG_RE.finditer(match.group("body")):
            pname = pmatch.group("pname").strip()
            if not pname:
                continue
            raw = (
                pmatch.group("cdata")
                if pmatch.group("cdata") is not None
                else pmatch.group("plain")
            )
            params[pname] = (raw or "").strip()
        calls.append({"action": name, "parameters": params})
    return calls or None


def extract_tool_calls(text: str) -> list[dict] | None:
    """提取 `<tool_call>` 或裸 JSON 中的动作对象。

    返回 None 表示没有工具调用（按普通文本处理）；返回空列表表示识别到
    空动作数组，由 `_parse_actions` 统一报错。
    """
    if not isinstance(text, str) or not text.strip():
        return None
    matches = list(_TOOL_CALL_RE.finditer(text))
    if matches:
        calls: list[dict] = []
        recognized = False
        for match in matches:
            payloads = _payloads_from_text(match.group(1))
            if payloads is None:
                continue
            recognized = True
            calls.extend(payloads)
        if recognized:
            return calls
    xml_calls = _payloads_from_function_xml(text)
    if xml_calls:
        return xml_calls
    return _payloads_from_text(text)


def _looks_like_action_attempt(text: str) -> bool:
    lowered = text.lower()
    if "<tool_call" in lowered or "<function" in lowered:
        return True
    return '"action"' in text


def _parse_actions(
    text: str,
    *,
    max_batch_actions: int,
    action_aliases: dict[str, str],
) -> list[AgentAction] | None:
    """解析标签化/裸 JSON 动作；无法识别时回落到最终自然语言。"""
    stripped = text.strip()
    if not stripped:
        return None
    payloads = extract_tool_calls(stripped)
    if payloads is None:
        if _looks_like_action_attempt(stripped):
            raise ValueError(
                "动作 JSON 解析失败：无法从输出中提取合法的工具调用"
            )
        return None
    if not payloads:
        raise ValueError("动作数组不能为空")
    if len(payloads) > max_batch_actions:
        raise ValueError(f"单次最多 {max_batch_actions} 个动作")
    return [
        _action_from_payload(item, action_aliases) for item in payloads
    ]


def _parse_action(
    text: str,
    *,
    max_batch_actions: int,
    action_aliases: dict[str, str],
) -> AgentAction | None:
    """兼容旧调用：只接受单动作；批量必须走 `_parse_actions`。"""
    actions = _parse_actions(
        text,
        max_batch_actions=max_batch_actions,
        action_aliases=action_aliases,
    )
    if actions is None:
        return None
    if len(actions) != 1:
        raise ValueError("该接口只接受单动作；批量动作请使用 _parse_actions")
    return actions[0]
