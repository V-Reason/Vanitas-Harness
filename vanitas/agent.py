"""Agent 受控闭环：解析动作、分级审批、执行并把结果回填给模型。

安全边界：
- 单动作与只读数组的 `read_file` / `list_dir` / `run_shell_ro` 自动执行；
- 修改动作 `write_file` / `delete_file` / `move_file` 必须确认；
- 含修改的动作数组只弹一次 [y/N]；全只读数组不确认；
- 工具层 `AgentTools` 以工作区为根，越界、坏 JSON 与执行失败都转成
  `<tool_result>` 纯文本反馈，避免原始动作 JSON 污染终端。
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from typing import Callable

from .config import AGENT_ACTION_NAMES, AppConfig, ModelSpec
from .errors import ToolDenied
from .render import format_text
from .repl import Repl
from .tools import AgentTools, ExecResult
from .ui import Palette

ACTION_NAMES = AGENT_ACTION_NAMES
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
_JSON_DECODER = json.JSONDecoder(strict=False)
_HEX = frozenset("0123456789abcdefABCDEF")
_VALID_JSON_ESCAPES = frozenset('"\\/bfnrt')


@dataclass(frozen=True)
class AgentAction:
    name: str
    parameters: dict


def _require_text(params: dict, key: str) -> str:
    value = params.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"parameters.{key} 必须是非空字符串")
    return value.strip()


def _build_action(name: str, params: dict, aliases: dict[str, str]) -> AgentAction:
    name = aliases.get(name, name)
    if name not in ACTION_NAMES:
        raise ValueError(f"未知动作：{name!r}")
    if name == "read_file":
        return AgentAction(name, {"path": _require_text(params, "path")})
    if name == "write_file":
        path = _require_text(params, "path")
        content = params.get("content")
        if not isinstance(content, str):
            raise ValueError("parameters.content 必须是字符串")
        return AgentAction(name, {"path": path, "content": content})
    if name == "move_file":
        src = params.get("src", params.get("source"))
        dst = params.get("dst", params.get("dest", params.get("destination")))
        if not isinstance(src, str) or not src.strip():
            raise ValueError("parameters.src 必须是非空字符串")
        if not isinstance(dst, str) or not dst.strip():
            raise ValueError("parameters.dst 必须是非空字符串")
        return AgentAction(
            name,
            {"src": src.strip(), "dst": dst.strip()},
        )
    if name == "delete_file":
        return AgentAction(name, {"path": _require_text(params, "path")})
    if name == "list_dir":
        path = params.get("path", ".")
        if not isinstance(path, str) or not path.strip():
            raise ValueError("parameters.path 必须是非空字符串")
        return AgentAction(name, {"path": path.strip()})
    return AgentAction(
        name, {"command": _require_text(params, "command")}
    )


def _action_from_payload(
    payload: object, aliases: dict[str, str]
) -> AgentAction:
    if not isinstance(payload, dict):
        raise ValueError("每个动作必须是 JSON 对象")
    if "action" not in payload:
        raise ValueError("动作对象缺少 action 字段")
    name = payload.get("action")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("action 必须是非空字符串")
    raw_params = payload.get("parameters", {})
    if not isinstance(raw_params, dict):
        raise ValueError("parameters 必须是 JSON 对象")
    return _build_action(name.strip(), raw_params, aliases)


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
        return calls if recognized else None
    return _payloads_from_text(text)


def _looks_like_action_attempt(text: str) -> bool:
    if "<tool_call" in text.lower():
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


def _summary(action: AgentAction) -> str:
    params = action.parameters
    if action.name == "run_shell_ro":
        return f"只读命令 {params['command']}"
    if action.name == "read_file":
        return f"读取 {params['path']}"
    if action.name == "list_dir":
        return f"列出 {params['path']}"
    if action.name == "write_file":
        return f"写入 {params['path']}"
    if action.name == "move_file":
        return f"重命名 {params['src']} -> {params['dst']}"
    return f"删除 {params['path']}"


def _describe(action: AgentAction) -> str:
    if action.name == "run_shell_ro":
        return f"将执行只读命令：{action.parameters['command']}"
    if action.name == "read_file":
        return f"将读取文件 {action.parameters['path']}"
    if action.name == "list_dir":
        return f"将列出目录 {action.parameters['path']}"
    if action.name == "write_file":
        return f"将写入文件 {action.parameters['path']}"
    if action.name == "move_file":
        return (
            f"将重命名 {action.parameters['src']} -> "
            f"{action.parameters['dst']}"
        )
    return f"将删除文件 {action.parameters['path']}"


def _needs_confirmation(
    actions: list[AgentAction], confirm_actions: frozenset[str]
) -> bool:
    # 审批策略来自 settings.agent；批量里有一个确认动作就只问一次。
    return any(action.name in confirm_actions for action in actions)


def _clip_feedback(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return text[:limit] + f"\n…已截断 {omitted} 字符"


class _AgentAnswerStream:
    """把 Agent 的流式输出分成“动作块”和“最终自然语言”。

    初始阶段先看首个非空白字符：`{` / `[` 进入 JSON 抑制，`<tool_call>`
    进入标签抑制。进入自然语言后，行首的 `{` / `[` 仍按旧兼容处理；
    `<tool_call>` 可在任意位置识别，且允许标签跨流式 chunk。
    """

    def __init__(self) -> None:
        self._target = sys.stdout
        self._buffer = ""
        self._suppressed = ""
        self._restore: list[str] = []
        self._mode = "initial"  # initial | text | probe_tag | suppress_tag | suppress_json
        self._line_start = False
        self._wrote = False

    @property
    def emitted(self) -> bool:
        return self._wrote

    @property
    def suppressed(self) -> bool:
        return bool(self._restore) or self._mode in {
            "probe_tag",
            "suppress_tag",
            "suppress_json",
        }

    def write(self, text: str) -> int:
        for char in text:
            self._feed(char)
        return len(text)

    def _feed(self, char: str) -> None:
        if self._mode == "suppress_tag":
            self._suppressed += char
            if self._suppressed.lower().endswith(TOOL_CALL_CLOSE):
                self._restore.append(self._suppressed)
                self._suppressed = ""
                self._mode = "text"
                self._line_start = False
            return
        if self._mode == "suppress_json":
            self._suppressed += char
            return
        if self._mode in {"initial", "probe_tag"}:
            self._probe(char)
            return
        if self._mode != "text":
            self._send(char)
            return
        if self._line_start:
            self._buffer += char
            if char in " \t":
                return
            stripped = self._buffer.lstrip(" \t")
            if stripped.startswith(("{", "[")):
                self._enter_json()
                return
            if stripped == "<":
                self._mode = "probe_tag"
                return
            if TOOL_CALL_OPEN.startswith(stripped.lower()):
                return
            pending = self._buffer
            self._buffer = ""
            self._line_start = pending.endswith(("\n", "\r"))
            self._send(pending)
            return

        if char == "<":
            self._mode = "probe_tag"
            self._buffer = char
            return
        if char in "\r\n":
            self._send(char)
            self._line_start = True
            return
        self._send(char)

    def _probe(self, char: str) -> None:
        self._buffer += char
        stripped = self._buffer.lstrip(" \t")
        if not stripped:
            return
        if stripped.startswith(("{", "[")):
            self._enter_json()
            return
        if TOOL_CALL_OPEN.startswith(stripped.lower()):
            if stripped.lower() == TOOL_CALL_OPEN:
                self._suppressed = self._buffer
                self._buffer = ""
                self._mode = "suppress_tag"
            return
        pending = self._buffer
        self._buffer = ""
        self._mode = "text"
        self._line_start = pending.endswith(("\n", "\r"))
        self._send(pending)

    def _enter_json(self) -> None:
        self._suppressed = self._buffer
        self._buffer = ""
        self._mode = "suppress_json"

    def flush(self) -> None:
        self._target.flush()

    def flush_suppressed(self) -> None:
        parts = list(self._restore)
        if self._mode in {"suppress_tag", "suppress_json"}:
            parts.append(self._suppressed)
        elif self._mode == "probe_tag":
            parts.append(self._buffer)
        if not parts:
            return
        pending = "".join(parts)
        self._restore = []
        self._suppressed = ""
        self._buffer = ""
        self._mode = "text"
        self._line_start = pending.endswith(("\n", "\r"))
        self._send(pending)

    def _send(self, text: str) -> None:
        if not text:
            return
        self._wrote = True
        self._target.write(text)
        self._target.flush()


def _shell_detail(result: ExecResult) -> str:
    pieces = [f"退出码 {result.exit_code}"]
    if result.stdout.strip():
        pieces.append(result.stdout.rstrip())
    if result.stderr.strip():
        pieces.append("stderr：" + result.stderr.rstrip())
    if len(pieces) == 1:
        pieces.append("无输出")
    return "\n".join(pieces)


class AgentRepl(Repl):
    """在 REPL 之上增加“模型提议 → 分级确认 → 执行 → 反馈”的闭环。"""

    def __init__(
        self,
        config: AppConfig,
        model: ModelSpec | None = None,
        profile: str | None = None,
        *,
        auto_start: bool = False,
        palette: Palette | None = None,
        verbose: bool = False,
        tools: AgentTools | None = None,
        max_steps: int | None = None,
        confirm: Callable[[str], bool] | None = None,
    ) -> None:
        super().__init__(
            config,
            model,
            profile,
            auto_start=auto_start,
            palette=palette,
            verbose=verbose,
        )
        self.agent_tools = tools or AgentTools(config)
        configured_steps = (
            config.settings.agent.max_steps if max_steps is None else max_steps
        )
        self.max_steps = max(1, int(configured_steps))
        self._confirm = confirm

    def _chat(self, text: str) -> None:
        stream = _AgentAnswerStream()
        result = self._run_chat(text, display=False, stream=stream)
        if result is None or result.interrupted:
            return
        steps = 0
        while True:
            try:
                actions = _parse_actions(
                    result.text,
                    max_batch_actions=self.config.settings.agent.max_batch_actions,
                    action_aliases=self.config.settings.agent.action_aliases,
                )
            except ValueError as exc:
                print(self.palette.err(str(exc)), file=sys.stderr)
                return
            if actions is None:
                if stream.suppressed:
                    stream.flush_suppressed()
                elif not stream.emitted:
                    self._print_final(result.text)
                return
            if steps >= self.max_steps:
                print(self.palette.warn(f"已达最大步数 {self.max_steps}"))
                return

            if _needs_confirmation(
                actions, self.config.settings.agent.confirm_actions
            ):
                if not self._confirm_actions(actions):
                    print(self.palette.meta("已取消"))
                    return
            else:
                self._announce_auto(actions)

            feedback = self._execute_actions(actions)
            steps += 1
            stream = _AgentAnswerStream()
            result = self._run_chat(
                "", display=False, feedback=feedback, stream=stream
            )
            if result is None or result.interrupted:
                return

    def _confirm_actions(self, actions: list[AgentAction]) -> bool:
        if len(actions) == 1:
            plan = _describe(actions[0])
        else:
            lines = [f"批量操作（{len(actions)} 项）："]
            lines.extend(
                f"{index}. {_summary(action)}"
                for index, action in enumerate(actions, start=1)
            )
            plan = "\n".join(lines)
        if self._confirm is not None:
            print(self.palette.accent(plan))
            return bool(self._confirm(plan))
        return self.confirm_plan(plan)

    def _announce_auto(self, actions: list[AgentAction]) -> None:
        if len(actions) == 1:
            print(self.palette.meta(f"自动执行：{_summary(actions[0])}"))
            return
        print(self.palette.meta(f"自动执行（{len(actions)} 项）："))
        for index, action in enumerate(actions, start=1):
            print(self.palette.meta(f"{index}. {_summary(action)}"))

    def _execute_actions(self, actions: list[AgentAction]) -> str:
        results: list[tuple[AgentAction, bool, str]] = []
        for action in actions:
            try:
                ok, detail = self._execute(action)
            except ToolDenied as exc:
                ok, detail = False, str(exc)
                print(self.palette.warn(f"工具拒绝：{detail}"), file=sys.stderr)
            except OSError as exc:
                ok, detail = False, f"工具执行失败：{exc}"
                print(self.palette.err(detail), file=sys.stderr)
            except (TypeError, ValueError) as exc:
                ok, detail = False, f"工具参数无效：{exc}"
                print(self.palette.err(detail), file=sys.stderr)
            results.append((action, ok, detail))

        if len(results) == 1:
            _action, ok, detail = results[0]
            status = "执行成功" if ok else "执行失败"
            body = f"{status}：{detail}"
        else:
            succeeded = sum(1 for _action, ok, _detail in results if ok)
            failed = len(results) - succeeded
            lines = [
                f"批量执行完成：{len(results)} 项，成功 {succeeded}，失败 {failed}"
            ]
            for index, (action, ok, detail) in enumerate(results, start=1):
                status = "成功" if ok else "失败"
                lines.append(f"{index}. {status}：{_summary(action)}")
                lines.append(f"   {detail.rstrip()}")
            body = "\n".join(lines)
        clipped = _clip_feedback(
            body, self.config.settings.agent.max_feedback_chars
        )
        return f"<tool_result>\n{clipped}\n</tool_result>"

    def _execute(self, action: AgentAction) -> tuple[bool, str]:
        params = action.parameters
        if action.name == "run_shell_ro":
            result = self.agent_tools.run_shell_ro(params["command"])
            return result.exit_code == 0, _shell_detail(result)
        if action.name == "read_file":
            path = params["path"]
            content = self.agent_tools.read_file(path)
            return True, f"读取 {path} · {len(content)} 字符\n{content}"
        if action.name == "write_file":
            path = params["path"]
            written = self.agent_tools.write_file(path, params["content"])
            return True, f"写入 {path} · {written} 字节"
        if action.name == "move_file":
            detail = self.agent_tools.move_file(params["src"], params["dst"])
            return True, detail
        if action.name == "delete_file":
            detail = self.agent_tools.delete_file(params["path"])
            return True, detail
        path = params["path"]
        names = self.agent_tools.list_dir(path)
        detail = f"列出 {path} · {len(names)} 项"
        if names:
            detail += "\n" + "\n".join(names)
        else:
            detail += "\n目录为空"
        return True, detail

    def _print_final(self, text: str) -> None:
        rendered = format_text(text)
        if not rendered.strip():
            print(self.palette.warn("模型未返回内容"))
            return
        sys.stdout.write(rendered)
        if not rendered.endswith("\n"):
            sys.stdout.write("\n")


def run_agent(
    config: AppConfig,
    model: ModelSpec | None = None,
    profile: str | None = None,
    *,
    system: str | None = None,
    palette: Palette | None = None,
    verbose: bool = False,
    max_steps: int | None = None,
    tools: AgentTools | None = None,
    confirm: Callable[[str], bool] | None = None,
) -> int:
    paint = palette or Palette({}, enabled=False)
    print(paint.meta("Agent 已就绪"))
    repl = AgentRepl(
        config,
        model,
        profile,
        auto_start=True,
        palette=paint,
        verbose=verbose,
        tools=tools,
        max_steps=max_steps,
        confirm=confirm,
    )
    repl.configure(system=system, task="agent")
    return repl.run()
