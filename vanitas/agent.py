"""Agent 受控闭环：解析动作、分级审批、执行并把结果回填给模型。

安全边界：
- 单动作 `read_file` / `list_dir` / `run_shell_ro` 自动执行；
- 修改动作 `write_file` / `delete_file` / `move_file` 必须确认；
- 动作数组统一打印清单并只弹一次 [y/N]，不因全只读而自动跳过；
- 工具层 `AgentTools` 以工作区为根，越界、坏 JSON 与执行失败都转为纯文本反馈。
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from typing import Callable

from .config import AppConfig, ModelSpec
from .errors import ToolDenied
from .render import format_text
from .repl import Repl
from .tools import AgentTools, ExecResult
from .ui import Palette

DEFAULT_MAX_STEPS = 5
MAX_FEEDBACK_CHARS = 8000
MAX_BATCH_ACTIONS = 100
AUTO_ACTIONS = frozenset({"read_file", "list_dir", "run_shell_ro"})
CONFIRM_ACTIONS = frozenset({"write_file", "delete_file", "move_file"})
ACTION_NAMES = AUTO_ACTIONS | CONFIRM_ACTIONS
_ACTION_ALIASES = {"shell": "run_shell_ro"}
_JSON_PATH_VALUE_RE = re.compile(
    r'("(?:path|src|dst|source|dest|destination|file|files|command)"\s*:\s*")'
    r'((?:[^"\\]|\\.)*)(")',
    re.DOTALL,
)
_JSON_LINE_START_RE = re.compile(r"(?m)^[ \t]*([\[{])")
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


def _build_action(name: str, params: dict) -> AgentAction:
    name = _ACTION_ALIASES.get(name, name)
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


def _action_from_payload(payload: object) -> AgentAction:
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
    return _build_action(name.strip(), raw_params)


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


def _load_action_json(text: str):
    """先按标准 JSON 解析，失败后只修复路径字段，最后再做全局转义兜底。"""
    try:
        return _json_loads(text)
    except json.JSONDecodeError as first_error:
        repaired = _repair_path_escapes(text)
        try:
            return _json_loads(repaired)
        except json.JSONDecodeError:
            second = _repair_invalid_json_escapes(repaired)
            try:
                return _json_loads(second)
            except json.JSONDecodeError:
                raise first_error


def _has_action_payload(payload: object) -> bool:
    if isinstance(payload, dict):
        return "action" in payload
    if isinstance(payload, list):
        return bool(payload) and all(
            isinstance(item, dict) and "action" in item for item in payload
        )
    return False


def _embedded_action_text(text: str) -> str | None:
    """提取“解释文字 + 尾部 JSON 动作块”中的动作块。

    只接受从某一行行首开始的 JSON 对象/数组，避免把正文里的普通 JSON
    示例当成动作；动作数组仍会由上层走一次清单确认。
    """
    for match in reversed(list(_JSON_LINE_START_RE.finditer(text))):
        if match.start() == 0:
            continue
        candidate = text[match.start():].strip()
        if candidate.endswith("```"):
            candidate = candidate[:-3].rstrip()
        try:
            payload = _load_action_json(candidate)
        except json.JSONDecodeError:
            continue
        if _has_action_payload(payload):
            return candidate
    return None


def _parse_actions(text: str) -> list[AgentAction] | None:
    """解析单动作/动作数组；支持前置解释文字后的尾部 JSON 动作块。"""
    stripped = text.strip()
    if not stripped:
        return None
    try:
        if stripped[0] in "[{":
            payload = _load_action_json(stripped)
        else:
            embedded = _embedded_action_text(stripped)
            if embedded is None:
                return None
            payload = _load_action_json(embedded)
    except json.JSONDecodeError as exc:
        if '"action"' in stripped or stripped.startswith("["):
            raise ValueError(
                f"动作 JSON 解析失败：第 {exc.lineno} 行第 {exc.colno} 列：{exc.msg}"
            ) from exc
        return None

    if isinstance(payload, list):
        if not payload:
            raise ValueError("动作数组不能为空")
        if len(payload) > MAX_BATCH_ACTIONS:
            raise ValueError(f"单次最多 {MAX_BATCH_ACTIONS} 个动作")
        return [_action_from_payload(item) for item in payload]

    if not isinstance(payload, dict):
        raise ValueError("动作 JSON 顶层必须是对象或数组")
    if "action" not in payload:
        # 允许模型用普通 JSON 作为最终答复；只有带 action 的才是工具调用。
        return None
    return [_action_from_payload(payload)]


def _parse_action(text: str) -> AgentAction | None:
    """兼容旧调用：只接受单动作；批量必须走 `_parse_actions`。"""
    actions = _parse_actions(text)
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
    actions: list[AgentAction], *, is_batch: bool = False
) -> bool:
    # 只读动作（含只读数组）直接执行；只有写/删/移动才确认。
    # 批量清单仍由 _confirm_actions 汇总，只要数组里有一个修改动作就只问一次。
    return any(action.name in CONFIRM_ACTIONS for action in actions)


def _clip_feedback(text: str) -> str:
    if len(text) <= MAX_FEEDBACK_CHARS:
        return text
    omitted = len(text) - MAX_FEEDBACK_CHARS
    return text[:MAX_FEEDBACK_CHARS] + f"\n…已截断 {omitted} 字符"


class _AgentAnswerStream:
    """把 Agent 的流式输出分成“动作 JSON”和“最终自然语言”。

    初始阶段先看首个非空白字符：`{` / `[` 直接进入动作模式，整段丢弃。
    进入自然语言后，行首若再次出现 `{` / `[`（典型是“解释文字 + 动作块”），
    暂停输出并缓存尾部；上层确认是动作后丢弃，不是动作则 `flush_suppressed()`。
    """

    def __init__(self) -> None:
        self._target = sys.stdout
        self._buffer = ""
        self._mode = "initial"  # initial | text | suppressed
        self._line_start = False
        self._wrote = False

    @property
    def emitted(self) -> bool:
        return self._wrote

    @property
    def suppressed(self) -> bool:
        return self._mode == "suppressed"

    def write(self, text: str) -> int:
        for char in text:
            if self._mode == "suppressed":
                self._buffer += char
                continue
            if self._mode == "initial":
                self._buffer += char
                stripped = self._buffer.lstrip()
                if not stripped:
                    continue
                if stripped[0] in "[{":
                    self._mode = "suppressed"
                    self._buffer = ""
                    continue
                self._mode = "text"
                pending = self._buffer
                self._buffer = ""
                self._line_start = pending.endswith(("\n", "\r"))
                self._send(pending)
                continue
            if char in "\r\n":
                self._send(char)
                self._buffer = ""
                self._line_start = True
                continue
            if self._line_start:
                self._buffer += char
                if char in " \t":
                    continue
                if char in "[{":
                    self._mode = "suppressed"
                    self._buffer = ""
                    continue
                pending = self._buffer
                self._buffer = ""
                self._line_start = False
                self._send(pending)
                continue
            self._send(char)
        return len(text)

    def flush(self) -> None:
        self._target.flush()

    def flush_suppressed(self) -> None:
        if self._mode != "suppressed":
            return
        pending = self._buffer
        self._buffer = ""
        self._mode = "text"
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
        max_steps: int = DEFAULT_MAX_STEPS,
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
        self.max_steps = max(1, int(max_steps))
        self._confirm = confirm

    def _chat(self, text: str) -> None:
        stream = _AgentAnswerStream()
        result = self._run_chat(text, display=False, stream=stream)
        if result is None or result.interrupted:
            return
        steps = 0
        while True:
            try:
                actions = _parse_actions(result.text)
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

            is_batch = result.text.lstrip().startswith("[")
            if _needs_confirmation(actions, is_batch=is_batch):
                if not self._confirm_actions(actions, is_batch=is_batch):
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

    def _confirm_actions(
        self, actions: list[AgentAction], *, is_batch: bool = False
    ) -> bool:
        if len(actions) == 1 and not is_batch:
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
            action, ok, detail = results[0]
            status = "执行成功" if ok else "执行失败"
            return _clip_feedback(f"[系统反馈] {status}：{detail}")

        succeeded = sum(1 for _action, ok, _detail in results if ok)
        failed = len(results) - succeeded
        lines = [
            f"[系统反馈] 批量执行完成：{len(results)} 项，成功 {succeeded}，失败 {failed}"
        ]
        for index, (action, ok, detail) in enumerate(results, start=1):
            status = "成功" if ok else "失败"
            lines.append(f"{index}. {status}：{_summary(action)}")
            if not ok:
                lines.append(f"   {detail}")
        return _clip_feedback("\n".join(lines))

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
    max_steps: int = DEFAULT_MAX_STEPS,
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
