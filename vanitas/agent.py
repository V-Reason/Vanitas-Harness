"""Agent 受控闭环：确认后执行工具，反馈回填，最多 max_steps 步。

安全边界：
- 模型每步只能提出一个动作；终端展示计划并要求 y/N。
- 工具层 `AgentTools` 以当前工作目录为沙盒根，越界路径直接拒绝。
- 用户拒绝、越界、坏 JSON 立即停止；只有已确认的动作会执行。
"""

from __future__ import annotations

import json
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
ACTION_NAMES = frozenset({"shell", "read_file", "write_file", "list_dir"})


@dataclass(frozen=True)
class AgentAction:
    name: str
    parameters: dict


def _require_text(params: dict, key: str) -> str:
    value = params.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"parameters.{key} 必须是非空字符串")
    return value.strip()


def _parse_action(text: str) -> AgentAction | None:
    stripped = text.strip()
    if not stripped.startswith("{"):
        return None
    if '"action"' not in stripped:
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"动作 JSON 解析失败：第 {exc.lineno} 行第 {exc.colno} 列：{exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError("动作 JSON 顶层必须是对象")
    name = payload.get("action")
    if not isinstance(name, str) or not name.strip():
        return None
    name = name.strip()
    if name not in ACTION_NAMES:
        raise ValueError(f"未知动作：{name!r}")
    raw_params = payload.get("parameters", {})
    if not isinstance(raw_params, dict):
        raise ValueError("parameters 必须是 JSON 对象")

    if name == "shell":
        return AgentAction(name, {"command": _require_text(raw_params, "command")})
    if name == "read_file":
        return AgentAction(name, {"path": _require_text(raw_params, "path")})
    if name == "write_file":
        path = _require_text(raw_params, "path")
        content = raw_params.get("content")
        if not isinstance(content, str):
            raise ValueError("parameters.content 必须是字符串")
        return AgentAction(name, {"path": path, "content": content})
    path = raw_params.get("path", ".")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("parameters.path 必须是非空字符串")
    return AgentAction(name, {"path": path.strip()})


def _describe(action: AgentAction) -> str:
    if action.name == "shell":
        return f"将执行 shell: {action.parameters['command']}"
    if action.name == "read_file":
        return f"将读取文件 {action.parameters['path']}"
    if action.name == "write_file":
        return f"将写入文件 {action.parameters['path']}"
    return f"将列出目录 {action.parameters['path']}"


def _clip_feedback(text: str) -> str:
    if len(text) <= MAX_FEEDBACK_CHARS:
        return text
    omitted = len(text) - MAX_FEEDBACK_CHARS
    return text[:MAX_FEEDBACK_CHARS] + f"\n…已截断 {omitted} 字符"


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
    """在 REPL 之上增加“模型提议 → 人工确认 → 执行 → 反馈”的闭环。"""

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
        result = self._run_chat(text, display=False)
        if result is None or result.interrupted:
            return
        steps = 0
        while True:
            try:
                action = _parse_action(result.text)
            except ValueError as exc:
                print(self.palette.err(str(exc)), file=sys.stderr)
                return
            if action is None:
                self._print_final(result.text)
                return
            if steps >= self.max_steps:
                print(self.palette.warn(f"已达最大步数 {self.max_steps}"))
                return
            if not self._confirm_action(action):
                print(self.palette.meta("已取消"))
                return
            try:
                ok, detail = self._execute(action)
            except ToolDenied as exc:
                print(self.palette.err(str(exc)), file=sys.stderr)
                return
            except OSError as exc:
                print(self.palette.err(f"工具执行失败：{exc}"), file=sys.stderr)
                return
            steps += 1
            status = "执行成功" if ok else "执行失败"
            feedback = _clip_feedback(f"[系统反馈] {status}：{detail}")
            result = self._run_chat("", display=False, feedback=feedback)
            if result is None or result.interrupted:
                return

    def _confirm_action(self, action: AgentAction) -> bool:
        plan = _describe(action)
        print(self.palette.accent(plan))
        if self._confirm is not None:
            return bool(self._confirm(plan))
        try:
            answer = input("[y/N] ")
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        return answer.strip().lower() in {"y", "yes", "是"}

    def _execute(self, action: AgentAction) -> tuple[bool, str]:
        if action.name == "shell":
            result = self.agent_tools.run_shell(action.parameters["command"])
            return result.exit_code == 0, _shell_detail(result)
        if action.name == "read_file":
            path = action.parameters["path"]
            content = self.agent_tools.read_file(path)
            return True, f"读取 {path} · {len(content)} 字符\n{content}"
        if action.name == "write_file":
            path = action.parameters["path"]
            written = self.agent_tools.write_file(path, action.parameters["content"])
            return True, f"写入 {path} · {written} 字节"
        path = action.parameters["path"]
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
