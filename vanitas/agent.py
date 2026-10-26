"""Agent 受控闭环：解析动作、分级审批、执行并把结果回填给模型。

安全边界：
- 单动作 `read_file` / `list_dir` / `run_shell_ro` 自动执行；
- 修改动作 `write_file` / `delete_file` / `move_file` 必须确认；
- 动作数组统一打印清单并只弹一次 [y/N]，不因全只读而自动跳过；
- 工具层 `AgentTools` 以工作区为根，越界、坏 JSON 与执行失败都转为纯文本反馈。
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
MAX_BATCH_ACTIONS = 100
AUTO_ACTIONS = frozenset({"read_file", "list_dir", "run_shell_ro"})
CONFIRM_ACTIONS = frozenset({"write_file", "delete_file", "move_file"})
ACTION_NAMES = AUTO_ACTIONS | CONFIRM_ACTIONS
_ACTION_ALIASES = {"shell": "run_shell_ro"}


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


def _parse_actions(text: str) -> list[AgentAction] | None:
    """解析单动作或动作数组；普通文本返回 None，非法 JSON 抛 ValueError。"""
    stripped = text.strip()
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        payload = json.loads(stripped)
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
    # 数组（即使只有一项）也走一次清单确认；单动作才按副作用分流。
    return is_batch or any(
        action.name in CONFIRM_ACTIONS for action in actions
    )


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
        result = self._run_chat(text, display=False)
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
            result = self._run_chat("", display=False, feedback=feedback)
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
        # 数组已统一进入确认分支，这里只会收到单个只读动作。
        print(self.palette.meta(f"自动执行：{_summary(actions[0])}"))

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
