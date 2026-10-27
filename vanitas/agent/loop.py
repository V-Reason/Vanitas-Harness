"""Agent 调度循环、REPL 绑定与入口。

AgentLoop 只做状态迁移与执行编排；解析、流式、审批、停滞、反馈均来自子模块。
"""
from __future__ import annotations

import sys
from collections import deque
from typing import Callable

from ..chat import TurnResult
from ..config import AppConfig, ModelSpec
from ..errors import ToolDenied
from ..render import format_text
from ..repl import Repl
from ..tools import AgentTools
from ..ui import Palette
from .actions import AgentAction, _action_signature, _describe, _summary
from .failed import FailedSignatures
from .feedback import _parse_error_feedback, _shell_detail, wrap_tool_result
from .parser import _parse_actions
from .policy import _needs_confirmation
from .stagnation import evaluate_stagnation
from .stream import _AgentAnswerStream


def _run_action(tools: AgentTools, action: AgentAction) -> tuple[bool, str]:
    params = action.parameters
    if action.name == "run_shell_ro":
        result = tools.run_shell_ro(params["command"])
        return result.exit_code == 0, _shell_detail(result)
    if action.name == "read_file":
        path = params["path"]
        content = tools.read_file(path)
        return True, f"读取 {path} · {len(content)} 字符\n{content}"
    if action.name == "read_file_range":
        path = params.get("path")
        if not isinstance(path, str) or not path:
            raise ToolDenied("read_file_range 缺少 path")
        content = tools.read_file_range(
            path,
            int(params.get("start_line", 1)),
            int(params.get("end_line", 100)),
        )
        return True, f"读取 {path} 行范围\n{content}"
    if action.name == "grep_workspace":
        pattern = params.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            raise ToolDenied("grep_workspace 缺少 pattern")
        raw_max = params.get("max_results")
        max_results = int(raw_max) if raw_max is not None else None
        matches = tools.grep_workspace(
            pattern,
            str(params.get("path", ".")),
            max_results=max_results,
        )
        detail = f"检索 {pattern} · {len(matches)} 项"
        return True, detail + "\n" + "\n".join(matches)
    if action.name == "patch_file":
        path = params.get("path")
        search = params.get("search")
        replace = params.get("replace")
        if not all(isinstance(item, str) for item in (path, search, replace)):
            raise ToolDenied("patch_file 缺少 path/search/replace")
        assert isinstance(path, str)
        assert isinstance(search, str)
        assert isinstance(replace, str)
        detail = tools.patch_file(path, search, replace)
        return True, detail
    if action.name == "write_file":
        path = params["path"]
        written = tools.write_file(path, params["content"])
        return True, f"写入 {path} · {written} 字节"
    if action.name == "move_file":
        detail = tools.move_file(params["src"], params["dst"])
        return True, detail
    if action.name in {"delete_file", "delete_path"}:
        detail = tools.delete_path(params["path"])
        return True, detail
    path = params["path"]
    names = tools.list_dir(path)
    detail = f"列出 {path} · {len(names)} 项"
    if names:
        detail += "\n" + "\n".join(names)
    else:
        detail += "\n目录为空"
    return True, detail

def _run_actions(
    actions: list[AgentAction],
    execute_one: Callable[[AgentAction], tuple[bool, str]],
    *,
    max_feedback_chars: int,
    palette: Palette,
    failed_sink: FailedSignatures | None = None,
    head_lines: int = 20,
    tail_lines: int = 40,
) -> str:
    results: list[tuple[AgentAction, bool, str]] = []
    for action in actions:
        try:
            ok, detail = execute_one(action)
        except ToolDenied as exc:
            ok, detail = False, str(exc)
            print(palette.warn(f"工具拒绝：{detail}"), file=sys.stderr)
        except OSError as exc:
            ok, detail = False, f"工具执行失败：{exc}"
            print(palette.err(detail), file=sys.stderr)
        except (TypeError, ValueError) as exc:
            ok, detail = False, f"工具参数无效：{exc}"
            print(palette.err(detail), file=sys.stderr)
        if failed_sink is not None:
            signature = _action_signature(action)
            if ok:
                failed_sink.record_success(signature)
            else:
                failed_sink.record_failure(signature, detail)
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
    return wrap_tool_result(
        body,
        max_feedback_chars,
        head_lines=head_lines,
        tail_lines=tail_lines,
    )
class AgentLoop:
    """Agent 调度状态机：解析自愈、停滞熔断、步数与执行编排。

    与 REPL UI 解耦：确认、播报、最终输出与执行都经由 `run()` 注入的钩子；
    自身只负责确定性状态迁移，并用构造时传入的 `AgentTools` 提供兜底执行。
    """

    def __init__(
        self,
        config: AppConfig,
        tools: AgentTools,
        *,
        max_steps: int | None = None,
        max_parse_retries: int | None = None,
        stagnation_limit: int | None = None,
        palette: Palette | None = None,
    ) -> None:
        self.config = config
        self.tools = tools
        configured_steps = (
            config.settings.agent.max_steps if max_steps is None else max_steps
        )
        self.max_steps = max(1, int(configured_steps))
        configured_retries = (
            config.settings.agent.max_parse_retries
            if max_parse_retries is None
            else max_parse_retries
        )
        self.max_parse_retries = max(0, int(configured_retries))
        configured_stagnation = (
            config.settings.agent.stagnation_limit
            if stagnation_limit is None
            else stagnation_limit
        )
        # 连续 1 次相同动作没有判别意义；下限固定为 2。
        self.stagnation_limit = max(2, int(configured_stagnation))
        self.palette = palette or Palette({}, enabled=False)
        self._recent: deque[tuple[str, str]] = deque(
            maxlen=max(6, self.stagnation_limit)
        )
        self.steps = 0
        self.parse_retries = 0
        self.stagnation_hits = 0
        self._stagnation_reason = ""
        self._failed_signatures = FailedSignatures()
        self._rounds: deque[tuple[str, ...]] = deque(maxlen=6)

    @property
    def recent_count(self) -> int:
        return len(self._recent)

    @property
    def stagnation_count(self) -> int:
        return min(len(self._recent), self.stagnation_limit)

    # ------------------------------------------------------------------ #
    # 执行（AgentRepl 复用与可注入钩子）
    # ------------------------------------------------------------------ #

    def execute_action(self, action: AgentAction) -> tuple[bool, str]:
        return _run_action(self.tools, action)

    def execute_actions(self, actions: list[AgentAction]) -> str:
        return _run_actions(
            actions,
            self.execute_action,
            max_feedback_chars=self.config.settings.agent.max_feedback_chars,
            palette=self.palette,
            failed_sink=self._failed_signatures,
            head_lines=self.config.settings.agent.feedback_head_lines,
            tail_lines=self.config.settings.agent.feedback_tail_lines,
        )

    # ------------------------------------------------------------------ #
    # 调度循环
    # ------------------------------------------------------------------ #

    def run(
        self,
        text: str,
        *,
        converse: Callable[
            [str, str | None, _AgentAnswerStream], TurnResult | None
        ],
        confirm: Callable[[list[AgentAction]], bool],
        announce: Callable[[list[AgentAction]], None],
        finalize: Callable[[str], None],
        execute: Callable[[list[AgentAction]], str] | None = None,
    ) -> None:
        execute_actions = execute or self.execute_actions
        self._recent.clear()
        self._rounds.clear()
        steps = 0
        parse_retries = 0
        self.steps = 0
        self.parse_retries = 0
        self.stagnation_hits = 0
        self._stagnation_reason = ""
        self._failed_signatures.clear()
        stream = _AgentAnswerStream()
        result = converse(text, None, stream)
        if result is None or result.interrupted:
            return
        while True:
            try:
                actions = _parse_actions(
                    result.text,
                    max_batch_actions=(
                        self.config.settings.agent.max_batch_actions
                    ),
                    action_aliases=self.config.settings.agent.action_aliases,
                )
            except ValueError as exc:
                if parse_retries >= self.max_parse_retries:
                    print(self.palette.err(str(exc)), file=sys.stderr)
                    return
                parse_retries += 1
                self.parse_retries = parse_retries
                print(
                    self.palette.warn(
                        "格式解析错误，触发自愈回灌"
                        f"（{parse_retries}/{self.max_parse_retries}）"
                    )
                )
                stream = _AgentAnswerStream()
                result = converse("", _parse_error_feedback(exc), stream)
                if result is None or result.interrupted:
                    return
                continue

            parse_retries = 0
            self.parse_retries = 0
            if actions is None:
                if stream.suppressed:
                    stream.flush_suppressed()
                elif not stream.emitted:
                    finalize(result.text)
                return

            if steps >= self.max_steps:
                print(self.palette.warn(f"已达最大步数 {self.max_steps}"))
                return
            if self._stagnates(actions):
                self.stagnation_hits += 1
                reason = (
                    f"连续 {self.stagnation_limit} 次"
                    if self._stagnation_reason == "连续"
                    else self._stagnation_reason
                )
                print(
                    self.palette.warn(
                        f"检测到动作{reason}停滞，"
                        "已触发硬熔断保护"
                    )
                )
                return

            blocked = self._failed_signatures.block(actions)
            if blocked:
                summary = "；".join(
                    f"{_summary(action)}（{reason[:200]}）"
                    for action, reason in blocked
                )
                print(
                    self.palette.warn(
                        f"拦截此前已失败动作，不再二次确认：{summary}"
                    )
                )
                steps += 1
                self.steps = steps
                feedback = (
                    "<tool_result>\n"
                    f"执行失败：该动作此前已失败（{summary}）。"
                    "严禁以相同参数重复提交；请修改参数、改用其他策略或直接收尾。\n"
                    "</tool_result>"
                )
                stream = _AgentAnswerStream()
                result = converse("", feedback, stream)
                if result is None or result.interrupted:
                    return
                continue

            if _needs_confirmation(
                actions, self.config.settings.agent.confirm_actions
            ):
                if not confirm(actions):
                    print(self.palette.meta("已取消"))
                    return
            else:
                announce(actions)

            feedback = execute_actions(actions)
            steps += 1
            self.steps = steps
            stream = _AgentAnswerStream()
            result = converse("", feedback, stream)
            if result is None or result.interrupted:
                return

    def _stagnates(self, actions: list[AgentAction]) -> bool:
        signatures = [_action_signature(action) for action in actions]
        for signature in signatures:
            self._recent.append(signature)
        self._rounds.append(tuple(sorted(signatures)))
        reason = evaluate_stagnation(
            list(self._recent), list(self._rounds), self.stagnation_limit
        )
        if reason is not None:
            self._stagnation_reason = reason
            return True
        return False
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
        max_parse_retries: int | None = None,
        stagnation_limit: int | None = None,
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
        self.loop = AgentLoop(
            config,
            self.agent_tools,
            max_steps=self.max_steps,
            max_parse_retries=max_parse_retries,
            stagnation_limit=stagnation_limit,
            palette=self.palette,
        )

    def _chat(self, text: str) -> None:
        self.loop.run(
            text,
            converse=self._converse,
            confirm=self._confirm_actions,
            announce=self._announce_auto,
            execute=self._execute_actions,
            finalize=self._print_final,
        )

    def _converse(
        self,
        text: str,
        feedback: str | None,
        stream: _AgentAnswerStream,
    ) -> TurnResult | None:
        return self._run_chat(
            text, display=False, feedback=feedback, stream=stream
        )

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
        return _run_actions(
            actions,
            self._execute,
            max_feedback_chars=self.config.settings.agent.max_feedback_chars,
            palette=self.palette,
            failed_sink=self.loop._failed_signatures,
            head_lines=self.config.settings.agent.feedback_head_lines,
            tail_lines=self.config.settings.agent.feedback_tail_lines,
        )

    def _execute(self, action: AgentAction) -> tuple[bool, str]:
        return _run_action(self.agent_tools, action)

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
