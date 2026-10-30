"""REPL 命令表：数据（Command）+ 分组（GROUPS）+ /help 渲染。

COMMANDS 是命令的权威定义；新增命令只改本文件。
"""
from __future__ import annotations

from dataclasses import dataclass

from ..ui import Palette


@dataclass(frozen=True)
class Command:
    name: str
    description: str
    aliases: tuple = ()
    arg_hint: str = ""


# D1：命令是数据，不是散落的 if 分支；/help 由本表生成。
COMMANDS: tuple[Command, ...] = (
    Command("help", "列出内置命令"),
    Command("model", "列出模型或切换模型（更新记忆）", (), "[id]"),
    Command("profile", "查看或切换档位（立即落盘记忆）", (), "[N]"),
    Command("restart", "重启服务以应用待生效档位", ("reload",)),
    Command("system", "设置系统提示", (), "[text|file]"),
    Command("task", "应用任务模板 config\\prompts\\task_<name>.md", (), "[name]"),
    Command("clear", "清空对话历史"),
    Command("undo", "回滚最近一轮对话", ("pop", "rollback")),
    Command("save", "保存当前会话 JSON", (), "[path]"),
    Command("load", "加载会话 JSON", (), "<path>"),
    Command("exec", "在用户原生 shell 中执行命令（无白名单）", (), "<cmd>"),
    Command("stop", "停止所有 llama-server", ("kill",)),
    Command("shutdown", "保存会话、停止服务并退出", ("halt",)),
    Command("status", "显示服务端 / 显存 / 当前模型 / 空闲倒计时"),
    Command("quit", "退出 REPL（后台服务保留）", ("exit",)),
)

COMMAND_MAP: dict[str, Command] = {}
for _command in COMMANDS:
    COMMAND_MAP[_command.name] = _command
    for _alias in _command.aliases:
        COMMAND_MAP[_alias] = _command


GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("会话", ("help", "clear", "undo", "save", "load", "quit")),
    ("模型与档位", ("model", "profile", "restart", "stop", "shutdown")),
    ("诊断", ("status", "system", "task")),
    ("工具", ("exec",)),
)


def render_help(palette: Palette) -> None:
    """按分组打印命令表；未知命令名不会崩溃（防御）。"""
    print(palette.accent("内置命令："))
    grouped: set[str] = set()
    for group_name, names in GROUPS:
        section_lines: list[str] = []
        for name in names:
            command = COMMAND_MAP.get(name)
            if command is None or command.name != name:
                continue
            if command.name in grouped:
                continue
            grouped.add(command.name)
            head = f"/{command.name}" + (
                f" {command.arg_hint}" if command.arg_hint else ""
            )
            section_lines.append(f"  {head:<24}{command.description}")
        if section_lines:
            print(palette.accent(group_name))
            for line in section_lines:
                print(line)
    leftovers = [c for c in COMMANDS if c.name not in grouped]
    if leftovers:
        print(palette.accent("其他"))
        for command in leftovers:
            head = f"/{command.name}" + (
                f" {command.arg_hint}" if command.arg_hint else ""
            )
            print(f"  {head:<24}{command.description}")


__all__ = ["Command", "COMMANDS", "COMMAND_MAP", "GROUPS", "render_help"]
