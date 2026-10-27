"""AgentAction 类型、参数强校验、动作签名与人类可读摘要。

本模块不读文件、不执行进程、不感知 UI；只有类型与纯函数。
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from ..config import AGENT_ACTION_NAMES

ACTION_NAMES = AGENT_ACTION_NAMES


@dataclass(frozen=True)
class AgentAction:
    name: str
    parameters: dict
def _require_text(params: dict, key: str, action: str) -> str:
    value = params.get(key)
    if not isinstance(value, str) or not value.strip():
        keys = "、".join(sorted(params)) or "（空）"
        raise ValueError(
            f"动作 {action!r} 缺少/非法 parameters.{key}；"
            f"收到参数键：{keys}"
        )
    return value.strip()


def _optional_int(params: dict, key: str, default: int, action: str) -> int:
    value = params.get(key, default)
    if isinstance(value, bool):
        raise ValueError(
            f"动作 {action!r} 的 parameters.{key} 必须是整数；收到 {value!r}"
        )
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"动作 {action!r} 的 parameters.{key} 必须是整数；收到 {value!r}"
        ) from exc


def _optional_text(
    params: dict, key: str, default: str, action: str
) -> str:
    value = params.get(key, default)
    if not isinstance(value, str) or not value.strip():
        keys = "、".join(sorted(params)) or "（空）"
        raise ValueError(
            f"动作 {action!r} 缺少/非法 parameters.{key}；"
            f"收到参数键：{keys}"
        )
    return value.strip()
def _build_action(name: str, params: dict, aliases: dict[str, str]) -> AgentAction:
    name = aliases.get(name, name)
    if name not in ACTION_NAMES:
        raise ValueError(f"未知动作：{name!r}")
    if name == "read_file":
        return AgentAction(name, {"path": _require_text(params, "path", name)})
    if name == "read_file_range":
        return AgentAction(
            name,
            {
                "path": _require_text(params, "path", name),
                "start_line": _optional_int(params, "start_line", 1, name),
                "end_line": _optional_int(params, "end_line", 100, name),
            },
        )
    if name == "grep_workspace":
        return AgentAction(
            name,
            {
                "pattern": _require_text(params, "pattern", name),
                "path": _optional_text(params, "path", ".", name),
                "max_results": _optional_int(
                    params, "max_results", 50, name
                ),
            },
        )
    if name == "patch_file":
        path = _require_text(params, "path", name)
        search = _require_text(params, "search", name)
        replace = params.get("replace")
        if not isinstance(replace, str):
            keys = "、".join(sorted(params)) or "（空）"
            raise ValueError(
                f"动作 {name!r} 缺少/非法 parameters.replace；"
                f"收到参数键：{keys}"
            )
        return AgentAction(
            name,
            {"path": path, "search": search, "replace": replace},
        )
    if name == "write_file":
        path = _require_text(params, "path", name)
        content = params.get("content")
        if not isinstance(content, str):
            keys = "、".join(sorted(params)) or "（空）"
            raise ValueError(
                f"动作 {name!r} 缺少/非法 parameters.content；"
                f"收到参数键：{keys}"
            )
        return AgentAction(name, {"path": path, "content": content})
    if name == "move_file":
        src = params.get("src", params.get("source"))
        dst = params.get("dst", params.get("dest", params.get("destination")))
        if not isinstance(src, str) or not src.strip():
            raise ValueError(f"动作 {name!r} 缺少/非法 parameters.src")
        if not isinstance(dst, str) or not dst.strip():
            raise ValueError(f"动作 {name!r} 缺少/非法 parameters.dst")
        return AgentAction(
            name,
            {"src": src.strip(), "dst": dst.strip()},
        )
    if name in {"delete_file", "delete_path"}:
        return AgentAction(
            name, {"path": _require_text(params, "path", name)}
        )
    if name == "list_dir":
        return AgentAction(
            name,
            {"path": _optional_text(params, "path", ".", name)},
        )
    if name == "run_shell_ro":
        return AgentAction(
            name,
            {"command": _require_text(params, "command", name)},
        )
    raise ValueError(f"动作 {name!r} 的参数校验未实现")


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
def _summary(action: AgentAction) -> str:
    params = action.parameters
    if action.name == "run_shell_ro":
        return f"只读命令 {params['command']}"
    if action.name == "read_file":
        return f"读取 {params['path']}"
    if action.name == "read_file_range":
        return (
            f"读取 {params.get('path', '?')} 行 "
            f"{params.get('start_line', 1)}-{params.get('end_line', 100)}"
        )
    if action.name == "grep_workspace":
        return f"检索 {params.get('pattern', '?')} @ {params.get('path', '.')}"
    if action.name == "patch_file":
        return f"局部修改 {params.get('path', '?')}"
    if action.name == "list_dir":
        return f"列出 {params['path']}"
    if action.name == "write_file":
        return f"写入 {params['path']}"
    if action.name == "move_file":
        return f"重命名 {params['src']} -> {params['dst']}"
    if action.name in {"delete_file", "delete_path"}:
        return f"删除文件/目录 {params.get('path', '?')}"
    return f"删除 {params.get('path', '?')}"


def _describe(action: AgentAction) -> str:
    if action.name == "run_shell_ro":
        return f"将执行只读命令：{action.parameters['command']}"
    if action.name == "read_file":
        return f"将读取文件 {action.parameters['path']}"
    if action.name == "read_file_range":
        return (
            f"将读取文件 {action.parameters.get('path', '?')} 第 "
            f"{action.parameters.get('start_line', 1)}–"
            f"{action.parameters.get('end_line', 100)} 行"
        )
    if action.name == "grep_workspace":
        return (
            f"将检索工作区 {action.parameters.get('path', '.')}："
            f"{action.parameters.get('pattern', '?')}"
        )
    if action.name == "patch_file":
        return f"将局部修改文件 {action.parameters.get('path', '?')}"
    if action.name == "list_dir":
        return f"将列出目录 {action.parameters['path']}"
    if action.name == "write_file":
        return f"将写入文件 {action.parameters['path']}"
    if action.name == "move_file":
        return (
            f"将重命名 {action.parameters['src']} -> "
            f"{action.parameters['dst']}"
        )
    if action.name in {"delete_file", "delete_path"}:
        return (
            "将删除文件或递归目录 "
            f"{action.parameters.get('path', '?')}"
        )
    return f"将删除文件 {action.parameters.get('path', '?')}"
def _action_signature(action: AgentAction) -> tuple[str, str]:
    return (
        action.name,
        json.dumps(action.parameters, sort_keys=True),
    )
