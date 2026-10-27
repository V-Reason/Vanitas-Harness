"""提示词模板加载与变量替换。

模板只认六个占位符：{input} {context} {context_history} {date} {model} {cwd}。
未知占位符直接报 ConfigError——模板写错比模型答错更难排查，必须在启动前拦住。
{date} 默认是秒级本地时间，{cwd} 由会话层传入，供 Agent 感知工作目录。
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from .errors import ConfigError

ALLOWED = ("input", "context", "context_history", "date", "model", "cwd")
NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
VAR_RE = re.compile(r"\{([a-z][a-z0-9_]*)\}")
DEFAULT_SYSTEM = "system_default"


def template_path(prompt_dir: Path, name: str) -> Path:
    """把模板名解析为 <prompt_dir>/<name>.md，拒绝任何路径穿越。"""
    if not NAME_RE.match(name):
        raise ConfigError(
            f"模板名非法 {name!r}；只允许小写字母、数字与下划线，且以字母开头"
        )
    path = Path(prompt_dir) / f"{name}.md"
    if not path.is_file():
        raise ConfigError(f"提示词模板不存在：{path}")
    return path


def placeholders(text: str) -> set[str]:
    return set(VAR_RE.findall(text))


def load_template(prompt_dir: Path, name: str) -> str:
    path = template_path(prompt_dir, name)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"提示词模板读取失败：{path}；{exc.strerror or exc}") from exc
    if not text.strip():
        raise ConfigError(f"提示词模板为空：{path}")

    found = placeholders(text)
    unknown = sorted(found - set(ALLOWED))
    if unknown:
        raise ConfigError(
            f"模板 {name}.md 含未知占位符 {unknown}；允许：{list(ALLOWED)}"
        )
    if name.startswith("task_") and "input" not in found:
        raise ConfigError(f"任务模板 {name}.md 必须包含 {{input}} 占位符")
    return text.rstrip("\n")


def now_stamp() -> str:
    """本地时间的秒级 ISO 风格快照，供 {date} 与运行时注入复用。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def render(template: str, values: dict | None = None) -> str:
    """替换占位符；缺失的键用空串（context 在 REPL 中就是空）。"""
    merged = {
        "input": "",
        "context": "",
        "context_history": "",
        "date": now_stamp(),
        "model": "",
        "cwd": "",
    }
    for key, value in (values or {}).items():
        if key not in merged:
            raise ConfigError(f"render() 收到未知变量 {key!r}；允许：{list(ALLOWED)}")
        merged[key] = "" if value is None else str(value)
    return VAR_RE.sub(lambda match: merged[match.group(1)], template)


def system_prompt(prompt_dir: Path) -> str:
    text = load_template(prompt_dir, DEFAULT_SYSTEM)
    found = placeholders(text)
    if found:
        raise ConfigError(
            f"系统提示 {DEFAULT_SYSTEM}.md 必须全静态，发现占位符 {sorted(found)}"
        )
    return text


def task_names(prompt_dir: Path) -> list[str]:
    directory = Path(prompt_dir)
    if not directory.is_dir():
        return []
    names: list[str] = []
    for path in sorted(directory.glob("task_*.md")):
        stem = path.stem
        name = stem[len("task_"):]
        if NAME_RE.match(name):
            names.append(name)
    return names


def load_task(prompt_dir: Path, name: str) -> str:
    if not name:
        raise ConfigError("任务名不能为空；可用任务：" + "、".join(task_names(prompt_dir)))
    return load_template(prompt_dir, f"task_{name}")
