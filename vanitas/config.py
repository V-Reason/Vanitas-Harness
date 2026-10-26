"""配置加载与校验。

红线：模型名、端口、上下文、KV 类型只允许出现在 config\\ 中。
本模块只做「读 + 校验 + dataclass 视图」，不替操作者猜默认值：
任何缺失、类型错误、语义冲突都以 ConfigError 抛出，CLI 统一映射为退出码 2。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .errors import ConfigError

BASE_DIR = Path(__file__).resolve().parents[1]
SHARD_RE = re.compile(r"^(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})$")
KV_TYPES = frozenset(
    {"f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "iq4_nl"}
)
MODEL_FIELDS = frozenset(
    {
        "id",
        "display",
        "path",
        "port",
        "alias",
        "shards",
        "role",
        "template_kwargs",
        "profiles",
        "default_profile",
        "repeat_penalty",
    }
)
PROFILE_FIELDS = frozenset({"ctx", "kv_k", "kv_v", "desc"})
SETTINGS_FIELDS = frozenset(
    {
        "version",
        "host",
        "request",
        "launcher",
        "repl",
        "chat",
        "agent",
        "tools",
        "batch",
        "translate",
        "ui",
    }
)
REQUEST_FIELDS = frozenset(
    {
        "timeout_sec",
        "max_retries",
        "backoff_base_ms",
        "backoff_jitter_ms",
        "health_timeout_sec",
        "status_probe_timeout_sec",
        "progress_interval_ms",
    }
)
LAUNCHER_FIELDS = frozenset(
    {
        "health_timeout_sec",
        "health_poll_interval_ms",
        "kill_wait_sec",
        "log_tail_lines",
    }
)
REPL_FIELDS = frozenset(
    {
        "history_file",
        "max_context_tokens",
        "stream",
        "context_reserve_tokens",
        "min_output_tokens",
    }
)
CHAT_FIELDS = frozenset({"temperature"})
AGENT_FIELDS = frozenset(
    {
        "max_steps",
        "max_batch_actions",
        "max_feedback_chars",
        "auto_actions",
        "confirm_actions",
        "action_aliases",
    }
)
TOOLS_FIELDS = frozenset(
    {
        "exec_enabled",
        "exec_require_confirm",
        "max_read_bytes",
        "max_list_entries",
        "move_illegal_chars",
        "manual_exec_prefixes",
        "destructive_commands",
        "readonly_shell_prefixes",
        "readonly_blocked_flags",
        "readonly_blocked_prefixes",
        "git_blocked_options",
        "git_blocked_prefixes",
    }
)
BATCH_FIELDS = frozenset(
    {"default_max_tokens", "max_tokens_limit", "temperature"}
)
TRANSLATE_FIELDS = frozenset(
    {
        "source_suffixes",
        "chunk_chars",
        "output_suffix",
        "context_window",
        "context_summary_chars",
        "min_output_tokens",
        "output_ratio",
        "prompt_safety_tokens",
        "monitor_lines",
        "temperature",
        "max_split_depth",
    }
)
UI_FIELDS = frozenset({"color", "spinner_interval_ms", "theme"})
THEME_KEYS = frozenset({"prompt", "success", "error", "warn", "info", "muted", "accent"})
COLOR_MODES = frozenset({"auto", "always", "never"})
# 与 vanitas/ui.py::COLOR_NAMES 逐字一致。刻意本地再定义一份：config 是最底层，
# 不 import ui，避免 ui 将来引入 config 时出现循环依赖。
COLOR_NAMES = frozenset(
    {
        "black",
        "red",
        "green",
        "yellow",
        "blue",
        "magenta",
        "cyan",
        "white",
        "bright_black",
        "bright_red",
        "bright_green",
        "bright_yellow",
        "bright_blue",
        "bright_magenta",
        "bright_cyan",
        "bright_white",
        "default",
    }
)
DEFAULT_THEME = {
    "prompt": "cyan",
    "success": "green",
    "error": "red",
    "warn": "yellow",
    "info": "cyan",
    "muted": "bright_black",
    "accent": "magenta",
}

# Agent 动作 schema 是固定协议；settings 只能调整审批策略和别名，不能新增动作。
AGENT_ACTION_NAMES = frozenset(
    {"read_file", "list_dir", "run_shell_ro", "write_file", "delete_file", "move_file"}
)
DEFAULT_AUTO_ACTIONS = frozenset({"read_file", "list_dir", "run_shell_ro"})
DEFAULT_CONFIRM_ACTIONS = frozenset({"write_file", "delete_file", "move_file"})
DEFAULT_ACTION_ALIASES = {"shell": "run_shell_ro"}

DEFAULT_MANUAL_EXEC_PREFIXES = (
    "echo",
    "dir",
    "type",
    "python --version",
    "git status",
    "git log",
    "git diff",
    "del",
    "erase",
    "ren",
    "rename",
    "copy",
    "move",
)
CORE_DESTRUCTIVE_COMMANDS = frozenset(
    {
        "del",
        "erase",
        "rd",
        "rmdir",
        "rm",
        "mv",
        "move",
        "ren",
        "rename",
        "copy",
        "xcopy",
        "robocopy",
        "mkdir",
        "md",
        "format",
    }
)
DEFAULT_READONLY_SHELL_PREFIXES = frozenset(
    {
        "dir",
        "ls",
        "type",
        "cat",
        "echo",
        "findstr",
        "find",
        "pwd",
        "where",
        "head",
        "tail",
        "wc",
        "grep",
        "git status",
        "git log",
        "git diff",
        "git show",
        "git rev-parse",
    }
)
CORE_READONLY_BLOCKED_FLAGS = frozenset(
    {"-o", "-delete", "-fls", "-ok", "-okdir"}
)
CORE_READONLY_BLOCKED_PREFIXES = (
    "-exec",
    "--output",
    "-fprint",
    "-fprintf",
)
CORE_GIT_BLOCKED_OPTIONS = frozenset(
    {
        "-c",
        "-C",
        "--git-dir",
        "--work-tree",
        "--exec-path",
        "--namespace",
        "--no-index",
        "--output",
    }
)
CORE_GIT_BLOCKED_PREFIXES = (
    "-c",
    "-C",
    "--git-dir=",
    "--work-tree=",
    "--exec-path=",
)
DEFAULT_MOVE_ILLEGAL_CHARS = '#<>"|?*'


@dataclass(frozen=True)
class ProfileSpec:
    key: str
    ctx: int
    kv_k: str
    kv_v: str
    desc: str


@dataclass(frozen=True)
class ModelSpec:
    id: str
    display: str
    path: Path
    port: int
    alias: str
    shards: int
    role: str
    template_kwargs: dict
    profiles: dict
    default_profile: str
    repeat_penalty: float = 1.0

    @property
    def profile_keys(self) -> list[str]:
        return sorted(self.profiles, key=int)


@dataclass(frozen=True)
class RequestSettings:
    timeout_sec: float
    max_retries: int
    backoff_base_ms: int
    backoff_jitter_ms: int
    health_timeout_sec: float = 3.0
    status_probe_timeout_sec: float = 0.6
    progress_interval_ms: int = 200


@dataclass(frozen=True)
class LauncherSettings:
    health_timeout_sec: float
    health_poll_interval_ms: int
    kill_wait_sec: float
    log_tail_lines: int = 15


@dataclass(frozen=True)
class ReplSettings:
    history_file: Path
    max_context_tokens: int
    stream: bool
    context_reserve_tokens: int = 512
    min_output_tokens: int = 256


@dataclass(frozen=True)
class ChatSettings:
    temperature: float = 0.2


@dataclass(frozen=True)
class AgentSettings:
    max_steps: int = 10
    max_batch_actions: int = 100
    max_feedback_chars: int = 8000
    auto_actions: frozenset[str] = DEFAULT_AUTO_ACTIONS
    confirm_actions: frozenset[str] = DEFAULT_CONFIRM_ACTIONS
    action_aliases: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_ACTION_ALIASES)
    )


@dataclass(frozen=True)
class BatchSettings:
    default_max_tokens: int = 512
    max_tokens_limit: int = 8192
    temperature: float = 0.2


@dataclass(frozen=True)
class ToolsSettings:
    exec_enabled: bool
    exec_require_confirm: bool
    max_read_bytes: int = 256 * 1024
    max_list_entries: int = 200
    move_illegal_chars: str = DEFAULT_MOVE_ILLEGAL_CHARS
    manual_exec_prefixes: tuple[str, ...] = DEFAULT_MANUAL_EXEC_PREFIXES
    destructive_commands: frozenset[str] = CORE_DESTRUCTIVE_COMMANDS
    readonly_shell_prefixes: frozenset[str] = DEFAULT_READONLY_SHELL_PREFIXES
    readonly_blocked_flags: frozenset[str] = CORE_READONLY_BLOCKED_FLAGS
    readonly_blocked_prefixes: tuple[str, ...] = CORE_READONLY_BLOCKED_PREFIXES
    git_blocked_options: frozenset[str] = CORE_GIT_BLOCKED_OPTIONS
    git_blocked_prefixes: tuple[str, ...] = CORE_GIT_BLOCKED_PREFIXES


@dataclass(frozen=True)
class TranslateSettings:
    source_suffixes: tuple[str, ...]
    chunk_chars: int
    output_suffix: str
    context_window: int = 5
    context_summary_chars: int = 200
    min_output_tokens: int = 512
    output_ratio: float = 2.5
    prompt_safety_tokens: int = 64
    monitor_lines: int = 5
    temperature: float = 0.2
    max_split_depth: int = 3


@dataclass(frozen=True)
class UiTheme:
    prompt: str
    success: str
    error: str
    warn: str
    info: str
    muted: str
    accent: str


@dataclass(frozen=True)
class UiSettings:
    color: str
    theme: UiTheme
    spinner_interval_ms: int = 300


def _default_ui() -> UiSettings:
    return UiSettings(
        color="auto",
        theme=UiTheme(**DEFAULT_THEME),
        spinner_interval_ms=300,
    )


@dataclass(frozen=True)
class Settings:
    version: int
    host: str
    request: RequestSettings
    launcher: LauncherSettings
    repl: ReplSettings
    chat: ChatSettings
    agent: AgentSettings
    tools: ToolsSettings
    batch: BatchSettings
    translate: TranslateSettings
    ui: UiSettings


@dataclass(frozen=True)
class AppConfig:
    base_dir: Path
    config_dir: Path
    models: tuple
    settings: Settings

    def model(self, model_id: str) -> ModelSpec:
        for item in self.models:
            if item.id == model_id:
                return item
        known = "、".join(item.id for item in self.models)
        raise ConfigError(f"未知模型 id {model_id!r}；可用：{known}")

    def model_by_role(self, role: str) -> ModelSpec:
        for item in self.models:
            if item.role == role:
                return item
        raise ConfigError(f"config\\models.json 中没有 role={role!r} 的模型")

    def effective_port(self, model: ModelSpec) -> int:
        override = os.environ.get("VANITAS_PORT")
        if override:
            return _parse_port(override, "环境变量 VANITAS_PORT")
        return model.port

    def resolve_profile(self, model: ModelSpec, key: str | None = None) -> ProfileSpec:
        chosen = (key or os.environ.get("VANITAS_PROFILE") or model.default_profile).strip()
        profile = model.profiles.get(chosen)
        if profile is None:
            available = "、".join(model.profile_keys)
            raise ConfigError(
                f"模型 {model.id} 没有档位 {chosen!r}；可用档位：{available}"
            )
        return profile


# --------------------------------------------------------------------------- #
# 基础校验工具：错误消息必须能直接指向出错的字段
# --------------------------------------------------------------------------- #


def _fail(label: str, message: str) -> None:
    raise ConfigError(f"{label}: {message}")


def _reject_unknown(mapping: dict, allowed: frozenset, label: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        _fail(label, f"含未知字段 {unknown}；允许的字段：{sorted(allowed)}")


def _as_mapping(value: object, label: str) -> dict:
    if not isinstance(value, dict):
        _fail(label, "必须是 JSON 对象")
    return value


def _as_str(mapping: dict, key: str, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        _fail(f"{label}.{key}", "必须是非空字符串")
    return value.strip()


def _as_int(mapping: dict, key: str, label: str, minimum: int | None = None,
            maximum: int | None = None) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{label}.{key}", f"必须是整数，实际为 {value!r}")
    if minimum is not None and value < minimum:
        _fail(f"{label}.{key}", f"必须 ≥ {minimum}，实际为 {value}")
    if maximum is not None and value > maximum:
        _fail(f"{label}.{key}", f"必须 ≤ {maximum}，实际为 {value}")
    return value


def _as_number(
    mapping: dict,
    key: str,
    label: str,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label}.{key}", f"必须是数字，实际为 {value!r}")
    if value < minimum:
        _fail(f"{label}.{key}", f"必须 ≥ {minimum}，实际为 {value}")
    if maximum is not None and value > maximum:
        _fail(f"{label}.{key}", f"必须 ≤ {maximum}，实际为 {value}")
    return float(value)


def _as_optional_int(
    mapping: dict,
    key: str,
    label: str,
    default: int,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if key not in mapping:
        return default
    return _as_int(mapping, key, label, minimum=minimum, maximum=maximum)


def _as_optional_number(
    mapping: dict,
    key: str,
    label: str,
    default: float,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> float:
    if key not in mapping:
        return default
    return _as_number(
        mapping, key, label, minimum=minimum, maximum=maximum
    )


def _as_str_list(
    mapping: dict,
    key: str,
    label: str,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    raw = mapping.get(key, list(default))
    if not isinstance(raw, list) or not raw:
        _fail(f"{label}.{key}", "必须是非空字符串数组")
    cleaned: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            _fail(f"{label}.{key}", f"每项必须是非空字符串，实际为 {item!r}")
        value = item.strip()
        if value not in cleaned:
            cleaned.append(value)
    return tuple(cleaned)


def _as_str_set(
    mapping: dict,
    key: str,
    label: str,
    default: frozenset[str],
) -> frozenset[str]:
    return frozenset(
        _as_str_list(mapping, key, label, tuple(sorted(default)))
    )


def _as_bool(mapping: dict, key: str, label: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        _fail(f"{label}.{key}", f"必须是布尔值，实际为 {value!r}")
    return value


def _require_superset(
    value: frozenset[str], required: frozenset[str], label: str
) -> None:
    missing = sorted(required - value)
    if missing:
        _fail(label, f"缺少安全底线条目 {missing}，只能追加，不能删除")


def _load_agent(entry: object | None) -> "AgentSettings":
    defaults = AgentSettings()
    if entry is None:
        return defaults
    label = "settings.agent"
    _as_mapping(entry, label)
    _reject_unknown(entry, AGENT_FIELDS, label)
    max_steps = _as_optional_int(
        entry, "max_steps", label, defaults.max_steps, minimum=1, maximum=100
    )
    max_batch_actions = _as_optional_int(
        entry,
        "max_batch_actions",
        label,
        defaults.max_batch_actions,
        minimum=1,
        maximum=1000,
    )
    max_feedback_chars = _as_optional_int(
        entry,
        "max_feedback_chars",
        label,
        defaults.max_feedback_chars,
        minimum=256,
    )
    auto_actions = frozenset(
        _as_str_list(
            entry,
            "auto_actions",
            label,
            tuple(sorted(defaults.auto_actions)),
        )
    )
    confirm_actions = frozenset(
        _as_str_list(
            entry,
            "confirm_actions",
            label,
            tuple(sorted(defaults.confirm_actions)),
        )
    )
    overlap = auto_actions & confirm_actions
    if overlap:
        _fail(label, f"auto_actions 与 confirm_actions 不能重叠：{sorted(overlap)}")
    unknown = (auto_actions | confirm_actions) - AGENT_ACTION_NAMES
    missing = AGENT_ACTION_NAMES - (auto_actions | confirm_actions)
    if unknown:
        _fail(label, f"包含未知动作：{sorted(unknown)}")
    if missing:
        _fail(label, f"审批策略缺少动作：{sorted(missing)}")

    raw_aliases = entry.get("action_aliases", defaults.action_aliases)
    if not isinstance(raw_aliases, dict):
        _fail(f"{label}.action_aliases", "必须是 JSON 对象")
    aliases: dict[str, str] = {}
    for alias, target in raw_aliases.items():
        if not isinstance(alias, str) or not alias.strip():
            _fail(f"{label}.action_aliases", "别名必须是非空字符串")
        if not isinstance(target, str) or target not in AGENT_ACTION_NAMES:
            _fail(
                f"{label}.action_aliases.{alias}",
                f"目标必须是正式动作，实际为 {target!r}",
            )
        alias = alias.strip()
        if alias in AGENT_ACTION_NAMES:
            _fail(f"{label}.action_aliases", f"别名不能与正式动作同名：{alias!r}")
        aliases[alias] = target
    return AgentSettings(
        max_steps=max_steps,
        max_batch_actions=max_batch_actions,
        max_feedback_chars=max_feedback_chars,
        auto_actions=auto_actions,
        confirm_actions=confirm_actions,
        action_aliases=aliases,
    )


def _load_chat(entry: object | None) -> "ChatSettings":
    defaults = ChatSettings()
    if entry is None:
        return defaults
    label = "settings.chat"
    _as_mapping(entry, label)
    _reject_unknown(entry, CHAT_FIELDS, label)
    return ChatSettings(
        temperature=_as_optional_number(
            entry,
            "temperature",
            label,
            defaults.temperature,
            minimum=0.0,
            maximum=2.0,
        )
    )


def _load_batch(entry: object | None) -> "BatchSettings":
    defaults = BatchSettings()
    if entry is None:
        return defaults
    label = "settings.batch"
    _as_mapping(entry, label)
    _reject_unknown(entry, BATCH_FIELDS, label)
    default_max_tokens = _as_optional_int(
        entry,
        "default_max_tokens",
        label,
        defaults.default_max_tokens,
        minimum=1,
    )
    max_tokens_limit = _as_optional_int(
        entry,
        "max_tokens_limit",
        label,
        defaults.max_tokens_limit,
        minimum=1,
    )
    if default_max_tokens > max_tokens_limit:
        _fail(
            label,
            "default_max_tokens 不能大于 max_tokens_limit",
        )
    return BatchSettings(
        default_max_tokens=default_max_tokens,
        max_tokens_limit=max_tokens_limit,
        temperature=_as_optional_number(
            entry,
            "temperature",
            label,
            defaults.temperature,
            minimum=0.0,
            maximum=2.0,
        ),
    )


def _load_tools(entry: object) -> "ToolsSettings":
    label = "settings.tools"
    _as_mapping(entry, label)
    _reject_unknown(entry, TOOLS_FIELDS, label)
    defaults = ToolsSettings(
        exec_enabled=_as_bool(entry, "exec_enabled", label),
        exec_require_confirm=_as_bool(entry, "exec_require_confirm", label),
    )
    max_read_bytes = _as_optional_int(
        entry, "max_read_bytes", label, defaults.max_read_bytes, minimum=1
    )
    max_list_entries = _as_optional_int(
        entry, "max_list_entries", label, defaults.max_list_entries, minimum=1
    )
    move_illegal_chars = entry.get(
        "move_illegal_chars", defaults.move_illegal_chars
    )
    if not isinstance(move_illegal_chars, str):
        _fail(f"{label}.move_illegal_chars", "必须是字符串")
    destructive = _as_str_set(
        entry,
        "destructive_commands",
        label,
        CORE_DESTRUCTIVE_COMMANDS,
    )
    _require_superset(destructive, CORE_DESTRUCTIVE_COMMANDS, label)
    readonly_blocked_flags = _as_str_set(
        entry,
        "readonly_blocked_flags",
        label,
        CORE_READONLY_BLOCKED_FLAGS,
    )
    _require_superset(
        readonly_blocked_flags, CORE_READONLY_BLOCKED_FLAGS, label
    )
    readonly_blocked_prefixes = _as_str_list(
        entry,
        "readonly_blocked_prefixes",
        label,
        CORE_READONLY_BLOCKED_PREFIXES,
    )
    _require_superset(
        frozenset(readonly_blocked_prefixes),
        frozenset(CORE_READONLY_BLOCKED_PREFIXES),
        label,
    )
    git_blocked_options = _as_str_set(
        entry,
        "git_blocked_options",
        label,
        CORE_GIT_BLOCKED_OPTIONS,
    )
    _require_superset(
        git_blocked_options, CORE_GIT_BLOCKED_OPTIONS, label
    )
    git_blocked_prefixes = _as_str_list(
        entry,
        "git_blocked_prefixes",
        label,
        CORE_GIT_BLOCKED_PREFIXES,
    )
    _require_superset(
        frozenset(git_blocked_prefixes),
        frozenset(CORE_GIT_BLOCKED_PREFIXES),
        label,
    )
    return ToolsSettings(
        exec_enabled=defaults.exec_enabled,
        exec_require_confirm=defaults.exec_require_confirm,
        max_read_bytes=max_read_bytes,
        max_list_entries=max_list_entries,
        move_illegal_chars=move_illegal_chars,
        manual_exec_prefixes=_as_str_list(
            entry,
            "manual_exec_prefixes",
            label,
            DEFAULT_MANUAL_EXEC_PREFIXES,
        ),
        destructive_commands=destructive,
        readonly_shell_prefixes=_as_str_set(
            entry,
            "readonly_shell_prefixes",
            label,
            DEFAULT_READONLY_SHELL_PREFIXES,
        ),
        readonly_blocked_flags=readonly_blocked_flags,
        readonly_blocked_prefixes=readonly_blocked_prefixes,
        git_blocked_options=git_blocked_options,
        git_blocked_prefixes=git_blocked_prefixes,
    )


def _parse_port(raw: object, label: str) -> int:
    try:
        port = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} 必须是整数端口，实际为 {raw!r}") from exc
    if not 1 <= port <= 65535:
        raise ConfigError(f"{label} 必须在 1–65535 之间，实际为 {port}")
    return port


def _read_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise ConfigError(f"{label} 不存在：{path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"{label} 读取失败：{path}；{exc.strerror or exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"{label} 不是合法 JSON：{path}；第 {exc.lineno} 行第 {exc.colno} 列：{exc.msg}"
        ) from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{label} 顶层必须是 JSON 对象：{path}")
    return data


# --------------------------------------------------------------------------- #
# 模型定义
# --------------------------------------------------------------------------- #


def _resolve_path(raw: str, base_dir: Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(raw))
    candidate = Path(expanded)
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve()


def _validate_shard_naming(path: Path, shards: int, label: str) -> None:
    match = SHARD_RE.match(path.stem)
    if shards > 1:
        if match is None:
            _fail(
                f"{label}.path",
                f"shards={shards} 要求首片文件名形如 <前缀>-00001-of-{shards:05d}.gguf，实际为 {path.name}",
            )
        index, total = int(match.group("index")), int(match.group("total"))
        if index != 1:
            _fail(f"{label}.path", f"必须指向第 1 个分片，实际收到第 {index} 片：{path.name}")
        if total != shards:
            _fail(
                f"{label}.path",
                f"文件名声明总分片 {total}，与 shards={shards} 不一致：{path.name}",
            )
    elif match is not None:
        _fail(
            f"{label}.path",
            f"shards=1 但文件名带分片后缀，分片模型必须声明 shards>1：{path.name}",
        )


def _load_profile(entry: dict, label: str) -> ProfileSpec:
    _as_mapping(entry, label)
    _reject_unknown(entry, PROFILE_FIELDS, label)
    ctx = _as_int(entry, "ctx", label, minimum=256)
    kv_k = _as_str(entry, "kv_k", label)
    kv_v = _as_str(entry, "kv_v", label)
    for field, value in (("kv_k", kv_k), ("kv_v", kv_v)):
        if value not in KV_TYPES:
            _fail(
                f"{label}.{field}",
                f"不支持的 KV 类型 {value!r}；允许：{sorted(KV_TYPES)}",
            )
    desc = _as_str(entry, "desc", label)
    return ProfileSpec(key="", ctx=ctx, kv_k=kv_k, kv_v=kv_v, desc=desc)


def _load_model(entry: object, index: int, base_dir: Path) -> ModelSpec:
    label = f"models[{index}]"
    _as_mapping(entry, label)
    _reject_unknown(entry, MODEL_FIELDS, label)
    model_id = _as_str(entry, "id", label)
    label = f"models[{index}:{model_id}]"
    display = _as_str(entry, "display", label)
    raw_path = _as_str(entry, "path", label)
    port = _parse_port(entry.get("port"), f"{label}.port")
    alias = _as_str(entry, "alias", label)
    shards = _as_int(entry, "shards", label, minimum=1, maximum=99999)
    role = _as_str(entry, "role", label)
    path = _resolve_path(raw_path, base_dir)
    _validate_shard_naming(path, shards, label)

    kwargs = _as_mapping(entry.get("template_kwargs"), f"{label}.template_kwargs")
    for key, value in kwargs.items():
        if not isinstance(key, str) or not key:
            _fail(f"{label}.template_kwargs", "键必须是非空字符串")
        if isinstance(value, (dict, list)):
            _fail(
                f"{label}.template_kwargs.{key}",
                "chat_template_kwargs 只接受标量值：字符串、数字、布尔、null",
            )

    raw_profiles = _as_mapping(entry.get("profiles"), f"{label}.profiles")
    if not raw_profiles:
        _fail(f"{label}.profiles", "至少需要一个档位")
    profiles: dict[str, ProfileSpec] = {}
    for key, value in raw_profiles.items():
        if not isinstance(key, str) or not key.isdigit():
            _fail(f"{label}.profiles", f"档位键必须是非负整数字符串，实际为 {key!r}")
        profile = _load_profile(value, f"{label}.profiles[{key}]")
        profiles[key] = ProfileSpec(
            key=key, ctx=profile.ctx, kv_k=profile.kv_k, kv_v=profile.kv_v, desc=profile.desc
        )

    default_profile = _as_str(entry, "default_profile", label)
    if default_profile not in profiles:
        _fail(
            f"{label}.default_profile",
            f"默认档位 {default_profile!r} 不在 profiles 中；可用：{sorted(profiles)}",
        )

    repeat_penalty = 1.0
    if "repeat_penalty" in entry:
        repeat_penalty = _as_number(
            entry, "repeat_penalty", label, minimum=1.0
        )

    return ModelSpec(
        id=model_id,
        display=display,
        path=path,
        port=port,
        alias=alias,
        shards=shards,
        role=role,
        template_kwargs=dict(kwargs),
        profiles=profiles,
        default_profile=default_profile,
        repeat_penalty=repeat_penalty,
    )


def _load_models(document: dict, base_dir: Path) -> tuple:
    if document.get("version") != 1:
        _fail("config\\models.json.version", "只支持 version=1")
    raw_models = document.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        _fail("config\\models.json.models", "必须是非空数组")
    models = tuple(
        _load_model(entry, index, base_dir) for index, entry in enumerate(raw_models)
    )

    seen_ids: set[str] = set()
    seen_ports: dict[int, str] = {}
    seen_aliases: set[str] = set()
    for model in models:
        if model.id in seen_ids:
            raise ConfigError(f"models[].id 重复：{model.id}")
        seen_ids.add(model.id)
        if model.port in seen_ports:
            raise ConfigError(
                f"端口 {model.port} 被 {seen_ports[model.port]} 与 {model.id} 同时占用；"
                "每个模型必须使用独立端口"
            )
        seen_ports[model.port] = model.id
        if model.alias in seen_aliases:
            raise ConfigError(f"models[].alias 重复：{model.alias}；/v1/models 会分不清模型")
        seen_aliases.add(model.alias)
    return models


# --------------------------------------------------------------------------- #
# 全局设置
# --------------------------------------------------------------------------- #


def _load_ui(entry: object) -> UiSettings:
    """严格校验 settings.ui；整段缺失时由 _load_settings 回落到默认主题。"""
    label = "settings.ui"
    _as_mapping(entry, label)
    _reject_unknown(entry, UI_FIELDS, label)

    raw_color = entry.get("color")
    color = raw_color.strip() if isinstance(raw_color, str) else ""
    if color not in COLOR_MODES:
        _fail(f"{label}.color", f"必须是 auto/always/never，实际为 {raw_color!r}")

    theme_label = f"{label}.theme"
    raw_theme = _as_mapping(entry.get("theme"), theme_label)
    _reject_unknown(raw_theme, THEME_KEYS, theme_label)
    missing = sorted(THEME_KEYS - set(raw_theme))
    if missing:
        _fail(theme_label, f"缺少字段 {missing}；必须完整给出 {sorted(THEME_KEYS)}")
    values: dict[str, str] = {}
    for key in sorted(THEME_KEYS):
        value = _as_str(raw_theme, key, theme_label)
        if value not in COLOR_NAMES:
            _fail(f"{theme_label}.{key}", f"不支持的颜色名 {value!r}；允许：{sorted(COLOR_NAMES)}")
        values[key] = value
    spinner_interval_ms = _as_optional_int(
        entry,
        "spinner_interval_ms",
        label,
        300,
        minimum=50,
        maximum=5000,
    )
    return UiSettings(
        color=color,
        theme=UiTheme(**values),
        spinner_interval_ms=spinner_interval_ms,
    )


def _load_settings(document: dict) -> Settings:
    _reject_unknown(document, SETTINGS_FIELDS, "config\\settings.json")
    if document.get("version") != 1:
        _fail("config\\settings.json.version", "只支持 version=1")
    host = _as_str(document, "host", "settings")

    request = _as_mapping(document.get("request"), "settings.request")
    _reject_unknown(request, REQUEST_FIELDS, "settings.request")
    request_settings = RequestSettings(
        timeout_sec=_as_number(request, "timeout_sec", "settings.request", minimum=0.1),
        max_retries=_as_int(request, "max_retries", "settings.request", minimum=0, maximum=10),
        backoff_base_ms=_as_int(request, "backoff_base_ms", "settings.request", minimum=0),
        backoff_jitter_ms=_as_int(request, "backoff_jitter_ms", "settings.request", minimum=0),
        health_timeout_sec=_as_optional_number(
            request,
            "health_timeout_sec",
            "settings.request",
            3.0,
            minimum=0.1,
        ),
        status_probe_timeout_sec=_as_optional_number(
            request,
            "status_probe_timeout_sec",
            "settings.request",
            0.6,
            minimum=0.05,
        ),
        progress_interval_ms=_as_optional_int(
            request,
            "progress_interval_ms",
            "settings.request",
            200,
            minimum=0,
        ),
    )

    launcher = _as_mapping(document.get("launcher"), "settings.launcher")
    _reject_unknown(launcher, LAUNCHER_FIELDS, "settings.launcher")
    launcher_settings = LauncherSettings(
        health_timeout_sec=_as_number(
            launcher, "health_timeout_sec", "settings.launcher", minimum=1.0
        ),
        health_poll_interval_ms=_as_int(
            launcher, "health_poll_interval_ms", "settings.launcher", minimum=50
        ),
        kill_wait_sec=_as_number(launcher, "kill_wait_sec", "settings.launcher", minimum=0.0),
        log_tail_lines=_as_optional_int(
            launcher,
            "log_tail_lines",
            "settings.launcher",
            15,
            minimum=1,
            maximum=100,
        ),
    )

    repl = _as_mapping(document.get("repl"), "settings.repl")
    _reject_unknown(repl, REPL_FIELDS, "settings.repl")
    history_raw = _as_str(repl, "history_file", "settings.repl")
    context_reserve = (
        _as_int(repl, "context_reserve_tokens", "settings.repl", minimum=64)
        if "context_reserve_tokens" in repl
        else 512
    )
    min_output = (
        _as_int(repl, "min_output_tokens", "settings.repl", minimum=16)
        if "min_output_tokens" in repl
        else 256
    )
    repl_settings = ReplSettings(
        history_file=Path(os.path.expanduser(history_raw)),
        max_context_tokens=_as_int(
            repl, "max_context_tokens", "settings.repl", minimum=256
        ),
        stream=_as_bool(repl, "stream", "settings.repl"),
        context_reserve_tokens=context_reserve,
        min_output_tokens=min_output,
    )

    chat_settings = _load_chat(document.get("chat"))
    agent_settings = _load_agent(document.get("agent"))
    tools_settings = _load_tools(document.get("tools"))
    batch_settings = _load_batch(document.get("batch"))

    translate = _as_mapping(document.get("translate"), "settings.translate")
    _reject_unknown(translate, TRANSLATE_FIELDS, "settings.translate")
    suffixes = translate.get("source_suffixes")
    if not isinstance(suffixes, list) or not suffixes:
        _fail("settings.translate.source_suffixes", "必须是非空字符串数组")
    cleaned_suffixes: list[str] = []
    for item in suffixes:
        if not isinstance(item, str) or not item.startswith(".") or len(item) < 2:
            _fail(
                "settings.translate.source_suffixes",
                f"后缀必须以 . 开头，实际为 {item!r}",
            )
        cleaned_suffixes.append(item.lower())
    output_suffix = _as_str(translate, "output_suffix", "settings.translate")
    if not output_suffix.startswith("."):
        _fail("settings.translate.output_suffix", "必须以 . 开头，例如 .zh")
    context_window = (
        _as_int(translate, "context_window", "settings.translate", minimum=0, maximum=10)
        if "context_window" in translate
        else 5
    )
    context_summary_chars = (
        _as_int(
            translate,
            "context_summary_chars",
            "settings.translate",
            minimum=40,
            maximum=2000,
        )
        if "context_summary_chars" in translate
        else 200
    )
    translate_min_output = (
        _as_int(translate, "min_output_tokens", "settings.translate", minimum=64)
        if "min_output_tokens" in translate
        else 512
    )
    output_ratio = (
        _as_number(translate, "output_ratio", "settings.translate", minimum=1.0)
        if "output_ratio" in translate
        else 2.5
    )
    prompt_safety = (
        _as_int(translate, "prompt_safety_tokens", "settings.translate", minimum=0)
        if "prompt_safety_tokens" in translate
        else 64
    )
    monitor_lines = (
        _as_int(translate, "monitor_lines", "settings.translate", minimum=1, maximum=10)
        if "monitor_lines" in translate
        else 5
    )
    translate_temperature = _as_optional_number(
        translate,
        "temperature",
        "settings.translate",
        0.2,
        minimum=0.0,
        maximum=2.0,
    )
    max_split_depth = _as_optional_int(
        translate,
        "max_split_depth",
        "settings.translate",
        3,
        minimum=1,
        maximum=10,
    )
    translate_settings = TranslateSettings(
        source_suffixes=tuple(cleaned_suffixes),
        chunk_chars=_as_int(translate, "chunk_chars", "settings.translate", minimum=50),
        output_suffix=output_suffix,
        context_window=context_window,
        context_summary_chars=context_summary_chars,
        min_output_tokens=translate_min_output,
        output_ratio=output_ratio,
        prompt_safety_tokens=prompt_safety,
        monitor_lines=monitor_lines,
        temperature=translate_temperature,
        max_split_depth=max_split_depth,
    )

    ui_settings = _load_ui(document["ui"]) if "ui" in document else _default_ui()

    return Settings(
        version=1,
        host=host,
        request=request_settings,
        launcher=launcher_settings,
        repl=repl_settings,
        chat=chat_settings,
        agent=agent_settings,
        tools=tools_settings,
        batch=batch_settings,
        translate=translate_settings,
        ui=ui_settings,
    )


def load_config(config_dir: Path | str | None = None, base_dir: Path | None = None) -> AppConfig:
    """读取并校验配置。

    优先级：显式参数 > 环境变量 VANITAS_CONFIG_DIR > <仓库根>/config。
    """
    root = Path(base_dir) if base_dir is not None else BASE_DIR
    if config_dir is not None:
        directory = Path(config_dir)
    else:
        env_dir = os.environ.get("VANITAS_CONFIG_DIR")
        directory = Path(env_dir) if env_dir else root / "config"
    directory = directory.expanduser().resolve()

    if not directory.is_dir():
        raise ConfigError(f"配置目录不存在：{directory}")
    prompt_dir = directory / "prompts"
    if not prompt_dir.is_dir():
        raise ConfigError(f"提示词目录不存在：{prompt_dir}")

    models_doc = _read_json(directory / "models.json", "config\\models.json")
    settings_doc = _read_json(directory / "settings.json", "config\\settings.json")
    models = _load_models(models_doc, root)
    settings = _load_settings(settings_doc)
    return AppConfig(
        base_dir=root, config_dir=directory, models=models, settings=settings
    )
