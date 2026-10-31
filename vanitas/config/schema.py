"""配置 schema：全部 dataclass 与字段常量。本模块不做任何 I/O。"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import ConfigError
from .preferences import Preferences

# 相对 vanitas/config/schema.py：parents[0]=config, parents[1]=vanitas, parents[2]=仓库根
BASE_DIR = Path(__file__).resolve().parents[2]

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
        "defaults",
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
        "idle_timeout_sec",
        "watchdog_interval_sec",
        "context_collapse_ratio",
        "context_collapse_keep_rounds",
        "context_collapse_min_chars",
        "ctx_warn_pct",
        "ctx_danger_pct",
        "status_bar_width",
    }
)
CHAT_FIELDS = frozenset({"temperature"})
AGENT_FIELDS = frozenset(
    {
        "max_steps",
        "max_batch_actions",
        "max_feedback_chars",
        "max_parse_retries",
        "stagnation_limit",
        "feedback_head_lines",
        "feedback_tail_lines",
        "auto_actions",
        "confirm_actions",
        "action_aliases",
    }
)
# exec_whitelist 已退役；故意不在允许字段中，配置出现即被 _reject_unknown 拒绝。
TOOLS_FIELDS = frozenset(
    {
        "exec_enabled",
        "exec_require_confirm",
        "project_prefixes",  # 项目级只读白名单
        "max_read_bytes",
        "max_list_entries",
        "max_patch_bytes",
        "max_range_lines",
        "max_grep_results",
        "max_grep_file_bytes",
        "grep_ignore_dirs",
        "max_capture_bytes",
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
        "top_p",
        "max_split_depth",
        "default_preset",
        "environment",
        "strip_trailing_periods",
        "enable_server_stop_tokens",
        "presets",
    }
)
ENVIRONMENT_NAMES = frozenset({"production", "experimental"})
PRESET_FIELDS = frozenset(
    {
        "task_template",
        "window_size",
        "max_window_chars",
        "lookahead_cues",
        "profile",
        "context_window",
        "context_summary_chars",
        "chunk_chars",
    }
)
DEFAULTS_FIELDS = frozenset({"agent", "chat", "translate", "batch"})
DEFAULTS_ROLE_FALLBACK = {
    "agent": "general",
    "chat": "general",
    "translate": "translate",
    "batch": "general",
}
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
    {
        "read_file",
        "read_file_range",
        "grep_workspace",
        "list_dir",
        "run_shell_ro",
        "write_file",
        "patch_file",
        "delete_file",
        "delete_path",
        "move_file",
    }
)
DEFAULT_AUTO_ACTIONS = frozenset(
    {"read_file", "read_file_range", "grep_workspace", "list_dir", "run_shell_ro"}
)
DEFAULT_CONFIRM_ACTIONS = frozenset(
    {"write_file", "patch_file", "delete_file", "delete_path", "move_file"}
)
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
        # 项目级白名单已迁至 settings.tools.project_prefixes。
        "python -m unittest",
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
    idle_timeout_sec: float = 600.0
    watchdog_interval_sec: float = 10.0
    context_collapse_ratio: float = 0.65
    context_collapse_keep_rounds: int = 2
    context_collapse_min_chars: int = 1000
    ctx_warn_pct: int = 60
    ctx_danger_pct: int = 80
    status_bar_width: int = 20


@dataclass(frozen=True)
class ChatSettings:
    temperature: float = 0.2


@dataclass(frozen=True)
class AgentSettings:
    max_steps: int = 10
    max_batch_actions: int = 100
    max_feedback_chars: int = 8000
    max_parse_retries: int = 2
    stagnation_limit: int = 3
    feedback_head_lines: int = 20
    feedback_tail_lines: int = 40
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
    max_patch_bytes: int = 256 * 1024
    max_range_lines: int = 400
    max_grep_results: int = 50
    max_grep_file_bytes: int = 1024 * 1024
    max_capture_bytes: int = 512 * 1024
    grep_ignore_dirs: tuple[str, ...] = (
        ".git",
        "__pycache__",
        ".venv",
        "node_modules",
    )
    move_illegal_chars: str = DEFAULT_MOVE_ILLEGAL_CHARS
    manual_exec_prefixes: tuple[str, ...] = DEFAULT_MANUAL_EXEC_PREFIXES
    destructive_commands: frozenset[str] = CORE_DESTRUCTIVE_COMMANDS
    readonly_shell_prefixes: frozenset[str] = DEFAULT_READONLY_SHELL_PREFIXES
    readonly_blocked_flags: frozenset[str] = CORE_READONLY_BLOCKED_FLAGS
    readonly_blocked_prefixes: tuple[str, ...] = CORE_READONLY_BLOCKED_PREFIXES
    git_blocked_options: frozenset[str] = CORE_GIT_BLOCKED_OPTIONS
    git_blocked_prefixes: tuple[str, ...] = CORE_GIT_BLOCKED_PREFIXES
    project_prefixes: tuple[str, ...] = ()


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
    temperature: float = 0.15
    top_p: float = 0.85
    max_split_depth: int = 3
    default_preset: str = "general"
    environment: str = "production"
    strip_trailing_periods: bool = True
    enable_server_stop_tokens: bool = True
    presets: dict = field(default_factory=dict)


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
class DefaultsSettings:
    agent: str = ""
    chat: str = ""
    translate: str = ""
    batch: str = ""


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
    defaults: DefaultsSettings = DefaultsSettings()


@dataclass(frozen=True)
class AppConfig:
    base_dir: Path
    config_dir: Path
    models: tuple
    settings: Settings
    preferences: Preferences = field(default_factory=Preferences)

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

    def default_model(self, mode: str) -> ModelSpec:
        """按模式返回配置的默认模型；缺省时回落 role 以兼容旧配置。"""
        if mode not in DEFAULTS_ROLE_FALLBACK:
            raise ConfigError(
                f"未知默认模型模式 {mode!r}；可用："
                f"{'、'.join(sorted(DEFAULTS_ROLE_FALLBACK))}"
            )
        model_id = getattr(self.settings.defaults, mode, "").strip()
        if model_id:
            return self.model(model_id)
        return self.model_by_role(DEFAULTS_ROLE_FALLBACK[mode])

    def effective_port(self, model: ModelSpec) -> int:
        override = os.environ.get("VANITAS_PORT")
        if override:
            return _parse_port(override, "环境变量 VANITAS_PORT")
        return model.port

    def resolve_profile(self, model: ModelSpec, key: str | None = None) -> ProfileSpec:
        chosen = (key or "").strip()
        if not chosen:
            chosen = (os.environ.get("VANITAS_PROFILE") or "").strip()
        if not chosen:
            preferred = self.preferences.profile_for(model.id)
            chosen = preferred if preferred in model.profiles else model.default_profile
        profile = model.profiles.get(chosen)
        if profile is None:
            available = "、".join(model.profile_keys)
            raise ConfigError(
                f"模型 {model.id} 没有档位 {chosen!r}；可用档位：{available}"
            )
        return profile


def _parse_port(raw: object, label: str) -> int:
    try:
        port = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} 必须是整数端口，实际为 {raw!r}") from exc
    if not 1 <= port <= 65535:
        raise ConfigError(f"{label} 必须在 1–65535 之间，实际为 {port}")
    return port
