"""vanitas.config 公开门面：re-export schema + loader + preferences。

所有旧的 `from vanitas.config import X` 保持可用；
内部函数（_load_*、_as_*）不 re-export，需要者从 vanitas.config.loader 导入。
"""

from __future__ import annotations

from .preferences import PREFERENCES_VERSION, Preferences
from .schema import (
    BASE_DIR, SHARD_RE, KV_TYPES,
    MODEL_FIELDS, PROFILE_FIELDS, SETTINGS_FIELDS, REQUEST_FIELDS,
    LAUNCHER_FIELDS, REPL_FIELDS, CHAT_FIELDS, AGENT_FIELDS,
    TOOLS_FIELDS, BATCH_FIELDS, TRANSLATE_FIELDS, PRESET_FIELDS,
    DEFAULTS_FIELDS, DEFAULTS_ROLE_FALLBACK,
    UI_FIELDS, THEME_KEYS, COLOR_MODES, COLOR_NAMES, DEFAULT_THEME,
    AGENT_ACTION_NAMES, DEFAULT_AUTO_ACTIONS, DEFAULT_CONFIRM_ACTIONS,
    DEFAULT_ACTION_ALIASES, DEFAULT_MANUAL_EXEC_PREFIXES,
    CORE_DESTRUCTIVE_COMMANDS, DEFAULT_READONLY_SHELL_PREFIXES,
    CORE_READONLY_BLOCKED_FLAGS, CORE_READONLY_BLOCKED_PREFIXES,
    CORE_GIT_BLOCKED_OPTIONS, CORE_GIT_BLOCKED_PREFIXES,
    DEFAULT_MOVE_ILLEGAL_CHARS,
    ProfileSpec, ModelSpec, RequestSettings, LauncherSettings,
    ReplSettings, ChatSettings, AgentSettings, BatchSettings,
    ToolsSettings, TranslateSettings, UiTheme, UiSettings,
    DefaultsSettings, Settings, AppConfig,
)
from .loader import load_config

__all__ = [
    "BASE_DIR", "SHARD_RE", "KV_TYPES",
    "MODEL_FIELDS", "PROFILE_FIELDS", "SETTINGS_FIELDS", "REQUEST_FIELDS",
    "LAUNCHER_FIELDS", "REPL_FIELDS", "CHAT_FIELDS", "AGENT_FIELDS",
    "TOOLS_FIELDS", "BATCH_FIELDS", "TRANSLATE_FIELDS", "PRESET_FIELDS",
    "DEFAULTS_FIELDS", "DEFAULTS_ROLE_FALLBACK",
    "UI_FIELDS", "THEME_KEYS", "COLOR_MODES", "COLOR_NAMES", "DEFAULT_THEME",
    "AGENT_ACTION_NAMES", "DEFAULT_AUTO_ACTIONS", "DEFAULT_CONFIRM_ACTIONS",
    "DEFAULT_ACTION_ALIASES", "DEFAULT_MANUAL_EXEC_PREFIXES",
    "CORE_DESTRUCTIVE_COMMANDS", "DEFAULT_READONLY_SHELL_PREFIXES",
    "CORE_READONLY_BLOCKED_FLAGS", "CORE_READONLY_BLOCKED_PREFIXES",
    "CORE_GIT_BLOCKED_OPTIONS", "CORE_GIT_BLOCKED_PREFIXES",
    "DEFAULT_MOVE_ILLEGAL_CHARS",
    "ProfileSpec", "ModelSpec", "RequestSettings", "LauncherSettings",
    "ReplSettings", "ChatSettings", "AgentSettings", "BatchSettings",
    "ToolsSettings", "TranslateSettings", "UiTheme", "UiSettings",
    "DefaultsSettings", "Settings", "AppConfig",
    "PREFERENCES_VERSION", "Preferences",
    "load_config",
]
