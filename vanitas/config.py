"""配置加载与校验。

红线：模型名、端口、上下文、KV 类型只允许出现在 config\\ 中。
本模块只做「读 + 校验 + dataclass 视图」，不替操作者猜默认值：
任何缺失、类型错误、语义冲突都以 ConfigError 抛出，CLI 统一映射为退出码 2。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
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
    }
)
PROFILE_FIELDS = frozenset({"ctx", "kv_k", "kv_v", "desc"})
SETTINGS_FIELDS = frozenset(
    {"version", "host", "request", "launcher", "repl", "tools", "translate"}
)
REQUEST_FIELDS = frozenset(
    {"timeout_sec", "max_retries", "backoff_base_ms", "backoff_jitter_ms"}
)
LAUNCHER_FIELDS = frozenset(
    {"health_timeout_sec", "health_poll_interval_ms", "kill_wait_sec"}
)
REPL_FIELDS = frozenset({"history_file", "max_context_tokens", "stream"})
TOOLS_FIELDS = frozenset({"exec_enabled", "exec_whitelist", "exec_require_confirm"})
TRANSLATE_FIELDS = frozenset({"source_suffixes", "chunk_chars", "output_suffix"})


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

    @property
    def profile_keys(self) -> list[str]:
        return sorted(self.profiles, key=int)


@dataclass(frozen=True)
class RequestSettings:
    timeout_sec: float
    max_retries: int
    backoff_base_ms: int
    backoff_jitter_ms: int


@dataclass(frozen=True)
class LauncherSettings:
    health_timeout_sec: float
    health_poll_interval_ms: int
    kill_wait_sec: float


@dataclass(frozen=True)
class ReplSettings:
    history_file: Path
    max_context_tokens: int
    stream: bool


@dataclass(frozen=True)
class ToolsSettings:
    exec_enabled: bool
    exec_whitelist: tuple[str, ...]
    exec_require_confirm: bool


@dataclass(frozen=True)
class TranslateSettings:
    source_suffixes: tuple[str, ...]
    chunk_chars: int
    output_suffix: str


@dataclass(frozen=True)
class Settings:
    version: int
    host: str
    request: RequestSettings
    launcher: LauncherSettings
    repl: ReplSettings
    tools: ToolsSettings
    translate: TranslateSettings


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


def _as_number(mapping: dict, key: str, label: str, minimum: float = 0.0) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label}.{key}", f"必须是数字，实际为 {value!r}")
    if value < minimum:
        _fail(f"{label}.{key}", f"必须 ≥ {minimum}，实际为 {value}")
    return float(value)


def _as_bool(mapping: dict, key: str, label: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        _fail(f"{label}.{key}", f"必须是布尔值，实际为 {value!r}")
    return value


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
        raise ConfigError(f"{label} 读取失败：{path}（{exc.strerror or exc}）") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"{label} 不是合法 JSON：{path}（第 {exc.lineno} 行第 {exc.colno} 列：{exc.msg}）"
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
                "chat_template_kwargs 只接受标量值（字符串/数字/布尔/null）",
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
            raise ConfigError(f"models[].alias 重复：{model.alias}（/v1/models 会分不清模型）")
        seen_aliases.add(model.alias)
    return models


# --------------------------------------------------------------------------- #
# 全局设置
# --------------------------------------------------------------------------- #


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
    )

    repl = _as_mapping(document.get("repl"), "settings.repl")
    _reject_unknown(repl, REPL_FIELDS, "settings.repl")
    history_raw = _as_str(repl, "history_file", "settings.repl")
    repl_settings = ReplSettings(
        history_file=Path(os.path.expanduser(history_raw)),
        max_context_tokens=_as_int(
            repl, "max_context_tokens", "settings.repl", minimum=256
        ),
        stream=_as_bool(repl, "stream", "settings.repl"),
    )

    tools = _as_mapping(document.get("tools"), "settings.tools")
    _reject_unknown(tools, TOOLS_FIELDS, "settings.tools")
    whitelist = tools.get("exec_whitelist")
    if not isinstance(whitelist, list) or not whitelist:
        _fail("settings.tools.exec_whitelist", "必须是非空字符串数组")
    normalized: list[str] = []
    for item in whitelist:
        if not isinstance(item, str) or not item.strip():
            _fail("settings.tools.exec_whitelist", f"元素必须是非空字符串，实际为 {item!r}")
        normalized.append(" ".join(item.split()).lower())
    tools_settings = ToolsSettings(
        exec_enabled=_as_bool(tools, "exec_enabled", "settings.tools"),
        exec_whitelist=tuple(normalized),
        exec_require_confirm=_as_bool(tools, "exec_require_confirm", "settings.tools"),
    )

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
    translate_settings = TranslateSettings(
        source_suffixes=tuple(cleaned_suffixes),
        chunk_chars=_as_int(translate, "chunk_chars", "settings.translate", minimum=50),
        output_suffix=output_suffix,
    )

    return Settings(
        version=1,
        host=host,
        request=request_settings,
        launcher=launcher_settings,
        repl=repl_settings,
        tools=tools_settings,
        translate=translate_settings,
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
