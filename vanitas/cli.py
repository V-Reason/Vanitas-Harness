"""CLI 入口：argparse 子命令分派 + 统一退出码。

退出码契约：
  0 成功 | 1 用户取消/中断 | 2 配置错误 | 3 守卫失败 | 4 参数错误 | 5 引擎/网络失败

本模块只做「解析参数 -> 加载配置 -> 分派 -> 映射退出码」，
所有用户可见文本经 Palette 上色；不在此拼接 ANSI。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

from . import __version__, agent as agent_mod, batch as batch_mod, environments, guard, launcher, prompts, translate
from .client import LlamaClient
from .config import (
    AppConfig,
    DEFAULT_THEME,
    ModelSpec,
    TranslateSettings,
    load_config,
)
from .errors import (
    ConfigError,
    EngineError,
    GuardError,
    LaunchError,
    TranslateError,
    VanitasError,
)
from .metrics import format_duration
from .monitor import FileReport, TranslationMonitor, TranslationReport
from .progress import Spinner
from .repl import Repl
from .ui import Palette, resolve_enabled

USAGE = (
    "用法：vanitas [--config-dir DIR] [-v] "
    "<models|status|serve|stop|chat|agent|translate|batch>"
)
VERBOSE_HELP = "显示 PID、日志、地址等诊断信息"


class _UsageError(VanitasError):
    """argparse 的参数错误：契约要求退出码 4，而 argparse 默认是 2。"""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # type: ignore[override]
        raise _UsageError(message)


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:  # 被重定向到非 TextIOWrapper 时跳过
            pass
    # 管道输入的默认编码是本地代码页（cp936），与 pwsh/CI 的 UTF-8 不一致；
    # 不统一会导致中文变成孤立代理字符并在 json.dumps 时崩溃。
    try:
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def _make_palette(config: AppConfig) -> Palette:
    """auto 要求 stdout 与 stderr 都是终端：任一流被重定向就整体降级为纯文本。"""
    mode = config.settings.ui.color
    enabled = resolve_enabled(mode, sys.stdout) and resolve_enabled(mode, sys.stderr)
    return Palette(asdict(config.settings.ui.theme), enabled)


def _add_verbose(
    parser: argparse.ArgumentParser, help_text: str = VERBOSE_HELP
) -> None:
    # SUPPRESS 让子命令不覆盖顶层 -v：`vanitas -v serve` 与 `vanitas serve -v` 等价
    parser.add_argument(
        "-v", "--verbose", action="store_true", default=argparse.SUPPRESS, help=help_text
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="vanitas", description="Vanitas-Harness 本地推理 CLI · llama.cpp 后端")
    parser.add_argument("--version", action="version", version=f"vanitas {__version__}")
    parser.add_argument("--config-dir", default=None, help="覆盖 config 目录，默认 <仓库>/config")
    parser.add_argument("-v", "--verbose", action="store_true", help=VERBOSE_HELP)
    sub = parser.add_subparsers(dest="command")

    chat = sub.add_parser("chat", help="进入 REPL，必要时自动启动服务端")
    chat.add_argument("--model", default=None, help="模型 id，见 vanitas models")
    chat.add_argument("--profile", default=None, help="档位编号")
    chat.add_argument("--system", default=None, help="系统提示文件路径")
    chat.add_argument("--task", default=None, help="任务模板名，task_<name>.md")
    _add_verbose(chat)

    agent = sub.add_parser(
        "agent", help="进入 Agent 闭环模式，修改类动作需确认"
    )
    agent.add_argument("--model", default=None, help="模型 id，见 vanitas models")
    agent.add_argument("--profile", default=None, help="档位编号")
    agent.add_argument("--system", default=None, help="系统提示文件路径")
    _add_verbose(agent)

    serve = sub.add_parser("serve", help="启动 llama-server 并等待 /health")
    serve.add_argument("model_id", help="模型 id")
    serve.add_argument("--profile", default=None, help="档位编号")
    mode = serve.add_mutually_exclusive_group()
    mode.add_argument("--foreground", action="store_true", help="前台运行，阻塞到进程退出")
    mode.add_argument("--bg", action="store_true", help="后台运行，默认")
    _add_verbose(serve)

    stop = sub.add_parser("stop", help="停止所有 llama-server")
    _add_verbose(stop)

    status = sub.add_parser("status", help="显示进程/端口/显存")
    _add_verbose(status)
    sub.add_parser("models", help="列出配置中的全部模型")

    translate_cmd = sub.add_parser("translate", help="批量翻译 txt/md/srt")
    translate_cmd.add_argument(
        "inputs", nargs="+", help="待翻译文件或目录；目录会递归展开其中的可翻译文件"
    )
    translate_cmd.add_argument(
        "--model", default=None, help="模型 id，默认 settings.defaults.translate"
    )
    translate_cmd.add_argument("--profile", default=None, help="档位编号")
    translate_cmd.add_argument(
        "--out-dir", default=None, help="输出目录；目录输入会保留相对层级，默认与原文件同目录"
    )
    translate_mode = translate_cmd.add_mutually_exclusive_group()
    translate_mode.add_argument(
        "--translate-name",
        "--translate-filename",
        dest="translate_name",
        action="store_true",
        help="翻译文件名（不加 .zh）并翻译文件内容",
    )
    translate_mode.add_argument(
        "--translate-filename-only",
        "--translate-name-only",
        dest="translate_name_only",
        action="store_true",
        help="只原地翻译文件名/目录名，不翻译内容；支持任意文件类型，不加 .zh",
    )
    translate_cmd.add_argument("--jobs", type=int, default=1, help="并发数；引擎为 -np 1，实际串行")
    translate_cmd.add_argument(
        "--preset",
        default=None,
        help=(
            "翻译场景预设：prose（小说/散文）| subtitle（普通字幕，默认）"
            " | asmr（ASMR）| general（旧版 4K/5）"
        ),
    )
    translate_cmd.add_argument(
        "--env",
        dest="environment",
        default=None,
        help="运行环境：production（默认）| experimental；experimental 当前仅 ASMR 实现",
    )
    translate_cmd.add_argument(
        "--resume",
        action="store_true",
        help="若存在与源时间轴对齐的 <输出>.part.srt 快照，从断点继续翻译",
    )
    _add_verbose(translate_cmd, "显示完整逐文件报告")

    batch_cmd = sub.add_parser("batch", help="按 JSONL 批量执行任务")
    batch_cmd.add_argument("tasks", help="tasks.jsonl 路径")
    batch_cmd.add_argument(
        "--model", default=None, help="模型 id，默认 settings.defaults.batch"
    )
    batch_cmd.add_argument("--profile", default=None, help="档位编号")

    return parser


def _invocation_cwd() -> Path:
    """调用方原始工作目录；vanitas.bat 会切到仓库根，必须优先读透传值。"""
    raw = os.environ.get("VANITAS_ORIG_CWD")
    if raw:
        candidate = Path(raw).expanduser()
        try:
            if candidate.is_dir():
                return candidate.resolve()
        except OSError:
            pass
    return Path.cwd()


def _resolve_cli_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = _invocation_cwd() / path
    return path


def _count_asmr_cues(paths: list[Path]) -> tuple[int, dict[str, int]]:
    """0 模型调用预扫描 .srt 非空 cue；非 SRT/不可读文件跳过。

    返回 ``(总非空 cue 数, {str(path): 文件非空 cue 数})``。只用于
    ``T1Budget.for_corpus`` 与 ``begin_file(file_cues)``；不解析正文语义。
    """
    total = 0
    counts: dict[str, int] = {}
    for raw in paths:
        path = Path(raw)
        if path.suffix.lower() != ".srt":
            continue
        try:
            text = path.read_bytes().decode("utf-8-sig")
        except (OSError, UnicodeDecodeError):
            continue
        eol = "\r\n" if "\r\n" in text else "\n"
        blocks = translate.parse_srt(text, eol)
        count = sum(
            1
            for subtitle, _separator in blocks
            if subtitle.index and subtitle.text.strip()
        )
        counts[str(path)] = count
        total += count
    return total, counts


def _resolve_model(
    config: AppConfig, model_id: str | None, mode: str
) -> ModelSpec:
    if model_id:
        return config.model(model_id)
    return config.default_model(mode)


def _ensure_engine(
    config: AppConfig, model: ModelSpec, profile, palette: Palette
) -> tuple[LlamaClient, int]:
    port = config.effective_port(model)
    client = LlamaClient.for_model(config, model, port)
    record = next(
        (
            item
            for item in launcher.read_records()
            if item.model_id == model.id and launcher.pid_running(item.pid)
        ),
        None,
    )
    if record is not None and record.profile != profile.key:
        print(
            palette.meta(
                f"预设要求档位 {profile.key}，当前为 {record.profile}；重启服务"
            )
        )
        launcher.stop_all(config)
        client = None
    elif client.health():
        return client, port
    record = launcher.start(config, model, profile, background=True, palette=palette)
    launcher.wait_ready(config, record, palette)
    return LlamaClient.for_model(config, model, record.port), record.port


def _cmd_models(config: AppConfig, args: argparse.Namespace, palette: Palette) -> int:
    header = f"{'id':<10}{'display':<28}{'port':>6}{'shards':>8}  role"
    print(palette.accent(header))
    for model in config.models:
        model_cell = palette.bold(palette.accent(f"{model.id:<10}"))
        port_cell = palette.info(f"{config.effective_port(model):>6}")
        print(
            f"{model_cell}{model.display:<28}{port_cell}"
            f"{model.shards:>8}  {model.role}"
        )
        for key in model.profile_keys:
            profile = model.profiles[key]
            marker = "*" if key == model.default_profile else " "
            print(
                f"  {marker}[{key}] ctx {profile.ctx:<6}"
                f" KV {profile.kv_k}/{profile.kv_v}  {profile.desc}"
            )
        print(palette.meta(f"alias {model.alias} · {model.path}"))
    routes = " · ".join(
        f"{mode}={config.default_model(mode).id}"
        for mode in ("agent", "chat", "translate", "batch")
    )
    print(palette.meta(f"默认 {routes}"))
    return 0


def _cmd_status(config: AppConfig, args: argparse.Namespace, palette: Palette) -> int:
    verbose = bool(getattr(args, "verbose", False))
    print(palette.meta(f"配置 {config.config_dir}"))
    records = {record.model_id: record for record in launcher.read_records()}
    for model in config.models:
        port = config.effective_port(model)
        client = LlamaClient.for_model(config, model, port)
        healthy = client.health(
            timeout=config.settings.request.status_probe_timeout_sec
        )
        record = records.get(model.id)
        if record is not None and launcher.pid_running(record.pid):
            state = palette.ok("运行")
        elif record is not None:
            state = palette.warn("残留")
        elif healthy:
            state = palette.ok("运行")
        else:
            state = palette.err("停止")
        print(f"{palette.meta(f'{model.id:<10} port {port:<5}')}  {state}")
        if verbose and record is not None:
            print(
                palette.meta(
                    f"{'':<10} PID {record.pid} · 日志 {record.log_path}"
                )
            )
    gpu = guard.gpu_status()
    print(
        palette.meta(gpu)
        if gpu
        else palette.meta("GPU 信息不可用，nvidia-smi 未找到")
    )
    return 0


def _cmd_stop(config: AppConfig, args: argparse.Namespace, palette: Palette) -> int:
    verbose = bool(getattr(args, "verbose", False))
    pids = launcher.stop_all(config)
    if pids:
        print(palette.ok(f"已停止 {len(pids)} 个服务"))
        if verbose:
            print(palette.meta("PID " + "、".join(str(pid) for pid in pids)))
    else:
        print(palette.meta("没有服务在运行"))
    return 0


def _cmd_serve(config: AppConfig, args: argparse.Namespace, palette: Palette) -> int:
    verbose = bool(getattr(args, "verbose", False))
    model = config.model(args.model_id)
    profile = config.resolve_profile(model, args.profile)
    if args.foreground:
        return launcher.run_foreground(
            config, model, profile, palette=palette, verbose=verbose
        )
    record = launcher.start(
        config, model, profile, background=True, palette=palette, verbose=verbose
    )
    launcher.wait_ready(config, record, palette)
    return 0


def _cmd_chat(config: AppConfig, args: argparse.Namespace, palette: Palette) -> int:
    verbose = bool(getattr(args, "verbose", False))
    model = _resolve_model(config, args.model, "chat")
    profile = config.resolve_profile(model, args.profile)
    repl = Repl(
        config,
        model,
        profile.key,
        auto_start=True,
        palette=palette,
        verbose=verbose,
    )
    system = _resolve_cli_path(args.system) if args.system else None
    repl.configure(system=str(system) if system else None, task=args.task)
    return repl.run()


def _cmd_agent(config: AppConfig, args: argparse.Namespace, palette: Palette) -> int:
    verbose = bool(getattr(args, "verbose", False))
    model = _resolve_model(config, getattr(args, "model", None), "agent")
    profile = config.resolve_profile(model, getattr(args, "profile", None))
    system_arg = getattr(args, "system", None)
    system = str(_resolve_cli_path(system_arg)) if system_arg else None
    return agent_mod.run_agent(
        config,
        model,
        profile.key,
        system=system,
        palette=palette,
        verbose=verbose,
    )


def _cmd_repl(config: AppConfig, args: argparse.Namespace, palette: Palette) -> int:
    verbose = bool(getattr(args, "verbose", False))
    return Repl(config, palette=palette, verbose=verbose).run()


def _resolve_translate_preset(
    settings: TranslateSettings,
    requested: str | None,
    prompt_dir: Path,
    *,
    environment: environments.EnvironmentProfile | None = None,
) -> tuple[str, dict, str]:
    """把 --preset 解析为 (预设名, 预设配置, 任务模板名)。

    environment 独立于 preset：production 原样使用 preset 模板；experimental
    由 profile 决定覆盖，未实现的 preset 直接 ConfigError，不 fallback。
    """
    profile = environment or environments.get_environment(settings.environment)
    presets = settings.presets or {}
    if requested is None:
        name = (settings.default_preset or "general").strip()
    else:
        name = requested.strip()
    entry = presets.get(name)
    if entry is None:
        if requested is not None:
            available = "、".join(sorted(presets)) or "无"
            raise ConfigError(f"未知翻译预设 {name!r}；可用：{available}")
        entry = {}
        template = "translate"
    else:
        template = str(entry.get("task_template") or "translate").strip()

    template = profile.resolve_task_template(name, template)
    try:
        prompts.load_task(prompt_dir, template)
    except ConfigError as exc:
        raise ConfigError(
            f"翻译预设 {name!r} 的模板 {template!r} 不可用：{exc}"
        ) from exc
    return name, entry, template


def _vanitas_home() -> Path:
    """``~/.vanitas`` 根；``VANITAS_HOME`` 可覆盖（测试与迁移用）。"""
    raw = os.environ.get("VANITAS_HOME")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".vanitas"


def _translate_artifacts_dir(stamp: str) -> Path:
    """单次 translate 的过程文件目录：``~/.vanitas/translate/<stamp>/``。

    ``translation_meta.json`` / ``t1_events.jsonl`` / ``t1_budget.json`` 一律放
    这里，绝不写进源目录或 ``--out-dir``：翻译产物目录只留译文，过程文件不
    污染用户工作区。
    """
    return _vanitas_home() / "translate" / stamp


def _write_translation_meta(
    directory: Path,
    *,
    environment: str,
    task_template: str,
    prompt_dir: Path,
    preset: str,
    temperature: float,
    model_id: str,
    started_at: str,
) -> Path:
    """写运行元数据（环境/模板/预设/模型），供 A/B 归因；不含正文或 blind 标识。"""
    prompt_path = Path(prompt_dir) / f"task_{task_template}.md"
    raw = prompt_path.read_bytes()
    payload = {
        "environment": environment,
        "task_template": task_template,
        "prompt_sha256": hashlib.sha256(raw).hexdigest().upper(),
        "prompt_bytes": len(raw),
        "preset": preset,
        "temperature": float(temperature),
        "model": str(model_id),
        "started_at": str(started_at),
    }
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "translation_meta.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _cmd_translate(config: AppConfig, args: argparse.Namespace, palette: Palette) -> int:
    if args.jobs < 1:
        print(palette.err("--jobs 必须 ≥ 1"), file=sys.stderr)
        return 4
    if args.jobs > 1:
        print(
            palette.warn("服务端为 -np 1，--jobs 只并行解析，请求仍然串行"),
            file=sys.stderr,
        )
    settings = config.settings.translate
    started = datetime.now().astimezone()
    started_at = started.isoformat(timespec="seconds")
    # 过程文件（meta / T1 事件 / T1 预算）统一落在 ~/.vanitas/translate/<stamp>/。
    artifacts_dir = _translate_artifacts_dir(started.strftime("%Y%m%d-%H%M%S"))
    requested_environment = getattr(args, "environment", None)
    try:
        environment = environments.get_environment(
            requested_environment
            if requested_environment is not None
            else settings.environment
        )
    except ConfigError as exc:
        print(palette.err(str(exc)), file=sys.stderr)
        return 4
    try:
        preset_name, preset_info, task_template = _resolve_translate_preset(
            settings,
            getattr(args, "preset", None),
            config.config_dir / "prompts",
            environment=environment,
        )
    except ConfigError as exc:
        print(palette.err(str(exc)), file=sys.stderr)
        return 4
    environment_name = environment.name
    effective = replace(
        settings,
        context_window=int(
            preset_info.get("context_window", settings.context_window)
        ),
        context_summary_chars=int(
            preset_info.get(
                "context_summary_chars", settings.context_summary_chars
            )
        ),
        chunk_chars=int(preset_info.get("chunk_chars", settings.chunk_chars)),
    )
    window_size = int(preset_info.get("window_size", 1))
    max_window_chars = int(preset_info.get("max_window_chars", 250))
    lookahead_cues = int(preset_info.get("lookahead_cues", 0))
    model = _resolve_model(config, args.model, "translate")
    profile_key = args.profile or preset_info.get("profile")
    profile = config.resolve_profile(model, profile_key)
    out_dir = _resolve_cli_path(args.out_dir) if args.out_dir else None
    if out_dir is not None and out_dir.exists() and not out_dir.is_dir():
        print(palette.err(f"输出路径不是目录：{out_dir}"), file=sys.stderr)
        return 4
    if args.translate_name_only:
        if out_dir is not None:
            print(
                palette.err("--translate-filename-only 为原地重命名，不支持 --out-dir"),
                file=sys.stderr,
            )
            return 4
        return _cmd_translate_names(
            config,
            args,
            palette,
            model,
            profile,
            preset_name,
            task_template,
            effective,
        )
    content_suffix = "" if args.translate_name else None
    try:
        sources = translate.collect_translate_inputs(
            [_resolve_cli_path(raw) for raw in args.inputs],
            effective,
            exclude_dir=out_dir,
        )
    except TranslateError as exc:
        print(palette.err(str(exc)), file=sys.stderr)
        return 5
    plans: list[translate.TranslationPlan] = []
    if not args.translate_name:
        try:
            plans = translate.plan_translation_outputs(
                sources, effective, out_dir, output_suffix=content_suffix
            )
        except TranslateError as exc:
            print(palette.err(str(exc)), file=sys.stderr)
            return 5
    try:
        _write_translation_meta(
            artifacts_dir,
            environment=environment_name,
            task_template=task_template,
            prompt_dir=config.config_dir / "prompts",
            preset=preset_name,
            temperature=effective.temperature,
            model_id=model.id,
            started_at=started_at,
        )
    except OSError as exc:
        print(
            palette.warn(
                f"translation_meta.json 写入失败：{artifacts_dir}；"
                f"{exc.strerror or exc}"
            ),
            file=sys.stderr,
        )
    monitor = TranslationMonitor(
        stream=sys.stderr,
        enabled=palette.enabled,
        capacity=effective.monitor_lines,
        palette=palette,
    )
    chat, _port = _translate_chat(
        config,
        model,
        profile,
        palette,
        on_stat=monitor.record,
        task_template=task_template,
        translate_settings=effective,
    )
    name_failures = 0
    if args.translate_name:
        translate_filename = getattr(chat, "translate_filename_result", None)
        if not callable(translate_filename):
            print(
                palette.err("当前翻译引擎不支持文件名翻译"),
                file=sys.stderr,
            )
            return 5
        translated: list[translate.TranslationSource] = []
        total = len(sources)
        for index, source in enumerate(sources, start=1):
            label = f"文件名 {index}/{total} {source.path.name}"
            try:
                result = _translate_one_name(
                    translate_filename,
                    source.path.name,
                    source.path.suffix,
                    f"文件名 {index}/{total}",
                    palette,
                    config.settings.ui.spinner_interval_ms,
                )
            except (EngineError, TranslateError) as exc:
                print(
                    palette.err(f"{source.path.name}：文件名翻译失败：{exc}"),
                    file=sys.stderr,
                )
                name_failures += 1
                continue
            if result.fallback_reason:
                print(
                    palette.warn(
                        f"保留原名：{label}（{result.fallback_reason}）"
                    ),
                    file=sys.stderr,
                )
                if out_dir is None:
                    print(
                        palette.err(
                            f"{label}：回退原名会覆盖源文件，已跳过；"
                            "请改用 --out-dir"
                        ),
                        file=sys.stderr,
                    )
                    name_failures += 1
                    continue
            translated.append(
                replace(source, translated_stem=result.name)
            )
        sources = translated
        try:
            plans = translate.plan_translation_outputs(
                sources, effective, out_dir, output_suffix=content_suffix
            )
        except TranslateError as exc:
            print(palette.err(str(exc)), file=sys.stderr)
            return 5
    verbose = bool(getattr(args, "verbose", False))
    output_hint = f"输出 {out_dir}" if out_dir is not None else "输出 同源目录"
    print(
        palette.meta(
            f"{model.display} · 档位 {profile.key} · 预设 {preset_name}"
            f" · {len(plans)} 个文件 · {output_hint}"
        )
    )
    report = TranslationReport()
    total_cues = 0
    file_cue_counts: dict[str, int] = {}
    if prompts.is_asmr_task(task_template) and plans:
        total_cues, file_cue_counts = _count_asmr_cues(
            [plan.source.path for plan in plans]
        )
        t1_budget = translate.T1Budget.for_corpus(total_cues)
    else:
        t1_budget = translate.T1Budget()
    t1_counts: dict[str, int] = {}
    t1_details: list[str] = []
    t1_records: list[dict] = []
    t1_unique: set[tuple[str, str]] = set()
    # 过程文件与输出目录解耦：一律写到 artifacts_dir，不落源目录/--out-dir。
    t1_events_path: Path = artifacts_dir / "t1_events.jsonl"
    t1_budget_path: Path = artifacts_dir / "t1_budget.json"
    total = len(plans)
    current_index = 0
    current_started = 0.0
    interrupted = False
    batch_started = time.perf_counter()
    try:
        for index, plan in enumerate(plans, start=1):
            current_index = index
            current_started = time.perf_counter()
            source = plan.source.path
            monitor.begin_file(index, total, plan.label)
            # 滑动上下文是文档级状态，不能把上一文件的标题/人称带进下一文件。
            reset_history = getattr(chat, "reset_history", None)
            if callable(reset_history):
                reset_history()
            will_overwrite = plan.target.exists()
            outcome = None
            error = None
            try:
                outcome = translate.translate_file(
                    chat,
                    source,
                    effective,
                    out_dir=plan.out_dir,
                    output_stem=plan.source.translated_stem,
                    output_suffix=content_suffix,
                    window_size=window_size,
                    max_window_chars=max_window_chars,
                    lookahead_cues=lookahead_cues,
                    asmr_guard=prompts.is_asmr_task(task_template),
                    t1_budget=t1_budget,
                    file_cues=file_cue_counts.get(str(source)),
                    join_rules=(
                        getattr(chat, "join_rules", ())
                        if window_size > 1
                        else ()
                    ),
                    on_segment=monitor.set_segment_progress,
                    resume=bool(getattr(args, "resume", False)),
                )
            except TranslateError as exc:
                error = str(exc)
            finally:
                # 成功、失败、中断与意外异常都先擦掉看板，避免残行。
                monitor.clear()
            elapsed = max(0.0, time.perf_counter() - current_started)
            tokens = monitor.tokens
            zh_chars = monitor.zh_chars
            if error is not None:
                monitor.print_line(
                    palette.err(
                        f"[{index}/{total}] {plan.label} · 失败：{error}"
                        f" · {format_duration(elapsed)}"
                    )
                )
                report.add(
                    FileReport(
                        index=index,
                        total=total,
                        label=plan.label,
                        output="—",
                        segments=0,
                        elapsed_s=elapsed,
                        tokens=tokens,
                        zh_chars=zh_chars,
                        error=error,
                    )
                )
                current_index = 0
                continue
            output = outcome.output or plan.target
            target = output.name
            if out_dir is not None:
                try:
                    target = output.relative_to(out_dir).as_posix()
                except ValueError:
                    pass
            flags: list[str] = []
            if outcome.failures:
                flags.append(f"{len(outcome.failures)} 段失败")
            if will_overwrite:
                flags.append("覆盖")
            suffix = " · " + " · ".join(flags) if flags else ""
            message = (
                f"[{index}/{total}] {plan.label} → {target}"
                f" · {outcome.segments} 段 · {format_duration(elapsed)}{suffix}"
            )
            if outcome.failures:
                monitor.print_line(palette.warn(message))
            else:
                monitor.print_line(palette.ok(message))
            report.add(
                FileReport(
                    index=index,
                    total=total,
                    label=plan.label,
                    output=target,
                    segments=outcome.segments,
                    elapsed_s=elapsed,
                    tokens=tokens,
                    zh_chars=zh_chars,
                    failures=tuple(
                        f"{failure.segment}：{failure.reason}"
                        for failure in outcome.failures
                    ),
                    overwrote=will_overwrite,
                )
            )
            events = getattr(outcome, "t1_events", ())
            if isinstance(events, (list, tuple)):
                outcome_source = getattr(outcome, "source", "")
                if not isinstance(outcome_source, (str, Path)):
                    outcome_source = ""
                for event in events:
                    key = getattr(event, "outcome", "")
                    if not key:
                        continue
                    event_source = str(
                        getattr(event, "source", "") or outcome_source
                    )
                    cue = getattr(event, "cue", "?")
                    code = getattr(event, "code", "?")
                    detail = str(getattr(event, "detail", "") or "")
                    t1_counts[key] = t1_counts.get(key, 0) + 1
                    t1_unique.add((event_source, str(cue)))
                    t1_records.append(
                        {
                            "source": event_source,
                            "cue": cue,
                            "code": code,
                            "outcome": key,
                            "detail": detail,
                        }
                    )
                    source_label = (
                        Path(event_source).name if event_source else "?"
                    )
                    t1_details.append(
                        f"{source_label} cue {cue} {code} {key}"
                    )
            current_index = 0
    except KeyboardInterrupt:
        interrupted = True
        if current_index:
            elapsed = max(0.0, time.perf_counter() - current_started)
            monitor.clear()
            report.add(
                FileReport(
                    index=current_index,
                    total=total,
                    label=plan.label,
                    output="—",
                    segments=0,
                    elapsed_s=elapsed,
                    tokens=monitor.tokens,
                    zh_chars=monitor.zh_chars,
                    error="用户中断",
                    interrupted=True,
                )
            )
    total_elapsed = max(0.0, time.perf_counter() - batch_started)
    write_t1 = bool(t1_records) or prompts.is_asmr_task(task_template)
    if write_t1:
        try:
            t1_budget_path.parent.mkdir(parents=True, exist_ok=True)
            t1_budget_path.write_text(
                json.dumps(
                    {
                        "total": t1_budget.total,
                        "max_total": t1_budget.max_total,
                        "per_file": t1_budget.per_file,
                        "total_cues": total_cues,
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
        except OSError as exc:
            print(
                palette.warn(
                    f"T1 预算写入失败：{t1_budget_path}；"
                    f"{exc.strerror or exc}"
                ),
                file=sys.stderr,
            )
    if write_t1:
        try:
            t1_events_path.parent.mkdir(parents=True, exist_ok=True)
            with t1_events_path.open(
                "w", encoding="utf-8", newline="\n"
            ) as handle:
                for record in t1_records:
                    handle.write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )
        except OSError as exc:
            print(
                palette.warn(
                    f"T1 事件写入失败：{t1_events_path}；"
                    f"{exc.strerror or exc}"
                ),
                file=sys.stderr,
            )
    print(
        "\n".join(
            report.render(
                palette,
                verbose=verbose,
                elapsed_s=total_elapsed,
                out_dir=out_dir,
            )
        )
    )
    if artifacts_dir.is_dir():
        # 过程文件搬出了工作区，就必须告诉用户去哪儿找。
        print(palette.muted(f"过程文件 {artifacts_dir}"))
    if t1_counts:
        labels = (
            ("accepted", "接受"),
            ("rejected", "拒绝"),
            ("error", "异常"),
            ("budget", "超预算"),
        )
        parts = [
            f"{label} {t1_counts[key]}"
            for key, label in labels
            if t1_counts.get(key)
        ]
        print(
            palette.muted(
                f"T1 守卫 · 事件 {sum(t1_counts.values())}"
                f" · 唯一 cue {len(t1_unique)}"
                f" · {' · '.join(parts)}"
                f" · 重译 {t1_budget.total}/{t1_budget.max_total}"
            )
        )
        if t1_details:
            print(palette.muted("  " + " · ".join(t1_details)))

    if name_failures:
        print(
            palette.warn(f"文件名翻译失败 {name_failures} 处"),
            file=sys.stderr,
        )
    if interrupted:
        print(
            palette.warn("已中断，未完成的文件未写出"),
            file=sys.stderr,
        )
        return 1
    if name_failures:
        return 5
    if any(item.status in ("partial", "failed") for item in report.items):
        return 5
    return 0


def _translate_one_name(
    translate_name,
    old_name: str,
    suffix: str,
    label: str,
    palette: Palette,
    interval_ms: int,
) -> translate.NameTranslation:
    """逐项翻译名称并即时显示进度；非 TTY 只输出完成行。"""
    with Spinner(
        f"{label} {old_name}",
        palette,
        interval_ms=interval_ms,
    ) as spinner:
        result = translate_name(old_name)
        final_name = f"{result.name}{suffix}"
        spinner.done(f"{label} {old_name} → {final_name}")
    return result


def _cmd_translate_names(
    config: AppConfig,
    args: argparse.Namespace,
    palette: Palette,
    model: ModelSpec,
    profile,
    preset_name: str,
    task_template: str,
    settings: TranslateSettings,
) -> int:
    try:
        targets = translate.collect_name_targets(
            [_resolve_cli_path(raw) for raw in args.inputs]
        )
    except TranslateError as exc:
        print(palette.err(str(exc)), file=sys.stderr)
        return 5
    chat, _port = _translate_chat(
        config,
        model,
        profile,
        palette,
        task_template=task_template,
        translate_settings=settings,
    )
    translate_filename = getattr(chat, "translate_filename_result", None)
    translate_dirname = getattr(chat, "translate_dirname_result", None)
    if not callable(translate_filename) or not callable(translate_dirname):
        print(
            palette.err("当前翻译引擎不支持文件名/目录名翻译"),
            file=sys.stderr,
        )
        return 5
    print(
        palette.meta(
            f"{model.display} · 档位 {profile.key} · 预设 {preset_name}"
            f" · {len(targets)} 个名称"
        )
    )
    translated: list[translate.NameTarget] = []
    total = len(targets)
    name_errors = 0
    name_fallbacks = 0
    for index, target in enumerate(targets, start=1):
        kind = "目录名" if target.is_dir else "文件名"
        label = f"{kind} {index}/{total} {target.path.name}"
        try:
            result = _translate_one_name(
                translate_dirname if target.is_dir else translate_filename,
                target.path.name,
                "" if target.is_dir else target.path.suffix,
                f"{kind} {index}/{total}",
                palette,
                config.settings.ui.spinner_interval_ms,
            )
        except (EngineError, TranslateError) as exc:
            print(
                palette.err(f"{target.path.name}：名称翻译失败：{exc}"),
                file=sys.stderr,
            )
            name_errors += 1
            translated.append(
                replace(target, translated_name=target.path.name)
            )
            continue
        if result.fallback_reason:
            name_fallbacks += 1
            print(
                palette.warn(
                    f"保留原名：{label}（{result.fallback_reason}）"
                ),
                file=sys.stderr,
            )
        translated.append(replace(target, translated_name=result.name))
    try:
        renames = translate.plan_name_renames(translated)
        translate.apply_name_renames(renames)
    except TranslateError as exc:
        print(palette.err(str(exc)), file=sys.stderr)
        return 5
    print(
        palette.ok(
            f"名称翻译完成 · 共 {len(targets)} 项 · 重命名 {len(renames)}"
            f" · 保留原名 {name_fallbacks} · 失败 {name_errors}"
        )
    )
    return 5 if name_errors else 0


def _translate_chat(
    config: AppConfig,
    model: ModelSpec,
    profile,
    palette: Palette,
    *,
    on_stat=None,
    task_template: str = "translate",
    translate_settings: TranslateSettings | None = None,
):
    client, port = _ensure_engine(config, model, profile, palette)
    settings = translate_settings or config.settings.translate
    chat = translate.make_engine_chat(
        client,
        config.config_dir / "prompts",
        prompts.system_prompt(config.config_dir / "prompts"),
        n_ctx=profile.ctx,
        context_window=settings.context_window,
        context_summary_chars=settings.context_summary_chars,
        min_output_tokens=settings.min_output_tokens,
        output_ratio=settings.output_ratio,
        prompt_safety_tokens=settings.prompt_safety_tokens,
        temperature=settings.temperature,
        max_split_depth=settings.max_split_depth,
        on_stat=on_stat,
        task_template=task_template,
        enable_server_stop_tokens=settings.enable_server_stop_tokens,
        top_p=settings.top_p,
    )
    return chat, port


def _cmd_batch(config: AppConfig, args: argparse.Namespace, palette: Palette) -> int:
    tasks = _resolve_cli_path(args.tasks)
    if not tasks.is_file():
        print(palette.err(f"批处理文件不存在：{tasks}"), file=sys.stderr)
        return 4
    if tasks.suffix.lower() != ".jsonl":
        print(palette.err(f"批处理文件必须是 .jsonl：{tasks}"), file=sys.stderr)
        return 4
    model = _resolve_model(config, args.model, "batch")
    profile = config.resolve_profile(model, args.profile)
    client, _port = _ensure_engine(config, model, profile, palette)
    system_default = prompts.system_prompt(config.config_dir / "prompts")

    def chat(system: str, prompt: str, max_tokens: int) -> str:
        messages = [
            {"role": "system", "content": system or system_default},
            {"role": "user", "content": prompt},
        ]
        return translate.extract_content(
            client.chat(
                messages,
                max_tokens=max_tokens,
                temperature=config.settings.batch.temperature,
            )
        )

    result = batch_mod.run_batch(
        tasks,
        chat,
        prompt_dir=config.config_dir / "prompts",
        system_default=system_default,
        default_max_tokens=config.settings.batch.default_max_tokens,
        max_tokens_limit=config.settings.batch.max_tokens_limit,
        palette=palette,
    )
    ok = sum(1 for item in result.items if item.ok)
    print(palette.ok(f"成功 {ok} / 共 {len(result.items)} · 结果 {result.output_path}"))
    if result.failures:
        for item in result.failures:
            print(palette.err(f"{item.id}：{item.error}"), file=sys.stderr)
        return 5
    return 0


HANDLERS = {
    "models": _cmd_models,
    "status": _cmd_status,
    "stop": _cmd_stop,
    "serve": _cmd_serve,
    "chat": _cmd_chat,
    "agent": _cmd_agent,
    "translate": _cmd_translate,
    "batch": _cmd_batch,
    None: _cmd_agent,
}


def _print_guard_error(exc: GuardError, palette: Palette) -> None:
    """守卫消息由 GuardError 携带（纯文本），这里统一上色并逐行落到 stderr。"""
    _stdout, stderr_lines = guard.format_for_cli(
        guard.GuardResult(exc.code, [], list(exc.messages)), palette
    )
    for message in stderr_lines:
        print(message, file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    parser = _build_parser()
    palette = Palette(DEFAULT_THEME, False)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # --version / --help
        return int(exc.code or 0)
    except _UsageError as exc:
        print(palette.err(str(exc)), file=sys.stderr)
        print(palette.hint(USAGE), file=sys.stderr)
        return 4
    try:
        config_dir = (
            _resolve_cli_path(args.config_dir) if args.config_dir else None
        )
        config = load_config(config_dir)
        palette = _make_palette(config)
        handler = HANDLERS.get(args.command)
        if handler is None:
            raise _UsageError(f"未知子命令 {args.command!r}")
        return handler(config, args, palette)
    except _UsageError as exc:
        print(palette.err(str(exc)), file=sys.stderr)
        print(palette.hint(USAGE), file=sys.stderr)
        return 4
    except ConfigError as exc:
        print(palette.err(str(exc)), file=sys.stderr)
        return 2
    except GuardError as exc:
        _print_guard_error(exc, palette)
        return 3
    except (LaunchError, EngineError, TranslateError, VanitasError) as exc:
        print(palette.err(str(exc)), file=sys.stderr)
        return 5
    except KeyboardInterrupt:
        print()
        return 1
