"""CLI 入口：argparse 子命令分派 + 统一退出码。

退出码契约：
  0 成功 | 1 用户取消/中断 | 2 配置错误 | 3 守卫失败 | 4 参数错误 | 5 引擎/网络失败

本模块只做「解析参数 -> 加载配置 -> 分派 -> 映射退出码」，
所有用户可见文本经 Palette 上色；不在此拼接 ANSI。
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path

from . import __version__, agent as agent_mod, batch as batch_mod, guard, launcher, prompts, translate
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
from .monitor import TranslationMonitor
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


def _add_verbose(parser: argparse.ArgumentParser) -> None:
    # SUPPRESS 让子命令不覆盖顶层 -v：`vanitas -v serve` 与 `vanitas serve -v` 等价
    parser.add_argument(
        "-v", "--verbose", action="store_true", default=argparse.SUPPRESS, help=VERBOSE_HELP
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
    translate_cmd.add_argument(
        "--name",
        "--output-name",
        dest="output_name",
        default=None,
        help="自定义输出文件名，仅限单个输入；不含扩展名时自动补源扩展名",
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
) -> tuple[str, dict, str]:
    """把 --preset 解析为 (预设名, 预设配置, 任务模板名)。"""
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
    try:
        prompts.load_task(prompt_dir, template)
    except ConfigError as exc:
        raise ConfigError(
            f"翻译预设 {name!r} 的模板 {template!r} 不可用：{exc}"
        ) from exc
    return name, entry, template


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
    try:
        preset_name, preset_info, task_template = _resolve_translate_preset(
            settings,
            getattr(args, "preset", None),
            config.config_dir / "prompts",
        )
    except ConfigError as exc:
        print(palette.err(str(exc)), file=sys.stderr)
        return 4
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
    model = _resolve_model(config, args.model, "translate")
    profile_key = args.profile or preset_info.get("profile")
    profile = config.resolve_profile(model, profile_key)
    out_dir = _resolve_cli_path(args.out_dir) if args.out_dir else None
    if out_dir is not None and out_dir.exists() and not out_dir.is_dir():
        print(palette.err(f"输出路径不是目录：{out_dir}"), file=sys.stderr)
        return 4
    try:
        sources = translate.collect_translate_inputs(
            [_resolve_cli_path(raw) for raw in args.inputs],
            effective,
            exclude_dir=out_dir,
        )
    except TranslateError as exc:
        print(palette.err(str(exc)), file=sys.stderr)
        return 5
    requested_name = args.output_name
    output_name: str | None = None
    if requested_name is not None:
        if len(sources) != 1:
            print(
                palette.err(
                    f"--name 只能用于单个输出文件；当前展开为 {len(sources)} 个文件"
                ),
                file=sys.stderr,
            )
            return 4
        try:
            output_name = translate.normalize_output_name(
                requested_name, sources[0].path.suffix
            )
        except TranslateError as exc:
            print(palette.err(str(exc)), file=sys.stderr)
            return 4
    try:
        plans = translate.plan_translation_outputs(
            sources, effective, out_dir, output_name=output_name
        )
    except TranslateError as exc:
        print(palette.err(str(exc)), file=sys.stderr)
        return 5
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
    print(
        palette.meta(
            f"{model.display} · 档位 {profile.key} · 预设 {preset_name}"
            f" · {len(plans)} 个文件"
        )
    )
    failures = 0
    for plan in plans:
        source = plan.source.path
        # 每个文件独立计时/计量；reset 也负责清掉上一文件的看板。
        monitor.reset()
        # 滑动上下文是文档级状态，不能把上一文件的标题/人称带进下一文件。
        reset_history = getattr(chat, "reset_history", None)
        if callable(reset_history):
            reset_history()
        try:
            outcome = translate.translate_file(
                chat,
                source,
                effective,
                out_dir=plan.out_dir,
                output_name=output_name,
                window_size=window_size,
                max_window_chars=max_window_chars,
            )
        except TranslateError as exc:
            print(palette.err(f"{plan.label}：{exc}"), file=sys.stderr)
            failures += 1
            if monitor.segments:
                monitor.finish()
            continue
        finally:
            # 成功、失败与意外异常都先擦掉看板，避免残行。
            monitor.clear()
        output = outcome.output or plan.target
        target = output.name
        if out_dir is not None:
            try:
                target = output.relative_to(out_dir).as_posix()
            except ValueError:
                pass
        monitor.print_line(
            palette.ok(f"{plan.label} → {target} · {outcome.segments} 段")
        )
        for failure in outcome.failures:
            print(
                palette.err(
                    f"{plan.label} {failure.segment}：{failure.reason}"
                ),
                file=sys.stderr,
            )
        failures += len(outcome.failures)
        monitor.finish()
    if failures:
        print(
            palette.warn(f"完成，{failures} 处失败，失败段保留原文"),
            file=sys.stderr,
        )
        return 5
    return 0


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
