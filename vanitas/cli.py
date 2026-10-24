"""CLI 入口：argparse 子命令分派 + 统一退出码。

退出码契约：
  0 成功 | 1 用户取消/中断 | 2 配置错误 | 3 守卫失败 | 4 参数错误 | 5 引擎/网络失败
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, batch as batch_mod, guard, launcher, prompts, translate
from .config import AppConfig, ModelSpec, load_config
from .engine import EngineClient
from .errors import (
    ConfigError,
    EngineError,
    GuardError,
    LaunchError,
    TranslateError,
    VanitasError,
)
from .repl import Repl


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


def _err(message: object) -> None:
    print(f"[ERROR] {message}", file=sys.stderr)


def _build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="vanitas", description="Vanitas-Harness 本地推理 CLI（llama.cpp 后端）")
    parser.add_argument("--version", action="version", version=f"vanitas {__version__}")
    parser.add_argument("--config-dir", default=None, help="覆盖 config 目录（默认 <仓库>/config）")
    sub = parser.add_subparsers(dest="command")

    chat = sub.add_parser("chat", help="进入 REPL（必要时自动启动服务端）")
    chat.add_argument("--model", default=None, help="模型 id（见 vanitas models）")
    chat.add_argument("--profile", default=None, help="档位编号")
    chat.add_argument("--system", default=None, help="系统提示文件路径")
    chat.add_argument("--task", default=None, help="任务模板名（task_<name>.md）")

    serve = sub.add_parser("serve", help="启动 llama-server 并等待 /health")
    serve.add_argument("model_id", help="模型 id")
    serve.add_argument("--profile", default=None, help="档位编号")
    mode = serve.add_mutually_exclusive_group()
    mode.add_argument("--foreground", action="store_true", help="前台运行，阻塞到进程退出")
    mode.add_argument("--bg", action="store_true", help="后台运行（默认）")

    sub.add_parser("stop", help="停止所有 llama-server")
    sub.add_parser("status", help="显示进程/端口/显存")
    sub.add_parser("models", help="列出配置中的全部模型")

    translate_cmd = sub.add_parser("translate", help="批量翻译 txt/md/srt")
    translate_cmd.add_argument("inputs", nargs="+", help="待翻译文件")
    translate_cmd.add_argument("--model", default=None, help="模型 id（默认 role=translate）")
    translate_cmd.add_argument("--profile", default=None, help="档位编号")
    translate_cmd.add_argument("--out-dir", default=None, help="输出目录（默认与原文件同目录）")
    translate_cmd.add_argument("--jobs", type=int, default=1, help="并发数；引擎为 -np 1，实际串行")

    batch_cmd = sub.add_parser("batch", help="按 JSONL 批量执行任务")
    batch_cmd.add_argument("tasks", help="tasks.jsonl 路径")
    batch_cmd.add_argument("--model", default=None, help="模型 id（默认 role=general）")
    batch_cmd.add_argument("--profile", default=None, help="档位编号")

    return parser


def _resolve_model(config: AppConfig, model_id: str | None, role: str) -> ModelSpec:
    if model_id:
        return config.model(model_id)
    return config.model_by_role(role)


def _ensure_engine(
    config: AppConfig, model: ModelSpec, profile
) -> tuple[EngineClient, int]:
    port = config.effective_port(model)
    client = EngineClient.for_model(config, model, port)
    if client.health():
        return client, port
    record = launcher.start(config, model, profile, background=True)
    elapsed = launcher.wait_ready(config, record)
    print(f"[SERVE] 就绪 {elapsed:.1f}s · {launcher.endpoint(config, record.port)}")
    return EngineClient.for_model(config, model, record.port), record.port


def _cmd_models(config: AppConfig, args: argparse.Namespace) -> int:
    print(f"{'id':<10}{'display':<28}{'port':>6}{'shards':>8}  role")
    for model in config.models:
        print(
            f"{model.id:<10}{model.display:<28}{config.effective_port(model):>6}"
            f"{model.shards:>8}  {model.role}"
        )
        for key in model.profile_keys:
            profile = model.profiles[key]
            marker = "*" if key == model.default_profile else " "
            print(
                f"  {marker}[{key}] ctx {profile.ctx:<6} KV {profile.kv_k}/{profile.kv_v}  {profile.desc}"
            )
        print(f"     alias {model.alias} · {model.path}")
    return 0


def _cmd_status(config: AppConfig, args: argparse.Namespace) -> int:
    print(f"[STATUS] 配置 {config.config_dir}")
    records = {record.model_id: record for record in launcher.read_records()}
    for model in config.models:
        port = config.effective_port(model)
        client = EngineClient.for_model(config, model, port)
        healthy = client.health(timeout=0.6)
        record = records.get(model.id)
        if record is not None and launcher.pid_running(record.pid):
            state = f"运行中 pid={record.pid} 档位 {record.profile}"
        elif record is not None:
            state = "残留记录（进程已退出）"
        elif healthy:
            state = "运行中（无记录）"
        else:
            state = "未运行"
        print(f"[STATUS] {model.id:<10} port {port:<5} {state}")
    gpu = guard.gpu_status()
    print(f"[STATUS] {gpu}" if gpu else "[STATUS] GPU 信息不可用（nvidia-smi 未找到）")
    return 0


def _cmd_stop(config: AppConfig, args: argparse.Namespace) -> int:
    pids = launcher.stop_all(config)
    if pids:
        print(f"[STOP] 已结束 PID {', '.join(str(pid) for pid in pids)}")
    else:
        print("[STOP] 没有运行中的 llama-server")
    return 0


def _cmd_serve(config: AppConfig, args: argparse.Namespace) -> int:
    model = config.model(args.model_id)
    profile = config.resolve_profile(model, args.profile)
    if args.foreground:
        return launcher.run_foreground(config, model, profile)
    record = launcher.start(config, model, profile, background=True)
    elapsed = launcher.wait_ready(config, record)
    print(
        f"[SERVE] {model.display} 就绪 {elapsed:.1f}s · "
        f"{launcher.endpoint(config, record.port)}/v1 · pid {record.pid}"
    )
    return 0


def _cmd_chat(config: AppConfig, args: argparse.Namespace) -> int:
    model = _resolve_model(config, args.model, "general")
    profile = config.resolve_profile(model, args.profile)
    repl = Repl(config, model, profile.key, auto_start=True)
    if args.system:
        path = Path(args.system).expanduser()
        try:
            repl.session.system = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ConfigError(
                f"系统提示文件读取失败：{path}（{exc.strerror or exc}）"
            ) from exc
    if args.task:
        prompts.load_task(repl.prompt_dir, args.task)
        repl.session.task = args.task
    return repl.run()


def _cmd_repl(config: AppConfig, args: argparse.Namespace) -> int:
    return Repl(config).run()


def _cmd_translate(config: AppConfig, args: argparse.Namespace) -> int:
    if args.jobs < 1:
        print("[ERROR] --jobs 必须 ≥ 1", file=sys.stderr)
        return 4
    if args.jobs > 1:
        print("[WARN] 服务端为 -np 1，--jobs 只并行解析，请求仍然串行", file=sys.stderr)
    model = _resolve_model(config, args.model, "translate")
    profile = config.resolve_profile(model, args.profile)
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else None
    chat, port = _translate_chat(config, model, profile)
    print(f"[TRANSLATE] {model.display} @ {launcher.endpoint(config, port)}")
    failures = 0
    for raw in args.inputs:
        source = Path(raw).expanduser()
        try:
            outcome = translate.translate_file(
                chat, source, config.settings.translate, out_dir=out_dir
            )
        except TranslateError as exc:
            print(f"[TRANSLATE-FAIL] {raw}: {exc}", file=sys.stderr)
            failures += 1
            continue
        if outcome.failures:
            for failure in outcome.failures:
                print(
                    f"[TRANSLATE-FAIL] {failure.source} {failure.segment}: {failure.reason}",
                    file=sys.stderr,
                )
            failures += len(outcome.failures)
        print(
            f"[TRANSLATE] {outcome.source.name} → {outcome.output.name if outcome.output else '—'}"
            f"（{outcome.segments} 段，失败 {len(outcome.failures)}）"
        )
    if failures:
        print(f"[TRANSLATE] 完成但有 {failures} 处失败（失败段保留原文）", file=sys.stderr)
        return 5
    return 0


def _translate_chat(config: AppConfig, model: ModelSpec, profile):
    client, port = _ensure_engine(config, model, profile)
    chat = translate.make_engine_chat(
        client, config.config_dir / "prompts", prompts.system_prompt(config.config_dir / "prompts")
    )
    return chat, port


def _cmd_batch(config: AppConfig, args: argparse.Namespace) -> int:
    tasks = Path(args.tasks).expanduser()
    if not tasks.is_file():
        print(f"[ERROR] 批处理文件不存在：{tasks}", file=sys.stderr)
        return 4
    if tasks.suffix.lower() != ".jsonl":
        print(f"[ERROR] 批处理文件必须是 .jsonl：{tasks}", file=sys.stderr)
        return 4
    model = _resolve_model(config, args.model, "general")
    profile = config.resolve_profile(model, args.profile)
    client, _port = _ensure_engine(config, model, profile)
    system_default = prompts.system_prompt(config.config_dir / "prompts")

    def chat(system: str, prompt: str, max_tokens: int) -> str:
        messages = [
            {"role": "system", "content": system or system_default},
            {"role": "user", "content": prompt},
        ]
        return translate.extract_content(
            client.chat(messages, max_tokens=max_tokens, temperature=0.2)
        )

    result = batch_mod.run_batch(
        tasks,
        chat,
        prompt_dir=config.config_dir / "prompts",
        system_default=system_default,
    )
    ok = sum(1 for item in result.items if item.ok)
    print(f"[BATCH] 成功 {ok} / 共 {len(result.items)}，结果 {result.output_path}")
    if result.failures:
        for item in result.failures:
            print(f"[BATCH-FAIL] {item.id}: {item.error}", file=sys.stderr)
        return 5
    return 0


HANDLERS = {
    "models": _cmd_models,
    "status": _cmd_status,
    "stop": _cmd_stop,
    "serve": _cmd_serve,
    "chat": _cmd_chat,
    "translate": _cmd_translate,
    "batch": _cmd_batch,
    None: _cmd_repl,
}


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # --version / --help
        return int(exc.code or 0)
    except _UsageError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        print(
            "用法：vanitas [--config-dir DIR] <models|status|serve|stop|chat|translate|batch>",
            file=sys.stderr,
        )
        return 4
    try:
        config = load_config(args.config_dir)
        handler = HANDLERS.get(args.command)
        if handler is None:
            raise _UsageError(f"未知子命令 {args.command!r}")
        return handler(config, args)
    except _UsageError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        print("用法：vanitas [--config-dir DIR] <models|status|serve|stop|chat|translate|batch>", file=sys.stderr)
        return 4
    except ConfigError as exc:
        _err(exc)
        return 2
    except GuardError as exc:
        messages = list(exc.messages)
        if messages:
            messages[0] = f"[GUARD-CODE={exc.code}] {messages[0]}"
        for message in messages:
            print(message, file=sys.stderr)
        return 3
    except LaunchError as exc:
        _err(exc)
        return 5
    except EngineError as exc:
        _err(exc)
        return 5
    except KeyboardInterrupt:
        print()
        return 1
    except VanitasError as exc:
        _err(exc)
        return 5
