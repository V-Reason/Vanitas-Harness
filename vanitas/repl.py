"""REPL：命令注册表（范式笔记 D1）+ 交互循环。

只有「以 / 开头且第 2 个字符不是空格」的行才当命令；其余整行都是用户输入，
这样用户想问 "/model 是什么意思" 时仍能正常发问。

所有用户可见文本都经 Palette 上色/加符号，本模块不拼接 ANSI 转义。
"""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from . import guard, launcher, prompts, tools
from .chat import ChatRunner, TurnResult
from .client import LlamaClient
from .config import AppConfig, ModelSpec, ProfileSpec
from .errors import ConfigError, EngineError, GuardError, LaunchError, ToolDenied
from .session import Session, default_save_path
from .ui import SYMBOLS, Palette, strip_ansi


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


def _context_bar(pct: float, width: int = 20) -> str:
    filled = max(0, min(width, int(round(pct / 100 * width))))
    return "█" * filled + "░" * (width - filled)


def _format_uptime(started_at: str) -> str:
    try:
        started = datetime.fromisoformat(started_at)
    except ValueError:
        return "-"
    seconds = max(0, int((datetime.now() - started).total_seconds()))
    return f"{seconds // 60}分{seconds % 60:02d}秒"


class IdleWatchdog:
    """单守护线程：空闲超时后回调一次；pause/resume 供重启与长命令使用。"""

    def __init__(
        self,
        timeout_sec: float,
        interval_sec: float,
        on_idle: Callable[[], None],
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.timeout_sec = max(0.0, float(timeout_sec))
        self.interval_sec = max(0.01, float(interval_sec))
        self._on_idle = on_idle
        self._clock = clock
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_active = clock()
        self._paused = False
        self._fired = False

    def start(self) -> None:
        if self.timeout_sec <= 0 or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, name="vanitas-idle-watchdog", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)

    def touch(self) -> None:
        with self._lock:
            self._last_active = self._clock()
            self._fired = False

    def pause(self) -> None:
        with self._lock:
            self._paused = True
            self._last_active = self._clock()

    def resume(self) -> None:
        with self._lock:
            self._paused = False
            self._last_active = self._clock()
            self._fired = False

    def remaining(self) -> float | None:
        with self._lock:
            if self.timeout_sec <= 0:
                return None
            if self._fired:
                return 0.0
            return max(0.0, self.timeout_sec - (self._clock() - self._last_active))

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_sec):
            with self._lock:
                if self._paused or self._fired or self.timeout_sec <= 0:
                    continue
                if self._clock() - self._last_active < self.timeout_sec:
                    continue
                self._fired = True
            try:
                self._on_idle()
            except Exception as exc:  # 守护线程不能因回调异常静默退出
                print(
                    Palette({}, enabled=False).err(f"空闲看门狗异常：{exc}"),
                    file=sys.stderr,
                )


class Repl:
    def __init__(
        self,
        config: AppConfig,
        model: ModelSpec | None = None,
        profile: str | None = None,
        *,
        auto_start: bool = False,
        palette: Palette | None = None,
        verbose: bool = False,
    ) -> None:
        self.config = config
        self.auto_start = auto_start
        self.palette = palette or Palette({}, enabled=False)
        self.verbose = verbose
        self.prompt_dir = config.config_dir / "prompts"
        default_model = model or config.models[0]
        self.profile: ProfileSpec = config.resolve_profile(default_model, profile)
        self.pending_profile: ProfileSpec | None = None
        self.model: ModelSpec = default_model
        self.port: int = config.effective_port(default_model)
        self.client: LlamaClient | None = None
        self.chat_runner: ChatRunner | None = None
        self.session = Session(
            model_id=default_model.id,
            profile=self.profile.key,
            system=prompts.system_prompt(self.prompt_dir),
        )
        self._prompt_session = None
        self._prompt_ready = False
        self._ansi_formatter = None
        self._state_lock = threading.RLock()
        self._chat_active = False
        self._sleeping = False
        self._watchdog = IdleWatchdog(
            config.settings.repl.idle_timeout_sec,
            config.settings.repl.watchdog_interval_sec,
            self._on_idle,
        )
        self.last_turn: TurnResult | None = None

    def _activate(
        self,
        model: ModelSpec,
        profile: ProfileSpec,
        client: LlamaClient | None,
        port: int,
    ) -> None:
        """统一收口“当前模型/档位/客户端/会话”的切换，避免四处复制。"""
        self.model = model
        self.profile = profile
        self.pending_profile = None
        self.client = client
        self.port = port
        self._sleeping = False
        self.config.preferences.last_used_model = model.id
        self.session.model_id = model.id
        self.session.profile = profile.key
        self.chat_runner = (
            ChatRunner(
                self.config,
                client,
                self.session,
                model,
                profile,
                self.palette,
                verbose=self.verbose,
            )
            if client is not None
            else None
        )

    def _remember_model(self, model: ModelSpec, profile: ProfileSpec) -> None:
        prefs = self.config.preferences
        prefs.last_used_model = model.id
        prefs.preferred_profiles[model.id] = profile.key
        try:
            prefs.save()
        except ConfigError as exc:
            print(self.palette.warn(str(exc)), file=sys.stderr)

    def configure(self, system: str | None = None, task: str | None = None) -> None:
        """应用 --system 文件与 --task 模板；供 CLI 与 agent 复用。"""
        if system:
            path = Path(system).expanduser()
            try:
                text = path.read_text(encoding="utf-8")
            except OSError as exc:
                raise ConfigError(
                    f"系统提示文件读取失败：{path}；{exc.strerror or exc}"
                ) from exc
            self.session.system = text.strip()
        if task:
            prompts.load_task(self.prompt_dir, task)
            self.session.task = task

    # ------------------------------------------------------------------ #
    # 主循环
    # ------------------------------------------------------------------ #

    def run(self) -> int:
        self._bootstrap()
        self._watchdog.start()
        print(self.palette.muted("[Enter] 输入，[/] 命令，[!] 原生命令"))
        try:
            while True:
                try:
                    line = self._read_line(self._prompt_text())
                except KeyboardInterrupt:
                    print()
                    continue
                except EOFError:
                    print()
                    break
                self._watchdog.touch()
                text = line.strip()
                if not text:
                    continue
                if text.startswith("!") and len(text) > 1:
                    self._cmd_exec(text[1:].strip())
                    continue
                if self._is_command(text):
                    if not self._handle_command(text):
                        break
                else:
                    self._chat(text)
        finally:
            self._watchdog.stop()
        print(self.palette.meta("再见"))
        return 0

    @staticmethod
    def _is_command(text: str) -> bool:
        return len(text) > 1 and text.startswith("/") and text[1] != " "

    def _prompt_text(self) -> str:
        used = self.session.message_tokens()
        total = self.profile.ctx
        pct = int((used / total) * 100) if total > 0 else 0
        tag = f"ctx: {pct}% | {used / 1000:.1f}k/{total / 1000:.1f}k"
        repl_settings = self.config.settings.repl
        if pct >= repl_settings.ctx_danger_pct:
            colored_tag = self.palette.error(f"[{tag}]")
        elif pct >= repl_settings.ctx_warn_pct:
            colored_tag = self.palette.warning(f"[{tag}]")
        else:
            colored_tag = self.palette.muted(f"[{tag}]")
        head = self.palette.accent(f"{self.model.id}@{self.profile.key}")
        if self._sleeping:
            head += " " + self.palette.muted("[休眠]")
        return f"{head} {colored_tag} {SYMBOLS['prompt']} "

    def confirm_plan(self, plan: str) -> bool:
        """展示动作清单并做一次 [y/N] 确认；无 TTY 的测试可注入 confirm。"""
        print(self.palette.accent(plan))
        try:
            answer = input("[y/N] ")
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        return answer.strip().lower() in {"y", "yes", "是"}

    def _read_line(self, prompt: str) -> str:
        try:
            interactive = bool(sys.stdin.isatty())
        except (AttributeError, OSError, ValueError):
            interactive = False
        session = self._get_prompt_session() if interactive else None
        if session is not None:
            # prompt_toolkit 必须收到 ANSI 格式化对象，不能把转义字符串
            # 直接交给它；否则 TTY 上会原样显示 \x1b[35m。
            formatter = self._ansi_formatter
            formatted = (
                formatter(prompt) if formatter is not None else strip_ansi(prompt)
            )
            return session.prompt(formatted)
        # 降级路径：提示符固定走 stderr，stdout 留给回答；必须剥离 ANSI，
        # 否则没有 prompt_toolkit 时 input()/终端会显示裸转义码。
        plain = strip_ansi(prompt)
        try:
            sys.stderr.write(plain)
            sys.stderr.flush()
        except (AttributeError, OSError, ValueError):
            pass
        return input()

    # ------------------------------------------------------------------ #
    # 启动引导（D6）
    # ------------------------------------------------------------------ #

    def _bootstrap(self) -> None:
        if self._adopt_any():
            return
        if self.auto_start:
            self._start_engine(
                self.model, self.pending_profile or self.profile
            )
            return
        if not sys.stdin.isatty():
            print(
                self.palette.hint(
                    "非交互模式且没有就绪的服务端：先运行 vanitas serve <id>"
                )
            )
            return
        chosen = self._prompt_model()
        if chosen is None:
            return
        pending = self.pending_profile
        profile = (
            self.config.resolve_profile(chosen, pending.key)
            if pending is not None and pending.key in chosen.profiles
            else self.config.resolve_profile(chosen)
        )
        port = self.config.effective_port(chosen)
        answer = input(
            f"启动 {chosen.display} · 档位 {profile.key} · {profile.desc}"
            f" · 端口 {port}？[Y/n] "
        ).strip().lower()
        if answer not in {"", "y", "yes", "是"}:
            print(
                self.palette.hint(
                    "已跳过启动；可稍后用 vanitas serve 启动，或 /model 切换"
                )
            )
            return
        self._start_engine(chosen, profile)

    def _start_engine(self, model: ModelSpec, profile: ProfileSpec) -> bool:
        try:
            record = launcher.start(
                self.config,
                model,
                profile,
                palette=self.palette,
                verbose=self.verbose,
            )
            launcher.wait_ready(self.config, record, self.palette)
        except GuardError as exc:
            _stdout, stderr_lines = guard.format_for_cli(
                guard.GuardResult(exc.code, [], list(exc.messages)), self.palette
            )
            for message in stderr_lines:
                print(message, file=sys.stderr)
            return False
        except LaunchError as exc:
            print(self.palette.err(str(exc)), file=sys.stderr)
            return False
        self._activate(
            model,
            profile,
            LlamaClient.for_model(self.config, model, record.port),
            record.port,
        )
        return True

    def _prompt_model(self) -> ModelSpec | None:
        print(self.palette.meta("可用模型："))
        for index, item in enumerate(self.config.models, start=1):
            print(
                f"  [{index}] {item.id:<10} {item.display}"
                f" · 端口 {self.config.effective_port(item)}"
            )
        try:
            raw = input("选择编号 [默认 1]：").strip()
        except EOFError:
            return None
        if not raw:
            return self.config.models[0]
        try:
            index = int(raw)
        except ValueError:
            print(self.palette.warn(f"编号必须是整数，收到 {raw!r}"))
            return None
        if not 1 <= index <= len(self.config.models):
            print(self.palette.warn(f"编号超出 1–{len(self.config.models)}"))
            return None
        return self.config.models[index - 1]

    def _probe(self, model: ModelSpec, port: int | None = None):
        target_port = port if port is not None else self.config.effective_port(model)
        client = LlamaClient.for_model(self.config, model, target_port)
        try:
            if not client.health():
                return None
            if model.alias not in client.model_ids():
                return None
        except EngineError:
            return None
        return client, target_port

    def _adopt_any(self) -> bool:
        for record in launcher.read_records():
            if not launcher.pid_running(record.pid):
                continue
            try:
                model = self.config.model(record.model_id)
            except ConfigError:
                continue
            probed = self._probe(model, record.port)
            if probed is None:
                continue
            client, port = probed
            profile = self.config.resolve_profile(model, record.profile)
            self._activate(model, profile, client, port)
            print(
                self.palette.ok(
                    f"已连接 {model.display} · 档位 {profile.key}"
                )
            )
            return True
        for model in self.config.models:
            probed = self._probe(model)
            if probed is None:
                continue
            client, port = probed
            profile = self.config.resolve_profile(model)
            self._activate(model, profile, client, port)
            print(
                self.palette.ok(
                    f"已连接 {model.display} · 档位 {profile.key}"
                )
            )
            return True
        return False

    @staticmethod
    def _autosave_path() -> Path:
        return Path.home() / ".vanitas" / "sessions" / "autosave.json"

    def _on_idle(self) -> None:
        with self._state_lock:
            if self._chat_active or self.client is None:
                return
            saved: Path | None = None
            if self.session.is_dirty and self.session.messages:
                try:
                    saved = self.session.save(self._autosave_path())
                except ConfigError as exc:
                    print(self.palette.err(str(exc)), file=sys.stderr)
                    return
            try:
                launcher.stop_all(self.config)
            except LaunchError as exc:
                print(self.palette.err(str(exc)), file=sys.stderr)
                return
            self.client = None
            self.chat_runner = None
            self._sleeping = True
        minutes = max(1, int(self.config.settings.repl.idle_timeout_sec // 60))
        message = f"· 已空闲 {minutes} 分钟，后台服务已休眠；输入任意内容自动唤醒"
        if saved is not None:
            message += f"；会话已保存至 {saved}"
        print(self.palette.meta(message), file=sys.stderr)

    def _ensure_client(self) -> bool:
        if self.client is not None and self.chat_runner is not None:
            return True
        if self._sleeping:
            print(self.palette.meta("· 服务已休眠，正在原地唤醒..."))
            if not self._start_engine(self.model, self.profile):
                return False
            self._sleeping = False
            self._watchdog.touch()
            print(self.palette.ok(f"✓ 已唤醒 · 档位 {self.profile.key}"))
            return True
        print(
            self.palette.hint(
                "服务端没有运行：先运行 vanitas serve <id>，或用 /model 查看可用模型"
            )
        )
        return False

    def _confirm_unsaved(self, action: str) -> bool:
        if not (self.session.is_dirty and self.session.messages):
            return True
        print(
            self.palette.warn(
                f"{action}前当前会话有 {len(self.session.messages)} 条未保存记录，"
                "是否保存？[Y/n/c]"
            )
        )
        try:
            answer = input("[Y/n/c] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        if answer in {"c", "cancel", "取消"}:
            print(self.palette.meta("已取消"))
            return False
        if answer in {"n", "no", "否"}:
            self.session.is_dirty = False
            print(self.palette.meta("已丢弃未保存变动"))
            return True
        try:
            saved = self.session.save(default_save_path())
        except ConfigError as exc:
            print(self.palette.err(str(exc)), file=sys.stderr)
            return False
        print(self.palette.ok(f"已保存到 {saved} · {len(self.session.messages)} 条消息"))
        return True

    # ------------------------------------------------------------------ #
    # 对话
    # ------------------------------------------------------------------ #

    def _run_chat(
        self,
        text: str,
        *,
        display: bool = True,
        feedback: str | None = None,
        stream: object | None = None,
    ) -> TurnResult | None:
        """运行一轮对话；Agent 复用本方法但不把原始 JSON 写屏。"""
        with self._state_lock:
            self._chat_active = True
        try:
            if not self._ensure_client():
                return None
            self._watchdog.pause()
            try:
                kwargs = {
                    "display": display,
                    "on_dropped": lambda count: print(
                        self.palette.warn(f"上下文已满，丢弃最早 {count} 轮对话")
                    ),
                }
                if stream is not None:
                    kwargs["stream"] = stream
                if feedback is None:
                    result = self.chat_runner.run(text, **kwargs)
                else:
                    result = self.chat_runner.run_followup(feedback, **kwargs)
            except ConfigError as exc:
                print(self.palette.err(str(exc)), file=sys.stderr)
                return None
            except EngineError as exc:
                print()
                print(self.palette.err(str(exc)), file=sys.stderr)
                return None
            self.last_turn = result
            if result.interrupted:
                print(self.palette.warn("已中止生成"))
                return result
            if result.finish_reason == "length":
                print(self.palette.warn("达到长度上限，输出被截断"))
            if not result.text.strip() and result.reasoning_chars:
                print(
                    self.palette.hint(
                        "仅返回 reasoning_content，content 为空："
                        "检查 chat_template_kwargs 或提高 max_tokens"
                    ),
                    file=sys.stderr,
                )
            if not display:
                return result

            pieces = [
                f"{result.tokens} tokens",
                f"{result.elapsed_s:.1f}s",
                f"{result.tps:.1f} t/s",
            ]
            if self.verbose and result.ttft_ms is not None:
                pieces.append(f"首字 {result.ttft_ms:.0f} ms")
            print(self.palette.meta(" · ".join(pieces)))
            return result
        finally:
            with self._state_lock:
                self._chat_active = False
            self._watchdog.resume()

    def _chat(self, text: str) -> None:
        self._run_chat(text)

    # ------------------------------------------------------------------ #
    # 命令分派
    # ------------------------------------------------------------------ #

    def _handle_command(self, line: str) -> bool:
        name, _, rest = line[1:].partition(" ")
        command = COMMAND_MAP.get(name.lower())
        if command is None:
            print(self.palette.hint(f"未知命令 /{name}，输入 /help 查看全部"))
            return True
        handler: Callable[[str], bool] = getattr(self, f"_cmd_{command.name}")
        return handler(rest.strip())

    def _cmd_help(self, rest: str) -> bool:
        print(self.palette.accent("内置命令："))
        for command in COMMANDS:
            head = f"/{command.name}" + (f" {command.arg_hint}" if command.arg_hint else "")
            print(f"  {head:<24}{command.description}")
        return True

    def _print_guard_error(self, exc: GuardError) -> None:
        _stdout, stderr_lines = guard.format_for_cli(
            guard.GuardResult(exc.code, [], list(exc.messages)), self.palette
        )
        for message in stderr_lines:
            print(message, file=sys.stderr)

    def _cmd_model(self, rest: str) -> bool:
        if not rest:
            records = {record.model_id: record for record in launcher.read_records()}
            for item in self.config.models:
                port = self.config.effective_port(item)
                marker = self.palette.accent("*") if item.id == self.model.id else " "
                record = records.get(item.id)
                if record is not None and launcher.pid_running(record.pid):
                    state = self.palette.ok("运行")
                elif record is not None:
                    state = self.palette.warn("残留")
                else:
                    state = self.palette.err("停止")
                model_cell = self.palette.bold(
                    self.palette.accent(f"{item.id:<10}")
                )
                port_cell = self.palette.info(f"port {port:<5}")
                print(
                    f" {marker} {model_cell} {item.display:<26} "
                    f"{port_cell} {state}"
                )
            print(
                self.palette.hint(
                    "带 id 切换模型；目标没有运行时会自动停旧启新"
                )
            )
            return True
        try:
            model = self.config.model(rest)
        except ConfigError as exc:
            print(self.palette.err(str(exc)), file=sys.stderr)
            return True
        if model.id == self.model.id:
            print(self.palette.meta(f"已经是当前模型 {model.id}"))
            return True
        probed = self._probe(model)
        if probed is not None:
            client, port = probed
            record = next(
                (item for item in launcher.read_records() if item.model_id == model.id),
                None,
            )
            try:
                profile = self.config.resolve_profile(
                    model, record.profile if record else None
                )
            except ConfigError as exc:
                print(self.palette.err(str(exc)), file=sys.stderr)
                return True
            self._activate(model, profile, client, port)
            self._remember_model(model, profile)
            print(
                self.palette.ok(
                    f"已切换到 {model.display} · 档位 {profile.key}"
                )
            )
            return True

        pending = self.pending_profile
        try:
            profile = (
                self.config.resolve_profile(model, pending.key)
                if pending is not None and pending.key in model.profiles
                else self.config.resolve_profile(model)
            )
        except ConfigError as exc:
            print(self.palette.err(str(exc)), file=sys.stderr)
            return True

        print(self.palette.spin("正在切换模型"))
        try:
            launcher.stop_all(self.config)
        except LaunchError as exc:
            print(self.palette.err(str(exc)), file=sys.stderr)
            return True
        self.client = None
        self.chat_runner = None
        try:
            record = launcher.start(
                self.config,
                model,
                profile,
                palette=self.palette,
                verbose=self.verbose,
            )
            launcher.wait_ready(self.config, record, self.palette)
        except GuardError as exc:
            self._print_guard_error(exc)
            return True
        except LaunchError as exc:
            print(self.palette.err(str(exc)), file=sys.stderr)
            return True
        self._activate(
            model,
            profile,
            LlamaClient.for_model(self.config, model, record.port),
            record.port,
        )
        self._remember_model(model, profile)
        print(self.palette.ok("已切换"))
        return True

    def _cmd_profile(self, rest: str) -> bool:
        if not rest:
            print(
                self.palette.meta(
                    f"当前档位 {self.profile.key}：{self.profile.desc}"
                )
            )
            if self.pending_profile is not None:
                print(
                    self.palette.meta(
                        f"待生效档位 {self.pending_profile.key}："
                        f"{self.pending_profile.desc}"
                    )
                )
            for key in self.model.profile_keys:
                profile = self.model.profiles[key]
                print(
                    f"  [{key}] ctx {profile.ctx:<6}"
                    f" KV {profile.kv_k}/{profile.kv_v}  {profile.desc}"
                )
            print(self.palette.hint("修改档位只影响下次 serve/切换"))
            return True
        try:
            profile = self.config.resolve_profile(self.model, rest)
        except ConfigError as exc:
            print(self.palette.err(str(exc)), file=sys.stderr)
            return True
        # 只改“待生效”档位：当前服务端的 ctx 不能凭命令幻觉改变。
        self.pending_profile = profile
        try:
            self.config.preferences.set_profile(self.model.id, profile.key)
        except ConfigError as exc:
            print(self.palette.err(str(exc)), file=sys.stderr)
            return True
        print(
            self.palette.meta(
                f"档位 {profile.key} 已暂存，重启生效；偏好已写入"
            )
        )
        print(self.palette.hint("执行 /restart 立即应用该档位"))
        return True

    def _cmd_restart(self, rest: str) -> bool:
        if not self._confirm_unsaved("重启"):
            return True
        target = self.pending_profile or self.profile
        with self._state_lock:
            self._chat_active = True
        self._watchdog.pause()
        started = time.monotonic()
        try:
            try:
                self.config.preferences.set_profile(self.model.id, target.key)
            except ConfigError as exc:
                print(self.palette.err(str(exc)), file=sys.stderr)
                return True
            print(self.palette.spin("正在切换档位并重启服务"))
            try:
                launcher.stop_all(self.config)
            except LaunchError as exc:
                print(self.palette.err(str(exc)), file=sys.stderr)
                return True
            self.client = None
            self.chat_runner = None
            try:
                record = launcher.start(
                    self.config,
                    self.model,
                    target,
                    palette=self.palette,
                    verbose=self.verbose,
                )
                launcher.wait_ready(self.config, record, self.palette)
            except GuardError as exc:
                self._print_guard_error(exc)
                return True
            except LaunchError as exc:
                print(self.palette.err(str(exc)), file=sys.stderr)
                return True
            self._activate(
                self.model,
                target,
                LlamaClient.for_model(self.config, self.model, record.port),
                record.port,
            )
            elapsed = time.monotonic() - started
            print(
                self.palette.ok(
                    f"✓ 服务端已重启 · 档位 {target.key} "
                    f"({target.ctx} ctx + {target.kv_k}/{target.kv_v} KV) "
                    f"· 耗时 {elapsed:.1f}s"
                )
            )
            return True
        finally:
            with self._state_lock:
                self._chat_active = False
            self._watchdog.resume()

    def _cmd_system(self, rest: str) -> bool:
        if not rest:
            current = self.session.system.strip()
            print(
                self.palette.meta(
                    f"系统提示 · {current if current else '空，使用模型默认行为'}"
                )
            )
            return True
        candidate = Path(rest).expanduser()
        try:
            is_file = candidate.is_file()
        except OSError:
            is_file = False
        if is_file:
            try:
                text = candidate.read_text(encoding="utf-8")
            except OSError as exc:
                print(
                    self.palette.err(
                        f"系统提示读取失败：{candidate}；{exc.strerror or exc}"
                    ),
                    file=sys.stderr,
                )
                return True
        else:
            text = rest
        self.session.system = text.strip()
        print(self.palette.ok(f"系统提示已更新 · {len(self.session.system)} 字符"))
        return True

    def _cmd_task(self, rest: str) -> bool:
        names = prompts.task_names(self.prompt_dir)
        if not rest:
            print(
                self.palette.meta(
                    f"当前任务：{self.session.task or '未应用'}"
                    f"；可用：{'、'.join(names) or '无'}"
                )
            )
            return True
        try:
            prompts.load_task(self.prompt_dir, rest)
        except ConfigError as exc:
            print(self.palette.err(str(exc)), file=sys.stderr)
            return True
        self.session.task = rest
        print(
            self.palette.ok(f"已应用 task_{rest}.md · 每轮输入填入 {{input}}")
        )
        return True

    def _cmd_undo(self, rest: str) -> bool:
        if not self.session.messages:
            print(self.palette.warn("当前没有任何对话历史可供回溯"))
            return True
        removed, released = self.session.pop_last_turn()
        if removed == 0:
            print(self.palette.warn("没有可回溯的完整轮次"))
            return True
        self.last_turn = None
        loop = getattr(self, "loop", None)
        if loop is not None:
            loop._recent.clear()
            loop._failed_signatures.clear()
            loop._stagnation_reason = ""
            loop.steps = 0
            loop.parse_retries = 0
            loop.stagnation_hits = 0
        used = self.session.message_tokens()
        total = self.profile.ctx
        pct = int((used / total) * 100) if total > 0 else 0
        # palette.ok 自动加单个 "✓ "，这里传纯文本，避免 "✓ ✓"。
        print(
            self.palette.ok(
                f"已回溯上一轮完整对话（移除 {removed} 条记录，"
                f"释放 {released} tokens）· 当前视窗已用降至 {pct}% "
                f"({used}/{total} tokens)"
            )
        )
        return True

    def _cmd_clear(self, rest: str) -> bool:
        self.session.clear_messages()
        print(self.palette.ok("对话历史已清空，系统提示与任务模板保留"))
        return True

    def _cmd_save(self, rest: str) -> bool:
        target = Path(rest).expanduser() if rest else default_save_path()
        try:
            saved = self.session.save(target)
        except ConfigError as exc:
            print(self.palette.err(str(exc)), file=sys.stderr)
            return True
        print(
            self.palette.ok(f"已保存到 {saved} · {len(self.session.messages)} 条消息")
        )
        return True

    def _cmd_load(self, rest: str) -> bool:
        if not rest:
            print(self.palette.hint("用法：/load <path>"), file=sys.stderr)
            return True
        try:
            loaded = Session.load(rest)
        except ConfigError as exc:
            print(self.palette.err(str(exc)), file=sys.stderr)
            return True
        self.session = loaded
        if loaded.system == "":
            self.session.system = prompts.system_prompt(self.prompt_dir)

        model = self.model
        if loaded.model_id:
            try:
                model = self.config.model(loaded.model_id)
            except ConfigError as exc:
                print(self.palette.warn(str(exc)), file=sys.stderr)

        profile = self.profile
        if loaded.profile:
            try:
                profile = self.config.resolve_profile(model, loaded.profile)
            except ConfigError as exc:
                print(self.palette.warn(str(exc)), file=sys.stderr)

        probed = self._probe(model)
        if probed is not None:
            client, port = probed
        else:
            client, port = None, self.config.effective_port(model)
        self._activate(model, profile, client, port)
        print(
            self.palette.ok(
                f"已载入 {len(self.session.messages)} 条消息 · 模型 {model.id}"
            )
        )
        return True

    def _cmd_exec(self, rest: str) -> bool:
        if not rest:
            print(self.palette.hint("用法：/exec <命令> 或 !<命令>"), file=sys.stderr)
            return True
        with self._state_lock:
            self._chat_active = True
        self._watchdog.pause()
        try:
            try:
                result = tools.run_exec(self.config, rest)
            except ToolDenied as exc:
                if str(exc) == tools.CANCEL_MESSAGE:
                    print(self.palette.meta(tools.CANCEL_MESSAGE))
                else:
                    print(self.palette.err(str(exc)), file=sys.stderr)
                return True
            if result.stdout:
                print(
                    result.stdout,
                    end="" if result.stdout.endswith("\n") else "\n",
                )
            if result.stderr:
                print(
                    result.stderr,
                    end="" if result.stderr.endswith("\n") else "\n",
                    file=sys.stderr,
                )
            print(self.palette.meta(f"退出码 {result.exit_code}"))
            return True
        finally:
            with self._state_lock:
                self._chat_active = False
            self._watchdog.resume()

    def _cmd_stop(self, rest: str) -> bool:
        try:
            pids = launcher.stop_all(self.config)
        except LaunchError as exc:
            print(self.palette.err(str(exc)), file=sys.stderr)
            return True
        self.client = None
        self.chat_runner = None
        self._sleeping = False
        self._watchdog.touch()
        if pids:
            print(self.palette.ok(f"已停止 {len(pids)} 个服务"))
            if self.verbose:
                print(
                    self.palette.meta("PID " + "、".join(str(pid) for pid in pids))
                )
        else:
            print(self.palette.meta("没有服务在运行"))
        return True

    def _cmd_status(self, rest: str) -> bool:
        detail = self.verbose or rest.strip() == "-v"
        print(self.palette.meta(f"模型 {self.model.id} · {self.model.display}"))
        print(
            self.palette.meta(
                f"档位 {self.profile.key} · {self.profile.desc}"
                f" · ctx {self.profile.ctx} · KV "
                f"{self.profile.kv_k}/{self.profile.kv_v}"
            )
        )
        if self.pending_profile is not None:
            print(
                self.palette.meta(
                    f"待生效档位 {self.pending_profile.key} · "
                    f"{self.pending_profile.desc}"
                )
            )
        ready = bool(self.client and self.client.health())
        state = self.palette.ok("就绪") if ready else self.palette.err("停止")
        head = self.palette.meta(f"端口 {self.port}")
        print(f"{head}  {state}")

        used = self.session.message_tokens()
        total = self.profile.ctx
        pct = (used / total * 100) if total > 0 else 0.0
        repl_settings = self.config.settings.repl
        bar = _context_bar(pct, repl_settings.status_bar_width)
        if pct >= repl_settings.ctx_danger_pct:
            bar_text = self.palette.error(bar)
        elif pct >= repl_settings.ctx_warn_pct:
            bar_text = self.palette.warning(bar)
        else:
            bar_text = self.palette.muted(bar)
        print(
            self.palette.meta(
                f"视窗监控 {bar_text} {pct:.1f}% · 已用 {used:,} / "
                f"上限 {total:,} tokens · 剩余 {max(0, total - used):,}"
            )
        )
        if self._sleeping:
            print(self.palette.meta("空闲休眠中；输入任意内容自动唤醒"))
        else:
            remaining = self._watchdog.remaining()
            if remaining is not None and self.client is not None:
                print(self.palette.meta(f"空闲倒计时 {remaining:.0f}s"))

        memory = guard.gpu_memory()
        if memory is not None:
            gpu_used, gpu_total = memory
            free = max(0, gpu_total - gpu_used)
            print(
                self.palette.meta(
                    f"显存物理 GPU 已用 {gpu_used:,} MiB / "
                    f"总量 {gpu_total:,} MiB · 余量 {free:,} MiB"
                )
            )
        else:
            gpu = guard.gpu_status()
            print(
                self.palette.meta(gpu)
                if gpu
                else self.palette.meta("GPU 信息不可用，nvidia-smi 未找到")
            )

        records = launcher.read_records()
        current = next(
            (
                record
                for record in records
                if record.model_id == self.model.id
                and launcher.pid_running(record.pid)
            ),
            None,
        )
        if current is not None:
            rss = guard.process_memory_kb(current.pid)
            rss_text = f"{rss / 1024:.0f} MiB" if rss is not None else "-"
            print(
                self.palette.meta(
                    f"系统进程 PID {current.pid} · RSS {rss_text} · "
                    f"存活 {_format_uptime(current.started_at)}"
                )
            )
        else:
            print(self.palette.meta("系统进程 - · 无当前模型运行记录"))
        if detail:
            for record in records:
                alive = launcher.pid_running(record.pid)
                print(
                    self.palette.meta(
                        f"{record.model_id} PID {record.pid} 端口 {record.port}"
                        f" 档位 {record.profile} {'运行' if alive else '停止'}"
                    )
                )

        if self.last_turn is not None:
            ttft = (
                f"{self.last_turn.ttft_ms:.0f} ms"
                if self.last_turn.ttft_ms is not None
                else "-"
            )
            print(
                self.palette.meta(
                    f"最近推理 TTFT {ttft} · 生成 {self.last_turn.tps:.1f} t/s"
                )
            )
        else:
            print(self.palette.meta("最近推理 TTFT - · 生成 -"))

        loop = getattr(self, "loop", None)
        if loop is not None:
            print(
                self.palette.meta(
                    f"Agent 步数配额 {loop.steps}/{loop.max_steps}"
                    f" · 熔断停滞计数 {loop.stagnation_count}/{loop.stagnation_limit}"
                )
            )
        else:
            print(
                self.palette.meta(
                    f"Agent 步数配额 -/{self.config.settings.agent.max_steps}"
                    " · 熔断停滞计数 -/3"
                )
            )
        rounds = sum(
            1 for item in self.session.messages if item.get("role") == "assistant"
        )
        print(
            self.palette.meta(
                f"会话 {len(self.session.messages)} 条消息 "
                f"({rounds} 轮交互) · 约 {used} tokens"
            )
        )
        return True

    def _cmd_shutdown(self, rest: str) -> bool:
        with self._state_lock:
            if self.session.is_dirty and self.session.messages:
                try:
                    saved = self.session.save(default_save_path())
                except ConfigError as exc:
                    print(self.palette.err(str(exc)), file=sys.stderr)
                    return True
                print(
                    self.palette.meta(
                        f"· 会话已自动保存至 {saved}"
                    )
                )
            try:
                pids = launcher.stop_all(self.config)
            except LaunchError as exc:
                print(self.palette.err(str(exc)), file=sys.stderr)
                return True
            self.client = None
            self.chat_runner = None
            self._sleeping = False
        if pids:
            print(
                self.palette.ok(
                    f"✓ 已停止后台推理服务 (PID "
                    f"{'、'.join(str(pid) for pid in pids)}) · 显存通道已完全释放"
                )
            )
        else:
            print(self.palette.meta("没有服务在运行"))
        self._watchdog.stop()
        return False

    def _cmd_quit(self, rest: str) -> bool:
        if rest.strip().lower() == "all":
            return self._cmd_shutdown("")
        if self.client is not None:
            print(
                self.palette.hint(
                    "后台服务仍驻留；完全退出请用 /shutdown 或 /exit all"
                )
            )
        return False

    # ------------------------------------------------------------------ #
    # prompt_toolkit（可选）
    # ------------------------------------------------------------------ #

    def _completion_candidates(self, text: str) -> list[tuple[str, int, str]]:
        """返回 (补全文本, start_position, 说明)；不依赖 prompt_toolkit。"""
        if not text.startswith("/"):
            return []
        if " " not in text:
            prefix = text.lower()
            candidates: list[tuple[str, int, str]] = []
            for command in COMMANDS:
                name = f"/{command.name}"
                if name.startswith(prefix):
                    candidates.append((name, -len(text), command.description))
                for alias in command.aliases:
                    alias_name = f"/{alias}"
                    if alias_name.startswith(prefix):
                        candidates.append(
                            (alias_name, -len(text), f"别名 → /{command.name}")
                        )
            return candidates

        body = text[1:]
        head, _, rest = body.partition(" ")
        if " " in rest:
            return []
        start = -len(rest) if rest else 0
        if head.lower() == "model":
            return [
                (model.id, start, model.display)
                for model in self.config.models
                if model.id.startswith(rest)
            ]
        if head.lower() == "profile":
            return [
                (key, start, self.model.profiles[key].desc)
                for key in self.model.profile_keys
                if key.startswith(rest)
            ]
        return []

    def _get_prompt_session(self):
        # D2：输入历史与业务会话分离；文件权限尽量收到 0600，失败只告警不阻断。
        if self._prompt_ready:
            return self._prompt_session
        self._prompt_ready = True
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.completion import Completer, Completion
            from prompt_toolkit.formatted_text import ANSI
            from prompt_toolkit.history import FileHistory
        except ImportError:
            print(
                self.palette.warn(
                    "未安装 prompt_toolkit，已降级为 input()；"
                    "pip install prompt_toolkit 可获得历史与行编辑"
                ),
                file=sys.stderr,
            )
            return None
        history = None
        path = Path(self.config.settings.repl.history_file).expanduser()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            history = FileHistory(str(path))
            if path.exists():
                os.chmod(path, 0o600)
        except OSError as exc:
            print(self.palette.warn(f"输入历史不可用：{exc}"), file=sys.stderr)

        class ReplCompleter(Completer):
            def __init__(self, repl: "Repl") -> None:
                self._repl = repl

            def get_completions(self, document, complete_event):
                for text, start, meta in self._repl._completion_candidates(
                    document.text_before_cursor
                ):
                    yield Completion(text, start_position=start, display_meta=meta)

        try:
            self._prompt_session = PromptSession(
                history=history, completer=ReplCompleter(self)
            )
            self._ansi_formatter = ANSI
        except (OSError, ValueError) as exc:
            print(
                self.palette.warn(f"prompt_toolkit 初始化失败，降级为 input()：{exc}"),
                file=sys.stderr,
            )
            self._prompt_session = None
        return self._prompt_session
