"""REPL：命令注册表（范式笔记 D1）+ 交互循环。

只有「以 / 开头且第 2 个字符不是空格」的行才当命令；其余整行都是用户输入，
这样用户想问 "/model 是什么意思" 时仍能正常发问。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import guard, launcher, prompts, tools
from .config import AppConfig, ModelSpec, ProfileSpec
from .engine import EngineClient
from .errors import ConfigError, EngineError, GuardError, LaunchError, ToolDenied
from .render import StreamRenderer
from .session import Session, context_budget, default_save_path


@dataclass(frozen=True)
class Command:
    name: str
    description: str
    aliases: tuple = ()
    arg_hint: str = ""


# D1：命令是数据，不是散落的 if 分支；/help 由本表生成。
COMMANDS: tuple[Command, ...] = (
    Command("help", "列出内置命令"),
    Command("model", "列出模型 / 切换模型（不会自动启停服务端）", (), "[id]"),
    Command("profile", "查看或切换档位（仅对下次 serve 生效）", (), "[N]"),
    Command("system", "设置系统提示（文本或文件路径）", (), "[text|file]"),
    Command("task", "应用任务模板 config\\prompts\\task_<name>.md", (), "[name]"),
    Command("clear", "清空对话历史（保留系统提示）"),
    Command("save", "保存当前会话 JSON", (), "[path]"),
    Command("load", "加载会话 JSON", (), "<path>"),
    Command("exec", "在白名单内执行本地命令（逐次确认）", (), "<cmd>"),
    Command("stop", "停止所有 llama-server", ("kill",)),
    Command("status", "显示服务端 / 显存 / 当前模型"),
    Command("quit", "退出 REPL", ("exit",)),
)

COMMAND_MAP: dict[str, Command] = {}
for _command in COMMANDS:
    COMMAND_MAP[_command.name] = _command
    for _alias in _command.aliases:
        COMMAND_MAP[_alias] = _command


class Repl:
    def __init__(
        self,
        config: AppConfig,
        model: ModelSpec | None = None,
        profile: str | None = None,
        *,
        auto_start: bool = False,
    ) -> None:
        self.config = config
        self.auto_start = auto_start
        self.prompt_dir = config.config_dir / "prompts"
        default_model = model or config.models[0]
        self.profile: ProfileSpec = config.resolve_profile(default_model, profile)
        self.model: ModelSpec = default_model
        self.port: int = config.effective_port(default_model)
        self.client: EngineClient | None = None
        self.session = Session(
            model_id=default_model.id,
            profile=self.profile.key,
            system=prompts.system_prompt(self.prompt_dir),
        )
        self._prompt_session = None
        self._prompt_ready = False

    # ------------------------------------------------------------------ #
    # 主循环
    # ------------------------------------------------------------------ #

    def run(self) -> int:
        self._bootstrap()
        print("[HINT] /help 查看命令，/quit 退出")
        while True:
            try:
                line = self._read_line(self._prompt_text())
            except KeyboardInterrupt:
                print()
                continue
            except EOFError:
                print()
                break
            text = line.strip()
            if not text:
                continue
            if self._is_command(text):
                if not self._handle_command(text):
                    break
            else:
                self._chat(text)
        print("[BYE] 已退出")
        return 0

    @staticmethod
    def _is_command(text: str) -> bool:
        return len(text) > 1 and text.startswith("/") and text[1] != " "

    def _prompt_text(self) -> str:
        return f"vanitas[{self.model.id}:{self.profile.key}]> "

    def _read_line(self, prompt: str) -> str:
        interactive = sys.stdin.isatty()
        session = self._get_prompt_session() if interactive else None
        if session is not None:
            return session.prompt(prompt)
        if not interactive:
            # 非 TTY：提示符走 stderr，stdout 留给回答，便于管道与测试抓取
            sys.stderr.write(prompt)
            sys.stderr.flush()
        return input()

    # ------------------------------------------------------------------ #
    # 启动引导（D6）
    # ------------------------------------------------------------------ #

    def _bootstrap(self) -> None:
        if self._adopt_any():
            return
        if self.auto_start:
            self._start_engine(self.model, self.profile)
            return
        if not sys.stdin.isatty():
            print("[HINT] 非交互模式且没有就绪的服务端：先运行 vanitas serve <id>")
            return
        chosen = self._prompt_model()
        if chosen is None:
            return
        profile = self.config.resolve_profile(chosen)
        port = self.config.effective_port(chosen)
        answer = input(
            f"启动 {chosen.display}（档位 {profile.key}：{profile.desc}）"
            f"于端口 {port}？[Y/n] "
        ).strip().lower()
        if answer not in {"", "y", "yes", "是"}:
            print("[HINT] 已跳过启动；可稍后用 vanitas serve 启动，或 /model 切换")
            return
        self._start_engine(chosen, profile)

    def _start_engine(self, model: ModelSpec, profile: ProfileSpec) -> bool:
        try:
            record = launcher.start(self.config, model, profile)
            elapsed = launcher.wait_ready(self.config, record)
        except (GuardError, LaunchError) as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return False
        print(f"[SERVE] 就绪 {elapsed:.1f}s")
        self.model = model
        self.profile = profile
        self.port = record.port
        self.client = EngineClient.for_model(self.config, model, record.port)
        self.session.model_id = model.id
        self.session.profile = profile.key
        return True

    def _prompt_model(self) -> ModelSpec | None:
        print("可用模型：")
        for index, item in enumerate(self.config.models, start=1):
            print(f"  [{index}] {item.id:<10} {item.display}（端口 {self.config.effective_port(item)}）")
        try:
            raw = input("选择编号 [默认 1]：").strip()
        except EOFError:
            return None
        if not raw:
            return self.config.models[0]
        try:
            index = int(raw)
        except ValueError:
            print(f"[WARN] 编号必须是整数，收到 {raw!r}")
            return None
        if not 1 <= index <= len(self.config.models):
            print(f"[WARN] 编号超出 1–{len(self.config.models)}")
            return None
        return self.config.models[index - 1]

    def _probe(self, model: ModelSpec, port: int | None = None):
        target_port = port if port is not None else self.config.effective_port(model)
        client = EngineClient.for_model(self.config, model, target_port)
        try:
            if not client.health(timeout=2.0):
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
            self.model = model
            self.port = port
            self.client = client
            self.profile = self.config.resolve_profile(model, record.profile)
            self.session.model_id = model.id
            self.session.profile = self.profile.key
            print(f"[ENGINE] 已连接 {model.display} · {launcher.endpoint(self.config, port)} · 档位 {self.profile.key}")
            return True
        for model in self.config.models:
            probed = self._probe(model)
            if probed is None:
                continue
            client, port = probed
            self.model = model
            self.port = port
            self.client = client
            self.profile = self.config.resolve_profile(model)
            self.session.model_id = model.id
            self.session.profile = self.profile.key
            print(f"[ENGINE] 已连接 {model.display} · {launcher.endpoint(self.config, port)} · 档位 {self.profile.key}")
            return True
        return False

    # ------------------------------------------------------------------ #
    # 对话
    # ------------------------------------------------------------------ #

    def _chat(self, text: str) -> None:
        if self.client is None:
            print("[HINT] 服务端未就绪：先运行 vanitas serve <id>，或用 /model 查看可用模型")
            return
        budget = context_budget(
            self.profile.ctx, self.config.settings.repl.max_context_tokens
        )
        try:
            user_message = self.session.build_user_message(
                text, self.prompt_dir, self.model.alias
            )
        except ConfigError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return
        self.session.add_user(user_message)
        dropped = self.session.trim(budget)
        if dropped:
            print(f"[CTX] 预算 {budget} tokens，已裁剪 {dropped} 轮最早对话")

        max_tokens = max(128, min(1024, self.profile.ctx // 8))
        renderer = StreamRenderer()
        usage: dict | None = None
        reasoning_chars = 0
        try:
            for event in self.client.chat_stream(
                self.session.api_messages(),
                max_tokens=max_tokens,
                temperature=0.2,
            ):
                if event["type"] == "delta":
                    renderer.feed(event["content"])
                elif event["type"] == "reasoning":
                    reasoning_chars += len(event["content"])
                elif event["type"] == "usage":
                    usage = event["usage"]
        except EngineError as exc:
            print()
            print(f"[ERROR] {exc}", file=sys.stderr)
            self.session.pop_last_user()
            return
        except KeyboardInterrupt:
            print()
            print("[CTX] 本轮已中断，未写入历史")
            self.session.pop_last_user()
            return

        meta = renderer.close()
        answer = renderer.text
        if answer.strip():
            self.session.add_assistant(answer)

        pieces: list[str] = []
        tokens = int((usage or {}).get("completion_tokens") or 0)
        pieces.append(f"{tokens} tokens" if tokens else f"{meta['chars']} 字符")
        seconds = meta["wall_ms"] / 1000.0
        if seconds > 0:
            pieces.append(f"{seconds:.1f}s")
        if meta["first_ms"] is not None:
            pieces.append(f"首字 {meta['first_ms']:.0f} ms")
        if tokens and seconds > 0:
            pieces.append(f"{tokens / seconds:.1f} t/s")
        print(f"[META] {' · '.join(pieces)}")
        if not answer.strip() and reasoning_chars:
            print(
                "[HINT] 仅返回 reasoning_content，content 为空："
                "检查模型的 chat_template_kwargs（enable_thinking）或提高 max_tokens",
                file=sys.stderr,
            )

    # ------------------------------------------------------------------ #
    # 命令分派
    # ------------------------------------------------------------------ #

    def _handle_command(self, line: str) -> bool:
        name, _, rest = line[1:].partition(" ")
        command = COMMAND_MAP.get(name.lower())
        if command is None:
            print(f"[HINT] 未知命令 /{name}；输入 /help 查看全部")
            return True
        handler: Callable[[str], bool] = getattr(self, f"_cmd_{command.name}")
        return handler(rest.strip())

    def _cmd_help(self, rest: str) -> bool:
        print("内置命令：")
        for command in COMMANDS:
            head = f"/{command.name}" + (f" {command.arg_hint}" if command.arg_hint else "")
            print(f"  {head:<24}{command.description}")
        return True

    def _cmd_model(self, rest: str) -> bool:
        if not rest:
            records = {record.model_id: record for record in launcher.read_records()}
            for item in self.config.models:
                port = self.config.effective_port(item)
                marker = "*" if item.id == self.model.id else " "
                record = records.get(item.id)
                if record is not None and launcher.pid_running(record.pid):
                    state = f"运行中 pid={record.pid} 档位 {record.profile}"
                elif record is not None:
                    state = "残留记录（进程已退出）"
                else:
                    state = "未运行"
                print(f" {marker} {item.id:<10} {item.display:<26} port {port:<5} {state}")
            print("[HINT] 切换模型只切换连接，不会自动启停服务端")
            return True
        try:
            model = self.config.model(rest)
        except ConfigError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return True
        if model.id == self.model.id:
            print(f"[MODEL] 已经是当前模型 {model.id}")
            return True
        probed = self._probe(model)
        if probed is None:
            print(
                f"[HINT] {model.display} 未就绪；先运行：vanitas serve {model.id} [--profile N]"
            )
            return True
        client, port = probed
        record = next(
            (item for item in launcher.read_records() if item.model_id == model.id), None
        )
        self.model = model
        self.port = port
        self.client = client
        self.profile = self.config.resolve_profile(model, record.profile if record else None)
        self.session.model_id = model.id
        self.session.profile = self.profile.key
        print(f"[MODEL] 已切换到 {model.display} · {launcher.endpoint(self.config, port)} · 档位 {self.profile.key}")
        return True

    def _cmd_profile(self, rest: str) -> bool:
        if not rest:
            print(f"[PROFILE] 当前 {self.profile.key}：{self.profile.desc}")
            for key in self.model.profile_keys:
                profile = self.model.profiles[key]
                print(f"  [{key}] ctx {profile.ctx:<6} KV {profile.kv_k}/{profile.kv_v}  {profile.desc}")
            print("[HINT] 修改档位只影响下次 serve")
            return True
        try:
            profile = self.config.resolve_profile(self.model, rest)
        except ConfigError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return True
        self.profile = profile
        self.session.profile = profile.key
        print(f"[PROFILE] 已设为 {profile.key}（{profile.desc}）；下次 serve 生效")
        return True

    def _cmd_system(self, rest: str) -> bool:
        if not rest:
            current = self.session.system.strip()
            print(f"[SYSTEM] {current if current else '（空：使用模型默认行为）'}")
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
                print(f"[ERROR] 系统提示读取失败：{candidate}（{exc.strerror or exc}）", file=sys.stderr)
                return True
        else:
            text = rest
        self.session.system = text.strip()
        print(f"[SYSTEM] 已设置 {len(self.session.system)} 字符")
        return True

    def _cmd_task(self, rest: str) -> bool:
        names = prompts.task_names(self.prompt_dir)
        if not rest:
            print(f"[TASK] 当前：{self.session.task or '（未应用）'}；可用：{'、'.join(names) or '无'}")
            return True
        try:
            prompts.load_task(self.prompt_dir, rest)
        except ConfigError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return True
        self.session.task = rest
        print(f"[TASK] 已应用 task_{rest}.md；每轮输入会填入 {{input}}")
        return True

    def _cmd_clear(self, rest: str) -> bool:
        self.session.clear_messages()
        print("[CLEAR] 对话历史已清空（系统提示与任务模板保留）")
        return True

    def _cmd_save(self, rest: str) -> bool:
        target = Path(rest).expanduser() if rest else default_save_path()
        try:
            saved = self.session.save(target)
        except ConfigError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return True
        print(f"[SAVE] {saved}（{len(self.session.messages)} 条消息）")
        return True

    def _cmd_load(self, rest: str) -> bool:
        if not rest:
            print("[ERROR] 用法：/load <path>", file=sys.stderr)
            return True
        try:
            loaded = Session.load(rest)
        except ConfigError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return True
        self.session = loaded
        if loaded.system == "":
            self.session.system = prompts.system_prompt(self.prompt_dir)
        if loaded.model_id:
            try:
                model = self.config.model(loaded.model_id)
            except ConfigError as exc:
                print(f"[WARN] {exc}", file=sys.stderr)
            else:
                self.model = model
                probed = self._probe(model)
                if probed is not None:
                    self.client, self.port = probed
                else:
                    self.client = None
        if loaded.profile:
            try:
                self.profile = self.config.resolve_profile(self.model, loaded.profile)
            except ConfigError as exc:
                print(f"[WARN] {exc}", file=sys.stderr)
        print(f"[LOAD] 已载入 {len(self.session.messages)} 条消息（模型 {self.model.id}）")
        return True

    def _cmd_exec(self, rest: str) -> bool:
        if not rest:
            print("[ERROR] 用法：/exec <命令>", file=sys.stderr)
            return True
        try:
            result = tools.run_exec(self.config, rest)
        except ToolDenied as exc:
            print(f"[DENY] {exc}", file=sys.stderr)
            return True
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        if result.stderr:
            print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)
        print(f"[EXEC] 退出码 {result.exit_code}")
        return True

    def _cmd_stop(self, rest: str) -> bool:
        try:
            pids = launcher.stop_all(self.config)
        except LaunchError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return True
        self.client = None
        if pids:
            print(f"[STOP] 已结束 PID {', '.join(str(pid) for pid in pids)}")
        else:
            print("[STOP] 没有运行中的 llama-server")
        return True

    def _cmd_status(self, rest: str) -> bool:
        print(f"[STATUS] 模型 {self.model.id} · {self.model.display}")
        print(
            f"[STATUS] 档位 {self.profile.key}（{self.profile.desc}）"
            f" · ctx {self.profile.ctx} · KV {self.profile.kv_k}/{self.profile.kv_v}"
        )
        ready = bool(self.client and self.client.health(timeout=2.0))
        print(
            f"[STATUS] 端口 {self.port} · {launcher.endpoint(self.config, self.port)}"
            f" · {'就绪' if ready else '未就绪'}"
        )
        records = launcher.read_records()
        if records:
            for record in records:
                alive = launcher.pid_running(record.pid)
                print(
                    f"[STATUS] 记录 {record.model_id} pid={record.pid} port={record.port}"
                    f" 档位 {record.profile} {'存活' if alive else '已退出'}"
                )
        else:
            print("[STATUS] 无运行记录（serve 会在 ~/.vanitas_run/ 下写入）")
        gpu = guard.gpu_status()
        print(f"[STATUS] {gpu}" if gpu else "[STATUS] GPU 信息不可用（nvidia-smi 未找到）")
        print(
            f"[STATUS] 会话 {len(self.session.messages)} 条消息 · 约 {self.session.message_tokens()} tokens"
        )
        return True

    def _cmd_quit(self, rest: str) -> bool:
        return False

    # ------------------------------------------------------------------ #
    # prompt_toolkit（可选）
    # ------------------------------------------------------------------ #

    def _get_prompt_session(self):
        # D2：输入历史与业务会话分离；文件权限尽量收到 0600，失败只告警不阻断。
        if self._prompt_ready:
            return self._prompt_session
        self._prompt_ready = True
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.history import FileHistory
        except ImportError:
            print(
                "[WARN] 未安装 prompt_toolkit，已降级为 input()："
                "pip install prompt_toolkit 可获得历史与行编辑",
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
            print(f"[WARN] 输入历史不可用：{exc}", file=sys.stderr)
        try:
            self._prompt_session = PromptSession(history=history)
        except (OSError, ValueError) as exc:
            print(f"[WARN] prompt_toolkit 初始化失败，降级为 input()：{exc}", file=sys.stderr)
            self._prompt_session = None
        return self._prompt_session
