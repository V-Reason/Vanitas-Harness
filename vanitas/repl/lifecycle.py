"""REPL 生命周期命令实现：模型切换、档位切换、重启、停止、关机、加载。

每个函数接收 Repl 实例（鸭子类型），返回 bool（继续 / 退出）。
Repl._cmd_* 方法为薄转发；lifecycle 不 import Repl 类型。
"""
from __future__ import annotations

import sys
import time
from typing import TYPE_CHECKING

from .. import launcher, prompts
from ..client import LlamaClient
from ..context import Session, default_save_path
from ..errors import ConfigError, GuardError, LaunchError

if TYPE_CHECKING:
    from .loop import Repl


def switch_model(repl: Repl, rest: str) -> bool:
    if not rest:
        records = {record.model_id: record for record in launcher.read_records()}
        for item in repl.config.models:
            port = repl.config.effective_port(item)
            marker = repl.palette.accent("*") if item.id == repl.model.id else " "
            record = records.get(item.id)
            if record is not None and launcher.pid_running(record.pid):
                state = repl.palette.ok("运行")
            elif record is not None:
                state = repl.palette.warn("残留")
            else:
                state = repl.palette.err("停止")
            model_cell = repl.palette.bold(
                repl.palette.accent(f"{item.id:<10}")
            )
            port_cell = repl.palette.info(f"port {port:<5}")
            print(
                f" {marker} {model_cell} {item.display:<26} "
                f"{port_cell} {state}"
            )
        print(
            repl.palette.hint(
                "带 id 切换模型；目标没有运行时会自动停旧启新"
            )
        )
        return True
    try:
        model = repl.config.model(rest)
    except ConfigError as exc:
        print(repl.palette.err(str(exc)), file=sys.stderr)
        return True
    if model.id == repl.model.id:
        print(repl.palette.meta(f"已经是当前模型 {model.id}"))
        return True
    probed = repl._probe(model)
    if probed is not None:
        client, port = probed
        record = next(
            (item for item in launcher.read_records() if item.model_id == model.id),
            None,
        )
        try:
            profile = repl.config.resolve_profile(
                model, record.profile if record else None
            )
        except ConfigError as exc:
            print(repl.palette.err(str(exc)), file=sys.stderr)
            return True
        repl._activate(model, profile, client, port)
        repl._remember_model(model, profile)
        print(
            repl.palette.ok(
                f"已切换到 {model.display} · 档位 {profile.key}"
            )
        )
        return True

    pending = repl.pending_profile
    try:
        profile = (
            repl.config.resolve_profile(model, pending.key)
            if pending is not None and pending.key in model.profiles
            else repl.config.resolve_profile(model)
        )
    except ConfigError as exc:
        print(repl.palette.err(str(exc)), file=sys.stderr)
        return True

    print(repl.palette.spin("停止旧服务"))
    try:
        launcher.stop_all(repl.config)
    except LaunchError as exc:
        print(repl.palette.err(str(exc)), file=sys.stderr)
        return True
    repl.client = None
    repl.chat_runner = None
    print(repl.palette.spin(f"启动 {model.id}"))
    try:
        record = launcher.start(
            repl.config,
            model,
            profile,
            palette=repl.palette,
            verbose=repl.verbose,
        )
        launcher.wait_ready(repl.config, record, repl.palette)
    except GuardError as exc:
        repl._print_guard_error(exc)
        return True
    except LaunchError as exc:
        print(repl.palette.err(str(exc)), file=sys.stderr)
        return True
    repl._activate(
        model,
        profile,
        LlamaClient.for_model(repl.config, model, record.port),
        record.port,
    )
    repl._remember_model(model, profile)
    print(repl.palette.ok("已切换"))
    return True


def switch_profile(repl: Repl, rest: str) -> bool:
    if not rest:
        print(
            repl.palette.meta(
                f"当前档位 {repl.profile.key}：{repl.profile.desc}"
            )
        )
        if repl.pending_profile is not None:
            print(
                repl.palette.meta(
                    f"待生效档位 {repl.pending_profile.key}："
                    f"{repl.pending_profile.desc}"
                )
            )
        for key in repl.model.profile_keys:
            profile = repl.model.profiles[key]
            print(
                f"  [{key}] ctx {profile.ctx:<6}"
                f" KV {profile.kv_k}/{profile.kv_v}  {profile.desc}"
            )
        print(repl.palette.hint("修改档位只影响下次 serve/切换"))
        return True
    try:
        profile = repl.config.resolve_profile(repl.model, rest)
    except ConfigError as exc:
        print(repl.palette.err(str(exc)), file=sys.stderr)
        return True
    # 只改“待生效”档位：当前服务端的 ctx 不能凭命令幻觉改变。
    repl.pending_profile = profile
    try:
        repl.config.preferences.set_profile(repl.model.id, profile.key)
    except ConfigError as exc:
        print(repl.palette.err(str(exc)), file=sys.stderr)
        return True
    print(
        repl.palette.meta(
            f"档位 {profile.key} 已暂存，重启生效；偏好已写入"
        )
    )
    print(repl.palette.hint("执行 /restart 立即应用该档位"))
    return True


def restart_service(repl: Repl, rest: str) -> bool:
    if not repl._confirm_unsaved("重启"):
        return True
    target = repl.pending_profile or repl.profile
    with repl._state_lock:
        repl._chat_active = True
    repl._watchdog.pause()
    started = time.monotonic()
    try:
        try:
            repl.config.preferences.set_profile(repl.model.id, target.key)
        except ConfigError as exc:
            print(repl.palette.err(str(exc)), file=sys.stderr)
            return True
        print(repl.palette.spin("正在切换档位并重启服务"))
        try:
            launcher.stop_all(repl.config)
        except LaunchError as exc:
            print(repl.palette.err(str(exc)), file=sys.stderr)
            return True
        repl.client = None
        repl.chat_runner = None
        try:
            record = launcher.start(
                repl.config,
                repl.model,
                target,
                palette=repl.palette,
                verbose=repl.verbose,
            )
            launcher.wait_ready(repl.config, record, repl.palette)
        except GuardError as exc:
            repl._print_guard_error(exc)
            return True
        except LaunchError as exc:
            print(repl.palette.err(str(exc)), file=sys.stderr)
            return True
        repl._activate(
            repl.model,
            target,
            LlamaClient.for_model(repl.config, repl.model, record.port),
            record.port,
        )
        elapsed = time.monotonic() - started
        print(
            repl.palette.ok(
                f"✓ 服务端已重启 · 档位 {target.key} "
                f"({target.ctx} ctx + {target.kv_k}/{target.kv_v} KV) "
                f"· 耗时 {elapsed:.1f}s"
            )
        )
        return True
    finally:
        with repl._state_lock:
            repl._chat_active = False
        repl._watchdog.resume()


def stop_services(repl: Repl, rest: str) -> bool:
    try:
        pids = launcher.stop_all(repl.config)
    except LaunchError as exc:
        print(repl.palette.err(str(exc)), file=sys.stderr)
        return True
    repl.client = None
    repl.chat_runner = None
    repl._sleeping = False
    repl._watchdog.touch()
    if pids:
        print(repl.palette.ok(f"已停止 {len(pids)} 个服务"))
        if repl.verbose:
            print(
                repl.palette.meta("PID " + "、".join(str(pid) for pid in pids))
            )
    else:
        print(repl.palette.meta("没有服务在运行"))
    return True


def shutdown(repl: Repl, rest: str) -> bool:
    with repl._state_lock:
        if repl.session.is_dirty and repl.session.messages:
            try:
                saved = repl.session.save(default_save_path())
            except ConfigError as exc:
                print(repl.palette.err(str(exc)), file=sys.stderr)
                return True
            print(
                repl.palette.meta(
                    f"· 会话已自动保存至 {saved}"
                )
            )
        try:
            pids = launcher.stop_all(repl.config)
        except LaunchError as exc:
            print(repl.palette.err(str(exc)), file=sys.stderr)
            return True
        repl.client = None
        repl.chat_runner = None
        repl._sleeping = False
    if pids:
        print(
            repl.palette.ok(
                f"✓ 已停止后台推理服务 (PID "
                f"{'、'.join(str(pid) for pid in pids)}) · 显存通道已完全释放"
            )
        )
    else:
        print(repl.palette.meta("没有服务在运行"))
    repl._watchdog.stop()
    return False


def load_session(repl: Repl, rest: str) -> bool:
    if not rest:
        print(repl.palette.hint("用法：/load <path>"), file=sys.stderr)
        return True
    try:
        loaded = Session.load(rest)
    except ConfigError as exc:
        print(repl.palette.err(str(exc)), file=sys.stderr)
        return True
    repl.session = loaded
    if loaded.system == "":
        repl.session.system = prompts.system_prompt(repl.prompt_dir)

    model = repl.model
    if loaded.model_id:
        try:
            model = repl.config.model(loaded.model_id)
        except ConfigError as exc:
            print(repl.palette.warn(str(exc)), file=sys.stderr)

    profile = repl.profile
    if loaded.profile:
        try:
            profile = repl.config.resolve_profile(model, loaded.profile)
        except ConfigError as exc:
            print(repl.palette.warn(str(exc)), file=sys.stderr)

    probed = repl._probe(model)
    if probed is not None:
        client, port = probed
    else:
        client, port = None, repl.config.effective_port(model)
    repl._activate(model, profile, client, port)
    print(
        repl.palette.ok(
            f"已载入 {len(repl.session.messages)} 条消息 · 模型 {model.id}"
        )
    )
    return True
